"""
pipeline.py
End-to-end Computational Global Macro pipeline.

Chains: Scenario Engine → Causal Filter → View Generator → Portfolio Optimizer

Usage
-----
    python pipeline.py

Or import and run programmatically:

    from pipeline import ComputationalGlobalMacroPipeline
    pipe = ComputationalGlobalMacroPipeline(config)
    result = pipe.run()
    result.report()
"""

from __future__ import annotations
import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import numpy as np
import json
from dataclasses import dataclass, field

from cgm.engines.cmab import (CMABSimulation, UCBAgent, ThompsonAgent,
                           Arm, Context)
from cgm.engines.qlearning import (TwoPlayerSimulation, QLearningAgent,
                                SARSAAgent, make_war_of_attrition,
                                make_trade_war)
from cgm.engines.causal import (NoisyOR, Cause, MacroView, ViewAggregator)
from cgm.portfolio.optimizer import (BlackLitterman, OrdinalBL, RobustMVO,
                                 causal_lp_trades, PortfolioResult)


# ---------------------------------------------------------------------------
# Pipeline config
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """
    All parameters needed to run the full CGM pipeline.
    Override any defaults to model a specific geopolitical scenario.
    """

    # --- Assets ---
    assets: list[str] = field(default_factory=lambda: [
        "US_Equity", "EU_Equity", "EM_Equity",
        "US_Bond", "Commodities"
    ])

    # --- Market data (use historical estimates or live data in production) ---
    hist_returns: list[float] = field(default_factory=lambda:
        [0.07, 0.05, 0.06, 0.03, 0.04])
    sigma_diagonal: list[float] = field(default_factory=lambda:
        [0.18, 0.20, 0.25, 0.06, 0.22])
    correlations: list[list[float]] = field(default_factory=lambda: [
        [1.00, 0.72, 0.65, -0.20,  0.15],
        [0.72, 1.00, 0.70, -0.18,  0.12],
        [0.65, 0.70, 1.00, -0.10,  0.25],
        [-0.20,-0.18,-0.10,  1.00, -0.05],
        [0.15, 0.12, 0.25, -0.05,  1.00],
    ])
    market_weights: list[float] = field(default_factory=lambda:
        [0.40, 0.20, 0.15, 0.20, 0.05])

    # --- BL parameters ---
    risk_aversion: float = 2.5
    tau: float = 0.025

    # --- Simulation parameters ---
    n_rounds_cmab: int = 500
    n_iters_cmab: int = 300
    n_rounds_ql: int = 30
    n_iters_ql: int = 500

    # --- Causal thresholds ---
    bull_threshold: float = 0.55
    bear_threshold: float = 0.45

    # --- Optimization bounds ---
    w_min: float = 0.02
    w_max: float = 0.40


# ---------------------------------------------------------------------------
# Pipeline result container
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    sim_cmab: dict
    sim_qlearn: dict
    nor_probs: dict[str, float]
    views: list[MacroView]
    bl_result: PortfolioResult
    obl_result: PortfolioResult
    robust_result: PortfolioResult
    lp_trades: dict[str, float]
    assets: list[str]

    def report(self):
        sep = "─" * 60
        print(f"\n{'═'*60}")
        print("  COMPUTATIONAL GLOBAL MACRO — PIPELINE REPORT")
        print(f"{'═'*60}\n")

        print("1. SCENARIO ENGINE")
        print(sep)
        ba = self.sim_cmab.get("best_arm_name", "N/A")
        print(f"   CMAB best long-run action : {ba}")
        print(f"   Q-learning avg reward P1  : "
              f"{self.sim_qlearn['cum_avg_p1'][-1]:.3f}")
        print(f"   Q-learning avg reward P2  : "
              f"{self.sim_qlearn['cum_avg_p2'][-1]:.3f}")
        print(f"   Best action P1 (final Q)  : "
              f"{_best_action(self.sim_qlearn['final_Q1'])}")
        print(f"   Best action P2 (final Q)  : "
              f"{_best_action(self.sim_qlearn['final_Q2'])}")

        print(f"\n2. CAUSAL LAYER (Noisy-OR)")
        print(sep)
        for asset, p in self.nor_probs.items():
            bar = "█" * int(p * 20) + "░" * (20 - int(p * 20))
        for asset, p in self.nor_probs.items():
            bar = "▓" * int(p * 20) + "░" * (20 - int(p * 20))
            print(f"   {asset:<14} P(positive return) = {p:.3f}  [{bar}]")

        print(f"\n3. MACRO VIEWS")
        print(sep)
        if self.views:
            for v in self.views:
                arrow = "▲" if v.direction > 0 else "▼"
                print(f"   {arrow} {v.asset:<14} "
                      f"confidence={v.confidence:.2f}  source={v.source}")
        else:
            print("   No views exceeded threshold — holding benchmark.")

        print(f"\n4. PORTFOLIO OPTIMIZATION")
        print(sep)
        for label, res in [("Black-Litterman", self.bl_result),
                            ("Ordinal BL", self.obl_result),
                            ("Robust MVO", self.robust_result)]:
            print(f"\n   [{label}]")
            for asset, w in zip(self.assets, res.weights):
                bar = "█" * int(abs(w) * 40)
                print(f"     {asset:<14} {w:+.1%}  {bar}")
            print(f"     Exp Return: {res.expected_return:.2%}  "
                  f"Vol: {res.volatility:.2%}  "
                  f"Sharpe: {res.sharpe:.2f}")

        print(f"\n5. CAUSAL LP TRADES")
        print(sep)
        if self.lp_trades:
            for asset, size in sorted(self.lp_trades.items(),
                                      key=lambda x: -abs(x[1])):
                arrow = "BUY " if size > 0 else "SELL"
                print(f"   {arrow} {asset:<14} {abs(size):.1%}")
        else:
            print("   LP solver did not converge — check inputs.")
        print(f"\n{'═'*60}\n")

    def to_dict(self) -> dict:
        return {
            "nor_probabilities": self.nor_probs,
            "views": [{"asset": v.asset, "direction": v.direction,
                       "confidence": v.confidence} for v in self.views],
            "bl_weights": dict(zip(self.assets, self.bl_result.weights.tolist())),
            "obl_weights": dict(zip(self.assets, self.obl_result.weights.tolist())),
            "robust_weights": dict(zip(self.assets,
                                       self.robust_result.weights.tolist())),
            "lp_trades": self.lp_trades,
        }


def _best_action(Q: np.ndarray) -> str:
    best = np.unravel_index(np.argmax(Q), Q.shape)
    return f"state={best[0]}, action={best[1]} (Q={Q[best]:.3f})"


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class ComputationalGlobalMacroPipeline:
    """
    Orchestrates the full four-layer pipeline:

        Layer 1: Scenario Engine  (CMAB + Q-learning simulations)
        Layer 2: Causal Filter    (Noisy-OR per asset)
        Layer 3: View Generator   (MacroView objects)
        Layer 4: Portfolio        (BL, OBL, Robust MVO, LP trades)

    Parameters
    ----------
    config : PipelineConfig (or None to use defaults)
    """

    def __init__(self, config: PipelineConfig | None = None):
        self.cfg = config or PipelineConfig()
        self._build_sigma()

    def _build_sigma(self):
        """Construct covariance matrix from correlations and volatilities."""
        vols = np.array(self.cfg.sigma_diagonal)
        corr = np.array(self.cfg.correlations)
        self.sigma = np.outer(vols, vols) * corr

    # ------------------------------------------------------------------
    # Layer 1: Scenario Engine
    # ------------------------------------------------------------------

    def run_cmab(self) -> dict:
        """
        Run CMAB simulation modeling geopolitical conflict (Ch 2).
        Default: Run-Pass game with Strong/Weak Passer contexts.
        """
        cfg = self.cfg
        strong_passer = Context(
            arms=[Arm("Pass", 10.0, 0.70), Arm("Run", 5.0, 0.80)],
            prior=0.30, label="Strong Passer"
        )
        weak_passer = Context(
            arms=[Arm("Pass", 3.0, 0.40), Arm("Run", 2.0, 0.30)],
            prior=0.70, label="Weak Passer"
        )
        agent = UCBAgent(n_arms=2, c=1.0)
        sim = CMABSimulation(
            contexts=[strong_passer, weak_passer],
            agent=agent,
            n_rounds=cfg.n_rounds_cmab,
            n_iters=cfg.n_iters_cmab,
        )
        return sim.run()

    def run_qlearning(self) -> dict:
        """
        Run two-player Q-learning simulation (Ch 4).
        Default: War of Attrition between two central banks.
        Player 1 = Q-learning (Fed), Player 2 = SARSA (ECB).
        """
        cfg = self.cfg
        spec = make_war_of_attrition()
        p1 = QLearningAgent(spec.n_states, spec.n_actions_p1,
                            alpha=0.1, gamma=0.9, epsilon=0.1)
        p2 = SARSAAgent(spec.n_states, spec.n_actions_p2,
                        alpha=0.1, gamma=0.9, epsilon=0.1)
        sim = TwoPlayerSimulation(
            spec=spec, agent1=p1, agent2=p2,
            n_rounds=cfg.n_rounds_ql,
            n_iters=cfg.n_iters_ql,
        )
        return sim.run()

    # ------------------------------------------------------------------
    # Layer 2: Causal Filter
    # ------------------------------------------------------------------

    def build_nor_models(self) -> dict[str, NoisyOR]:
        """
        Build one Noisy-OR model per asset encoding expert causal beliefs
        about whether each asset will deliver positive returns.

        In production: load these from a database or config file.
        Here: sensible defaults representing a "hawkish Fed / geopolitical
        tension" macro environment.
        """
        return {
            "US_Equity": NoisyOR([
                Cause("Strong US earnings growth",     0.55, "positive"),
                Cause("AI productivity tailwind",      0.50, "positive"),
                Cause("Fed rate cuts ahead",           0.35, "positive"),
                Cause("Geopolitical risk premium",     0.40, "negative"),
                Cause("High valuations",               0.45, "negative"),
            ], effect_label="US equity positive return"),

            "EU_Equity": NoisyOR([
                Cause("ECB dovish pivot",              0.45, "positive"),
                Cause("Europe earnings recovery",      0.35, "positive"),
                Cause("Russia-Ukraine drag",           0.55, "negative"),
                Cause("EUR weakness headwind",         0.30, "negative"),
                Cause("Energy price shock risk",       0.40, "negative"),
            ], effect_label="EU equity positive return"),

            "EM_Equity": NoisyOR([
                Cause("China reopening momentum",      0.40, "positive"),
                Cause("Commodity cycle upturn",        0.45, "positive"),
                Cause("USD strength headwind",         0.50, "negative"),
                Cause("EM political instability",      0.35, "negative"),
                Cause("Fed tightening spillover",      0.45, "negative"),
            ], effect_label="EM equity positive return"),

            "US_Bond": NoisyOR([
                Cause("Recession / flight-to-quality", 0.40, "positive"),
                Cause("Fed cutting cycle",             0.35, "positive"),
                Cause("Inflation persistence",         0.50, "negative"),
                Cause("US fiscal deficit expansion",   0.45, "negative"),
            ], effect_label="US bond positive return"),

            "Commodities": NoisyOR([
                Cause("OPEC supply discipline",        0.50, "positive"),
                Cause("China demand recovery",         0.45, "positive"),
                Cause("Supply chain disruptions",      0.40, "positive"),
                Cause("Global growth slowdown",        0.35, "negative"),
                Cause("Strong USD",                    0.40, "negative"),
            ], effect_label="Commodities positive return"),
        }

    # ------------------------------------------------------------------
    # Layer 3: View Generator
    # ------------------------------------------------------------------

    def generate_views(self, nor_models: dict[str, NoisyOR]) -> list[MacroView]:
        cfg = self.cfg
        aggregator = ViewAggregator(
            assets=cfg.assets,
            threshold_bull=cfg.bull_threshold,
            threshold_bear=cfg.bear_threshold,
        )
        return aggregator.from_nor(nor_models)

    # ------------------------------------------------------------------
    # Layer 4: Portfolio
    # ------------------------------------------------------------------

    def run_portfolio(self, views: list[MacroView]) -> tuple[
            PortfolioResult, PortfolioResult, PortfolioResult, dict]:
        cfg = self.cfg
        assets = cfg.assets
        mu = np.array(cfg.hist_returns)
        w_mkt = np.array(cfg.market_weights)

        # 4a. Black-Litterman
        bl = BlackLitterman(assets, self.sigma, w_mkt,
                            cfg.risk_aversion, cfg.tau)
        bl_result = bl.optimize(views, cfg.w_min, cfg.w_max)

        # 4b. Ordinal BL — derive ranks from NOR probabilities
        nor_probs = {a: 0.5 for a in assets}  # fallback
        for v in views:
            nor_probs[v.asset] = v.confidence if v.direction > 0 \
                                 else 1.0 - v.confidence
        return_ranks  = np.array([nor_probs[a] for a in assets])
        variance_ranks = 1.0 - np.array(cfg.sigma_diagonal)  # prefer low vol
        obl = OrdinalBL(assets, mu, self.sigma, w_mkt, cfg.tau)
        obl_result = obl.optimize(return_ranks, variance_ranks,
                                  w_min=cfg.w_min, w_max=cfg.w_max)

        # 4c. Robust MVO — estimation error = 1 std-error of mean return
        n_years = 10
        estimation_error = np.array(cfg.sigma_diagonal) / np.sqrt(n_years)
        bl_mu, _ = bl.posterior(views) if views else (mu, self.sigma)
        robust = RobustMVO(assets, self.sigma, cfg.risk_aversion,
                           estimation_error)
        robust_result = robust.optimize(bl_mu, cfg.w_min, cfg.w_max)

        # 4d. Causal LP trades (Ch 6.2) — example with 3 scenarios
        causal_distances = {"G1": 0.66, "G2": 0.83, "G3": 0.50}
        # Assign buy/sell assets per scenario based on views
        view_dirs = {v.asset: v.direction for v in views}
        sorted_by_conviction = sorted(
            cfg.assets,
            key=lambda a: abs(view_dirs.get(a, 0.0)),
            reverse=True
        )
        buy_pool  = [a for a in sorted_by_conviction
                     if view_dirs.get(a, 0) > 0][:3]
        sell_pool = [a for a in sorted_by_conviction
                     if view_dirs.get(a, 0) < 0][:3]
        # Pad to length 3 if needed
        while len(buy_pool) < 3: buy_pool.append(buy_pool[-1] if buy_pool else cfg.assets[0])
        while len(sell_pool) < 3: sell_pool.append(sell_pool[-1] if sell_pool else cfg.assets[-1])

        lp_trades = causal_lp_trades(
            causal_distances,
            buy_pool[:3], sell_pool[:3],
            total_budget=0.05,
        )

        return bl_result, obl_result, robust_result, lp_trades

    # ------------------------------------------------------------------
    # Run full pipeline
    # ------------------------------------------------------------------

    def run(self) -> PipelineResult:
        print("Running Layer 1: Scenario Engine (CMAB)...")
        sim_cmab = self.run_cmab()

        print("Running Layer 1: Scenario Engine (Q-learning)...")
        sim_ql = self.run_qlearning()

        print("Running Layer 2: Causal Filter (Noisy-OR)...")
        nor_models = self.build_nor_models()
        nor_probs = {a: nor_models[a].net_probability()
                     for a in nor_models}

        print("Running Layer 3: View Generator...")
        views = self.generate_views(nor_models)

        print("Running Layer 4: Portfolio Optimization...")
        bl_r, obl_r, robust_r, lp_trades = self.run_portfolio(views)

        return PipelineResult(
            sim_cmab=sim_cmab,
            sim_qlearn=sim_ql,
            nor_probs=nor_probs,
            views=views,
            bl_result=bl_r,
            obl_result=obl_r,
            robust_result=robust_r,
            lp_trades=lp_trades,
            assets=self.cfg.assets,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pipe = ComputationalGlobalMacroPipeline()
    result = pipe.run()
    result.report()

    # Optionally export to JSON
    out_path = os.path.join(os.path.dirname(__file__), "pipeline_output.json")
    with open(out_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"JSON output written to: {out_path}")