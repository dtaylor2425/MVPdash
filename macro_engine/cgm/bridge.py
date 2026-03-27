"""
cgm/bridge.py
═══════════════════════════════════════════════════════════════════════════════
Bridge between the Macro Engine and the Computational Global Macro pipeline.

This module translates your live FRED/regime data into the data structures
the CGM pipeline expects, so the pipeline never needs hardcoded defaults.

What gets translated:
  • Regime components (z-scores, contributions) → Noisy-OR cause probabilities
  • Regime score + label → geopolitical scenario context for CMAB
  • Asset return / vol / correlation → PipelineConfig market data fields
  • Macro signal directions → MacroView objects (feeding BL directly)

Sign convention carried through from regime.py:
  Higher z-score on credit  = spreads widening  = BEARISH (negative cause)
  Higher z-score on curve   = steeper curve     = BULLISH for cyclicals
  Higher z-score on real_yl = higher real yield = BEARISH for equities/GLD
  Higher z-score on dollar  = stronger dollar   = BEARISH for commodities/EM
  Higher z-score on risk_ap = better breadth    = BULLISH
"""

from __future__ import annotations
import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any

from cgm.engines.causal import NoisyOR, Cause, MacroView, ViewAggregator
from cgm.pipeline import PipelineConfig


# ── Asset universe that the CGM pipeline will optimise ───────────────────────
# These map to ETF proxies your engine already tracks.
# The BL model operates on these 7 broad exposures.
CGM_ASSETS = [
    "US_Equity",     # proxy: SPY
    "Growth_Equity", # proxy: QQQ
    "SmallCap",      # proxy: IWM
    "Long_Bond",     # proxy: TLT
    "HY_Credit",     # proxy: HYG
    "Gold",          # proxy: GLD
    "Copper",        # proxy: CPER — industrial cycle / Dr. Copper macro signal
]

# ETF proxy map: CGM asset → ticker in your px DataFrame
PROXY_MAP = {
    "US_Equity":     "SPY",
    "Growth_Equity": "QQQ",
    "SmallCap":      "IWM",
    "Long_Bond":     "TLT",
    "HY_Credit":     "HYG",
    "Gold":          "GLD",
    "Copper":        "CPER",
}

# Human-friendly display labels for the pipeline UI
ASSET_LABELS = {
    "US_Equity":     "Large Cap",
    "Growth_Equity": "Growth",
    "SmallCap":      "Small Cap",
    "Long_Bond":     "Long Bond",
    "HY_Credit":     "High Yield",
    "Gold":          "Gold",
    "Copper":        "Copper",
}

# Default market-cap-style benchmark weights for the 7 assets.
# Calibrated to a typical 60/40-ish global macro portfolio:
# US equity heavy, moderate bond allocation, smaller alternatives.
# These drive the BL implied equilibrium returns — realistic weights
# prevent BL from producing extreme bond-heavy allocations.
DEFAULT_MKT_WEIGHTS = [0.35, 0.18, 0.12, 0.12, 0.08, 0.08, 0.07]
#                      SPY   QQQ   IWM   TLT   HYG   GLD  CPER (Energy)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Extract cause probabilities from regime components
# ─────────────────────────────────────────────────────────────────────────────

def _z_to_prob(z: float | None, invert: bool = False) -> float:
    """
    Convert a z-score to a probability in [0.10, 0.90] via a logistic curve.
    This prevents extreme z-scores from producing trivial 0/1 probabilities
    and keeps the Noisy-OR model well-conditioned.

    invert=True means a higher z-score is a NEGATIVE cause (e.g. real yields:
    higher = more restrictive = bearish for most assets).
    """
    if z is None:
        return 0.50  # no information → 50/50
    z = float(z)
    if invert:
        z = -z
    prob = 1.0 / (1.0 + np.exp(-0.6 * z))   # logistic with moderate slope
    return float(np.clip(prob, 0.10, 0.90))


def build_nor_models_from_regime(
    components: dict[str, dict],
    macro: pd.DataFrame,
    cur_score: int,
) -> dict[str, NoisyOR]:
    """
    Build one NoisyOR model per CGM asset using your regime component z-scores
    as the underlying cause probabilities.

    The book (Ch 5) requires each cause to have a name, probability, and
    direction (+/-). We derive these directly from your engine's live z-scores,
    meaning the causal DAG is updated every time your FRED cache refreshes.

    Parameters
    ----------
    components  : regime.components dict from compute_regime_v3()
    macro       : FRED macro DataFrame (for reading raw levels)
    cur_score   : current regime score (0-100)

    Returns
    -------
    dict mapping each CGM asset name to a NoisyOR model
    """
    # Extract component z-scores with safe fallback
    def _z(key: str) -> float | None:
        c = components.get(key, {})
        return c.get("zscore") if isinstance(c, dict) else None

    def _roc(key: str) -> float | None:
        c = components.get(key, {})
        return c.get("roc_zscore") if isinstance(c, dict) else None

    credit_z  = _z("credit")
    real_z    = _z("real_yields")
    real_roc  = _roc("real_yields")
    curve_z   = _z("curve")
    dollar_z  = _z("dollar")
    risk_z    = _z("risk_appetite")
    infl_z    = _z("cpi_momentum")

    # Regime itself as a probability of being in a good macro state
    regime_bull_p = float(np.clip(cur_score / 100.0, 0.10, 0.90))
    regime_bear_p = float(np.clip(1.0 - cur_score / 100.0, 0.10, 0.90))

    # ── US Equity (SPY) ───────────────────────────────────────────────────────
    # Bullish: good regime, tight credit, accommodative real yields
    # Bearish: restrictive real yields, dollar headwind, credit stress
    us_eq = NoisyOR([
        Cause("Bullish macro regime",
              _z_to_prob(cur_score / 10.0 - 5.0),         "positive"),
        Cause("Credit spreads tightening",
              _z_to_prob(credit_z, invert=True),           "positive"),
        Cause("Positive breadth (IWM/SPY)",
              _z_to_prob(risk_z),                          "positive"),
        Cause("Restrictive real yields",
              _z_to_prob(real_z),                          "negative"),
        Cause("Dollar headwind",
              _z_to_prob(dollar_z),                        "negative"),
        Cause("Credit stress",
              _z_to_prob(credit_z),                        "negative"),
    ], effect_label="US_Equity positive return", leak_prob=0.05)

    # ── Growth Equity (QQQ) ───────────────────────────────────────────────────
    # QQQ is the purest real yield play — duration sensitivity is highest here
    growth_eq = NoisyOR([
        Cause("Accommodative real yields",
              _z_to_prob(real_z, invert=True),             "positive"),
        Cause("Falling real yield trend",
              _z_to_prob(real_roc, invert=True) if real_roc else 0.50,
                                                           "positive"),
        Cause("Risk-on regime",
              _z_to_prob(cur_score / 10.0 - 5.0),         "positive"),
        Cause("Rising real yields",
              _z_to_prob(real_z),                          "negative"),
        Cause("Flattening/inverting curve",
              _z_to_prob(curve_z, invert=True),            "negative"),
        Cause("Credit widening",
              _z_to_prob(credit_z),                        "negative"),
    ], effect_label="Growth_Equity positive return", leak_prob=0.05)

    # ── Small Cap (IWM) ───────────────────────────────────────────────────────
    # Cyclical, needs steep curve + risk-on + tight credit
    small_cap = NoisyOR([
        Cause("Steep yield curve",
              _z_to_prob(curve_z),                         "positive"),
        Cause("Risk appetite (breadth)",
              _z_to_prob(risk_z),                          "positive"),
        Cause("Bullish macro regime",
              _z_to_prob(cur_score / 10.0 - 5.0),         "positive"),
        Cause("Restrictive real yields",
              _z_to_prob(real_z),                          "negative"),
        Cause("Credit stress",
              _z_to_prob(credit_z),                        "negative"),
        Cause("Dollar headwind",
              _z_to_prob(dollar_z),                        "negative"),
    ], effect_label="SmallCap positive return", leak_prob=0.05)

    # ── Long Bond (TLT) ───────────────────────────────────────────────────────
    # Inverse of equities — benefits from risk-off and FALLING real yields
    long_bond = NoisyOR([
        Cause("Falling real yields",
              _z_to_prob(real_roc, invert=True) if real_roc else 0.50,
                                                           "positive"),
        Cause("Risk-off flight to quality",
              regime_bear_p,                               "positive"),
        Cause("Credit stress bid",
              _z_to_prob(credit_z),                        "positive"),
        Cause("Inflation persistence",
              _z_to_prob(infl_z) if infl_z else 0.50,     "negative"),
        Cause("Rising real yields",
              _z_to_prob(real_z),                          "negative"),
        Cause("Steepening curve (supply/fiscal)",
              _z_to_prob(curve_z),                         "negative"),
    ], effect_label="Long_Bond positive return", leak_prob=0.03)

    # ── HY Credit (HYG) ──────────────────────────────────────────────────────
    # Needs tight spreads + risk-on + supportive growth
    hy_credit = NoisyOR([
        Cause("Tight credit spreads",
              _z_to_prob(credit_z, invert=True),           "positive"),
        Cause("Bullish macro regime",
              _z_to_prob(cur_score / 10.0 - 5.0),         "positive"),
        Cause("Risk appetite",
              _z_to_prob(risk_z),                          "positive"),
        Cause("Credit widening",
              _z_to_prob(credit_z),                        "negative"),
        Cause("Recession / flight to quality",
              regime_bear_p,                               "negative"),
    ], effect_label="HY_Credit positive return", leak_prob=0.04)

    # ── Gold (GLD) ────────────────────────────────────────────────────────────
    # Classic: low/negative real yields + weak dollar + stress bid
    gold = NoisyOR([
        Cause("Low/negative real yields",
              _z_to_prob(real_z, invert=True),             "positive"),
        Cause("Dollar weakening",
              _z_to_prob(dollar_z, invert=True),           "positive"),
        Cause("Risk-off / safe-haven bid",
              regime_bear_p,                               "positive"),
        Cause("Credit stress premium",
              _z_to_prob(credit_z),                        "positive"),
        Cause("Restrictive real yields",
              _z_to_prob(real_z),                          "negative"),
        Cause("Strong dollar",
              _z_to_prob(dollar_z),                        "negative"),
    ], effect_label="Gold positive return", leak_prob=0.04)

    # ── Copper (CPER) — Dr. Copper industrial cycle proxy ────────────────────
    # Copper is driven by: global PMIs, Chinese demand, curve steepness
    # (growth proxy), dollar (inverted), and the regime score.
    # Unlike oil it has no OPEC/geopolitical noise — pure demand signal.
    # Low correlation with GLD in risk-off (gold rises, copper falls) makes
    # it a genuinely diversifying seventh asset.
    copper = NoisyOR([
        Cause("Bullish growth regime",
              _z_to_prob(cur_score / 10.0 - 5.0),         "positive"),
        Cause("Steep yield curve (PMI proxy)",
              _z_to_prob(curve_z),                         "positive"),
        Cause("Dollar weakening",
              _z_to_prob(dollar_z, invert=True),           "positive"),
        Cause("Positive market breadth",
              _z_to_prob(risk_z),                          "positive"),
        Cause("Rising inflation expectations",
              _z_to_prob(infl_z) if infl_z else 0.50,     "positive"),
        Cause("Global growth slowdown / recession",
              regime_bear_p,                               "negative"),
        Cause("Dollar headwind",
              _z_to_prob(dollar_z),                        "negative"),
        Cause("Credit stress (demand destruction)",
              _z_to_prob(credit_z),                        "negative"),
    ], effect_label="Copper positive return", leak_prob=0.04)

    return {
        "US_Equity":     us_eq,
        "Growth_Equity": growth_eq,
        "SmallCap":      small_cap,
        "Long_Bond":     long_bond,
        "HY_Credit":     hy_credit,
        "Gold":          gold,
        "Copper":        copper,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Build PipelineConfig from live market data
# ─────────────────────────────────────────────────────────────────────────────

def build_pipeline_config(
    px: pd.DataFrame,
    cur_score: int,
    cur_label: str,
    risk_aversion: float = 2.5,
    tau: float = 0.025,
    bull_threshold: float = 0.55,
    bear_threshold: float = 0.45,
    w_min: float = 0.02,
    w_max: float = 0.40,
    n_years: int = 3,
) -> PipelineConfig:
    """
    Construct a PipelineConfig using live price data from your engine.

    Uses trailing n_years of price data to compute:
      - Annualised historical returns per proxy ETF
      - Annualised volatilities (sigma_diagonal)
      - Correlation matrix

    Falls back to sensible defaults if any proxy is missing from px.
    """
    assets = CGM_ASSETS
    proxies = [PROXY_MAP[a] for a in assets]
    trading_days = 252

    hist_returns = []
    sigma_diag   = []
    ret_matrix   = {}

    cutoff = int(n_years * trading_days)

    for ticker in proxies:
        if ticker in px.columns:
            s = px[ticker].dropna().iloc[-cutoff:]
            if len(s) >= 60:
                daily_ret = s.pct_change().dropna()
                ann_ret   = float(daily_ret.mean() * trading_days)
                ann_vol   = float(daily_ret.std() * np.sqrt(trading_days))
                hist_returns.append(max(ann_ret, -0.30))  # floor at -30%
                sigma_diag.append(max(ann_vol, 0.02))     # floor at 2% vol
                ret_matrix[ticker] = daily_ret
            else:
                hist_returns.append(0.05)
                sigma_diag.append(0.15)
        else:
            hist_returns.append(0.05)
            sigma_diag.append(0.15)

    # Build correlation matrix from available return series
    n = len(proxies)
    corr = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            ti, tj = proxies[i], proxies[j]
            if ti in ret_matrix and tj in ret_matrix:
                ri = ret_matrix[ti]
                rj = ret_matrix[tj]
                idx = ri.index.intersection(rj.index)
                if len(idx) >= 60:
                    c = float(np.corrcoef(ri.loc[idx], rj.loc[idx])[0, 1])
                    c = float(np.clip(c, -0.95, 0.95))
                    corr[i, j] = c
                    corr[j, i] = c
    correlations = corr.tolist()

    # Adjust simulation intensity based on regime score
    # More uncertain regimes (near 50) → more iterations to converge
    uncertainty = 1.0 - abs(cur_score - 50) / 50.0
    n_iters_cmab = int(200 + uncertainty * 200)   # 200–400
    n_iters_ql   = int(300 + uncertainty * 300)   # 300–600

    return PipelineConfig(
        assets=assets,
        hist_returns=hist_returns,
        sigma_diagonal=sigma_diag,
        correlations=correlations,
        market_weights=DEFAULT_MKT_WEIGHTS,
        risk_aversion=risk_aversion,
        tau=tau,
        n_rounds_cmab=400,
        n_iters_cmab=n_iters_cmab,
        n_rounds_ql=24,
        n_iters_ql=n_iters_ql,
        bull_threshold=bull_threshold,
        bear_threshold=bear_threshold,
        w_min=w_min,
        w_max=w_max,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Scenario context builder — wires regime label into CMAB contexts
# ─────────────────────────────────────────────────────────────────────────────

def build_cmab_contexts_from_regime(
    cur_label: str,
    cur_score: int,
    components: dict,
) -> list:
    """
    Build CMAB Context objects that reflect the current macro regime.

    The CMAB simulation (book Ch 2-3) simulates an agent choosing between
    'risk-on' and 'risk-off' actions under two type-states drawn from a prior.

    We set the prior probabilities from your regime score and the payoffs
    from the component z-scores, making the CMAB contextually live.
    """
    from cgm.engines.cmab import Arm, Context

    def _z(key): return float((components.get(key) or {}).get("zscore") or 0.0)

    # Regime score → prior probability of being in a "bullish" type-state
    bull_prior = float(np.clip(cur_score / 100.0, 0.15, 0.85))
    bear_prior = 1.0 - bull_prior

    # Payoffs for risk-on vs risk-off under each context
    credit_z = _z("credit");  real_z = _z("real_yields")
    curve_z  = _z("curve");   risk_z = _z("risk_appetite")

    # Bullish context: risk-on pays well, risk-off has opportunity cost
    bull_risk_on_payoff  = max(3.0 + risk_z * 0.5 - real_z * 0.3, 0.5)
    bull_risk_off_payoff = max(1.5 - risk_z * 0.3, 0.2)
    bull_risk_on_prob    = float(np.clip(0.60 + risk_z * 0.05, 0.30, 0.90))

    # Bearish context: risk-off is safer, risk-on punished
    bear_risk_on_payoff  = max(1.0 - abs(credit_z) * 0.4, 0.1)
    bear_risk_off_payoff = max(3.5 + abs(credit_z) * 0.3, 1.0)
    bear_risk_on_prob    = float(np.clip(0.35 - abs(credit_z) * 0.05, 0.10, 0.65))

    bullish_ctx = Context(
        arms=[
            Arm("Risk-On",  bull_risk_on_payoff,  bull_risk_on_prob),
            Arm("Risk-Off", bull_risk_off_payoff, 0.85),
        ],
        prior=bull_prior,
        label=f"Bullish type ({cur_label})",
    )

    bearish_ctx = Context(
        arms=[
            Arm("Risk-On",  bear_risk_on_payoff,  bear_risk_on_prob),
            Arm("Risk-Off", bear_risk_off_payoff, 0.80),
        ],
        prior=bear_prior,
        label=f"Bearish type ({cur_label})",
    )

    return [bullish_ctx, bearish_ctx]


# ─────────────────────────────────────────────────────────────────────────────
# 4. Map PipelineResult weights back to your engine's ETF tickers
# ─────────────────────────────────────────────────────────────────────────────

def result_to_etf_weights(
    pipeline_result,
    method: str = "bl",
) -> dict[str, float]:
    """
    Convert PipelineResult portfolio weights back to your ETF ticker space.

    Parameters
    ----------
    pipeline_result : PipelineResult from ComputationalGlobalMacroPipeline.run()
    method          : "bl" | "obl" | "robust"

    Returns
    -------
    dict mapping ETF ticker → weight (e.g. {"SPY": 0.28, "TLT": 0.18, ...})
    """
    result_map = {
        "bl":     pipeline_result.bl_result,
        "obl":    pipeline_result.obl_result,
        "robust": pipeline_result.robust_result,
    }
    port = result_map.get(method, pipeline_result.bl_result)
    return {
        PROXY_MAP[asset]: float(w)
        for asset, w in zip(CGM_ASSETS, port.weights)
    }


def nor_probs_summary(nor_probs: dict[str, float]) -> list[dict]:
    """
    Format NOR probabilities into a sorted list of signal dicts for display.
    Uses ASSET_LABELS for clean display names in the UI.
    """
    rows = []
    for asset, p in nor_probs.items():
        ticker = PROXY_MAP.get(asset, asset)
        label  = ASSET_LABELS.get(asset, asset)
        direction = +1 if p >= 0.55 else (-1 if p <= 0.45 else 0)
        rows.append({
            "asset":        asset,
            "ticker":       ticker,
            "display_name": label,
            "nor_prob":     round(p, 3),
            "direction":    direction,
            "label":        ("BULLISH" if direction > 0
                             else "BEARISH" if direction < 0 else "NEUTRAL"),
            "color":        ("#1f7a4f" if direction > 0
                             else "#b42318" if direction < 0 else "#6b7280"),
        })
    rows.sort(key=lambda x: abs(x["nor_prob"] - 0.5), reverse=True)
    return rows