"""
engines/causal.py
Noisy-OR causal model and view generator (Chapter 5).

Encodes expert causal beliefs as probabilistic DAGs and computes the
net probability of a macroeconomic effect materialising given a set
of promoting and inhibiting causes.

The output (causal probabilities per asset/scenario) feeds directly
into the portfolio layer as Black-Litterman view confidences.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Noisy-OR model (Chapter 5, Section 5.4)
# ---------------------------------------------------------------------------

@dataclass
class Cause:
    """
    A single causal factor with a name, direction, and probability.

    name        : e.g. "Fed Funds Rate Cut"
    probability : P(this cause is active AND capable of producing the effect)
    direction   : "positive" (promotes effect) or "negative" (inhibits effect)
    weight      : optional weight for weighted NOR variant (default 1.0)
    """
    name: str
    probability: float
    direction: str = "positive"   # "positive" | "negative"
    weight: float = 1.0

    def __post_init__(self):
        assert 0.0 <= self.probability <= 1.0, "Probability must be in [0,1]."
        assert self.direction in ("positive", "negative")
        assert self.weight > 0


class NoisyOR:
    """
    Noisy-OR model for combining multiple causes.

    Implements equations (5.15) and (5.16) from the book:

      P(E | C+) = 1 - ∏(1 - p_i)   for positive causes
      P(E | C-) = 1 - ∏(1 - p_i)   for negative causes  [= P(¬E | C-)]
      P(E | C+, C-) = [P(E|C+) + 1 - P(¬E|C-)] / 2

    Also supports a "leak probability" (Henrion 1987) for background
    probability of the effect even without any listed causes.

    Parameters
    ----------
    causes          : list of Cause objects
    leak_prob       : background probability of effect (default 0.0)
    effect_label    : human-readable label for the effect
    """

    def __init__(self, causes: list[Cause],
                 leak_prob: float = 0.0,
                 effect_label: str = "Effect"):
        self.causes = causes
        self.leak_prob = leak_prob
        self.effect_label = effect_label

    @property
    def positive_causes(self) -> list[Cause]:
        return [c for c in self.causes if c.direction == "positive"]

    @property
    def negative_causes(self) -> list[Cause]:
        return [c for c in self.causes if c.direction == "negative"]

    def _nor_prob(self, causes: list[Cause]) -> float:
        """P(E | causes active) = 1 - ∏(1 - p_i * w_i / max_w)"""
        if not causes:
            return 0.0
        prob_no_effect = 1.0
        for c in causes:
            # weighted variant: scale individual probability by weight
            effective_p = min(c.probability * c.weight, 1.0)
            prob_no_effect *= (1.0 - effective_p)
        return 1.0 - prob_no_effect

    def net_probability(self,
                        active_positive: list[str] | None = None,
                        active_negative: list[str] | None = None) -> float:
        """
        Compute net P(effect materialises).

        Parameters
        ----------
        active_positive : names of positive causes currently active.
                          If None, all positive causes are treated as active.
        active_negative : names of negative causes currently active.
                          If None, all negative causes are treated as active.

        Returns
        -------
        float in [0, 1]
        """
        pos = self.positive_causes
        neg = self.negative_causes

        if active_positive is not None:
            pos = [c for c in pos if c.name in active_positive]
        if active_negative is not None:
            neg = [c for c in neg if c.name in active_negative]

        p_pos = self._nor_prob(pos)
        p_neg = self._nor_prob(neg)          # = P(¬E | negative causes)

        # Add leak probability contribution
        if self.leak_prob > 0:
            p_pos = 1.0 - (1.0 - p_pos) * (1.0 - self.leak_prob)

        # Equation (5.16)
        net = (p_pos + 1.0 - p_neg) / 2.0
        return float(np.clip(net, 0.0, 1.0))

    def sensitivity_table(self) -> list[dict]:
        """
        Return a table showing how each cause individually impacts the net
        probability (all others held at their base state).
        """
        rows = []
        baseline = self.net_probability()
        for cause in self.causes:
            if cause.direction == "positive":
                without = self.net_probability(
                    active_positive=[c.name for c in self.positive_causes
                                     if c.name != cause.name])
            else:
                without = self.net_probability(
                    active_negative=[c.name for c in self.negative_causes
                                     if c.name != cause.name])
            rows.append({
                "cause": cause.name,
                "direction": cause.direction,
                "probability": cause.probability,
                "marginal_impact": round(baseline - without, 4),
            })
        rows.sort(key=lambda x: abs(x["marginal_impact"]), reverse=True)
        return rows

    def __repr__(self):
        return (f"NoisyOR(effect='{self.effect_label}', "
                f"n_causes={len(self.causes)}, "
                f"net_prob={self.net_probability():.3f})")


# ---------------------------------------------------------------------------
# View generator: converts simulation + NOR outputs into BL view inputs
# ---------------------------------------------------------------------------

@dataclass
class MacroView:
    """
    A single directional view on an asset, ready for Black-Litterman input.

    asset       : asset identifier (matches portfolio asset list)
    direction   : +1 (bullish) or -1 (bearish)
    confidence  : in [0,1] — drives the Omega (view uncertainty) matrix
    source      : human-readable provenance string
    causal_prob : raw NOR output (optional, stored for audit trail)
    """
    asset: str
    direction: float          # +1.0 or -1.0
    confidence: float         # in [0,1]
    source: str = ""
    causal_prob: float | None = None

    def view_return(self, base_magnitude: float = 0.02) -> float:
        """Convert direction + confidence into an expected excess return view."""
        return self.direction * confidence_to_magnitude(self.confidence,
                                                         base_magnitude)


def confidence_to_magnitude(confidence: float,
                             base: float = 0.02,
                             scale: float = 3.0) -> float:
    """
    Non-linear mapping from confidence in [0,1] to view magnitude.
    confidence=0.5 → base return; confidence=1.0 → base*scale.
    """
    return base * (1 + (scale - 1) * (confidence - 0.5) * 2) if confidence > 0.5 \
        else base * confidence * 2


class ViewAggregator:
    """
    Combines NOR causal probabilities and simulation payoffs into a
    coherent list of MacroViews.

    Parameters
    ----------
    assets          : list of asset names matching the portfolio
    threshold_bull  : net probability above this → bullish view
    threshold_bear  : net probability below this → bearish view
    """

    def __init__(self, assets: list[str],
                 threshold_bull: float = 0.55,
                 threshold_bear: float = 0.45):
        self.assets = assets
        self.threshold_bull = threshold_bull
        self.threshold_bear = threshold_bear

    def from_nor(self, nor_map: dict[str, NoisyOR]) -> list[MacroView]:
        """
        Convert {asset_name: NoisyOR} dict into MacroView list.

        Parameters
        ----------
        nor_map : maps each asset to its NoisyOR model for "will this
                  asset deliver positive returns?"
        """
        views = []
        for asset, model in nor_map.items():
            p = model.net_probability()
            if p >= self.threshold_bull:
                views.append(MacroView(
                    asset=asset, direction=+1.0,
                    confidence=p, source=f"NOR({model.effect_label})",
                    causal_prob=p))
            elif p <= self.threshold_bear:
                views.append(MacroView(
                    asset=asset, direction=-1.0,
                    confidence=1.0 - p, source=f"NOR({model.effect_label})",
                    causal_prob=p))
        return views

    def from_simulation(self,
                        sim_results: dict,
                        asset_arm_map: dict[str, int]) -> list[MacroView]:
        """
        Convert CMAB/Q-learning simulation results into MacroViews.

        Parameters
        ----------
        sim_results     : output dict from CMABSimulation.run() or
                          TwoPlayerSimulation.run()
        asset_arm_map   : maps asset name to the arm/action index whose
                          long-run payoff determines the view direction
        """
        views = []
        cum = sim_results.get("cum_payoffs") or sim_results.get("cum_avg_p1")
        if cum is None:
            return views

        final_payoffs = cum[-1] if cum.ndim > 1 else cum

        # Normalise payoffs to [0,1] confidence scale
        min_p, max_p = final_payoffs.min(), final_payoffs.max()
        rng = max_p - min_p if max_p > min_p else 1.0

        for asset, arm_idx in asset_arm_map.items():
            if arm_idx >= len(final_payoffs):
                continue
            normalised = (final_payoffs[arm_idx] - min_p) / rng  # [0,1]
            direction = +1.0 if normalised >= 0.5 else -1.0
            confidence = normalised if direction > 0 else 1.0 - normalised
            views.append(MacroView(
                asset=asset, direction=direction,
                confidence=float(confidence),
                source="simulation"))
        return views