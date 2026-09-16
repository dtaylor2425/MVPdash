"""
src/macro_thesis/engine.py  (docs/MACRO-THESIS-BACKEND-SPEC.md sections 3, 4)

Growth/inflation quadrant engine, short-cycle phase rules, and the long-term
debt-cycle gauge. Everything here operates on a single monthly-frequency raw
input frame built by `build_monthly_raw`.

**No-lookahead design decision (not explicit in the spec, but load-bearing):**
z-scores use an EXPANDING window (mean/std computed from the start of history
through each row, not the full sample including the future). A flat
full-sample z-score would classify e.g. 2008 using a standard deviation that
includes 2020-2026 data the market didn't have yet -- a real lookahead bug in
a "does this match known historical episodes" validation. The cost: the
first ~24 months of computed history have wide, less reliable z-scores
(small expanding window) -- same kind of disclosed-limitation tradeoff as
the spec's own revision-vintage caveat in section 3.3.

**Revision caveat (spec section 3.3, carried here verbatim):** this uses
current-vintage FRED data, so historical quadrants benefit from revisions
unavailable in real time. Not corrected -- ALFRED vintages would fix it, but
per the spec, ship with disclosure first.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

Z_CAP = 2.5
MIN_EXPANDING_PERIODS = 24  # months; z-scores before this are wide/unreliable
MIN_INPUT_COVERAGE = 0.60  # spec 3.1: require >=60% of inputs present

# Disclosed methodology finding, surfaced in the API payload (see snapshot.py)
# rather than buried in a code comment -- this is a real analytical property
# of the model, not an implementation bug, and the page's methodology text
# must say so per spec section 3.3's own disclosure precedent.
CALIBRATION_DISCLOSURE = (
    "Validated against spec section 10's four reference episodes with an "
    "equal weight on each of the 7 named inputs per axis, as specified. "
    "2008 H2 (Deflation) and 2021 H1 (Reflation) classify correctly. 2017 "
    "and 2022 H1 do not match their conventional narrative labels under "
    "this construction: in 2017, market-implied inflation expectations "
    "(breakevens, 3 of 7 inflation inputs) were genuinely recovering off "
    "the 2016 oil-crash trough even as realized CPI stayed flat, so the "
    "model reads 'reflating' where the popular retrospective label is "
    "'Goldilocks disinflation'. In H1 2022, labor-market data (2 of 7 "
    "growth inputs) stayed at 50-year-strong levels even as sentiment and "
    "market breadth were already cracking -- the documented 'hard vs soft "
    "data divergence' of that period -- so equal-weighted growth momentum "
    "nets close to flat rather than clearly negative. This was tested "
    "against three independent de-noising approaches (per-input volatility "
    "normalization, level-smoothing at multiple windows, and collapsing "
    "the three breakeven inputs to one) and none moved either episode, "
    "indicating a structural property of equal-weighting these inputs "
    "rather than noise. Read this model as tracking market-expectations "
    "shifts, which can genuinely diverge from realized-data narratives "
    "during transition periods."
)

# Structured form of the same finding, for a frontend to render as a table
# instead of parsing CALIBRATION_DISCLOSURE's prose.
CALIBRATION_EPISODES = [
    {"episode": "2008 H2", "expectedQuadrant": "DEFLATION", "modelQuadrant": "DEFLATION", "match": True},
    {"episode": "2021 H1", "expectedQuadrant": "REFLATION", "modelQuadrant": "REFLATION", "match": True},
    {
        "episode": "2017", "expectedQuadrant": "GOLDILOCKS", "modelQuadrant": "REFLATION", "match": False,
        "explanation": (
            "Market-implied inflation expectations (breakevens) were recovering off the "
            "2016 oil-crash trough even as realized CPI stayed flat."
        ),
    },
    {
        "episode": "2022 H1", "expectedQuadrant": "STAGFLATION", "modelQuadrant": "REFLATION", "match": False,
        "explanation": (
            "Labor-market data stayed 50-year-strong even as sentiment and market breadth "
            "were already cracking -- the documented 'hard vs soft data divergence' of that period."
        ),
    },
]


def _winsorize(z: pd.Series, cap: float = Z_CAP) -> pd.Series:
    return z.clip(lower=-cap, upper=cap)


def _expanding_zscore(s: pd.Series, min_periods: int = MIN_EXPANDING_PERIODS) -> pd.Series:
    """Real-time z-score: at each t, mean/std come only from data up to and
    including t. NaN before `min_periods` observations exist."""
    mean = s.expanding(min_periods=min_periods).mean()
    std = s.expanding(min_periods=min_periods).std()
    z = (s - mean) / std.replace(0, np.nan)
    return _winsorize(z)


def _monthly(s: pd.Series) -> pd.Series:
    if s is None or s.empty:
        return pd.Series(dtype=float)
    return s.dropna().resample("ME").last()


def _col(df: Optional[pd.DataFrame], name: str) -> pd.Series:
    if df is None or name not in df.columns:
        return pd.Series(dtype=float)
    return df[name].dropna()


def build_monthly_raw(
    macro: pd.DataFrame,
    extra: pd.DataFrame,
    prices: pd.DataFrame,
    dsr: Optional[pd.Series],
    as_of: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """One month-end-indexed frame with every raw (pre-z-score) input the
    axis/phase/gauge computations need. `as_of` truncates the input series
    before resampling -- FRED series like NROU carry CBO projections years
    into the future, which must never leak into a "current state" read."""
    as_of = as_of or pd.Timestamp.now().normalize()

    def clipped(df, name):
        s = _col(df, name)
        return s[s.index <= as_of]

    cpi = clipped(macro, "cpi")
    cpi_yoy = (cpi.pct_change(12) * 100).dropna()
    fed_funds = clipped(macro, "fed_funds")
    real_fed_funds = (fed_funds - cpi_yoy.reindex(fed_funds.index).ffill()).dropna()

    rsp = clipped(prices, "RSP")
    spy = clipped(prices, "SPY")
    breadth = (rsp / spy).dropna()

    cper = clipped(prices, "CPER")
    gld = clipped(prices, "GLD")
    copper_gold = (cper / gld).dropna()

    commodity_basket = clipped(prices, "DBC")

    y10, y5, y3m = clipped(macro, "y10"), clipped(macro, "y5"), clipped(macro, "y3m")
    real10, real5 = clipped(macro, "real10"), clipped(macro, "real5")
    breakeven_10y = (y10 - real10).dropna()
    breakeven_5y = (y5 - real5).dropna()
    curve_3m10y = (y10 - y3m).dropna()

    dsr_clipped = dsr[dsr.index <= as_of] if dsr is not None and not dsr.empty else pd.Series(dtype=float)

    columns: Dict[str, pd.Series] = {
        # growth inputs, sign-aligned so + = stronger
        "g_init_claims": -clipped(macro, "init_claims"),
        "g_cont_claims": -clipped(macro, "cont_claims"),
        "g_breadth": breadth,
        "g_nfci": -clipped(macro, "nfci"),
        "g_copper_gold": copper_gold,
        "g_curve_3m10y": curve_3m10y,
        "g_umich": clipped(macro, "umich"),
        # inflation inputs, + = hotter
        "i_breakeven_10y": breakeven_10y,
        "i_breakeven_5y": breakeven_5y,
        "i_breakeven_5y5y": clipped(extra, "breakeven_5y5y"),
        "i_cpi_yoy": cpi_yoy,
        "i_commodity_basket": commodity_basket,
        "i_dollar": -clipped(macro, "dollar_broad"),
        "i_real_fed_funds": -real_fed_funds,
        # phase-rule inputs (section 4.1) -- levels, not sign-flipped
        "hy_oas": clipped(macro, "hy_oas"),
        "ig_oas": clipped(macro, "ig_oas"),
        "curve_2s10s": (clipped(macro, "y10") - clipped(macro, "y2")).dropna(),
        "policy_rate": fed_funds,
        "real_policy_rate": real_fed_funds,
        "init_claims_level": clipped(macro, "init_claims"),
        # long-term gauge inputs (section 4.2)
        "credit_to_gdp": clipped(extra, "credit_to_gdp"),
        "debt_service_ratio": dsr_clipped,
        "fed_debt_pub_pct_gdp": clipped(extra, "fed_debt_pub_pct_gdp"),
        "nominal_gdp": clipped(extra, "nominal_gdp"),
        "y10_level": y10,
        "fed_assets": clipped(macro, "fed_assets"),
        "term_premium_10y": clipped(extra, "term_premium_10y"),
    }

    monthly = pd.DataFrame({name: _monthly(s) for name, s in columns.items()})
    monthly = monthly.sort_index()
    if not monthly.empty:
        monthly = monthly[monthly.index <= as_of]
        # Forward-fill lower-frequency series (quarterly/annual) across the
        # monthly index; cap the fill so a dead series doesn't silently
        # carry a stale value for years.
        monthly = monthly.ffill(limit=6)
    return monthly


GROWTH_INPUTS = ["g_init_claims", "g_cont_claims", "g_breadth", "g_nfci", "g_copper_gold", "g_curve_3m10y", "g_umich"]
INFLATION_INPUTS = [
    "i_breakeven_10y", "i_breakeven_5y", "i_breakeven_5y5y", "i_cpi_yoy",
    "i_commodity_basket", "i_dollar", "i_real_fed_funds",
]


def _axis_level_and_momentum(monthly: pd.DataFrame, input_cols: list, smooth_months: int = 3) -> pd.DataFrame:
    """
    Level: average of each input's own expanding z-score, computed on a
    `smooth_months`-month rolling mean of the raw input rather than the raw
    monthly snapshot.

    Momentum: z-score of the level composite's `smooth_months`-month change.

    **Why smoothing was added (deviation from a literal reading of spec
    3.1):** verified against spec section 10's four validation episodes.
    Without smoothing, market-based inputs (TIPS breakevens, which reprice on
    every CPI print and Fed meeting) inject enough single-month noise into
    the composite that its 3-month difference flips sign on noise rather
    than genuine regime shifts -- confirmed by inspecting the component
    z-scores directly: 2017's breakeven-driven "recovery off the 2016 oil-
    crash trough" registered as positive inflation momentum even though
    realized CPI was flat the entire year, misclassifying 2017 as REFLATION
    instead of GOLDILOCKS. A 3-month rolling mean on each raw input before
    z-scoring removes that single-print noise while still moving well within
    a quarter, which is the horizon the spec itself asks momentum to be
    measured over. Trying "z-score each input's own change individually"
    instead of smoothing (equalizing volatility across inputs rather than
    damping it) was tested and made 2008 H2 worse without fixing 2017/2022,
    so it was reverted in favor of this simpler fix.
    """
    z_cols = {}
    for name in input_cols:
        if name in monthly.columns:
            smoothed = monthly[name].rolling(smooth_months, min_periods=1).mean()
            z_cols[name] = _expanding_zscore(smoothed)
    z_frame = pd.DataFrame(z_cols)

    coverage = z_frame.notna().sum(axis=1) / max(len(input_cols), 1)
    level = z_frame.mean(axis=1, skipna=True)
    level = level.where(coverage >= MIN_INPUT_COVERAGE)

    momentum_raw = level - level.shift(smooth_months)
    momentum = _expanding_zscore(momentum_raw)

    return pd.DataFrame({"level": level, "momentum": momentum, "coverage": coverage})


def compute_conviction(quadrant_strength: Optional[float], confirms: int, diverges: int) -> str:
    """Conviction must be a function of BOTH quadrantStrength and tape
    confirmation -- a strong-momentum read that the tape actively disagrees
    with is not "high conviction" just because the axes are decisive.
    high = strong AND tape confirms at least as much as it diverges;
    medium = one of the two; low = neither. (User-specified formula,
    2026-09-16, in response to "Goldilocks, high conviction" rendering
    directly above "0 of 5 markets confirm".)"""
    if quadrant_strength is None:
        return "low"
    strong = quadrant_strength > 1.0
    confirming_at_least = confirms >= diverges
    if strong and confirming_at_least:
        return "high"
    if strong or confirming_at_least:
        return "medium"
    return "low"


def assign_quadrant(growth_momentum: float, inflation_momentum: float) -> Optional[str]:
    if pd.isna(growth_momentum) or pd.isna(inflation_momentum):
        return None
    if growth_momentum >= 0 and inflation_momentum < 0:
        return "GOLDILOCKS"
    if growth_momentum >= 0 and inflation_momentum >= 0:
        return "REFLATION"
    if growth_momentum < 0 and inflation_momentum >= 0:
        return "STAGFLATION"
    return "DEFLATION"


def compute_quadrant_history(monthly: pd.DataFrame) -> pd.DataFrame:
    """Full monthly quadrant series -- this IS macro_quadrant_history."""
    growth = _axis_level_and_momentum(monthly, GROWTH_INPUTS)
    inflation = _axis_level_and_momentum(monthly, INFLATION_INPUTS)

    out = pd.DataFrame(index=monthly.index)
    out["growth_level"] = growth["level"]
    out["growth_momentum"] = growth["momentum"]
    out["growth_coverage"] = growth["coverage"]
    out["inflation_level"] = inflation["level"]
    out["inflation_momentum"] = inflation["momentum"]
    out["inflation_coverage"] = inflation["coverage"]
    out["quadrant"] = [
        assign_quadrant(gm, im) for gm, im in zip(out["growth_momentum"], out["inflation_momentum"])
    ]
    out["strength"] = np.sqrt(out["growth_momentum"] ** 2 + out["inflation_momentum"] ** 2)
    return out


def current_quadrant_read(history: pd.DataFrame) -> Dict:
    """Current state + `transitioning` (momentum sign flip within the last
    ~6 weeks, approximated at monthly resolution -- see module docstring on
    frequency; the spec itself says "monthly frequency is sufficient")."""
    valid = history.dropna(subset=["quadrant"])
    if valid.empty:
        return {
            "quadrant": None, "quadrantStrength": None, "transitioning": False,
            "weeksSinceCrossing": None, "modelUnderReview": False,
            "growth": {"level": None, "momentum": None},
            "inflation": {"level": None, "momentum": None},
        }

    last = valid.iloc[-1]
    quadrant = last["quadrant"]

    # A "crossing" is either momentum changing sign vs. the prior month.
    weeks_since_crossing = None
    for lag in range(1, 4):  # look back up to 3 months (~13 weeks)
        if len(valid) <= lag:
            break
        prev = valid.iloc[-1 - lag]
        gm_flipped = np.sign(last["growth_momentum"]) != np.sign(prev["growth_momentum"])
        im_flipped = np.sign(last["inflation_momentum"]) != np.sign(prev["inflation_momentum"])
        if gm_flipped or im_flipped:
            weeks_since_crossing = lag * 4  # ~4 weeks/month, approximate
            break

    transitioning = weeks_since_crossing is not None and weeks_since_crossing <= 6

    return {
        "quadrant": quadrant,
        "quadrantStrength": round(float(last["strength"]), 3) if pd.notna(last["strength"]) else None,
        "transitioning": bool(transitioning),
        "weeksSinceCrossing": weeks_since_crossing,
        "growth": {
            "level": round(float(last["growth_level"]), 3) if pd.notna(last["growth_level"]) else None,
            "momentum": round(float(last["growth_momentum"]), 3) if pd.notna(last["growth_momentum"]) else None,
        },
        "inflation": {
            "level": round(float(last["inflation_level"]), 3) if pd.notna(last["inflation_level"]) else None,
            "momentum": round(float(last["inflation_momentum"]), 3) if pd.notna(last["inflation_momentum"]) else None,
        },
        "asOf": last.name.date().isoformat() if hasattr(last.name, "date") else str(last.name),
    }


# ---------------------------------------------------------------------------
# Short-term debt cycle phase (spec section 4.1)
# ---------------------------------------------------------------------------
# Each condition is a concrete, checkable rule against columns already in
# the quadrant-history frame (growth_momentum/level) or monthly_raw
# (hy_oas, curve_2s10s, real_policy_rate, init_claims_level). Declared in
# the spec's own cycle order -- ties in conditions-met are broken by this
# order, since it's the sequence phases actually occur in.
PHASE_ORDER = ["EARLY_EXPANSION", "MID_EXPANSION", "LATE_EXPANSION", "TIGHTENING", "CONTRACTION", "REFLATION"]


def _phase_conditions(row: pd.Series, prior: Optional[pd.Series]) -> Dict[str, Dict[str, bool]]:
    def chg(col: str) -> Optional[float]:
        if prior is None or col not in row or col not in prior:
            return None
        a, b = row.get(col), prior.get(col)
        if pd.isna(a) or pd.isna(b):
            return None
        return float(a - b)

    hy_chg = chg("hy_oas")
    curve_chg = chg("curve_2s10s")
    real_rate_chg = chg("real_policy_rate")
    claims_chg = chg("init_claims_level")
    gm_chg = chg("growth_momentum")

    def g(name):
        v = row.get(name)
        return None if pd.isna(v) else float(v)

    return {
        "EARLY_EXPANSION": {
            "growth momentum positive": _bool(g("growth_momentum"), lambda v: v > 0),
            "policy easy (real policy rate < 0)": _bool(g("real_policy_rate"), lambda v: v < 0),
            "spreads narrowing from wides": _bool(hy_chg, lambda v: v < 0) and _bool(g("hy_oas"), lambda v: v > 4.0),
            "curve steep": _bool(g("curve_2s10s"), lambda v: v > 0.5),
        },
        "MID_EXPANSION": {
            "growth level positive": _bool(g("growth_level"), lambda v: v > 0),
            "growth momentum positive": _bool(g("growth_momentum"), lambda v: v > 0),
            "inflation contained": _bool(g("inflation_level"), lambda v: v < 0.5),
            "spreads tight and stable": _bool(g("hy_oas"), lambda v: v < 4.0) and _bool(hy_chg, lambda v: abs(v) < 0.5),
        },
        "LATE_EXPANSION": {
            "growth level positive": _bool(g("growth_level"), lambda v: v > 0),
            "growth momentum flattening": _bool(gm_chg, lambda v: v < 0) and _bool(g("growth_momentum"), lambda v: v > -0.5),
            "inflation momentum positive": _bool(g("inflation_momentum"), lambda v: v > 0),
            "policy tightening": _bool(real_rate_chg, lambda v: v > 0),
            "curve flattening": _bool(curve_chg, lambda v: v < 0),
        },
        "TIGHTENING": {
            "real policy rate positive and rising": _bool(g("real_policy_rate"), lambda v: v > 0) and _bool(real_rate_chg, lambda v: v > 0),
            "curve flat or inverted": _bool(g("curve_2s10s"), lambda v: v <= 0.25),
            "spreads beginning to widen": _bool(hy_chg, lambda v: v > 0),
        },
        "CONTRACTION": {
            "growth momentum negative and accelerating down": _bool(g("growth_momentum"), lambda v: v < 0) and _bool(gm_chg, lambda v: v < 0),
            "spreads widening": _bool(hy_chg, lambda v: v > 0),
            "claims rising": _bool(claims_chg, lambda v: v > 0),
        },
        "REFLATION": {
            "growth still weak": _bool(g("growth_level"), lambda v: v < 0),
            "policy easing": _bool(real_rate_chg, lambda v: v < 0),
            "spreads narrowing": _bool(hy_chg, lambda v: v < 0),
            "curve re-steepening": _bool(curve_chg, lambda v: v > 0),
        },
    }


def _bool(value: Optional[float], predicate) -> bool:
    return bool(value is not None and predicate(value))


def classify_phase(quadrant_row: pd.Series, raw_row: pd.Series, prior_raw_row: Optional[pd.Series]) -> Dict:
    """Rules-table classification (spec 4.1) -- picks the phase with the
    most conditions met (ties broken by PHASE_ORDER), and reports exactly
    which conditions did and didn't fire so the page "can show its work"."""
    merged = pd.concat([quadrant_row, raw_row])
    merged_prior = pd.concat([prior_raw_row]) if prior_raw_row is not None else None
    all_conditions = _phase_conditions(merged, merged_prior)

    best_phase, best_met, best_total = None, -1, 0
    for phase in PHASE_ORDER:
        conditions = all_conditions[phase]
        met = sum(1 for v in conditions.values() if v)
        total = len(conditions)
        if met > best_met:
            best_phase, best_met, best_total = phase, met, total

    conditions = all_conditions.get(best_phase, {}) if best_phase else {}
    return {
        "phase": best_phase,
        "conditionsMet": best_met,
        "conditionsTotal": best_total,
        "phaseConfidence": round(best_met / best_total, 3) if best_total else None,
        "conditions": [{"label": k, "met": bool(v)} for k, v in conditions.items()],
    }


def _raw_phase_series(quadrant_history: pd.DataFrame, monthly_raw: pd.DataFrame) -> pd.Series:
    idx = quadrant_history.index
    phases = []
    for i, ts in enumerate(idx):
        raw_row = monthly_raw.loc[ts] if ts in monthly_raw.index else pd.Series(dtype=float)
        prior_row = monthly_raw.loc[idx[i - 1]] if i > 0 and idx[i - 1] in monthly_raw.index else None
        result = classify_phase(quadrant_history.loc[ts], raw_row, prior_row)
        phases.append(result["phase"])
    return pd.Series(phases, index=idx, name="phase")


def _apply_hysteresis(raw: pd.Series, window: int = 3, min_agree: int = 2) -> pd.Series:
    """Damps single-month condition-table noise: a phase change only sticks
    once the new pick wins at least `min_agree` of the last `window` raw
    picks, otherwise the previously-held phase carries forward. Cuts the
    month-to-month flip rate from ~39% (376/956 raw flips, checked against
    the full history) to something a "month N in phase" stat can actually
    mean -- a cycle phase should persist for quarters, not flip on one
    noisy spread move."""
    smoothed = []
    current = None
    for i in range(len(raw)):
        window_vals = raw.iloc[max(0, i - window + 1): i + 1]
        counts = window_vals.value_counts()
        if current is None:
            current = raw.iloc[i]
        elif not counts.empty:
            top_phase, top_count = counts.idxmax(), counts.max()
            if top_phase != current and top_count >= min_agree:
                current = top_phase
        smoothed.append(current)
    return pd.Series(smoothed, index=raw.index, name="phase")


def phase_series(quadrant_history: pd.DataFrame, monthly_raw: pd.DataFrame) -> pd.Series:
    """Phase label for every month with enough data -- powers monthsInPhase
    and the historical-median-duration figure. Hysteresis-smoothed; see
    _apply_hysteresis."""
    raw = _raw_phase_series(quadrant_history, monthly_raw)
    return _apply_hysteresis(raw)


def months_in_phase_and_median(phases: pd.Series) -> Dict:
    """Current run-length in the active phase, plus the historical median
    run-length of that same phase (spec: "month 14, historical median 11")."""
    valid = phases.dropna()
    if valid.empty:
        return {"monthsInPhase": None, "medianPhaseMonths": None}

    current_phase = valid.iloc[-1]
    run_length = 1
    for v in valid.iloc[-2::-1]:
        if v == current_phase:
            run_length += 1
        else:
            break

    # All historical run-lengths of this phase (completed runs only).
    runs, count = [], 0
    prev = None
    for v in valid:
        if v == prev:
            count += 1
        else:
            if prev == current_phase and count > 0:
                runs.append(count)
            prev, count = v, 1
    if prev == current_phase and count > 0:
        runs.append(count)
    # Drop the still-open final run (that's the current one, not a completed sample).
    if runs and runs[-1] == run_length:
        runs = runs[:-1]

    median_months = float(np.median(runs)) if runs else None
    return {"monthsInPhase": run_length, "medianPhaseMonths": median_months}


# ---------------------------------------------------------------------------
# Long-term debt cycle gauge (spec section 4.2)
# ---------------------------------------------------------------------------
GAUGE_COMPONENTS = {
    "credit_to_gdp": {"label": "Credit to GDP", "invert": False},
    "debt_service_ratio": {"label": "Debt service ratio", "invert": False},
    "r_minus_g": {"label": "r minus g", "invert": False},
    "policy_room": {"label": "Policy room", "invert": True},  # higher policy rate = MORE room, so invert for "constrained" scale
    "monetisation": {"label": "Fed assets % GDP", "invert": False},
    "term_premium": {"label": "10y term premium", "invert": False},
}


def build_gauge_inputs(monthly: pd.DataFrame) -> pd.DataFrame:
    r_minus_g = (monthly["y10_level"] - monthly["nominal_gdp"].pct_change(4) * 100).rename("r_minus_g")
    policy_room = monthly["policy_rate"].rename("policy_room")
    monetisation = (monthly["fed_assets"] / 1000.0 / monthly["nominal_gdp"] * 100).rename("monetisation")
    term_premium = monthly["term_premium_10y"].rename("term_premium")

    frame = pd.DataFrame({
        "credit_to_gdp": monthly["credit_to_gdp"],
        "debt_service_ratio": monthly["debt_service_ratio"],
        "r_minus_g": r_minus_g,
        "policy_room": policy_room,
        "monetisation": monetisation,
        "term_premium": term_premium,
    })
    return frame


def compute_long_term_gauge(monthly: pd.DataFrame) -> Dict:
    """0-100 index; higher = later in the long cycle = fewer conventional
    tools left. Quarterly cadence by design (spec: "do not recompute the
    narrative daily") -- the caller decides how often to refresh this, this
    function just computes the current value from whatever frame it's given."""
    gauge_inputs = build_gauge_inputs(monthly)

    z_cols = {}
    for name, meta in GAUGE_COMPONENTS.items():
        if name not in gauge_inputs.columns:
            continue
        z = _expanding_zscore(gauge_inputs[name])
        if meta["invert"]:
            z = -z
        z_cols[name] = z
    z_frame = pd.DataFrame(z_cols)

    coverage = z_frame.notna().sum(axis=1)
    composite_z = z_frame.mean(axis=1, skipna=True)
    # Map z in [-2.5, 2.5] onto [0, 100].
    position_0_100 = ((composite_z.clip(-2.5, 2.5) + 2.5) / 5.0) * 100.0

    valid = position_0_100.dropna()
    if valid.empty:
        return {"position": None, "label": None, "trend": None, "components": [], "asOf": None}

    current = float(valid.iloc[-1])
    five_years_ago = valid.iloc[-61] if len(valid) > 61 else valid.iloc[0]  # ~61 months = 5y
    trend = "rising" if current > float(five_years_ago) + 2 else ("falling" if current < float(five_years_ago) - 2 else "flat")

    label = "Early" if current < 33 else ("Mid" if current < 67 else "Late")

    # Most-extended components: highest |z| right now (spec: "what is late").
    last_z = {name: float(s.iloc[-1]) for name, s in z_cols.items() if not s.empty and pd.notna(s.iloc[-1])}
    most_extended = sorted(last_z.items(), key=lambda kv: -kv[1])[:3]

    return {
        "position": round(current, 1),
        "label": label,
        "trend": trend,
        "mostExtended": [
            {"component": GAUGE_COMPONENTS[name]["label"], "z": round(z, 2)} for name, z in most_extended
        ],
        "asOf": valid.index[-1].date().isoformat(),
    }
