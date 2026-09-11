"""
src/fx/series_map.py

Single source of truth for every external series ID the FX model touches.

READ docs/FX-BACKEND-SPEC.md section 3.3 before trusting anything in here.
FRED's OECD-sourced international series are periodically renamed or
discontinued. Every ID below is a *candidate to verify*, not a fact. Run
`python scripts/validate_fx_series.py` and fix whatever it reports before
shipping a snapshot.

Confidence tags travel with each ID so the validator report and the snapshot
`meta.seriesHealth` block can show which inputs are shaky.
"""

from __future__ import annotations

from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Universe (spec section 2)
# ---------------------------------------------------------------------------
SCORED = ["USD", "EUR", "JPY", "GBP", "CHF", "CAD", "AUD", "NZD", "SEK", "NOK"]
DISPLAY_ONLY = ["CNY"]  # managed float -- shown, never scored

CURRENCY_NAMES = {
    "USD": "US Dollar",
    "EUR": "Euro",
    "JPY": "Japanese Yen",
    "GBP": "British Pound",
    "CHF": "Swiss Franc",
    "CAD": "Canadian Dollar",
    "AUD": "Australian Dollar",
    "NZD": "New Zealand Dollar",
    "SEK": "Swedish Krona",
    "NOK": "Norwegian Krone",
    "CNY": "Chinese Yuan",
}

# ISO-4217 quote list for Frankfurter (lower-case, USD base).
FRANKFURTER_QUOTES = [c.lower() for c in SCORED + DISPLAY_ONLY if c != "USD"]

# ---------------------------------------------------------------------------
# Central-bank inflation targets (spec section 4.1 -- "its own target")
# Do NOT compare every country to 2%.
# ---------------------------------------------------------------------------
CB_INFLATION_TARGET: Dict[str, float] = {
    "USD": 2.0,   # Fed, 2% PCE
    "EUR": 2.0,   # ECB
    "JPY": 2.0,   # BoJ
    "GBP": 2.0,   # BoE
    "CHF": 1.0,   # SNB, 0-2% band midpoint
    "CAD": 2.0,   # BoC, 1-3% band midpoint
    "AUD": 2.5,   # RBA, 2-3% band midpoint
    "NZD": 2.0,   # RBNZ, 1-3% band midpoint
    "SEK": 2.0,   # Riksbank
    "NOK": 2.0,   # Norges Bank
}

# ---------------------------------------------------------------------------
# FRED series per currency.
#
#   policy_rate : central-bank / immediate (call money) rate  -- carry
#   y2          : 2y government yield                          -- carry, policyMomentum
#   y10         : 10y government yield (2y fallback)           -- carry, policyMomentum
#   cpi         : consumer price series                        -- macroVsMandate, real carry
#   cpi_kind    : "index" (compute YoY here) | "yoy" (already a YoY %)
#
# A value of None means "no free source found" -- the calculators fall back
# (y2 -> y10) and set proxy:true, or drop the sub-input entirely.
# ---------------------------------------------------------------------------
# These have been probed live against FRED (2026-09-10) with
# scripts/validate_fx_series.py -- see the comment on each entry. Re-run the
# validator periodically; OECD's international feeds move under you.
FRED_SERIES: Dict[str, Dict[str, Optional[str]]] = {
    "USD": {
        "policy_rate": "DFF",           # High -- live, daily
        "y2": "DGS2",                   # High -- live, daily
        "y10": "DGS10",                 # High -- live, daily
        "cpi": "CPIAUCSL",              # High -- live, monthly
        "cpi_kind": "index",
    },
    "EUR": {
        "policy_rate": "ECBDFR",             # High -- ECB deposit facility rate, live daily
        "y2": None,
        "y10": "IRLTLT01EZM156N",            # Low  -- OECD, runs ~8mo behind
        "cpi": "CP0000EZ19M086NEST",         # Medium -- HICP index, live monthly
        "cpi_kind": "index",
    },
    "JPY": {
        "policy_rate": "IRSTCI01JPM156N",   # Low -- OECD immediate rate, ~3mo lag
        "y2": None,
        "y10": "IRLTLT01JPM156N",           # Low -- OECD, ~3mo lag
        "cpi": "FPCPITOTLZGJPN",            # Low -- World Bank annual YoY; OECD MEI CPI
                                             #        feed for JPY is dead on FRED (verified)
        "cpi_kind": "yoy",
    },
    "GBP": {
        "policy_rate": "IRSTCI01GBM156N",   # Low -- ~3mo lag
        "y2": None,
        "y10": "IRLTLT01GBM156N",           # Low -- ~3mo lag
        "cpi": "CPALTT01GBM659N",           # Low -- OECD MEI CPI feed stopped updating
                                             #        ~Feb 2025 (verified); will read as stale
        "cpi_kind": "yoy",
    },
    "CHF": {
        "policy_rate": "IR3TIB01CHM156N",   # Low -- 3m interbank proxy; IRSTCI01CH is dead
        "y2": None,
        "y10": "IRLTLT01CHM156N",           # Low -- ~3mo lag
        "cpi": "CPALTT01CHM659N",           # Low -- OECD MEI CPI feed stopped ~Feb 2025
        "cpi_kind": "yoy",
    },
    "CAD": {
        "policy_rate": "IRSTCI01CAM156N",   # Low -- ~3mo lag
        "y2": None,
        "y10": "IRLTLT01CAM156N",           # Low -- ~3mo lag
        "cpi": "CPALTT01CAM659N",           # Low -- OECD MEI CPI feed stopped ~Feb 2025
        "cpi_kind": "yoy",
    },
    "AUD": {
        "policy_rate": "IRSTCI01AUM156N",   # Low -- ~3mo lag
        "y2": None,
        "y10": "IRLTLT01AUM156N",           # Low -- ~3mo lag
        "cpi": "AUSCPIALLQINMEI",           # Low -- quarterly index; OECD feed stopped ~2025
        "cpi_kind": "index",
    },
    "NZD": {
        "policy_rate": "IR3TIB01NZM156N",   # Low -- 3m interbank proxy; IRSTCI01NZ is dead
        "y2": None,
        "y10": "IRLTLT01NZM156N",           # Low -- ~3mo lag
        "cpi": "NZLCPIALLQINMEI",           # Low -- quarterly index; OECD feed stopped ~2025
        "cpi_kind": "index",
    },
    "SEK": {
        "policy_rate": "IR3TIB01SEM156N",   # Low -- 3m interbank proxy; IRSTCI01SE is dead
        "y2": None,
        "y10": "IRLTLT01SEM156N",           # Low -- ~3mo lag
        "cpi": "CPALTT01SEM659N",           # Low -- OECD MEI CPI feed stopped ~Feb 2025
        "cpi_kind": "yoy",
    },
    "NOK": {
        "policy_rate": "IRSTCI01NOM156N",   # Low -- ~3mo lag
        "y2": None,
        "y10": "IRLTLT01NOM156N",           # Low -- ~3mo lag
        "cpi": "CPALTT01NOM659N",           # Low -- OECD MEI CPI feed stopped ~Feb 2025
        "cpi_kind": "yoy",
    },
    # Display-only. Scored=false, but we still surface the raw data.
    "CNY": {
        "policy_rate": "IR3TIB01CNM156N",   # Low -- ~4mo lag
        "y2": None,
        "y10": None,                        # No free source found (INTGSTCNM193N is a 404)
        "cpi": "CHNCPIALLMINMEI",           # Low -- OECD feed stopped ~2025
        "cpi_kind": "index",
    },
}

# Per-ID confidence, for the validator report / meta.seriesHealth.
# Anything not listed here is treated as "low".
SERIES_CONFIDENCE = {
    "DFF": "high", "DGS2": "high", "DGS10": "high", "CPIAUCSL": "high",
    "DTWEXBGS": "high", "DCOILBRENTEU": "high", "DCOILWTICO": "high",
    "ECBDFR": "high",
    "CP0000EZ19M086NEST": "medium",
    "PCOPPUSDM": "medium", "PIORECRUSDM": "medium", "PALUMUSDM": "medium",
    "DHHNGSP": "medium", "PFOODINDEXM": "medium",
}

# ---------------------------------------------------------------------------
# Publication frequency per FRED series -- drives the staleness cutoff used by
# src/fx/components.py so an OECD feed that quietly stopped updating months
# ago reads as unavailable, not as a silently-repeated last value (spec 3.3).
# Normal publication lag is NOT staleness: OECD MEI series routinely land
# ~90-100 days behind. Unlisted series default to "monthly".
# ---------------------------------------------------------------------------
SERIES_FREQ: Dict[str, str] = {
    "DFF": "daily", "DGS2": "daily", "DGS10": "daily", "ECBDFR": "daily",
    "ECBMRRFR": "daily", "DTWEXBGS": "daily",
    "DCOILBRENTEU": "daily", "DCOILWTICO": "daily", "DHHNGSP": "daily",
    "CPIAUCSL": "monthly", "CP0000EZ19M086NEST": "monthly",
    "PCOPPUSDM": "monthly", "PIORECRUSDM": "monthly", "PALUMUSDM": "monthly",
    "PFOODINDEXM": "monthly",
    "AUSCPIALLQINMEI": "quarterly", "NZLCPIALLQINMEI": "quarterly",
    "FPCPITOTLZGJPN": "annual",
}

# Max age (days) a series' most recent *real* (non-forward-filled) observation
# may be before the component treats it as unavailable rather than stale-but-usable.
STALE_DAYS: Dict[str, int] = {
    "daily": 10, "monthly": 120, "quarterly": 280, "annual": 450,
}


def freq_for(series_id: Optional[str]) -> str:
    if not series_id:
        return "monthly"
    if series_id in SERIES_FREQ:
        return SERIES_FREQ[series_id]
    if series_id.endswith(("Q156N", "Q659N", "Q657N", "QINMEI")):
        return "quarterly"
    return "monthly"


def max_age_for(series_id: Optional[str]) -> int:
    return STALE_DAYS[freq_for(series_id)]

# ---------------------------------------------------------------------------
# Commodity price series (FRED) for termsOfTrade.
# ---------------------------------------------------------------------------
COMMODITY_SERIES: Dict[str, str] = {
    "brent": "DCOILBRENTEU",     # $/bbl, daily
    "wti": "DCOILWTICO",         # $/bbl, daily
    "natgas": "DHHNGSP",         # Henry Hub $/MMBtu, daily
    "copper": "PCOPPUSDM",       # $/mt, monthly
    "iron_ore": "PIORECRUSDM",   # $/mt, monthly
    "aluminum": "PALUMUSDM",     # $/mt, monthly
    "food": "PFOODINDEXM",       # global food price index, monthly (NZD dairy proxy)
}

# Broad USD index used only for reconciliation with the existing macro signal.
DOLLAR_BROAD_SERIES = "DTWEXBGS"

# ---------------------------------------------------------------------------
# Export-weighted commodity baskets (spec section 4.1).
# Positive weight = the country exports it (rising price helps the currency).
# Negative weight = net importer (rising price hurts the currency).
# Weights are export-share approximations; they do not need to sum to 1.
# ---------------------------------------------------------------------------
TERMS_OF_TRADE_BASKET: Dict[str, Dict[str, float]] = {
    "NOK": {"brent": 0.60, "natgas": 0.30, "aluminum": 0.10},
    "CAD": {"wti": 0.55, "copper": 0.15, "aluminum": 0.10, "food": 0.20},
    "AUD": {"iron_ore": 0.45, "copper": 0.20, "brent": 0.15, "food": 0.20},
    "NZD": {"food": 0.90, "brent": -0.10},
    "SEK": {"iron_ore": 0.30, "brent": -0.35, "food": -0.05},
    "USD": {"brent": -0.05, "natgas": 0.05},
    "GBP": {"brent": -0.20, "natgas": -0.15},
    "EUR": {"brent": -0.45, "natgas": -0.55},
    "CHF": {"brent": -0.40, "natgas": -0.20},
    "JPY": {"brent": -0.60, "natgas": -0.40},
}

# ---------------------------------------------------------------------------
# Trade-weighted "trend" basket. A proper NEER per currency has no free daily
# source, so `trend` is built from Frankfurter cross-rates against the other
# scored currencies using these static weights (roughly trade-share ordered).
# proxy:true is recorded on the component because the weights are static.
# ---------------------------------------------------------------------------
TREND_PARTNER_WEIGHTS: Dict[str, Dict[str, float]] = {
    "USD": {"EUR": 0.32, "CAD": 0.18, "JPY": 0.15, "GBP": 0.10, "CHF": 0.06,
            "AUD": 0.06, "SEK": 0.04, "NOK": 0.04, "NZD": 0.05},
    "EUR": {"USD": 0.30, "GBP": 0.20, "CHF": 0.14, "JPY": 0.10, "SEK": 0.09,
            "NOK": 0.06, "CAD": 0.05, "AUD": 0.04, "NZD": 0.02},
    "JPY": {"USD": 0.35, "EUR": 0.24, "AUD": 0.10, "GBP": 0.08, "CAD": 0.07,
            "CHF": 0.06, "NZD": 0.04, "SEK": 0.03, "NOK": 0.03},
    "GBP": {"EUR": 0.42, "USD": 0.25, "CHF": 0.08, "JPY": 0.07, "CAD": 0.06,
            "SEK": 0.05, "AUD": 0.04, "NOK": 0.02, "NZD": 0.01},
    "CHF": {"EUR": 0.48, "USD": 0.22, "GBP": 0.10, "JPY": 0.08, "CAD": 0.04,
            "SEK": 0.03, "AUD": 0.03, "NOK": 0.01, "NZD": 0.01},
    "CAD": {"USD": 0.62, "EUR": 0.14, "JPY": 0.08, "GBP": 0.06, "CHF": 0.03,
            "AUD": 0.03, "NOK": 0.02, "SEK": 0.01, "NZD": 0.01},
    "AUD": {"USD": 0.28, "JPY": 0.20, "EUR": 0.18, "NZD": 0.12, "GBP": 0.08,
            "CAD": 0.06, "CHF": 0.04, "SEK": 0.02, "NOK": 0.02},
    "NZD": {"AUD": 0.34, "USD": 0.22, "JPY": 0.14, "EUR": 0.14, "GBP": 0.08,
            "CAD": 0.04, "CHF": 0.02, "SEK": 0.01, "NOK": 0.01},
    "SEK": {"EUR": 0.44, "USD": 0.16, "NOK": 0.14, "GBP": 0.10, "JPY": 0.06,
            "CHF": 0.05, "CAD": 0.02, "AUD": 0.02, "NZD": 0.01},
    "NOK": {"EUR": 0.42, "SEK": 0.16, "USD": 0.16, "GBP": 0.12, "JPY": 0.06,
            "CHF": 0.04, "CAD": 0.02, "AUD": 0.01, "NZD": 0.01},
}

# Currencies whose authorities actively manage FX -- hard-coded intervention
# watch (spec section 4.4).
INTERVENTION_WATCH = {"JPY", "CHF"}


def confidence_for(series_id: Optional[str]) -> str:
    if not series_id:
        return "none"
    return SERIES_CONFIDENCE.get(series_id, "low")


def logical_max_age() -> Dict[str, int]:
    """{logical_column_name -> max_age_days} for every FRED series in the model."""
    return {logical: max_age_for(sid) for logical, sid in all_series_ids().items()}


def all_series_ids() -> Dict[str, str]:
    """Flat {logical_name: fred_series_id} for every FRED ID the model uses."""
    out: Dict[str, str] = {}
    for ccy, m in FRED_SERIES.items():
        for key, sid in m.items():
            if key == "cpi_kind" or not sid:
                continue
            out[f"{ccy}__{key}"] = sid
    for name, sid in COMMODITY_SERIES.items():
        out[f"cmdty__{name}"] = sid
    out["macro__dollar_broad"] = DOLLAR_BROAD_SERIES
    return out
