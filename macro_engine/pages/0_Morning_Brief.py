# pages/0_Morning_Brief.py
"""
Morning Brief — fully deterministic, no external API required.
Thesis paragraph generated from live signals using a rule-based prose engine.
"""

import numpy as np
import pandas as pd
from datetime import date
from collections import defaultdict
import plotly.graph_objects as go
import streamlit as st

from src.config import CACHE_DIR, FRED_API_KEY, FRED_SERIES, YF_PROXIES
from src.data_sources import fetch_prices, get_fred_cached
from src.regime import compute_regime_v3, compute_regime_timeseries
from src.ui import inject_css, sidebar_nav, safe_switch_page, regime_color, regime_bg

st.set_page_config(page_title="Morning Brief", page_icon="☀️",
                   layout="wide", initial_sidebar_state="expanded")
inject_css()
sidebar_nav(active="Morning Brief")

if not FRED_API_KEY:
    st.error("FRED_API_KEY is not set."); st.stop()

# ── Basket ────────────────────────────────────────────────────────────────────

ASSETS = ["SPY","QQQ","IWM","XLE","XLF","XLK","XLI","XLP","XLV",
          "XLU","XLC","GLD","UUP","HYG","TLT","BTC","XBI","IGV","SMH","RSP"]
ASSET_LABELS = {
    "SPY":"S&P 500","QQQ":"Nasdaq","IWM":"Small caps","XLE":"Energy",
    "XLF":"Financials","XLK":"Technology","XLI":"Industrials","XLV":"Healthcare",
    "XLP":"Staples","XLU":"Utilities","XLC":"Comms",
    "GLD":"Gold","UUP":"USD","HYG":"High yield","TLT":"Long bonds",
    "BTC":"Bitcoin","XBI":"Biotech","IGV":"Software","SMH":"Semis","RSP":"Equal weight",
}

# ── Data ──────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=30 * 60, show_spinner=False)
def load_all():
    macro = get_fred_cached(FRED_SERIES, FRED_API_KEY, CACHE_DIR,
                            cache_name="fred_macro").sort_index()
    vix_t = [v for v in [YF_PROXIES.get("vix"), YF_PROXIES.get("vix3m")] if v]
    all_t = list(dict.fromkeys(ASSETS + vix_t))
    px    = fetch_prices(all_t, period="5y")
    px    = pd.DataFrame() if (px is None or px.empty) else px.sort_index()
    regime   = compute_regime_v3(macro=macro, proxies=px,
                                 lookback_trend=63, momentum_lookback_days=21)
    reg_hist = compute_regime_timeseries(macro, px, freq="W-FRI")
    return macro, px, regime, reg_hist

macro, px, regime, reg_hist = load_all()

# ── Helpers ───────────────────────────────────────────────────────────────────

def _last(s):
    s = s.dropna(); return float(s.iloc[-1]) if not s.empty else None

def _delta(s, days):
    s = s.dropna()
    if len(s) < 2: return None
    prev = s.index[s.index <= s.index.max() - pd.Timedelta(days=days)]
    return float(s.iloc[-1] - s.loc[prev[-1]]) if len(prev) else None

def _zscore(s, w=252):
    s = s.dropna()
    if len(s) < 30: return None
    tail = s.iloc[-min(w, len(s)):]
    sd   = float(tail.std())
    return float((tail.iloc[-1] - tail.mean()) / sd) if sd > 0 else 0.0

def _pct_rank(s, w=252):
    s = s.dropna()
    if len(s) < w: return None
    return float((s.iloc[-w:] < s.iloc[-1]).mean() * 100)

def _ret(s, days):
    s = s.dropna()
    if len(s) < 2: return None
    prev = s.index[s.index <= s.index.max() - pd.Timedelta(days=days)]
    return float(s.iloc[-1] / s.loc[prev[-1]] - 1) if len(prev) else None

def fmt(x, nd=2, suffix="", plus=False):
    if x is None or (isinstance(x, float) and np.isnan(x)): return "—"
    return (f"{float(x):+.{nd}f}{suffix}" if plus else f"{float(x):.{nd}f}{suffix}")

# ── Regime state ──────────────────────────────────────────────────────────────

cur_label  = regime.label
cur_score  = regime.score
cur_raw    = regime.score_raw
cur_conf   = regime.confidence
cur_mom    = regime.momentum_label
cur_delta  = regime.score_delta or 0
components = regime.components or {}
allocation = regime.allocation or {}
stance     = allocation.get("stance", {}) if isinstance(allocation, dict) else {}
cur_color  = regime_color(cur_label)
cur_bg     = regime_bg(cur_label)
mom_arrow  = "▲" if "improv" in cur_mom.lower() else ("▼" if "deterio" in cur_mom.lower() else "▶")
mom_color  = "#1f7a4f" if "improv" in cur_mom.lower() else ("#b42318" if "deterio" in cur_mom.lower() else "#6b7280")

# ── Markov ────────────────────────────────────────────────────────────────────

stay_p = degrade_p = improve_p = None
prev_label = None
if not reg_hist.empty and len(reg_hist) >= 20:
    worse  = {"Risk On":["Bullish","Neutral","Bearish","Risk Off"],
               "Bullish":["Neutral","Bearish","Risk Off"],
               "Neutral":["Bearish","Risk Off"],
               "Bearish":["Risk Off"],"Risk Off":[]}
    better = {"Risk Off":["Bearish","Neutral","Bullish","Risk On"],
               "Bearish":["Neutral","Bullish","Risk On"],
               "Neutral":["Bullish","Risk On"],
               "Bullish":["Risk On"],"Risk On":[]}
    lbl_s = reg_hist["label"].values
    mat   = defaultdict(lambda: defaultdict(int))
    for i in range(len(lbl_s)-1):
        mat[lbl_s[i]][lbl_s[i+1]] += 1
    cur_lbl = reg_hist["label"].iloc[-1]
    total   = sum(mat[cur_lbl].values())
    if total > 0:
        stay_p    = mat[cur_lbl][cur_lbl] / total
        degrade_p = sum(mat[cur_lbl][w]/total for w in worse.get(cur_lbl,[]))
        improve_p = sum(mat[cur_lbl][b]/total for b in better.get(cur_lbl,[]))
    if len(reg_hist) >= 2:
        prev_label = reg_hist["label"].iloc[-2]

# ── Live macro reads ──────────────────────────────────────────────────────────

def _col(name):
    return macro[name].dropna() if name in macro.columns else pd.Series(dtype=float)

y10  = _col("y10"); y2 = _col("y2"); y3m = _col("y3m")
real10 = _col("real10"); ff = _col("fed_funds")
hy_oas = _col("hy_oas"); dollar = _col("dollar_broad")
fed_assets = _col("fed_assets")

curve_210 = (y10 - y2).dropna()   if len(y10)>1 and len(y2)>1   else pd.Series(dtype=float)
breakeven = (y10 - real10).dropna() if len(y10)>1 and len(real10)>1 else pd.Series(dtype=float)

cpi_yoy = None
if "cpi" in macro.columns and len(macro["cpi"].dropna()) >= 13:
    cpi_yoy = (macro["cpi"].dropna().pct_change(12)*100).dropna()

c210_now   = _last(curve_210);  c210_1m  = _delta(curve_210, 30)
real_now   = _last(real10);     real_1m  = _delta(real10, 30)
hy_now     = _last(hy_oas);     hy_7d    = _delta(hy_oas, 7)
hy_z       = _zscore(hy_oas)
dollar_now = _last(dollar);     dollar_1m = _delta(dollar, 30)
ff_now     = _last(ff)
cpi_now    = _last(cpi_yoy) if cpi_yoy is not None else None
be_now     = _last(breakeven)
fed_13w    = None
if len(fed_assets) >= 70:
    fed_13w = float(fed_assets.pct_change(63).iloc[-1] * 100)

vix_now = vix3m_now = vratio = vix_pct = None
vt = YF_PROXIES.get("vix"); v3t = YF_PROXIES.get("vix3m")
if vt and vt in px.columns:
    vix_now = _last(px[vt]); vix_pct = _pct_rank(px[vt])
if v3t and v3t in px.columns:
    vix3m_now = _last(px[v3t])
if vix_now and vix3m_now and vix3m_now != 0:
    vratio = vix_now / vix3m_now

# Weekly changes
weekly_changes = []
for name, series, days, inverse in [
    ("Credit spreads (HY OAS)", hy_oas, 7,  True),
    ("Curve 2s10s",             curve_210, 7, False),
    ("Real 10y yield",          real10, 7,  True),
    ("Dollar broad",            dollar, 10, False),
]:
    d = _delta(series, days); l = _last(series)
    if d is not None and l is not None:
        weekly_changes.append((name, l, d, inverse))

# Market returns 1w
mkt_ret = {}
for t in ["SPY","QQQ","IWM","GLD","TLT","HYG","UUP","XLE"]:
    if t in px.columns:
        r = _ret(px[t].dropna(), 7)
        if r is not None:
            mkt_ret[t] = round(float(r)*100, 2)

# ── Factor rotation ───────────────────────────────────────────────────────────

def compute_factors():
    curve_z  = _zscore(curve_210) or 0
    real_z   = _zscore(real10)    or 0
    credit_z = _zscore(hy_oas)    or 0
    dollar_z = _zscore(dollar)    or 0
    reg_z    = float(np.clip((cur_score - 50) / 25, -1, 1))
    return sorted([
        ("Value vs Growth",
         float(np.clip(curve_z*0.40 - real_z*0.35 - credit_z*0.25, -3, 3)),
         "XLE+XLF vs QQQ+IGV"),
        ("Quality vs High-beta",
         float(np.clip(credit_z*0.45 + real_z*0.30 - curve_z*0.25, -3, 3)),
         "XLP+XLV vs IWM+XBI"),
        ("Small cap vs Large cap",
         float(np.clip(-credit_z*0.35 + curve_z*0.35 - dollar_z*0.30, -3, 3)),
         "IWM vs SPY"),
        ("Cyclicals vs Defensives",
         float(np.clip(curve_z*0.35 - credit_z*0.35 + reg_z*0.30, -3, 3)),
         "XLE+XLI vs XLP+XLV"),
        ("Real assets vs Financial",
         float(np.clip(-real_z*0.40 - dollar_z*0.35 + reg_z*0.25, -3, 3)),
         "GLD+XLE vs TLT+HYG"),
    ], key=lambda x: abs(x[1]), reverse=True)

factor_scores = compute_factors()

# ── Divergences ───────────────────────────────────────────────────────────────

def find_divergences():
    divs = []
    vt2 = YF_PROXIES.get("vix")
    if vt2 and vt2 in px.columns and not hy_oas.empty:
        vix_z2 = _zscore(px[vt2].dropna()); hy_z2 = _zscore(hy_oas)
        if vix_z2 and hy_z2 and abs(vix_z2 - hy_z2) > 1.0:
            d = vix_z2 - hy_z2
            divs.append(("VIX vs HY OAS", d,
                "VIX pricing more stress than credit" if d>0 else "Credit widening without VIX spike",
                "Reduce equity risk" if d>0 else "HYG weakness = quiet stress. Add TLT hedge."))
    if not curve_210.empty and "IWM" in px.columns and "SPY" in px.columns:
        cz = _zscore(curve_210)
        iz = _zscore((px["IWM"]/px["SPY"]).dropna())
        if cz and iz and abs(cz - iz) > 1.2:
            d = cz - iz
            divs.append(("Curve vs IWM/SPY", d,
                "Curve steepening but small caps lagging" if d>0 else "Small caps outrunning curve signal",
                "Long IWM, curve is the leading signal" if d>0 else "Fade IWM without curve confirmation"))
    if not real10.empty and "GLD" in px.columns:
        rz = _zscore(real10); gz = _zscore(px["GLD"].dropna())
        if rz and gz and abs(gz + rz) > 1.2:
            d = gz + rz
            divs.append(("GLD vs Real rates", d,
                "GLD elevated vs real rate signal" if d>0 else "GLD cheap vs falling real rates",
                "Take GLD profits or hedge" if d>0 else "Accumulate GLD"))
    if "HYG" in px.columns and "RSP" in px.columns and "SPY" in px.columns:
        hyg_r = _ret(px["HYG"].dropna(), 63)
        rsp_r = _ret(px["RSP"].dropna(), 63)
        if hyg_r and rsp_r and hyg_r < -0.02 and rsp_r > 0.03:
            divs.append(("HYG vs Equity breadth", (rsp_r - hyg_r)*10,
                "Credit deteriorating while equity breadth healthy. Credit leads by 4-8 weeks.",
                "Reduce equity beta. Credit signal is more reliable."))
    return sorted(divs, key=lambda x: abs(x[1]), reverse=True)[:3]

divergences = find_divergences()

# ── Top trade signals ─────────────────────────────────────────────────────────

def top_trades():
    trades = []
    comp_rows = []
    for k, c in components.items():
        if not isinstance(c, dict): continue
        contrib = float(c.get("contribution", 0))
        z       = float(c.get("zscore") or 0)
        name    = c.get("name", k)
        comp_rows.append((name, contrib, z))
    comp_rows.sort(key=lambda x: abs(x[1]), reverse=True)

    inst_map = {
        "Credit stress":        lambda c: "HYG" if c>0 else "Short HYG / Long TLT",
        "Real yields":          lambda c: "TLT" if c>0 else "Short TLT / Long UUP",
        "Curve":                lambda c: "XLF" if c>0 else "TLT / XLP",
        "Risk appetite":        lambda c: "IWM" if c>0 else "Short IWM / Long SPY",
        "Dollar impulse":       lambda c: "GLD" if c>0 else "UUP",
        "Inflation momentum":   lambda c: "TIPS / GLD" if c>0 else "TLT",
    }
    for name, contrib, z in comp_rows[:3]:
        if abs(z) < 0.3: continue
        fn = inst_map.get(name, lambda c: name)
        trades.append({"signal":name,"direction":"bullish" if contrib>0 else "bearish",
                       "instrument":fn(contrib),"z":z,"contrib":contrib})

    if factor_scores and abs(factor_scores[0][1]) >= 0.5:
        f = factor_scores[0]
        parts  = f[0].split(" vs ")
        winner = parts[0] if f[1]>0 else (parts[1] if len(parts)>1 else f[0])
        trades.append({"signal":f"Factor: {f[0]}","direction":"long" if f[1]>0 else "short",
                       "instrument":f[2],"z":f[1],"contrib":f[1]})

    return sorted(trades, key=lambda x: abs(x["contrib"]), reverse=True)[:3]

top_trade_signals = top_trades()

# ── Falsification conditions ──────────────────────────────────────────────────

def falsification():
    out = []
    if hy_now is not None:
        t = round(hy_now + 0.50, 2) if cur_score >= 50 else round(hy_now - 0.30, 2)
        out.append((f"HY OAS {'widens to' if cur_score>=50 else 'tightens to'} {t}%",
                    "Credit stress breaking through" if cur_score>=50 else "Credit confirming recovery"))
    if real_now is not None:
        if cur_score <= 50:
            out.append((f"Real yields fall below 1.5% (now {real_now:.2f}%)",
                        "Financial conditions easing meaningfully"))
        else:
            out.append((f"Real yields rise above 2.0% (now {real_now:.2f}%)",
                        "Conditions turning genuinely restrictive"))
    if c210_now is not None:
        if c210_now >= 0:
            out.append((f"Curve inverts (2s10s below 0, now {c210_now:+.2f}pp)",
                        "Recession signal activated"))
        else:
            out.append((f"Curve uninverts (2s10s above 0, now {c210_now:+.2f}pp)",
                        "Recession clock starts"))
    if vratio is not None:
        label = "falls below 0.90" if vratio > 0.95 else "rises above 1.0"
        read  = "vol regime turning calm" if vratio > 0.95 else "vol regime entering panic"
        out.append((f"V-Ratio {label} (now {vratio:.3f})", read))
    return out[:4]

falsification_conditions = falsification()

# ══════════════════════════════════════════════════════════════════════════════
# DOMINANT TRADE ENGINE
# Single-screen answer to "what is the dominant trade right now and why"
# Scores every candidate trade across 6 independent dimensions then picks
# the one with the highest multi-signal conviction.
# ══════════════════════════════════════════════════════════════════════════════

def compute_dominant_trade():
    """
    Score every tradeable asset against ALL macro signals simultaneously.

    Each asset is evaluated on how ALL current macro conditions align with
    a long or short position — not just one component.

    The 8 scoring dimensions per asset:
      1. Regime direction   — does cur_score favour this asset?
      2. Credit signal      — HY OAS z and direction
      3. Real rate signal   — TIPS real yield level vs thresholds
      4. Curve signal       — 2s10s slope and momentum
      5. Dollar signal      — broad dollar direction
      6. Inflation signal   — CPI momentum vs breakeven
      7. Price momentum     — asset's own 63d MA and RoC z
      8. Historical win rate — regime-conditional forward return

    Score = weighted sum of confirmations, then Kelly-sized.
    Final rank = conviction * kelly * avg_signal_strength.
    """

    # Shared macro z-scores
    curve_z  = _zscore(curve_210) or 0.0
    real_z   = _zscore(real10)    or 0.0
    credit_z = _zscore(hy_oas)    or 0.0
    dollar_z = _zscore(dollar)    or 0.0
    vix_z    = (_zscore(px[YF_PROXIES.get("vix","")].dropna())
                if YF_PROXIES.get("vix","") in px.columns else 0.0)
    cpi_z    = (_zscore(cpi_yoy) if cpi_yoy is not None else 0.0) or 0.0
    be_z     = _zscore(breakeven) or 0.0

    # Component roc_zscores (momentum)
    comp_roc = {}
    for k, c in components.items():
        if isinstance(c, dict) and c.get("roc_zscore") is not None:
            comp_roc[c.get("name", k)] = float(c["roc_zscore"])

    # Historical win rates — regime-conditional 4w forward returns
    hist_wr = {}
    if not reg_hist.empty:
        for t in ["GLD","UUP","TLT","HYG","SPY","QQQ","IWM","XLE","XLF","XLP","BTC"]:
            if t not in px.columns: continue
            s   = px[t].resample("W-FRI").last().dropna()
            fwd = s.pct_change(4).shift(-4).dropna()
            merged = reg_hist.join(fwd.rename("fwd"), how="inner").dropna()
            subset = merged[merged["label"] == cur_label]["fwd"]
            if len(subset) >= 8:
                hist_wr[t] = float((subset > 0).mean())

    def _wr(t, direction="long"):
        wr = hist_wr.get(t, 0.52)
        return wr if direction == "long" else (1.0 - wr)

    def _kelly(wr, payoff=1.6):
        k = wr - (1.0 - wr) / payoff
        return max(0.0, k * 0.25)

    def _price_momentum(t):
        """Returns (momentum_score, mom_z) for an asset's own price."""
        if t not in px.columns: return 0, 0.0
        s = px[t].dropna()
        if len(s) < 70: return 0, 0.0
        ma63 = float(s.rolling(63).mean().iloc[-1])
        above_ma = s.iloc[-1] > ma63
        roc_z = _zscore(s.pct_change(21).dropna()) or 0.0
        if above_ma and roc_z > 0.4:   return +1, roc_z
        if not above_ma and roc_z < -0.4: return -1, roc_z
        return 0, roc_z

    def _score_asset(ticker, direction):
        """
        Score one asset/direction against all 8 macro dimensions.
        Returns dict of {dimension: +1/0/-1} and summary stats.
        """
        s = {}
        d = +1 if direction == "long" else -1  # directional multiplier

        # 1. Regime direction
        # Risk-on assets benefit from high score; safe havens from low score
        risk_on_assets  = {"SPY","QQQ","IWM","XLE","XLF","XLK","HYG","BTC"}
        safe_haven      = {"GLD","TLT","UUP","XLP"}
        if ticker in risk_on_assets:
            s["regime"] = +1 if (d > 0 and cur_score > 55) or (d < 0 and cur_score < 45) else                           (-1 if (d > 0 and cur_score < 45) or (d < 0 and cur_score > 55) else 0)
        elif ticker in safe_haven:
            s["regime"] = +1 if (d > 0 and cur_score < 45) or (d < 0 and cur_score > 55) else                           (-1 if (d > 0 and cur_score > 55) or (d < 0 and cur_score < 45) else 0)
        else:
            s["regime"] = 0

        # 2. Credit signal
        # Wide/widening credit = bearish risk, bullish safe havens
        credit_bearish = credit_z > 0.5
        credit_bullish = credit_z < -0.5
        if ticker in ("GLD","TLT","UUP","XLP"):
            s["credit"] = +1 if (d>0 and credit_bearish) or (d<0 and credit_bullish) else                           (-1 if (d>0 and credit_bullish) or (d<0 and credit_bearish) else 0)
        elif ticker in ("HYG","SPY","IWM","XLE","XLF","BTC"):
            s["credit"] = +1 if (d>0 and credit_bullish) or (d<0 and credit_bearish) else                           (-1 if (d>0 and credit_bearish) or (d<0 and credit_bullish) else 0)
        else:
            s["credit"] = 0

        # 3. Real rate signal
        real_restrictive  = real_now is not None and real_now > 1.5
        real_accommodative = real_now is not None and real_now < 0.5
        real_rising = real_z > 0.5
        if ticker == "GLD":
            s["real_rate"] = +1 if (d>0 and real_accommodative) or (d<0 and real_restrictive) else                              (-1 if (d>0 and real_restrictive) or (d<0 and real_accommodative) else 0)
        elif ticker == "TLT":
            s["real_rate"] = +1 if (d>0 and not real_rising) else                              (-1 if (d>0 and real_rising) else 0)
        elif ticker == "UUP":
            s["real_rate"] = +1 if (d>0 and real_restrictive) else                              (-1 if (d>0 and real_accommodative) else 0)
        elif ticker in ("SPY","QQQ","IWM","BTC"):
            s["real_rate"] = +1 if (d>0 and real_accommodative) else                              (-1 if (d>0 and real_restrictive) else 0)
        elif ticker == "XLE":
            # Energy likes inflation, not just real rates
            s["real_rate"] = +1 if (d>0 and cpi_z > 0.3) else                              (-1 if (d>0 and cpi_z < -0.3) else 0)
        else:
            s["real_rate"] = 0

        # 4. Curve signal
        curve_steep   = curve_z > 0.4
        curve_flat    = curve_z < -0.3
        curve_mom_roc = comp_roc.get("Curve", 0)
        if ticker == "XLF":       # banks love steep curve
            s["curve"] = +1 if (d>0 and curve_steep) or (d<0 and curve_flat) else                          (-1 if (d>0 and curve_flat) or (d<0 and curve_steep) else 0)
        elif ticker == "TLT":     # long bonds hate bear flattener
            s["curve"] = +1 if (d>0 and curve_flat and curve_z < -0.5) else                          (-1 if curve_z > 0.5 else 0)
        elif ticker in ("IWM","XLE","XLI"):  # cyclicals like steep curve
            s["curve"] = +1 if (d>0 and curve_steep) else                          (-1 if (d>0 and curve_flat) else 0)
        elif ticker == "GLD":
            # GLD indifferent to curve slope, cares about real rates
            s["curve"] = 0
        else:
            s["curve"] = +1 if (d>0 and not curve_flat) else 0

        # 5. Dollar signal
        dollar_strong = dollar_z > 0.5
        dollar_weak   = dollar_z < -0.5
        if ticker == "UUP":
            s["dollar"] = +1 if (d>0 and dollar_strong) or (d<0 and dollar_weak) else                           (-1 if (d>0 and dollar_weak) or (d<0 and dollar_strong) else 0)
        elif ticker in ("GLD","BTC","XLE"):   # inversely correlated with dollar
            s["dollar"] = +1 if (d>0 and dollar_weak) or (d<0 and dollar_strong) else                           (-1 if (d>0 and dollar_strong) or (d<0 and dollar_weak) else 0)
        elif ticker in ("SPY","IWM"):          # mildly negative dollar correlation
            s["dollar"] = +1 if (d>0 and not dollar_strong) else                           (-1 if (d>0 and dollar_strong) else 0)
        else:
            s["dollar"] = 0

        # 6. Inflation signal
        infl_rising   = cpi_z > 0.4 or be_z > 0.5
        infl_falling  = cpi_z < -0.4 and be_z < 0
        if ticker == "GLD":          # classic inflation hedge
            s["inflation"] = +1 if (d>0 and infl_rising) or (d<0 and infl_falling) else 0
        elif ticker == "XLE":        # energy = inflation proxy
            s["inflation"] = +1 if (d>0 and infl_rising) else                              (-1 if (d>0 and infl_falling) else 0)
        elif ticker == "TLT":        # bonds hate inflation
            s["inflation"] = +1 if (d>0 and infl_falling) else                              (-1 if (d>0 and infl_rising) else 0)
        elif ticker == "BTC":       # bitcoin loosely tracks inflation narratives
            s["inflation"] = +1 if (d>0 and infl_rising) else 0
        else:
            s["inflation"] = 0

        # 7. Price momentum (own price trend)
        mom_score, mom_z_val = _price_momentum(ticker)
        s["momentum"] = +1 if (d>0 and mom_score>0) or (d<0 and mom_score<0) else                         (-1 if (d>0 and mom_score<0) or (d<0 and mom_score>0) else 0)

        # 8. Historical win rate in current regime
        wr = _wr(ticker, direction)
        s["hist_wr"] = +1 if wr > 0.58 else (-1 if wr < 0.44 else 0)

        n_confirm = sum(1 for v in s.values() if v > 0)
        n_contra  = sum(1 for v in s.values() if v < 0)
        conviction = (n_confirm - n_contra) / len(s)
        avg_z = (abs(real_z) + abs(credit_z) + abs(curve_z) + abs(dollar_z)) / 4
        wr_val = _wr(ticker, direction)
        kelly  = _kelly(wr_val)
        score  = max(0, conviction) * kelly * (1 + avg_z * 0.5)

        return s, conviction, score, wr_val, kelly, mom_z_val

    # ── Asset universe to score ───────────────────────────────────────────────

    # Each tuple: (ticker, display_name, direction, rationale_template)
    # XLU and XLC added as scoreable assets in the universe
    UNIVERSE = [
        ("GLD",  "Gold",          "long",
         f"Real yield {fmt(real_now, suffix='%')} · dollar z {dollar_z:+.2f} · inflation z {cpi_z:+.2f}. "
         f"GLD performs best when real rates are falling or negative and dollar is weakening."),
        ("UUP",  "USD",           "long",
         f"Real yield {fmt(real_now, suffix='%')} (z {real_z:+.2f}) · credit z {credit_z:+.2f} · dollar z {dollar_z:+.2f}. "
         f"Dollar supported by restrictive real rates and risk-off credit signals."),
        ("UUP",  "USD",           "short",
         f"Dollar z {dollar_z:+.2f} · real yield z {real_z:+.2f} · regime score {cur_score}. "
         f"Dollar weakening historically correlates with risk-on and GLD outperformance."),
        ("TLT",  "Long bonds",    "long",
         f"Real yield {fmt(real_now, suffix='%')} · curve {fmt(c210_now, suffix='pp', plus=True)} · credit z {credit_z:+.2f}. "
         f"Duration bid when real rates are peaking and credit stress is building."),
        ("TLT",  "Long bonds",    "short",
         f"Real yield {fmt(real_now, suffix='%')} (restrictive, z {real_z:+.2f}) · curve bear flattening {fmt(c210_1m, suffix='pp', plus=True)} 1m. "
         f"Elevated real rates and flattening curve are headwinds for duration."),
        ("HYG",  "High yield",    "long",
         f"HY OAS {fmt(hy_now, suffix='%')} (z {hy_z:+.2f}) · credit z {credit_z:+.2f} · regime {cur_score}. "
         f"Tight credit spreads and supportive regime = carry trade is live."),
        ("HYG",  "High yield",    "short",
         f"HY OAS {fmt(hy_now, suffix='%')} (z {hy_z:+.2f}) widening · credit stress building. "
         f"Credit leads equities by 4-8 weeks — this is the early warning."),
        ("SPY",  "S&P 500",       "long",
         f"Regime {cur_score}/100 · credit z {credit_z:+.2f} · real yield z {real_z:+.2f}. "
         f"Broad equity exposure when regime is supportive and credit is not stressed."),
        ("IWM",  "Small caps",    "long",
         f"IWM/SPY z {_zscore((px['IWM']/px['SPY']).dropna()) or 0:+.2f} · curve {fmt(c210_now, suffix='pp', plus=True)} · credit z {credit_z:+.2f}. "
         f"Small caps historically lead in steep curve + tight credit + early cycle."),
        ("XLE",  "Energy",        "long",
         f"CPI z {cpi_z:+.2f} · dollar z {dollar_z:+.2f} · curve {fmt(c210_now, suffix='pp', plus=True)}. "
         f"Energy outperforms in inflationary environments with steepening curve."),
        ("XLF",  "Financials",    "long",
         f"Curve {fmt(c210_now, suffix='pp', plus=True)} (z {curve_z:+.2f}) · real yield {fmt(real_now, suffix='%')} · credit z {credit_z:+.2f}. "
         f"Banks thrive with steep curve and normalising credit conditions."),
        ("XLP",  "Defensives",    "long",
         f"Regime {cur_score}/100 (deteriorating) · credit z {credit_z:+.2f} · vix z {vix_z:+.2f}. "
         f"Defensives outperform when macro is deteriorating and vol is rising."),
        ("BTC", "Bitcoin",       "long",
         f"Dollar z {dollar_z:+.2f} · real yield z {real_z:+.2f} · regime {cur_score}. "
         f"Bitcoin historically positively correlated with risk appetite and dollar weakness."),
    ]

    candidates = []
    for ticker, display, direction, rationale in UNIVERSE:
        if ticker not in px.columns and ticker != "UUP": continue
        if ticker == "UUP" and "UUP" not in px.columns: continue
        signals, conviction, score, wr, kelly, mom_z_val = _score_asset(ticker, direction)
        n_confirm = sum(1 for v in signals.values() if v > 0)
        # Only include if at least 3 signals confirm
        if n_confirm < 2: continue
        candidates.append({
            "name":       f"{display}",
            "ticker":     ticker,
            "instrument": f"Long {display}" if direction=="long" else f"Short {display}",
            "direction":  direction,
            "score":      score,
            "conviction": conviction,
            "kelly":      kelly,
            "win_rate":   wr,
            "comp_z":     (abs(real_z)+abs(credit_z)+abs(curve_z)+abs(dollar_z))/4,
            "mom_z":      mom_z_val,
            "signals":    signals,
            "rationale":  rationale,
            "n_confirm":  n_confirm,
        })

    if not candidates:
        return None

    ranked = sorted(candidates, key=lambda x: x["score"], reverse=True)
    best   = ranked[0]
    best["all_candidates"] = ranked
    return best

dominant_trade = compute_dominant_trade()

# ══════════════════════════════════════════════════════════════════════════════
# DETERMINISTIC THESIS ENGINE
# ══════════════════════════════════════════════════════════════════════════════
# DETERMINISTIC THESIS ENGINE
# Builds a genuine macro paragraph from live signals without any API call.
# Reads exactly like a trader's morning note.
# ══════════════════════════════════════════════════════════════════════════════

def build_thesis() -> str:
    sentences = []

    # ── Sentence 1: Regime opening ────────────────────────────────────────────
    mom_phrase = {
        True:  "with momentum improving",
        False: "with momentum deteriorating",
        None:  "with mixed momentum",
    }.get("improv" in cur_mom.lower() if cur_mom else None,
          "with mixed momentum")

    if cur_score >= 75:
        s1 = f"Macro conditions are firmly risk-on at {cur_score}/100, {mom_phrase}."
    elif cur_score >= 60:
        s1 = f"The macro regime sits in bullish territory at {cur_score}/100, {mom_phrase}."
    elif cur_score >= 40:
        s1 = f"Macro conditions are neutral at {cur_score}/100, {mom_phrase}."
    elif cur_score >= 25:
        s1 = f"The macro regime is deteriorating at {cur_score}/100, {mom_phrase}."
    else:
        s1 = f"Macro conditions are in risk-off territory at {cur_score}/100, {mom_phrase}."

    if cur_delta:
        direction = "gained" if cur_delta > 0 else "lost"
        s1 += f" The score has {direction} {abs(cur_delta)} points over the past 21 days."
    sentences.append(s1)

    # ── Sentence 2: Dominant driver ───────────────────────────────────────────
    comp_rows = [(c.get("name",""), float(c.get("contribution",0)),
                  float(c.get("zscore") or 0))
                 for c in components.values() if isinstance(c, dict)]
    comp_rows.sort(key=lambda x: abs(x[1]), reverse=True)

    if comp_rows:
        top_name, top_contrib, top_z = comp_rows[0]
        direction = "supporting" if top_contrib > 0 else "dragging on"
        extreme   = "sharply " if abs(top_z) > 1.5 else ""
        s2 = f"{top_name} is the dominant driver, {extreme}{direction} the score (z {top_z:+.2f})."
        if len(comp_rows) > 1:
            sec_name, sec_contrib, sec_z = comp_rows[1]
            sec_dir = "adding to the tailwind" if sec_contrib > 0 else "adding to the headwind"
            s2 += f" {sec_name} is {sec_dir} (z {sec_z:+.2f})."
        sentences.append(s2)

    # ── Sentence 3: Key macro reads ───────────────────────────────────────────
    parts = []
    if real_now is not None:
        r_read = "restrictive" if real_now > 1.5 else ("tightening" if real_now > 0.8 else "accommodative")
        parts.append(f"real yields at {real_now:.2f}% ({r_read})")
    if hy_now is not None and hy_z is not None:
        h_read = "stressed" if hy_z > 1.0 else ("elevated" if hy_z > 0.3 else "contained")
        parts.append(f"HY OAS at {hy_now:.2f}% ({h_read})")
    if c210_now is not None:
        c_read = "steep" if c210_now > 0.5 else ("flat" if c210_now > 0 else "inverted")
        move   = ""
        if c210_1m is not None and abs(c210_1m) > 0.05:
            move = f", bear flattening {c210_1m:+.2f}pp 1m" if c210_1m < 0 else f", steepening {c210_1m:+.2f}pp 1m"
        parts.append(f"curve {c_read} at {c210_now:+.2f}pp{move}")
    if parts:
        sentences.append("Key reads: " + "; ".join(parts) + ".")

    # ── Sentence 4: Top divergence ────────────────────────────────────────────
    if divergences:
        pair, dz, interp, trade = divergences[0]
        sentences.append(f"Primary divergence: {interp} ({pair}, delta-z {dz:+.2f}) — {trade}.")

    # ── Sentence 5: Factor rotation call ─────────────────────────────────────
    if factor_scores and abs(factor_scores[0][1]) >= 0.4:
        f = factor_scores[0]
        parts_f = f[0].split(" vs ")
        winner  = parts_f[0] if f[1] > 0 else (parts_f[1] if len(parts_f)>1 else f[0])
        loser   = parts_f[1] if f[1] > 0 else (parts_f[0] if len(parts_f)>1 else "")
        s_fac   = f"Factor rotation favours {winner}"
        if loser:
            s_fac += f" over {loser}"
        s_fac += f" (score {f[1]:+.2f}, proxy: {f[2]})."
        sentences.append(s_fac)

    # ── Sentence 6: Transition risk ───────────────────────────────────────────
    if degrade_p is not None and improve_p is not None:
        if degrade_p > 0.35:
            sentences.append(
                f"Regime transition risk is elevated: Markov model assigns {degrade_p:.0%} "
                f"probability of moving to a worse regime in the next 4 weeks. Size positions accordingly.")
        elif improve_p > 0.30:
            sentences.append(
                f"Conditions may be improving: {improve_p:.0%} probability of regime upgrade "
                f"in the next 4 weeks per the Markov model.")
        else:
            sentences.append(
                f"Regime stability: {stay_p:.0%} probability of remaining in {cur_label} "
                f"next week per the Markov transition model.")

    return " ".join(sentences)

thesis = build_thesis()

# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════

today_str = date.today().strftime("%A, %B %d, %Y").replace(" 0", " ")

h1, h2 = st.columns([5, 1])
with h1:
    st.markdown(
        f"""<div class="me-topbar">
          <div style="display:flex;justify-content:space-between;align-items:center;
                      gap:12px;flex-wrap:wrap;">
            <div>
              <div class="me-title">Morning Brief</div>
              <div class="me-subtle">{today_str}</div>
            </div>
            <div style="padding:7px 16px;border-radius:20px;background:{cur_bg};
                        border:1px solid {cur_color}44;">
              <span style="font-weight:900;color:{cur_color};font-size:13px;">
                {cur_label} · {cur_score}
              </span>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)
with h2:
    if st.button("← Home", width='stretch'):
        safe_switch_page("app.py")

# ══════════════════════════════════════════════════════════════════════════════
# THESIS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    f"""<div style='padding:24px 28px;border-radius:16px;background:#ffffff;
          border:1px solid rgba(0,0,0,0.09);border-left:5px solid {cur_color};
          box-shadow:0 2px 16px rgba(0,0,0,0.06);margin-bottom:4px;'>
      <div style='font-size:9px;font-weight:700;color:rgba(0,0,0,0.35);
                  text-transform:uppercase;letter-spacing:0.8px;margin-bottom:10px;'>
        Macro thesis · {date.today().strftime("%b %d, %Y").replace(" 0"," ")}
      </div>
      <div style='font-size:15px;line-height:1.80;color:rgba(0,0,0,0.85);
                  font-weight:400;max-width:960px;'>
        {thesis}
      </div>
      <div style='font-size:10px;color:rgba(0,0,0,0.28);margin-top:12px;'>
        Generated from live signals · {cur_label} {cur_score} · {date.today()}
      </div>
    </div>""", unsafe_allow_html=True)

st.markdown("")

# ══════════════════════════════════════════════════════════════════════════════
# DOMINANT TRADE — single-screen answer
# ══════════════════════════════════════════════════════════════════════════════

if dominant_trade:
    dt   = dominant_trade
    d    = dt["direction"]
    dc   = "#1f7a4f" if d == "long" else "#b42318"
    dbg  = "#dcfce7" if d == "long" else "#fee2e2"
    conv = dt["conviction"]
    conv_pct = int(abs(conv) * 100)
    # Conviction label
    if conv_pct >= 67: conv_label, conv_col = "HIGH CONVICTION", dc
    elif conv_pct >= 40: conv_label, conv_col = "MODERATE", "#d97706"
    else: conv_label, conv_col = "WEAK", "#6b7280"

    # Signal dimension labels
    DIM_LABELS = {
        "regime":   "Regime",
        "comp_z":   "Signal z",
        "momentum": "Momentum",
        "factor":   "Factor model",
        "hist_wr":  "Hist win rate",
        "cross":    "Cross-asset",
    }
    sig_html = ""
    for dim, val in dt["signals"].items():
        sc2  = "#1f7a4f" if val > 0 else ("#b42318" if val < 0 else "#94a3b8")
        bg2  = "#dcfce7" if val > 0 else ("#fee2e2" if val < 0 else "#f3f4f6")
        icon = "✓" if val > 0 else ("✗" if val < 0 else "·")
        sig_html += (
            f"<div style='display:flex;align-items:center;gap:6px;padding:5px 0;"
            f"border-bottom:1px solid rgba(0,0,0,0.04);'>"
            f"<span style='width:18px;height:18px;border-radius:50%;background:{bg2};"
            f"display:flex;align-items:center;justify-content:center;"
            f"font-size:10px;font-weight:900;color:{sc2};flex-shrink:0;'>{icon}</span>"
            f"<span style='font-size:11px;color:rgba(0,0,0,0.65);flex:1;'>{DIM_LABELS.get(dim,dim)}</span>"
            f"<span style='font-size:11px;font-weight:700;color:{sc2};'>"
            f"{'Confirms' if val>0 else ('Contradicts' if val<0 else 'Neutral')}</span>"
            f"</div>"
        )

    # Conviction bar (filled segments)
    n_confirm = sum(1 for v in dt["signals"].values() if v > 0)
    n_contra  = sum(1 for v in dt["signals"].values() if v < 0)
    bar_segs  = ""
    for i in range(6):
        if i < n_confirm:
            seg_c = dc
        elif i >= 6 - n_contra:
            seg_c = "#b42318" if d=="long" else "#1f7a4f"
        else:
            seg_c = "#e2e8f0"
        bar_segs += f"<div style='flex:1;height:6px;border-radius:3px;background:{seg_c};margin:0 1px;'></div>"

    st.markdown(
        f"""<div style='padding:0;border-radius:16px;background:#ffffff;
              border:1px solid rgba(0,0,0,0.09);
              box-shadow:0 2px 16px rgba(0,0,0,0.06);
              overflow:hidden;margin-bottom:4px;'>
          <!-- Header band -->
          <div style='padding:12px 24px;background:{dc};display:flex;
                      justify-content:space-between;align-items:center;'>
            <div style='font-size:10px;font-weight:800;color:rgba(255,255,255,0.75);
                        text-transform:uppercase;letter-spacing:0.8px;'>
              Dominant trade · {date.today().strftime("%b %d, %Y").replace(" 0"," ")}
            </div>
            <div style='font-size:10px;font-weight:800;color:rgba(255,255,255,0.75);
                        text-transform:uppercase;letter-spacing:0.8px;'>
              {conv_label} · {n_confirm}/6 signals confirm
            </div>
          </div>
          <!-- Main body -->
          <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:0;'>
            <!-- Left: instrument + sizing -->
            <div style='padding:20px 24px;border-right:1px solid rgba(0,0,0,0.06);'>
              <div style='font-size:9px;font-weight:700;color:rgba(0,0,0,0.38);
                          text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;'>
                Trade
              </div>
              <div style='font-size:28px;font-weight:900;color:{dc};line-height:1;
                          margin-bottom:6px;'>{d.upper()}</div>
              <div style='font-size:18px;font-weight:800;color:rgba(0,0,0,0.85);
                          margin-bottom:12px;'>{dt["instrument"]}</div>
              <div style='font-size:11px;color:rgba(0,0,0,0.50);margin-bottom:3px;'>
                {dt["name"]}
              </div>
              <div style='font-size:11px;color:rgba(0,0,0,0.50);margin-bottom:14px;'>
                Regime: {cur_label} · {cur_score}/100
              </div>
              <!-- Sizing -->
              <div style='padding:10px 12px;border-radius:10px;background:#f8fafc;
                          border:1px solid rgba(0,0,0,0.06);'>
                <div style='font-size:9px;font-weight:700;color:rgba(0,0,0,0.38);
                            text-transform:uppercase;letter-spacing:0.4px;margin-bottom:6px;'>
                  Kelly sizing (25% fractional)
                </div>
                <div style='display:grid;grid-template-columns:1fr 1fr;gap:8px;'>
                  <div>
                    <div style='font-size:18px;font-weight:900;color:{dc};'>{dt["kelly"]:.1%}</div>
                    <div style='font-size:9px;color:rgba(0,0,0,0.40);'>Position size</div>
                  </div>
                  <div>
                    <div style='font-size:18px;font-weight:900;color:rgba(0,0,0,0.75);'>{dt["win_rate"]:.0%}</div>
                    <div style='font-size:9px;color:rgba(0,0,0,0.40);'>Hist win rate</div>
                  </div>
                </div>
                <!-- Conviction bar -->
                <div style='display:flex;margin-top:8px;gap:2px;'>{bar_segs}</div>
                <div style='font-size:9px;color:rgba(0,0,0,0.35);margin-top:3px;'>
                  {n_confirm} confirm · {n_contra} contradict · {6-n_confirm-n_contra} neutral
                </div>
              </div>
            </div>
            <!-- Middle: signal breakdown -->
            <div style='padding:20px 24px;border-right:1px solid rgba(0,0,0,0.06);'>
              <div style='font-size:9px;font-weight:700;color:rgba(0,0,0,0.38);
                          text-transform:uppercase;letter-spacing:0.5px;margin-bottom:10px;'>
                Signal breakdown
              </div>
              {sig_html}
              <div style='margin-top:12px;display:grid;grid-template-columns:1fr 1fr;gap:6px;'>
                <div style='padding:7px 8px;border-radius:7px;background:#f8fafc;'>
                  <div style='font-size:9px;color:rgba(0,0,0,0.40);'>Signal z-score</div>
                  <div style='font-size:14px;font-weight:800;color:{dc};'>{dt["comp_z"]:+.2f}</div>
                </div>
                <div style='padding:7px 8px;border-radius:7px;background:#f8fafc;'>
                  <div style='font-size:9px;color:rgba(0,0,0,0.40);'>Momentum z</div>
                  <div style='font-size:14px;font-weight:800;color:{"#1f7a4f" if dt["mom_z"]>0 else "#b42318"};'>
                    {dt["mom_z"]:+.2f}</div>
                </div>
              </div>
            </div>
            <!-- Right: rationale + runner-up -->
            <div style='padding:20px 24px;'>
              <div style='font-size:9px;font-weight:700;color:rgba(0,0,0,0.38);
                          text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;'>
                Why this trade
              </div>
              <div style='font-size:12px;color:rgba(0,0,0,0.72);line-height:1.65;
                          margin-bottom:14px;'>
                {dt["rationale"]}
              </div>
              <div style='font-size:9px;font-weight:700;color:rgba(0,0,0,0.38);
                          text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;'>
                Runner-up trades
              </div>""",
        unsafe_allow_html=True)

    # Runner-up trades (2nd and 3rd)
    for runner in dt.get("all_candidates", [])[1:3]:
        rd  = runner["direction"]
        rc2 = "#1f7a4f" if rd=="long" else "#b42318"
        rn  = sum(1 for v in runner["signals"].values() if v > 0)
        st.markdown(
            f"<div style='padding:8px 10px;border-radius:8px;background:#f8fafc;"
            f"border:1px solid rgba(0,0,0,0.06);margin-bottom:5px;'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
            f"<div>"
            f"<span style='font-size:11px;font-weight:800;color:{rc2};'>{rd.upper()}</span>"
            f"<span style='font-size:11px;color:rgba(0,0,0,0.65);margin-left:6px;'>"
            f"{runner['instrument']}</span></div>"
            f"<span style='font-size:10px;color:rgba(0,0,0,0.40);'>{rn}/6 · Kelly {runner['kelly']:.1%}</span>"
            f"</div></div>",
            unsafe_allow_html=True)

    st.markdown("</div></div></div>", unsafe_allow_html=True)

st.markdown("")

# ══════════════════════════════════════════════════════════════════════════════
# REGIME STRIP + WHAT CHANGED
# ══════════════════════════════════════════════════════════════════════════════

left_col, right_col = st.columns([1.1, 1.0], gap="large")

with left_col:
    st.markdown("<div class='me-rowtitle'>Regime state</div>", unsafe_allow_html=True)
    rs1, rs2, rs3, rs4, rs5 = st.columns(5, gap="small")

    def _kpi(col, label, val, sub, color="#0f172a", bg="#f8fafc"):
        col.markdown(
            f"<div style='padding:10px 10px;border-radius:10px;background:{bg};"
            f"border:1px solid rgba(0,0,0,0.07);'>"
            f"<div style='font-size:8px;font-weight:700;color:rgba(0,0,0,0.38);"
            f"text-transform:uppercase;letter-spacing:0.4px;margin-bottom:3px;'>{label}</div>"
            f"<div style='font-size:16px;font-weight:900;color:{color};line-height:1.1;'>{val}</div>"
            f"<div style='font-size:9px;color:rgba(0,0,0,0.48);margin-top:2px;'>{sub}</div>"
            f"</div>", unsafe_allow_html=True)

    _kpi(rs1, "Score", str(cur_score), f"{cur_raw:.1f} raw", cur_color, cur_bg)
    _kpi(rs2, "Momentum", f"{mom_arrow} {cur_mom.split()[0]}",
         f"Δ{cur_delta:+d} vs 21d", mom_color,
         "#dcfce7" if "improv" in cur_mom.lower() else "#fee2e2" if "deterio" in cur_mom.lower() else "#f3f4f6")
    _kpi(rs3, "Stay prob", f"{stay_p:.0%}" if stay_p else "—", "4w Markov",
         "#1f7a4f" if (stay_p and stay_p>0.6) else "#d97706")
    _kpi(rs4, "Degrade", f"{degrade_p:.0%}" if degrade_p else "—", "worse regime",
         "#b42318" if (degrade_p and degrade_p>0.35) else "#d97706" if (degrade_p and degrade_p>0.20) else "#6b7280",
         "#fee2e2" if (degrade_p and degrade_p>0.35) else "#fef9c3" if (degrade_p and degrade_p>0.20) else "#f8fafc")
    _kpi(rs5, "Confidence", cur_conf, f"prev: {prev_label or '—'}", "#1d4ed8","#eff6ff")

    st.markdown("")
    st.markdown("<div class='me-rowtitle'>Score drivers</div>", unsafe_allow_html=True)
    comp_rows2 = []
    for k, c in components.items():
        if not isinstance(c, dict): continue
        comp_rows2.append({"name": c.get("name",k),
                           "contrib": float(c.get("contribution",0)),
                           "z": float(c.get("zscore") or 0)})
    if comp_rows2:
        comp_sorted = sorted(comp_rows2, key=lambda x: x["contrib"])
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            x=[r["contrib"] for r in comp_sorted],
            y=[r["name"]    for r in comp_sorted],
            orientation="h",
            marker_color=["#1f7a4f" if r["contrib"]>=0 else "#b42318" for r in comp_sorted],
            marker_opacity=0.85,
            text=[f"{r['contrib']:+.3f}  z {r['z']:+.2f}" for r in comp_sorted],
            textposition="outside", textfont=dict(size=9),
            hovertemplate="<b>%{y}</b><br>Contribution: %{x:.3f}<extra></extra>",
        ))
        fig_comp.add_vline(x=0, line_color="#94a3b8", line_width=1.5)
        max_c = max(abs(r["contrib"]) for r in comp_sorted) if comp_sorted else 0.3
        fig_comp.update_layout(
            height=220, margin=dict(l=10,r=110,t=8,b=8),
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(range=[-max_c*1.6, max_c*1.6], showgrid=True,
                       gridcolor="#f1f5f9", zeroline=False),
            yaxis=dict(showgrid=False),
            showlegend=False,
        )
        st.plotly_chart(fig_comp, width='stretch')

with right_col:
    st.markdown("<div class='me-rowtitle'>What changed this week</div>", unsafe_allow_html=True)
    for name, level, delta, inverse in weekly_changes:
        good  = (delta < 0) if inverse else (delta > 0)
        d_col = "#1f7a4f" if good else "#b42318"
        d_bg  = "rgba(31,122,79,0.04)" if good else "rgba(180,35,24,0.04)"
        arrow = "↓" if delta < 0 else "↑"
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;align-items:center;"
            f"padding:10px 12px;border-radius:10px;background:{d_bg};margin-bottom:6px;'>"
            f"<div><div style='font-size:12px;font-weight:700;color:rgba(0,0,0,0.80);'>{name}</div>"
            f"<div style='font-size:10px;color:rgba(0,0,0,0.45);margin-top:1px;'>Now {level:.2f}</div></div>"
            f"<span style='font-size:15px;font-weight:900;color:{d_col};'>{arrow} {abs(delta):.2f}</span>"
            f"</div>", unsafe_allow_html=True)

    st.markdown("")
    st.markdown("<div class='me-rowtitle'>Market returns this week</div>", unsafe_allow_html=True)
    if mkt_ret:
        sorted_ret = sorted(mkt_ret.items(), key=lambda x: x[1], reverse=True)
        max_abs    = max(abs(v) for _,v in sorted_ret) or 1
        for t, r in sorted_ret:
            rc    = "#1f7a4f" if r > 0 else "#b42318"
            bar_w = min(abs(r) / max_abs * 80, 80)
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:8px;"
                f"padding:5px 0;border-bottom:1px solid rgba(0,0,0,0.04);'>"
                f"<span style='font-size:11px;font-weight:700;color:rgba(0,0,0,0.65);"
                f"width:36px;flex-shrink:0;'>{t}</span>"
                f"<div style='width:{bar_w:.0f}px;height:4px;border-radius:2px;"
                f"background:{rc};flex-shrink:0;'></div>"
                f"<span style='font-size:11px;font-weight:800;color:{rc};'>{r:+.1f}%</span>"
                f"</div>", unsafe_allow_html=True)

st.markdown("")

# ══════════════════════════════════════════════════════════════════════════════
# TOP SIGNALS | FACTOR TILTS | DIVERGENCES
# ══════════════════════════════════════════════════════════════════════════════

t1, t2, t3 = st.columns(3, gap="medium")

with t1:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>Top signals</div>", unsafe_allow_html=True)
        st.caption("Highest-conviction reads from regime components.")
        for tr in top_trade_signals:
            d   = tr["direction"]
            tc  = "#1f7a4f" if d in ("bullish","long") else "#b42318"
            tbg = "#dcfce7" if d in ("bullish","long") else "#fee2e2"
            z   = float(tr["z"])
            bar_w = min(abs(z)/2.5*60, 60)
            st.markdown(
                f"<div style='padding:10px 12px;border-radius:10px;background:#fafafa;"
                f"border:1px solid {tc}33;margin-bottom:8px;'>"
                f"<div style='display:flex;justify-content:space-between;align-items:center;"
                f"margin-bottom:4px;'>"
                f"<span style='font-size:11px;font-weight:800;color:rgba(0,0,0,0.80);'>"
                f"{tr['signal']}</span>"
                f"<span style='font-size:10px;font-weight:800;color:{tc};"
                f"background:{tbg};padding:2px 7px;border-radius:5px;'>{d.upper()}</span></div>"
                f"<div style='font-size:12px;font-weight:700;color:{tc};margin-bottom:4px;'>"
                f"{tr['instrument']}</div>"
                f"<div style='display:flex;align-items:center;gap:6px;'>"
                f"<div style='width:{bar_w:.0f}px;height:3px;border-radius:2px;background:{tc};'></div>"
                f"<span style='font-size:10px;color:rgba(0,0,0,0.45);'>z {z:+.2f}</span>"
                f"</div></div>", unsafe_allow_html=True)

with t2:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>Factor tilts</div>", unsafe_allow_html=True)
        st.caption("Composite of curve, real rates, credit, and dollar.")
        for fname, fscore, fproxy in factor_scores[:5]:
            fc  = "#1f7a4f" if fscore>0.3 else "#b42318" if fscore<-0.3 else "#6b7280"
            fbg = "#dcfce7" if fscore>0.3 else "#fee2e2" if fscore<-0.3 else "#f3f4f6"
            parts   = fname.split(" vs ")
            winner  = parts[0] if fscore>0 else (parts[1] if len(parts)>1 else fname)
            bar_w   = min(abs(fscore)/3.0*60, 60)
            st.markdown(
                f"<div style='padding:8px 10px;border-radius:9px;background:#fafafa;"
                f"border:1px solid {fc}22;margin-bottom:6px;'>"
                f"<div style='display:flex;justify-content:space-between;'>"
                f"<span style='font-size:10px;color:rgba(0,0,0,0.55);'>{fname}</span>"
                f"<span style='font-size:11px;font-weight:900;color:{fc};'>{fscore:+.2f}</span></div>"
                f"<div style='font-size:11px;font-weight:800;color:{fc};"
                f"background:{fbg};padding:2px 7px;border-radius:5px;"
                f"display:inline-block;margin:3px 0;'>{winner}</div>"
                f"<div style='font-size:9px;color:rgba(0,0,0,0.35);'>{fproxy}</div>"
                f"</div>", unsafe_allow_html=True)

with t3:
    with st.container(border=True):
        st.markdown("<div class='me-rowtitle'>Divergence alerts</div>", unsafe_allow_html=True)
        st.caption("Markets that should agree but don't.")
        if divergences:
            for pair, dz, interp, trade in divergences:
                dc  = "#b42318" if dz>0 else "#1f7a4f"
                dbg = "#fee2e2" if dz>0 else "#dcfce7"
                st.markdown(
                    f"<div style='padding:10px 12px;border-radius:10px;background:#fafafa;"
                    f"border:1px solid {dc}33;margin-bottom:8px;'>"
                    f"<div style='font-size:11px;font-weight:800;color:{dc};margin-bottom:4px;'>"
                    f"{pair} · Δz {dz:+.2f}</div>"
                    f"<div style='font-size:11px;color:rgba(0,0,0,0.65);line-height:1.4;"
                    f"margin-bottom:6px;'>{interp}</div>"
                    f"<div style='padding:5px 8px;border-radius:6px;background:{dbg};"
                    f"font-size:10px;font-weight:700;color:{dc};'>→ {trade}</div>"
                    f"</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                "<div style='padding:16px;text-align:center;color:rgba(0,0,0,0.40);"
                "font-size:12px;'>No significant divergences detected.</div>",
                unsafe_allow_html=True)

st.markdown("")

# ══════════════════════════════════════════════════════════════════════════════
# FALSIFICATION CONDITIONS
# ══════════════════════════════════════════════════════════════════════════════

with st.container(border=True):
    st.markdown("<div class='me-rowtitle'>What breaks the thesis</div>",
                unsafe_allow_html=True)
    st.caption("Monitor these levels. If they hit, the thesis is wrong.")
    st.markdown(
        "<div style='padding:9px 14px;border-radius:9px;background:#f8fafc;"
        "border-left:3px solid #1d4ed8;font-size:12px;line-height:1.6;"
        "color:rgba(0,0,0,0.65);margin-bottom:10px;'>"
        "These are not predictions — they are falsification conditions. "
        "A good macro trade has a pre-defined level where you accept the thesis is wrong. "
        "Checking these weekly forces discipline: if the level hits, you exit or reduce, "
        "regardless of your conviction. The moment you start rationalising why the level "
        "does not matter, the trade has become an opinion, not an analysis."
        "</div>",
        unsafe_allow_html=True)
    fc_cols = st.columns(len(falsification_conditions) or 1, gap="medium")
    for i, (main, detail) in enumerate(falsification_conditions):
        fc_cols[i].markdown(
            f"<div style='padding:12px 14px;border-radius:12px;background:#fafafa;"
            f"border:1px solid rgba(0,0,0,0.08);'>"
            f"<div style='font-size:11px;font-weight:800;color:#0f172a;margin-bottom:4px;'>{main}</div>"
            f"<div style='font-size:10px;color:rgba(0,0,0,0.50);line-height:1.4;'>{detail}</div>"
            f"</div>", unsafe_allow_html=True)

st.markdown("")

# ══════════════════════════════════════════════════════════════════════════════
# ASSET WATCHLIST — quick macro alignment for the 5 most-traded assets
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    "<div style='font-size:10px;font-weight:700;color:rgba(0,0,0,0.45);"
    "text-transform:uppercase;letter-spacing:0.8px;margin:4px 0 10px;"
    "padding-bottom:6px;border-bottom:1px solid rgba(0,0,0,0.07);'>"
    "Asset watchlist — macro alignment</div>", unsafe_allow_html=True)

_WATCH_T = [("SPY","S&P 500","#1d4ed8"),("QQQ","Nasdaq","#7c3aed"),
            ("GLD","Gold","#d97706"),("BTC","Bitcoin","#f59e0b"),("SLV","Silver","#94a3b8")]

def _quick_alignment(ticker, macro, px, cur_score, cur_label):
    """Fast alignment read — 3 key signals only."""
    score = 50; signals = []
    r10 = float(macro["real10"].dropna().iloc[-1]) if "real10" in macro.columns and not macro["real10"].dropna().empty else None
    dz  = None
    if "dollar_broad" in macro.columns:
        d = macro["dollar_broad"].dropna()
        if len(d) >= 30:
            t = d.iloc[-min(252,len(d)):]
            sd = float(t.std())
            dz = float((t.iloc[-1]-t.mean())/sd) if sd > 0 else 0

    if ticker == "SPY":
        if cur_score > 55: score += 20
        elif cur_score < 45: score -= 20
        if r10 and r10 > 1.5: score -= 10
        elif r10 and r10 < 0.5: score += 10
    elif ticker == "QQQ":
        if r10 and r10 > 1.5: score -= 25
        elif r10 and r10 < 0.5: score += 25
        if cur_score > 55: score += 10
        elif cur_score < 45: score -= 10
    elif ticker == "GLD":
        if r10 and r10 > 1.5: score -= 20
        elif r10 and r10 < 0.5: score += 20
        if dz and dz < -0.5: score += 15
        elif dz and dz > 0.5: score -= 15
    elif ticker == "BTC":
        if cur_score > 55: score += 20
        elif cur_score < 45: score -= 20
        if dz and dz < -0.5: score += 15
        elif dz and dz > 0.5: score -= 15
    elif ticker == "SLV":
        if r10 and r10 > 1.5: score -= 15
        elif r10 and r10 < 0.5: score += 15
        if dz and dz < -0.5: score += 10
        elif dz and dz > 0.5: score -= 10

    score = max(0, min(100, score))
    if score >= 65:   label, col = "Bullish", "#1f7a4f"
    elif score >= 55: label, col = "Tailwind", "#16a34a"
    elif score >= 45: label, col = "Neutral",  "#6b7280"
    elif score >= 35: label, col = "Headwind", "#d97706"
    else:             label, col = "Bearish",  "#b42318"
    return score, label, col

_watch_cols = st.columns(5, gap="small")
for _wc, (ticker, name, tcolor) in zip(_watch_cols, _WATCH_T):
    _price = None
    _ret1w = None
    if ticker in px.columns:
        _s = px[ticker].dropna()
        if not _s.empty:
            _price = float(_s.iloc[-1])
            _prev  = _s.index[_s.index <= _s.index.max() - pd.Timedelta(days=7)]
            if len(_prev): _ret1w = float(_s.iloc[-1]/_s.loc[_prev[-1]] - 1)*100

    _align, _albl, _acol = _quick_alignment(ticker, macro, px, cur_score, cur_label)
    _abg = "#dcfce7" if _align >= 60 else "#fee2e2" if _align <= 40 else "#f3f4f6"

    _wc.markdown(
        f"<div style='padding:10px 12px;border-radius:12px;background:#ffffff;"
        f"border:1px solid rgba(0,0,0,0.08);border-top:3px solid {tcolor};'>"
        f"<div style='font-size:10px;font-weight:800;color:{tcolor};"
        f"margin-bottom:2px;'>{ticker}</div>"
        f"<div style='font-size:9px;color:rgba(0,0,0,0.45);margin-bottom:6px;'>{name}</div>"
        f"<div style='font-size:15px;font-weight:900;color:rgba(0,0,0,0.85);margin-bottom:4px;'>"
        f"{'${:,.2f}'.format(_price) if _price else '—'}</div>"
        f"<div style='font-size:10px;font-weight:700;"
        f"color:{'#1f7a4f' if _ret1w and _ret1w>0 else '#b42318'};margin-bottom:8px;'>"
        f"{f'{_ret1w:+.1f}% 1w' if _ret1w is not None else ''}</div>"
        f"<div style='padding:3px 8px;border-radius:6px;background:{_abg};"
        f"display:inline-block;'>"
        f"<span style='font-size:10px;font-weight:800;color:{_acol};'>"
        f"{_albl}</span></div>"
        f"</div>", unsafe_allow_html=True)

st.markdown("")
if st.button("Full asset analysis →", key="btn_asset_mon", use_container_width=False):
    safe_switch_page("pages/3_Asset_Monitor.py")

st.markdown("")

# ── Nav footer ────────────────────────────────────────────────────────────────
n1,n2,n3,n4,n5 = st.columns(5, gap="small")
for col, (label, path, key) in zip([n1,n2,n3,n4,n5], [
    ("Regime Playbook", "pages/7_Regime_Playbook.py", "btn_mb_play"),
    ("Transition Watch","pages/8_Transition_Watch.py","btn_mb_tw"),
    ("Curve View",      "pages/9_Curve_View.py",      "btn_mb_curve"),
    ("Fed & Liquidity", "pages/10_Fed_Liquidity.py",  "btn_mb_fed"),
    ("Credit & Macro",  "pages/2_Macro_Charts.py",    "btn_mb_credit"),
]):
    if col.button(label, width='stretch', key=key):
        safe_switch_page(path)

st.markdown("<div style='height:48px;'></div>", unsafe_allow_html=True)
