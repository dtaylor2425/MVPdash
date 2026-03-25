# pages/3_Asset_Monitor.py
"""
Asset Monitor
═══════════════════════════════════════════════════════════════════════════════
Macro-conditional analysis for the most-traded assets:
SPY · QQQ · GLD · IBIT (Bitcoin) · SLV (Silver)

For each asset:
  1. Macro alignment signal — does the current regime support this asset?
  2. Historical return distribution in the current regime label
  3. Key macro driver relationship (real yields for QQQ, dollar+real for GLD, etc.)
  4. Price vs MA with regime overlay
  5. One-click content generator — grab a ready-to-post comment
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import date
from collections import defaultdict
import streamlit as st

from src.config import CACHE_DIR, FRED_API_KEY, FRED_SERIES, YF_PROXIES
from src.data_sources import fetch_prices, get_fred_cached
from src.regime import compute_regime_v3, compute_regime_timeseries
from src.ui import inject_css, sidebar_nav, safe_switch_page, regime_color, regime_bg

st.set_page_config(page_title="Asset Monitor", page_icon="📡",
                   layout="wide", initial_sidebar_state="expanded")
inject_css()
sidebar_nav(active="Asset Monitor")

if not FRED_API_KEY:
    st.error("FRED_API_KEY is not set."); st.stop()

# ── Assets ────────────────────────────────────────────────────────────────────

WATCH = {
    "SPY":  {"name":"S&P 500",      "color":"#1d4ed8","emoji":"🏦",
             "macro_driver":"Regime score + credit spreads + real yields",
             "driver_key":"regime"},
    "QQQ":  {"name":"Nasdaq 100",   "color":"#7c3aed","emoji":"💻",
             "macro_driver":"Real yields (long-duration sensitivity)",
             "driver_key":"real"},
    "GLD":  {"name":"Gold",         "color":"#d97706","emoji":"🥇",
             "macro_driver":"Real yields (inverted) + dollar direction",
             "driver_key":"gold"},
    "IBIT": {"name":"Bitcoin ETF",  "color":"#f59e0b","emoji":"₿",
             "macro_driver":"Global liquidity + dollar + risk appetite",
             "driver_key":"btc"},
    "SLV":  {"name":"Silver ETF",   "color":"#94a3b8","emoji":"🥈",
             "macro_driver":"Real yields + industrial demand + dollar",
             "driver_key":"silver"},
    "XLU":  {"name":"Utilities",    "color":"#0d9488","emoji":"⚡",
             "macro_driver":"Bond proxy — real yields + curve slope (inverted)",
             "driver_key":"xlu"},
    "XLC":  {"name":"Comms/Media",  "color":"#c026d3","emoji":"📡",
             "macro_driver":"Growth + real yields + regime score",
             "driver_key":"xlc"},
}
ALL_T = list(WATCH.keys())

# ── Data ──────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=30*60, show_spinner=False)
def load_all():
    macro = get_fred_cached(FRED_SERIES, FRED_API_KEY, CACHE_DIR,
                            cache_name="fred_macro").sort_index()
    extra = ALL_T + ["SPY","IWM","RSP","TLT","HYG","UUP"]
    vix_t = [v for v in [YF_PROXIES.get("vix"), YF_PROXIES.get("vix3m")] if v]
    px    = fetch_prices(list(dict.fromkeys(extra + vix_t)), period="5y")
    px    = pd.DataFrame() if (px is None or px.empty) else px.sort_index()
    regime   = compute_regime_v3(macro=macro, proxies=px,
                                 lookback_trend=63, momentum_lookback_days=21)
    reg_hist = compute_regime_timeseries(macro, px, freq="W-FRI")
    return macro, px, regime, reg_hist

macro, px, regime, reg_hist = load_all()

cur_label = regime.label
cur_score = regime.score
cur_color = regime_color(cur_label)
cur_bg    = regime_bg(cur_label)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _last(s):
    s = s.dropna(); return float(s.iloc[-1]) if not s.empty else None

def _ret(s, days):
    s = s.dropna()
    if len(s) < 2: return None
    prev = s.index[s.index <= s.index.max() - pd.Timedelta(days=days)]
    return float(s.iloc[-1]/s.loc[prev[-1]] - 1) if len(prev) else None

def _zscore(s, w=252):
    s = s.dropna()
    if len(s) < 30: return None
    tail = s.iloc[-min(w,len(s)):]
    sd   = float(tail.std())
    return float((tail.iloc[-1]-tail.mean())/sd) if sd > 0 else 0.0

def _pct_rank(s, w=252):
    s = s.dropna()
    if len(s) < w: return None
    return float((s.iloc[-w:] < s.iloc[-1]).mean()*100)

def _ma(s, n):
    s = s.dropna()
    return s.rolling(n, min_periods=n//2).mean()

def fmt_pct(x, nd=1):
    if x is None: return "—"
    return f"{x*100:+.{nd}f}%"

def fmt_num(x, nd=2):
    if x is None: return "—"
    return f"{float(x):.{nd}f}"

# ── Regime-conditional return engine ─────────────────────────────────────────

def regime_returns(ticker, reg_hist, px, fwd_days=21):
    """
    For each week in reg_hist, compute the fwd_days forward return of ticker.
    Return dict: {label: [returns]} for historical win rates and distributions.
    """
    if ticker not in px.columns or reg_hist.empty:
        return {}
    s   = px[ticker].resample("W-FRI").last().dropna()
    fwd = s.pct_change(4).shift(-4).dropna()
    merged = reg_hist.join(fwd.rename("fwd"), how="inner").dropna()
    out = defaultdict(list)
    for _, row in merged.iterrows():
        out[row["label"]].append(float(row["fwd"]))
    return dict(out)

def macro_alignment(ticker, regime, macro, px):
    """
    Score this asset's macro alignment 0-100 based on current signals.
    Returns (score, signals_list, summary_text, color).
    """
    signals = []
    score   = 50

    r10  = _last(macro["real10"].dropna()) if "real10" in macro.columns else None
    r10z = _zscore(macro["real10"]) if "real10" in macro.columns else None
    dz   = _zscore(macro["dollar_broad"]) if "dollar_broad" in macro.columns else None
    hyz  = _zscore(macro["hy_oas"]) if "hy_oas" in macro.columns else None
    fed_roc = None
    if "fed_assets" in macro.columns:
        fa = macro["fed_assets"].dropna()
        if len(fa) >= 70:
            fed_roc = float(fa.pct_change(63).iloc[-1]*100)

    if ticker == "SPY":
        # Regime score direction
        if cur_score > 55:   signals.append(("Regime",  +1, f"Score {cur_score} — bullish macro backdrop"))
        elif cur_score < 45: signals.append(("Regime",  -1, f"Score {cur_score} — bearish macro backdrop"))
        else:                signals.append(("Regime",   0, f"Score {cur_score} — neutral"))
        # Credit
        if hyz is not None:
            if hyz < -0.5:   signals.append(("Credit",  +1, f"HY OAS tightening (z {hyz:+.2f}) — supportive"))
            elif hyz > 0.5:  signals.append(("Credit",  -1, f"HY OAS widening (z {hyz:+.2f}) — headwind"))
            else:            signals.append(("Credit",   0, f"HY OAS neutral (z {hyz:+.2f})"))
        # Real yields
        if r10 is not None:
            if r10 < 0.5:    signals.append(("Real yields", +1, f"{r10:.2f}% — accommodative for equities"))
            elif r10 > 1.5:  signals.append(("Real yields", -1, f"{r10:.2f}% — restrictive, compressing multiples"))
            else:            signals.append(("Real yields",  0, f"{r10:.2f}% — neutral range"))
        # Momentum
        if "SPY" in px.columns:
            spy_s = px["SPY"].dropna()
            ma200 = _ma(spy_s, 200)
            if not ma200.dropna().empty:
                above = float(spy_s.iloc[-1]) > float(ma200.dropna().iloc[-1])
                signals.append(("200d MA",  +1 if above else -1,
                    f"{'Above' if above else 'Below'} 200d MA — {'trend intact' if above else 'trend broken'}"))

    elif ticker == "QQQ":
        # QQQ is pure real yield play
        if r10 is not None:
            if r10 < 0.5:    signals.append(("Real yields", +1, f"{r10:.2f}% — historically most bullish for QQQ"))
            elif r10 > 1.5:  signals.append(("Real yields", -1, f"{r10:.2f}% — restrictive, QQQ underperforms vs SPY"))
            else:            signals.append(("Real yields",  0, f"{r10:.2f}% — neutral for QQQ duration premium"))
        if r10z is not None:
            if r10z > 0.5:   signals.append(("Real yield trend", -1, f"Rising real yields (z {r10z:+.2f}) — QQQ headwind"))
            elif r10z < -0.5:signals.append(("Real yield trend", +1, f"Falling real yields (z {r10z:+.2f}) — QQQ tailwind"))
            else:            signals.append(("Real yield trend",  0, f"Real yields stable (z {r10z:+.2f})"))
        # QQQ vs SPY ratio (growth premium)
        if "QQQ" in px.columns and "SPY" in px.columns:
            ratio = (px["QQQ"]/px["SPY"]).dropna()
            rz    = _zscore(ratio)
            if rz is not None:
                if rz > 1.0:  signals.append(("Growth premium", -1, f"QQQ/SPY ratio extended (z {rz:+.2f}) — mean reversion risk"))
                elif rz < -1.0:signals.append(("Growth premium",+1, f"QQQ/SPY ratio depressed (z {rz:+.2f}) — undervalued vs history"))
                else:          signals.append(("Growth premium",  0, f"QQQ/SPY ratio normal (z {rz:+.2f})"))
        if cur_score > 55:    signals.append(("Regime",  +1, f"Score {cur_score} — risk-on supports tech"))
        elif cur_score < 45:  signals.append(("Regime",  -1, f"Score {cur_score} — risk-off hurts high-duration tech"))
        else:                 signals.append(("Regime",   0, f"Score {cur_score} — neutral"))

    elif ticker == "GLD":
        # GLD = real yield (inverted) + dollar (inverted)
        if r10 is not None:
            if r10 < 0.5:    signals.append(("Real yields", +1, f"{r10:.2f}% — low/negative real yields = GLD's best environment"))
            elif r10 > 1.5:  signals.append(("Real yields", -1, f"{r10:.2f}% — restrictive real yields = GLD headwind"))
            else:            signals.append(("Real yields",  0, f"{r10:.2f}% — neutral for GLD"))
        if dz is not None:
            if dz < -0.5:    signals.append(("Dollar",     +1, f"Dollar weakening (z {dz:+.2f}) — tailwind for GLD"))
            elif dz > 0.5:   signals.append(("Dollar",     -1, f"Dollar strengthening (z {dz:+.2f}) — headwind for GLD"))
            else:            signals.append(("Dollar",      0, f"Dollar neutral (z {dz:+.2f})"))
        if hyz is not None:
            if hyz > 0.5:    signals.append(("Credit stress",+1, f"HY OAS widening (z {hyz:+.2f}) — safe haven bid for GLD"))
            else:            signals.append(("Credit stress", 0, f"Credit contained — no stress bid"))
        # GLD vs real yield regression deviation
        if "GLD" in px.columns and "real10" in macro.columns:
            gld_s = px["GLD"].dropna()
            gldz  = _zscore(gld_s)
            if gldz is not None:
                if gldz > 1.5: signals.append(("GLD vs history",  -1, f"GLD at {_pct_rank(gld_s):.0f}th pct — stretched vs history"))
                elif gldz < -1:signals.append(("GLD vs history",  +1, f"GLD depressed vs history — potential mean reversion"))
                else:           signals.append(("GLD vs history",   0, f"GLD within normal historical range"))

    elif ticker == "IBIT":
        # BTC = global liquidity + dollar (inv) + risk appetite
        if fed_roc is not None:
            if fed_roc > 0.5:   signals.append(("Fed liquidity", +1, f"Balance sheet expanding {fed_roc:+.1f}% 13w — liquidity tailwind for BTC"))
            elif fed_roc < -0.5:signals.append(("Fed liquidity", -1, f"Balance sheet contracting {fed_roc:+.1f}% 13w — liquidity headwind"))
            else:               signals.append(("Fed liquidity",  0, f"Balance sheet flat {fed_roc:+.1f}% 13w"))
        if dz is not None:
            if dz < -0.5:    signals.append(("Dollar",  +1, f"Dollar weakening (z {dz:+.2f}) — historically BTC positive"))
            elif dz > 0.5:   signals.append(("Dollar",  -1, f"Dollar strengthening (z {dz:+.2f}) — BTC headwind"))
            else:            signals.append(("Dollar",   0, f"Dollar neutral (z {dz:+.2f})"))
        if cur_score > 55:   signals.append(("Risk appetite", +1, f"Score {cur_score} — risk-on, BTC historically performs well"))
        elif cur_score < 45: signals.append(("Risk appetite", -1, f"Score {cur_score} — risk-off, BTC correlates with drawdowns"))
        else:                signals.append(("Risk appetite",  0, f"Score {cur_score} — mixed"))
        # BTC vs GLD ratio
        if "IBIT" in px.columns and "GLD" in px.columns:
            ibit_gld = (px["IBIT"]/px["GLD"]).dropna()
            bgrz = _zscore(ibit_gld)
            if bgrz is not None:
                if bgrz > 1.0:  signals.append(("BTC/GLD ratio", -1, f"BTC expensive vs GLD (z {bgrz:+.2f}) — rotation risk to GLD"))
                elif bgrz < -1: signals.append(("BTC/GLD ratio", +1, f"BTC cheap vs GLD (z {bgrz:+.2f}) — historical entry zone"))
                else:           signals.append(("BTC/GLD ratio",  0, f"BTC/GLD ratio normal (z {bgrz:+.2f})"))

    elif ticker == "SLV":
        # Silver = GLD signals + industrial component (growth)
        if r10 is not None:
            if r10 < 0.5:    signals.append(("Real yields", +1, f"{r10:.2f}% — low real yields support precious metals"))
            elif r10 > 1.5:  signals.append(("Real yields", -1, f"{r10:.2f}% — restrictive real rates weigh on SLV"))
            else:            signals.append(("Real yields",  0, f"{r10:.2f}% — neutral"))
        if dz is not None:
            if dz < -0.5:    signals.append(("Dollar",     +1, f"Dollar weakening (z {dz:+.2f}) — SLV tailwind"))
            elif dz > 0.5:   signals.append(("Dollar",     -1, f"Dollar strengthening (z {dz:+.2f}) — SLV headwind"))
            else:            signals.append(("Dollar",      0, f"Dollar neutral"))
        # Silver industrial component — likes growth / curve steepening
        c210 = None
        if "y10" in macro.columns and "y2" in macro.columns:
            curve = (macro["y10"]-macro["y2"]).dropna()
            c210 = _last(curve)
            cz   = _zscore(curve)
            if cz is not None:
                if cz > 0.3:  signals.append(("Curve/growth", +1, f"Steepening curve (z {cz:+.2f}) — industrial demand supportive"))
                elif cz < -0.3:signals.append(("Curve/growth",-1, f"Flattening curve (z {cz:+.2f}) — industrial headwind for SLV"))
                else:          signals.append(("Curve/growth",  0, f"Curve neutral (z {cz:+.2f})"))
        # SLV vs GLD ratio — silver's relative value
        if "SLV" in px.columns and "GLD" in px.columns:
            sgr  = (px["SLV"]/px["GLD"]).dropna()
            sgrz = _zscore(sgr)
            if sgrz is not None:
                if sgrz < -1.0: signals.append(("SLV/GLD ratio", +1, f"Silver cheap vs gold (z {sgrz:+.2f}) — historically mean-reverts higher"))
                elif sgrz > 1.0:signals.append(("SLV/GLD ratio", -1, f"Silver expensive vs gold (z {sgrz:+.2f}) — rotation risk"))
                else:           signals.append(("SLV/GLD ratio",  0, f"Silver/gold ratio normal"))

    # Compute alignment score from signals
    # Score represents the net directional balance of macro signals for this asset.
    # Floor at 15, ceiling at 85 — a score of 0 or 100 implies perfect certainty
    # which the model never has. Real range in practice: ~20-80.
    n    = len(signals)
    pos  = sum(1 for _,v,_ in signals if v > 0)
    neg  = sum(1 for _,v,_ in signals if v < 0)
    neu  = sum(1 for _,v,_ in signals if v == 0)
    raw  = (pos - neg) / n if n > 0 else 0
    # Apply floor/ceiling: max bearish = 15, max bullish = 85
    score_out = int(max(15, min(85, 50 + raw * 35)))

    # Labels reflect the balance, not binary good/bad
    n_confirm = pos; n_contra = neg
    if score_out >= 68:   color_out = "#1f7a4f"; summary = f"BULLISH · {n_confirm}/{n} signals confirm"
    elif score_out >= 55: color_out = "#16a34a"; summary = f"TAILWIND · {n_confirm}/{n} signals confirm"
    elif score_out >= 45: color_out = "#6b7280"; summary = f"NEUTRAL · signals split {n_confirm}✓ {n_contra}✗"
    elif score_out >= 33: color_out = "#d97706"; summary = f"HEADWIND · {n_contra}/{n} signals against"
    else:                 color_out = "#b42318"; summary = f"BEARISH · {n_contra}/{n} signals against"

    return score_out, signals, summary, color_out

def build_macro_card(ticker, align_score, align_color, summary,
                     signals, regime_rets, macro, px):
    """
    Visual Bloomberg-style macro snapshot card for social media attachment.
    Hierarchy: (1) big verdict headline, (2) win rate stat, (3) key drivers,
    (4) signal dots, (5) falsification condition.
    """
    meta  = WATCH[ticker]
    r10   = _last(macro["real10"].dropna())   if "real10"       in macro.columns else None
    hy    = _last(macro["hy_oas"].dropna())   if "hy_oas"       in macro.columns else None
    dz    = _zscore(macro["dollar_broad"])    if "dollar_broad" in macro.columns else None
    fed_r = None
    if "fed_assets" in macro.columns:
        fa = macro["fed_assets"].dropna()
        if len(fa) >= 70: fed_r = float(fa.pct_change(63).iloc[-1]*100)
    c210 = None
    if "y10" in macro.columns and "y2" in macro.columns:
        c210 = _last((macro["y10"]-macro["y2"]).dropna())

    price = ma200_pct = ret_1w = ret_1m = None
    if ticker in px.columns:
        s      = px[ticker].dropna()
        price  = _last(s)
        ret_1w = _ret(s, 7); ret_1m = _ret(s, 30)
        ma200  = _ma(s, 200)
        if price and not ma200.dropna().empty:
            ma200_pct = (price / float(ma200.dropna().iloc[-1]) - 1)*100

    cur_rets = regime_rets.get(cur_label, [])
    wr  = (sum(1 for r in cur_rets if r > 0)/len(cur_rets)) if len(cur_rets) >= 8 else None
    med = float(np.median(cur_rets))*100 if cur_rets else None
    p25 = float(np.percentile(cur_rets,25))*100 if len(cur_rets) >= 8 else None
    p75 = float(np.percentile(cur_rets,75))*100 if len(cur_rets) >= 8 else None

    pos_sigs = [(l,v,t) for l,v,t in signals if v > 0]
    neg_sigs = [(l,v,t) for l,v,t in signals if v < 0]
    neu_sigs = [(l,v,t) for l,v,t in signals if v == 0]
    n_pos = len(pos_sigs); n_neg = len(neg_sigs); n_tot = len(signals)

    # ── Headline verdict ──────────────────────────────────────────────────────
    if align_score >= 65:
        verdict = "MACRO ALIGNED ▲"
        verdict_color = "#4aba6e"
        tagline = f"Regime + macro both support this trade"
    elif align_score >= 55:
        verdict = "MILD TAILWIND ▲"
        verdict_color = "#4aba6e"
        tagline = f"More signals in favour than against"
    elif align_score >= 45:
        verdict = "SIGNALS MIXED →"
        verdict_color = "#f79400"
        tagline = f"{n_pos} confirm · {n_neg} against — no directional edge"
    elif align_score >= 33:
        verdict = "MILD HEADWIND ▼"
        verdict_color = "#f79400"
        tagline = f"More macro signals working against than for"
    else:
        verdict = "MACRO AGAINST ▼"
        verdict_color = "#e84040"
        tagline = f"{n_neg} of {n_tot} signals pointing against this trade"

    # ── Win rate headline ─────────────────────────────────────────────────────
    if wr is not None:
        wr_color = "#4aba6e" if wr >= 0.58 else "#e84040" if wr < 0.45 else "#f79400"
        wr_str = f"{wr:.0%} WIN RATE"
        wr_sub = f"in {cur_label} regimes · median {med:+.1f}% · n={len(cur_rets)}"
    else:
        wr_color = "#555"; wr_str = "—"; wr_sub = "insufficient history"

    # ── Three key drivers (most readable, not all data) ───────────────────────
    drivers = []
    # Real yields — always show for most assets
    if r10 is not None:
        rc = "#e84040" if r10 > 1.5 else "#4aba6e" if r10 < 0.5 else "#f79400"
        ctx = "restrictive" if r10 > 1.5 else "accommodative" if r10 < 0.5 else "neutral"
        drivers.append(("REAL YIELD", f"{r10:.2f}%", ctx.upper(), rc))
    # HY OAS / credit
    if hy is not None:
        rc = "#e84040" if hy > 4.5 else "#f79400" if hy > 3.5 else "#4aba6e"
        ctx = "stress" if hy > 4.5 else "elevated" if hy > 3.5 else "contained"
        drivers.append(("HY SPREADS", f"{hy:.2f}%", ctx.upper(), rc))
    # Price vs 200d MA
    if ma200_pct is not None:
        rc = "#4aba6e" if ma200_pct > 0 else "#e84040"
        ctx = "above 200d" if ma200_pct > 0 else "below 200d"
        drivers.append(("TREND", f"{ma200_pct:+.1f}%", ctx.upper(), rc))
    # Dollar for GLD/BTC/SLV
    if ticker in ("GLD","IBIT","SLV") and dz is not None:
        rc = "#4aba6e" if dz < -0.5 else "#e84040" if dz > 0.5 else "#f79400"
        ctx = "weakening" if dz < -0.5 else "strengthening" if dz > 0.5 else "stable"
        drivers.append(("DOLLAR", f"z {dz:+.2f}", ctx.upper(), rc))
    # Fed BS for BTC
    if ticker == "IBIT" and fed_r is not None:
        rc = "#4aba6e" if fed_r > 0 else "#e84040"
        ctx = "injecting" if fed_r > 0.5 else "draining" if fed_r < -0.5 else "flat"
        drivers.append(("FED BS 13W", f"{fed_r:+.1f}%", ctx.upper(), rc))
    drivers = drivers[:3]  # max 3 for readability

    drivers_html = ""
    for d_label, d_val, d_ctx, d_color in drivers:
        drivers_html += f"""
    <div style="background:#111;border-radius:6px;padding:8px 10px;flex:1;min-width:0;">
      <div style="font-size:7px;color:#555;letter-spacing:0.6px;margin-bottom:3px;">{d_label}</div>
      <div style="font-size:13px;font-weight:900;color:{d_color};line-height:1;">{d_val}</div>
      <div style="font-size:7px;color:{d_color};margin-top:2px;opacity:0.8;">{d_ctx}</div>
    </div>"""

    # ── Signal dots ───────────────────────────────────────────────────────────
    dots_html = ""
    for lbl, val, txt in signals:
        dc = "#4aba6e" if val > 0 else "#e84040" if val < 0 else "#333"
        short = lbl[:3].upper()
        dots_html += (f"<div style='display:flex;flex-direction:column;"
                      f"align-items:center;gap:3px;'>"
                      f"<div style='width:8px;height:8px;border-radius:50%;"
                      f"background:{dc};'></div>"
                      f"<span style='font-size:6px;color:#555;'>{short}</span>"
                      f"</div>")

    # ── Falsification / watch level ───────────────────────────────────────────
    falsif = ""
    if ticker == "SPY":
        if align_score < 50 and hy is not None:
            falsif = f"Bull case needs: HY OAS < 3.5% + regime score > 55"
        elif align_score >= 50:
            falsif = f"Bear watch: HY OAS > 4.5% or real yields > 2.0%"
    elif ticker == "QQQ":
        falsif = f"Key: real yields direction — falling = tailwind, rising = headwind"
    elif ticker == "GLD":
        falsif = f"Bull case holds while real yields below 1.5% or dollar weak"
    elif ticker == "IBIT":
        falsif = f"Risk: dollar strength or credit stress (HY OAS > 4.5%) = BTC headwind"
    elif ticker in ("SLV","XLU","XLC"):
        falsif = f"Watch real yield trend — direction matters more than level"

    # ── Regime bar ────────────────────────────────────────────────────────────
    seg_labels = ["Risk Off","Bearish","Neutral","Bullish","Risk On"]
    seg_colors_map = {"Risk Off":"#b42318","Bearish":"#d97706","Neutral":"#6b7280",
                      "Bullish":"#16a34a","Risk On":"#1f7a4f"}
    segs_html = ""
    for lbl in seg_labels:
        sc = seg_colors_map[lbl]
        is_cur = (lbl == cur_label)
        segs_html += (f"<div style='flex:1;display:flex;flex-direction:column;"
                      f"align-items:center;gap:3px;'>"
                      f"<div style='width:100%;height:5px;border-radius:2px;"
                      f"background:{sc};opacity:{"1" if is_cur else "0.2"};'></div>"
                      f"<span style='font-size:6px;color:{sc if is_cur else "#333"};"
                      f"font-weight:{"800" if is_cur else "400"};'>"
                      f"{lbl.split()[0].upper()}</span>"
                      f"</div>")

    card_html = f"""
<div style="background:#0a0a0a;border:1px solid #222;border-radius:12px;
     padding:18px 20px;font-family:'Courier New',monospace;
     max-width:400px;box-shadow:0 8px 32px rgba(0,0,0,0.6);">

  <!-- HEADER: ticker + date -->
  <div style="display:flex;justify-content:space-between;align-items:flex-start;
              margin-bottom:10px;">
    <div>
      <span style="font-size:18px;font-weight:900;color:#f0f0f0;
                   letter-spacing:2px;">{ticker}</span>
      <span style="font-size:10px;color:#444;margin-left:8px;
                   letter-spacing:1px;">{meta["name"].upper()}</span>
    </div>
    <div style="font-size:7px;color:#444;text-align:right;letter-spacing:0.5px;
                padding-top:4px;">
      MACRO ENGINE<br>{date.today().strftime("%b %d %Y").upper()}
    </div>
  </div>

  <!-- HEADLINE VERDICT — the thing that makes people stop scrolling -->
  <div style="background:#111;border-radius:8px;padding:12px 14px;
              margin-bottom:12px;border-left:3px solid {verdict_color};">
    <div style="font-size:15px;font-weight:900;color:{verdict_color};
                letter-spacing:1.5px;margin-bottom:3px;">{verdict}</div>
    <div style="font-size:8px;color:#666;letter-spacing:0.3px;">{tagline}</div>
  </div>

  <!-- WIN RATE — the stat that grabs attention -->
  <div style="display:flex;align-items:center;gap:12px;
              margin-bottom:12px;padding-bottom:12px;
              border-bottom:1px solid #1a1a1a;">
    <div>
      <div style="font-size:22px;font-weight:900;color:{wr_color};
                  letter-spacing:1px;line-height:1;">{wr_str}</div>
      <div style="font-size:8px;color:#555;margin-top:2px;">{wr_sub}</div>
    </div>
    {"" if price is None else f'<div style="margin-left:auto;text-align:right;"><div style="font-size:13px;font-weight:900;color:#f0f0f0;">${price:,.2f}</div><div style="font-size:8px;color:{("#4aba6e" if ret_1w and ret_1w>0 else "#e84040") if ret_1w is not None else "#555"};">{f"{ret_1w*100:+.1f}% 1W" if ret_1w is not None else ""}</div></div>'}
  </div>

  <!-- KEY DRIVERS — 3 boxes -->
  <div style="display:flex;gap:6px;margin-bottom:12px;">
    {drivers_html}
  </div>

  <!-- SIGNAL DOTS — compact visual -->
  <div style="margin-bottom:12px;">
    <div style="font-size:7px;color:#444;letter-spacing:0.6px;
                margin-bottom:6px;">SIGNAL BREAKDOWN</div>
    <div style="display:flex;gap:10px;align-items:flex-start;flex-wrap:wrap;">
      {dots_html}
    </div>
  </div>

  <!-- REGIME POSITION -->
  <div style="margin-bottom:12px;">
    <div style="display:flex;gap:4px;">{segs_html}</div>
  </div>

  {"" if not falsif else f'<div style="font-size:8px;color:#555;border-top:1px solid #1a1a1a;padding-top:8px;margin-bottom:8px;font-style:italic;">{falsif}</div>'}

  <!-- FOOTER -->
  <div style="font-size:7px;color:#2a2a2a;letter-spacing:0.4px;">
    macroengine.io &nbsp;·&nbsp; not financial advice &nbsp;·&nbsp;
    {n_tot} signals · {n_pos} confirm · {n_neg} against
  </div>
</div>"""
    return card_html

# ══════════════════════════════════════════════════════════════════════════════
# TOPBAR
# ══════════════════════════════════════════════════════════════════════════════

h1, h2 = st.columns([5, 1])
with h1:
    st.markdown(
        f"""<div class="me-topbar">
          <div style="display:flex;justify-content:space-between;align-items:center;
                      gap:12px;flex-wrap:wrap;">
            <div>
              <div class="me-title">Asset Monitor</div>
              <div class="me-subtle">
                Macro-conditional analysis · SPY · QQQ · GLD · BTC · SLV
                &nbsp;·&nbsp; regime: <b style="color:{cur_color}">{cur_label} {cur_score}</b>
              </div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)
with h2:
    if st.button("← Home", width='stretch'):
        safe_switch_page("app.py")

# ══════════════════════════════════════════════════════════════════════════════
# ASSET TABS
# ══════════════════════════════════════════════════════════════════════════════

tab_labels = [f"{WATCH[t]['emoji']} {t}" for t in ALL_T]
tabs       = st.tabs(tab_labels)

for tab, ticker in zip(tabs, ALL_T):
    with tab:
        meta = WATCH[ticker]

        # Pre-compute
        align, signals, summary, align_color = macro_alignment(ticker, regime, macro, px)
        reg_rets = regime_returns(ticker, reg_hist, px)
        cur_rets = reg_rets.get(cur_label, [])

        if ticker in px.columns:
            price_s = px[ticker].dropna()
            price   = _last(price_s)
            ret_1w  = _ret(price_s, 7)
            ret_1m  = _ret(price_s, 30)
            ret_3m  = _ret(price_s, 63)
            ma50    = _ma(price_s, 50)
            ma200   = _ma(price_s, 200)
            pct_rank_1y = _pct_rank(price_s)
            price_z = _zscore(price_s)
        else:
            price = ret_1w = ret_1m = ret_3m = None
            ma50 = ma200 = pd.Series(dtype=float)
            pct_rank_1y = price_z = None

        wr   = (sum(1 for r in cur_rets if r > 0)/len(cur_rets)) if len(cur_rets) >= 8 else None
        med  = float(np.median(cur_rets))*100 if cur_rets else None
        p25  = float(np.percentile(cur_rets, 25))*100 if cur_rets else None
        p75  = float(np.percentile(cur_rets, 75))*100 if cur_rets else None

        # ── ROW 1: KPI strip + macro alignment ───────────────────────────────
        kc1, kc2, kc3, kc4, kc5, kc6 = st.columns(6, gap="small")

        def _kpi(col, label, val, sub, color="#0f172a", bg="#f8fafc"):
            col.markdown(
                f"<div style='padding:10px 10px;border-radius:10px;background:{bg};"
                f"border:1px solid rgba(0,0,0,0.07);'>"
                f"<div style='font-size:8px;font-weight:700;color:rgba(0,0,0,0.38);"
                f"text-transform:uppercase;letter-spacing:0.4px;margin-bottom:3px;'>{label}</div>"
                f"<div style='font-size:16px;font-weight:900;color:{color};line-height:1.1;'>{val}</div>"
                f"<div style='font-size:9px;color:rgba(0,0,0,0.48);margin-top:2px;'>{sub}</div>"
                f"</div>", unsafe_allow_html=True)

        _kpi(kc1, "Price",
             f"${price:.2f}" if price else "—", "current",
             meta["color"])
        _kpi(kc2, "1W Return",
             fmt_pct(ret_1w), "7 days",
             "#1f7a4f" if ret_1w and ret_1w > 0 else "#b42318")
        _kpi(kc3, "1M Return",
             fmt_pct(ret_1m), "30 days",
             "#1f7a4f" if ret_1m and ret_1m > 0 else "#b42318")
        _kpi(kc4, "52w Pct Rank",
             f"{pct_rank_1y:.0f}th" if pct_rank_1y else "—",
             "price vs 1y history",
             "#b42318" if pct_rank_1y and pct_rank_1y > 80 else "#1f7a4f" if pct_rank_1y and pct_rank_1y < 20 else "#6b7280")
        _kpi(kc5, "Macro Alignment",
             f"{align}/100", summary,
             align_color,
             "#dcfce7" if align >= 60 else "#fee2e2" if align <= 40 else "#f3f4f6")
        if wr is not None:
            _kpi(kc6, f"Win Rate ({cur_label})",
                 f"{wr:.0%}", f"n={len(cur_rets)} weeks · med {med:+.1f}%",
                 "#1f7a4f" if wr >= 0.58 else "#b42318" if wr < 0.45 else "#6b7280")
        else:
            _kpi(kc6, "Win Rate", "—", "need 8+ regime weeks")

        st.markdown("")

        # ── ROW 2: Price chart + Signal breakdown ─────────────────────────────
        chart_col, signal_col = st.columns([2.0, 1.0], gap="large")

        with chart_col:
            with st.container(border=True):
                # Price chart with MA overlay and regime background
                rng_key = f"am_{ticker}_range"
                rng = st.selectbox("Range", ["3m","6m","1y","2y","5y"],
                                   index=2, key=rng_key)
                days_map = {"3m":63,"6m":126,"1y":252,"2y":504,"5y":1260}
                nd = days_map.get(rng, 252)

                if price and not price_s.empty:
                    sl = price_s.iloc[-nd:]
                    fig = go.Figure()

                    # Regime background — colour by regime label
                    if not reg_hist.empty:
                        reg_colors = {
                            "Risk On":"rgba(31,122,79,0.08)",
                            "Bullish":"rgba(22,163,74,0.06)",
                            "Neutral":"rgba(107,114,128,0.04)",
                            "Bearish":"rgba(217,119,6,0.06)",
                            "Risk Off":"rgba(180,35,24,0.08)",
                        }
                        rh_sl = reg_hist[reg_hist.index >= sl.index[0]]
                        prev_d = sl.index[0]; prev_l = None
                        for d, row in rh_sl.iterrows():
                            if d > sl.index[-1]: break
                            if prev_l and prev_l != row["label"]:
                                fig.add_vrect(x0=prev_d, x1=d,
                                              fillcolor=reg_colors.get(prev_l,"rgba(0,0,0,0)"),
                                              line_width=0)
                                prev_d = d
                            prev_l = row["label"]
                        if prev_l:
                            fig.add_vrect(x0=prev_d, x1=sl.index[-1],
                                          fillcolor=reg_colors.get(prev_l,"rgba(0,0,0,0)"),
                                          line_width=0)

                    # MAs
                    ma50_sl  = ma50.loc[ma50.index >= sl.index[0]]  if not ma50.empty  else pd.Series()
                    ma200_sl = ma200.loc[ma200.index >= sl.index[0]] if not ma200.empty else pd.Series()

                    if not ma200_sl.dropna().empty:
                        fig.add_trace(go.Scatter(x=ma200_sl.index, y=ma200_sl.values,
                            mode="lines", name="200d MA",
                            line=dict(color="#94a3b8", width=1.4, dash="dash")))
                    if not ma50_sl.dropna().empty:
                        fig.add_trace(go.Scatter(x=ma50_sl.index, y=ma50_sl.values,
                            mode="lines", name="50d MA",
                            line=dict(color="#d97706", width=1.2, dash="dot")))

                    # Price line
                    fig.add_trace(go.Scatter(x=sl.index, y=sl.values,
                        mode="lines", name=ticker,
                        line=dict(color=meta["color"], width=2.2)))

                    lo = float(sl.min())*0.97; hi = float(sl.max())*1.03
                    fig.update_layout(
                        height=320, margin=dict(l=10,r=20,t=10,b=10),
                        plot_bgcolor="white", paper_bgcolor="white",
                        hovermode="x unified",
                        legend=dict(orientation="h", y=1.04, x=0, font_size=10),
                        yaxis=dict(range=[lo,hi], showgrid=True, gridcolor="#f1f5f9"),
                        xaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
                    )
                    st.plotly_chart(fig, width='stretch')
                    st.caption(f"Background colour = macro regime at that time. "
                               f"Green = bullish regime · Red = bearish · Grey = neutral.")

        with signal_col:
            with st.container(border=True):
                st.markdown("<div class='me-rowtitle'>Macro signal breakdown</div>",
                            unsafe_allow_html=True)
                st.caption(f"Key macro drivers for {meta['name']} right now.")

                # Alignment gauge bar
                gauge_c = align_color
                st.markdown(
                    f"<div style='margin-bottom:12px;'>"
                    f"<div style='display:flex;justify-content:space-between;"
                    f"margin-bottom:4px;'>"
                    f"<span style='font-size:11px;font-weight:800;color:{gauge_c};'>{summary}</span>"
                    f"<span style='font-size:11px;font-weight:700;color:{gauge_c};'>{align}/100</span>"
                    f"</div>"
                    f"<div style='background:rgba(0,0,0,0.06);border-radius:4px;height:6px;'>"
                    f"<div style='width:{align}%;background:{gauge_c};height:100%;border-radius:4px;'></div>"
                    f"</div></div>", unsafe_allow_html=True)

                for sig_label, sig_val, sig_text in signals:
                    sc = "#1f7a4f" if sig_val>0 else ("#b42318" if sig_val<0 else "#6b7280")
                    sb = "#dcfce7" if sig_val>0 else ("#fee2e2" if sig_val<0 else "#f3f4f6")
                    icon = "✓" if sig_val>0 else ("✗" if sig_val<0 else "·")
                    st.markdown(
                        f"<div style='display:flex;gap:8px;align-items:flex-start;"
                        f"padding:6px 0;border-bottom:1px solid rgba(0,0,0,0.04);'>"
                        f"<span style='width:18px;height:18px;border-radius:50%;"
                        f"background:{sb};display:flex;align-items:center;"
                        f"justify-content:center;font-size:10px;font-weight:900;"
                        f"color:{sc};flex-shrink:0;'>{icon}</span>"
                        f"<div><div style='font-size:11px;font-weight:700;"
                        f"color:rgba(0,0,0,0.75);'>{sig_label}</div>"
                        f"<div style='font-size:10px;color:rgba(0,0,0,0.50);"
                        f"line-height:1.4;'>{sig_text}</div>"
                        f"</div></div>", unsafe_allow_html=True)

        st.markdown("")

        # ── ROW 3: Return distribution + Content generator ────────────────────
        dist_col, content_col = st.columns([1.0, 1.2], gap="large")

        with dist_col:
            with st.container(border=True):
                st.markdown("<div class='me-rowtitle'>4-week return distribution by regime</div>",
                            unsafe_allow_html=True)
                if reg_rets:
                    labels_ord = ["Risk On","Bullish","Neutral","Bearish","Risk Off"]
                    fig_dist = go.Figure()
                    for lbl in labels_ord:
                        rets = reg_rets.get(lbl, [])
                        if len(rets) < 4: continue
                        rc = {"Risk On":"#1f7a4f","Bullish":"#16a34a",
                              "Neutral":"#6b7280","Bearish":"#d97706",
                              "Risk Off":"#b42318"}.get(lbl,"#6b7280")
                        is_cur = (lbl == cur_label)
                        fig_dist.add_trace(go.Box(
                            y=[r*100 for r in rets],
                            name=lbl,
                            marker_color=rc,
                            line_width=2.0 if is_cur else 1.0,
                            fillcolor=f"rgba({int(rc[1:3],16)},{int(rc[3:5],16)},{int(rc[5:7],16)},{'0.25' if is_cur else '0.10'})",
                            boxpoints="outliers",
                            marker_size=3,
                        ))
                    fig_dist.add_hline(y=0, line_color="#94a3b8", line_width=1)
                    fig_dist.update_layout(
                        height=260, margin=dict(l=10,r=10,t=10,b=10),
                        plot_bgcolor="white", paper_bgcolor="white",
                        yaxis=dict(title="4w return %", showgrid=True,
                                   gridcolor="#f1f5f9", zeroline=False),
                        xaxis=dict(showgrid=False),
                        showlegend=False,
                        hovermode="closest",
                    )
                    st.plotly_chart(fig_dist, width='stretch')
                    st.caption(f"Current regime ({cur_label}) shown with thicker border. "
                               f"Each box = p25/median/p75 of 4-week forward returns.")
                else:
                    st.caption("Not enough regime history to compute distributions.")

        with content_col:
            with st.container(border=True):
                st.markdown("<div class='me-rowtitle'>📸 Macro snapshot card</div>",
                            unsafe_allow_html=True)
                st.caption("Screenshot this card and attach it to your comment. "
                           "Tap the ↻ button to refresh after market moves.")

                card_html = build_macro_card(
                    ticker, align, align_color, summary,
                    signals, reg_rets, macro, px)

                st.markdown(card_html, unsafe_allow_html=True)

                st.markdown("")
                c1, c2 = st.columns(2, gap="small")
                with c1:
                    if st.button("↻ Refresh", key=f"refresh_{ticker}",
                                 width="stretch"):
                        st.rerun()
                with c2:
                    st.markdown(
                        f"<div style='font-size:10px;color:rgba(0,0,0,0.45);"
                        f"padding:6px 0;'>"
                        f"Right-click → Save image, or screenshot the card above"
                        f"</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

st.markdown("<div style='height:48px;'></div>", unsafe_allow_html=True)
