#!/usr/bin/env python3
"""
generate_content.py
═══════════════════════════════════════════════════════════════════════════════
Run this whenever you want ready-to-revise post content.

Usage:
    python scripts/generate_content.py

Output:
    Prints X post + Substack note to terminal.
    Saves to scripts/output/YYYY-MM-DD_HHMM_<signal>.md

Setup:
    pip install anthropic fredapi yfinance pandas numpy scipy python-dotenv
    Add FRED_API_KEY and ANTHROPIC_API_KEY to your .env file
"""

import os, sys, json
from pathlib import Path
from datetime import date, datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import numpy as np
import pandas as pd

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
FRED_API_KEY      = os.environ.get("FRED_API_KEY", "")

for name, val in [("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY),
                  ("FRED_API_KEY",      FRED_API_KEY)]:
    if not val:
        print(f"ERROR: {name} not set. Add it to your .env file.")
        sys.exit(1)

# ── Load data ─────────────────────────────────────────────────────────────────

print("Loading live data...")

from src.config import CACHE_DIR, FRED_SERIES, YF_PROXIES
from src.data_sources import fetch_prices, get_fred_cached
from src.regime import compute_regime_v3, compute_regime_timeseries

macro    = get_fred_cached(FRED_SERIES, FRED_API_KEY, CACHE_DIR,
                           cache_name="fred_macro").sort_index()
TICKERS  = ["SPY","IWM","QQQ","XLE","XLF","XLK","XLP","XLV",
            "GLD","UUP","HYG","TLT","IBIT","XBI"]
vix_t    = [v for v in [YF_PROXIES.get("vix"), YF_PROXIES.get("vix3m")] if v]
px       = fetch_prices(list(dict.fromkeys(TICKERS + vix_t)), period="5y")
px       = pd.DataFrame() if (px is None or px.empty) else px.sort_index()

regime   = compute_regime_v3(macro=macro, proxies=px,
                             lookback_trend=63, momentum_lookback_days=21)
reg_hist = compute_regime_timeseries(macro, px, freq="W-FRI")

print(f"Regime: {regime.label} (score {regime.score})")

# ── Signal helpers ────────────────────────────────────────────────────────────

def _last(s):
    s = s.dropna()
    return float(s.iloc[-1]) if not s.empty else None

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

# ── Compute live reads ────────────────────────────────────────────────────────

curve_2_10 = (macro["y10"] - macro["y2"]).dropna() \
             if "y10" in macro.columns and "y2" in macro.columns else pd.Series(dtype=float)
curve_3m10 = (macro["y10"] - macro["y3m"]).dropna() \
             if "y10" in macro.columns and "y3m" in macro.columns else pd.Series(dtype=float)

c210_now    = _last(curve_2_10)
c3m10_now   = _last(curve_3m10)
c210_1m     = _delta(curve_2_10, 30)
real_now    = _last(macro["real10"])    if "real10"       in macro.columns else None
real_1m     = _delta(macro["real10"],30) if "real10"      in macro.columns else None
hy_now      = _last(macro["hy_oas"])    if "hy_oas"       in macro.columns else None
hy_7d       = _delta(macro["hy_oas"],7)  if "hy_oas"      in macro.columns else None
hy_z        = _zscore(macro["hy_oas"])   if "hy_oas"      in macro.columns else None
dollar_now  = _last(macro["dollar_broad"]) if "dollar_broad" in macro.columns else None
dollar_1m   = _delta(macro["dollar_broad"],30) if "dollar_broad" in macro.columns else None
fed_now     = _last(macro["fed_assets"])   if "fed_assets" in macro.columns else None
fed_13w     = None
if "fed_assets" in macro.columns:
    fa = macro["fed_assets"].dropna()
    if len(fa) >= 70:
        fed_13w = float(fa.pct_change(63).iloc[-1] * 100)
cpi_now = None
if "cpi" in macro.columns and len(macro["cpi"].dropna()) >= 13:
    cpi_now = _last((macro["cpi"].dropna().pct_change(12)*100).dropna())

vix_now = vix3m_now = vratio = vix_pct = None
vt = YF_PROXIES.get("vix"); v3t = YF_PROXIES.get("vix3m")
if vt and vt in px.columns:
    vix_now = _last(px[vt])
    vix_pct = _pct_rank(px[vt])
if v3t and v3t in px.columns:
    vix3m_now = _last(px[v3t])
if vix_now and vix3m_now:
    vratio = vix_now / vix3m_now

iwm_spy_z = None
if "IWM" in px.columns and "SPY" in px.columns:
    iwm_spy_z = _zscore((px["IWM"]/px["SPY"]).dropna())

score_delta  = getattr(regime, "score_delta", 0) or 0
prev_label   = reg_hist["label"].iloc[-2] \
               if not reg_hist.empty and len(reg_hist) >= 2 else None

# Markov transition probs
stay_p = degrade_p = improve_p = None
if not reg_hist.empty and len(reg_hist) >= 20:
    worse  = {"Risk On":["Bullish","Neutral","Bearish","Risk Off"],
               "Bullish":["Neutral","Bearish","Risk Off"],
               "Neutral":["Bearish","Risk Off"],
               "Bearish":["Risk Off"], "Risk Off":[]}
    better = {"Risk Off":["Bearish","Neutral","Bullish","Risk On"],
               "Bearish":["Neutral","Bullish","Risk On"],
               "Neutral":["Bullish","Risk On"],
               "Bullish":["Risk On"], "Risk On":[]}
    lbl_s  = reg_hist["label"].values
    mat    = defaultdict(lambda: defaultdict(int))
    for i in range(len(lbl_s)-1):
        mat[lbl_s[i]][lbl_s[i+1]] += 1
    cur    = reg_hist["label"].iloc[-1]
    total  = sum(mat[cur].values())
    if total > 0:
        stay_p    = mat[cur][cur] / total
        degrade_p = sum(mat[cur][w]/total for w in worse.get(cur,[]))
        improve_p = sum(mat[cur][b]/total for b in better.get(cur,[]))

# Market returns 1w
mkt_ret = {}
for t in ["SPY","QQQ","IWM","XLE","XLF","GLD","HYG","TLT"]:
    if t in px.columns:
        s = px[t].dropna()
        if len(s) >= 6:
            mkt_ret[t] = round(float(s.iloc[-1]/s.iloc[-6]-1)*100, 2)

# Component summary
comp_summary = {}
if isinstance(regime.components, dict):
    for k, c in regime.components.items():
        if not isinstance(c, dict): continue
        comp_summary[c.get("name",k)] = {
            "z":      round(float(c["zscore"]),   2) if c.get("zscore")     is not None else None,
            "mom_z":  round(float(c["roc_zscore"]),2) if c.get("roc_zscore") is not None else None,
            "contrib":round(float(c.get("contribution",0)),3),
        }

# ── Pick top signal ───────────────────────────────────────────────────────────

candidates = []

if prev_label and prev_label != regime.label:
    candidates.append(("regime_change", 10))
if vratio and vratio > 1.0:
    candidates.append(("vratio_panic", 9))
if c210_now is not None and len(curve_2_10) >= 2:
    c_prev = float(curve_2_10.iloc[-2])
    if (c210_now < 0 and c_prev >= 0) or (c210_now >= 0 and c_prev < 0):
        candidates.append(("curve_cross", 8))
if degrade_p and degrade_p > 0.40:
    candidates.append(("degrade_warning", 7))
if abs(score_delta) >= 8:
    candidates.append(("score_momentum", 6))
if hy_7d and abs(hy_7d) > 0.25:
    candidates.append(("credit_move", 5))
if vix_pct and vix_pct > 80:
    candidates.append(("vix_elevated", 4))
if c210_1m and abs(c210_1m) > 0.20:
    candidates.append(("curve_momentum", 3))
candidates.append(("weekly_summary", 1))

top_signal = sorted(candidates, key=lambda x: x[1], reverse=True)[0][0]
print(f"Top signal: {top_signal}")

# ── Build context ─────────────────────────────────────────────────────────────

def r(x, nd=2):
    return round(float(x), nd) if x is not None else None

ctx = {
    "today":       str(date.today()),
    "signal_type": top_signal,
    "regime": {
        "label":       regime.label,
        "score":       regime.score,
        "score_raw":   r(regime.score_raw, 1),
        "confidence":  regime.confidence,
        "momentum":    regime.momentum_label,
        "score_delta": score_delta,
        "prev_label":  prev_label,
    },
    "macro": {
        "curve_2_10":   r(c210_now),
        "curve_3m10":   r(c3m10_now),
        "curve_1m_chg": r(c210_1m),
        "real_yield":   r(real_now),
        "real_1m_chg":  r(real_1m),
        "hy_oas":       r(hy_now),
        "hy_7d_chg":    r(hy_7d),
        "hy_zscore":    r(hy_z),
        "dollar":       r(dollar_now, 1),
        "dollar_1m":    r(dollar_1m, 1),
        "cpi_yoy":      r(cpi_now, 1),
        "fed_assets_T": r(fed_now/1e6, 2) if fed_now else None,
        "fed_13w_pct":  r(fed_13w, 1),
    },
    "volatility": {
        "vix":      r(vix_now, 1),
        "vix3m":    r(vix3m_now, 1),
        "vratio":   r(vratio, 3),
        "vix_pct":  r(vix_pct, 0),
    },
    "breadth":    {"iwm_spy_z": r(iwm_spy_z)},
    "transition": {"stay": r(stay_p), "degrade": r(degrade_p), "improve": r(improve_p)},
    "components": comp_summary,
    "mkt_1w":     mkt_ret,
}

# ── Generate with Claude ──────────────────────────────────────────────────────

print("Generating with Claude...")

import anthropic
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

SYSTEM = """You write macro market commentary for Macro Engine — a real-time 
macro regime scoring platform for serious traders.

Voice: Direct. Data-first. No filler. Sound like a macro trader explaining 
a position to a peer. Never use "fascinating", "dive into", "it's worth noting".
No emojis except optionally one at the start of an X post.

Return ONLY valid JSON with these exact keys:
{
  "signal_headline": "one line — what is actually happening right now",
  "x_post": "standalone tweet, max 265 chars",
  "x_thread": ["tweet 1", "tweet 2", "tweet 3"] or [],
  "substack_title": "post title",
  "substack_body": "150-300 words with markdown formatting",
  "post_angle": "one sentence — the specific angle you chose and why"
}

x_thread: only if the signal genuinely needs depth. 3-4 tweets max.
x_post: always populate — it's the standalone single-tweet version.
The signal_type is a hint. If something else in the data is more 
interesting, write about that instead."""

response = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=1500,
    system=SYSTEM,
    messages=[{"role": "user", "content":
        f"Signal: {top_signal}\n\nData:\n{json.dumps(ctx, indent=2)}\n\n"
        f"Find the most interesting angle. Write the content."}]
)

raw = response.content[0].text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()

try:
    out = json.loads(raw)
except json.JSONDecodeError:
    print("\nRaw Claude output (not valid JSON):\n")
    print(raw)
    sys.exit(1)

# ── Print ─────────────────────────────────────────────────────────────────────

S = "─" * 60
print(f"\n{S}")
print(f"SIGNAL:  {out.get('signal_headline','')}")
print(f"ANGLE:   {out.get('post_angle','')}")
print(S)

x_post = out.get("x_post","")
print(f"\n── X  ({len(x_post)} chars) {'✓' if len(x_post)<=280 else '✗ TOO LONG'}")
print(x_post)

if out.get("x_thread"):
    print(f"\n── X THREAD ({len(out['x_thread'])} tweets)")
    for i, t in enumerate(out["x_thread"], 1):
        print(f"\n[{i}] {t}")

print(f"\n── SUBSTACK")
print(f"Title: {out.get('substack_title','')}\n")
print(out.get("substack_body",""))
print(S)

# ── Save ──────────────────────────────────────────────────────────────────────

out_dir = Path(__file__).parent / "output"
out_dir.mkdir(exist_ok=True)
ts      = datetime.now().strftime("%Y-%m-%d_%H%M")
outfile = out_dir / f"{ts}_{top_signal}.md"

md = f"""# {out.get('substack_title','')}
*{datetime.now().strftime("%Y-%m-%d %H:%M")} · signal: {top_signal}*

---

## Signal
**{out.get('signal_headline','')}**
*Angle: {out.get('post_angle','')}*

---

## X post ({len(x_post)} chars)
```
{x_post}
```
"""

if out.get("x_thread"):
    md += "\n## X thread\n"
    for i, t in enumerate(out["x_thread"], 1):
        md += f"\n**[{i}]** {t}\n"

md += f"""
---

## Substack

### {out.get('substack_title','')}

{out.get('substack_body','')}

---
*Raw data: {json.dumps(ctx)}*
"""

outfile.write_text(md, encoding="utf-8")
print(f"\nSaved → {outfile.name}")