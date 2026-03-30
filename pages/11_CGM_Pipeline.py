# pages/11_CGM_Pipeline.py
"""
Computational Global Macro Pipeline
═══════════════════════════════════════════════════════════════════════════════
Integrates the four-layer Simonian (2025) pipeline with live macro engine data.

Layer 1 — Scenario Engine : CMAB (UCB) + Q-learning War of Attrition
Layer 2 — Causal Filter   : Noisy-OR models per asset (z-scores → cause probs)
Layer 3 — View Generator  : MacroView objects with direction + confidence
Layer 4 — Portfolio       : Black-Litterman, Ordinal BL, Robust MVO weights

All inputs are derived from your live FRED data and regime components —
no hardcoded assumptions.
"""

import sys
import os

# Ensure the project root (macro_engine/) is on sys.path so that
# `cgm`, `src`, etc. are all importable regardless of where Streamlit
# launches from.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import CACHE_DIR, FRED_API_KEY, FRED_SERIES, YF_PROXIES
from src.data_sources import fetch_prices, get_fred_cached
from src.regime import compute_regime_v3
from src.ui import inject_css, sidebar_nav, safe_switch_page, regime_color, regime_bg

st.set_page_config(
    page_title="CGM Pipeline",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()
sidebar_nav(active="CGM Pipeline")

if not FRED_API_KEY:
    st.error("FRED_API_KEY is not set.")
    st.stop()

# ── Data ──────────────────────────────────────────────────────────────────────

TICKERS = ["SPY","QQQ","IWM","TLT","HYG","GLD","CPER","^VIX","^VIX3M"]

@st.cache_data(ttl=30*60, show_spinner=False)
def load_data():
    macro = get_fred_cached(FRED_SERIES, FRED_API_KEY, CACHE_DIR,
                            cache_name="fred_macro").sort_index()
    px = fetch_prices(TICKERS, period="5y")
    px = pd.DataFrame() if (px is None or px.empty) else px.sort_index()
    regime = compute_regime_v3(macro=macro, proxies=px,
                               lookback_trend=63, momentum_lookback_days=21)
    return macro, px, regime

macro, px, regime = load_data()

cur_label  = regime.label
cur_score  = regime.score
cur_color  = regime_color(cur_label)
cur_bg     = regime_bg(cur_label)
components = getattr(regime, "components", {}) or {}

# ── Topbar ────────────────────────────────────────────────────────────────────

h1, h2 = st.columns([5, 1])
with h1:
    st.markdown(
        f"""<div class="me-topbar">
          <div style="display:flex;justify-content:space-between;
                      align-items:center;gap:12px;flex-wrap:wrap;">
            <div>
              <div class="me-title">Computational Global Macro Pipeline</div>
              <div class="me-subtle">
                Simonian (2025) · CMAB · Q-learning · Noisy-OR · Black-Litterman
                &nbsp;·&nbsp; Regime: <b style="color:{cur_color}">{cur_label} {cur_score}</b>
              </div>
            </div>
          </div>
        </div>""",
        unsafe_allow_html=True)
with h2:
    if st.button("← Home", width="stretch"):
        safe_switch_page("app.py")

# ── Educational header ────────────────────────────────────────────────────────

with st.expander("What is this? How does the CGM pipeline work?", expanded=False):
    st.markdown("""
**Computational Global Macro** (Simonian, 2025) formalises thematic macro investing
using four layers of quantitative machinery, each feeding the next:

| Layer | Method | What it produces |
|-------|--------|-----------------|
| 1 — Scenario Engine | CMAB (UCB/Thompson) + Q-learning | Which macro 'action' (risk-on / risk-off) is optimal given the game-theoretic payoff structure |
| 2 — Causal Filter | Noisy-OR probabilistic DAG | P(positive return) per asset, combining multiple promoting and inhibiting causes |
| 3 — View Generator | Confidence thresholding | MacroView objects: direction (+1/-1) and confidence score per asset |
| 4 — Portfolio | Black-Litterman / Ordinal BL / Robust MVO | Bayesian posterior weights blending market equilibrium with causal views |

**Integration with this engine:** The cause probabilities in Layer 2 are derived directly
from your live FRED z-scores — so the causal DAG updates every time your data refreshes.
The BL model uses your trailing ETF returns and correlations as the prior. No hardcoded inputs.
    """)

st.markdown("")

# ── Run button ────────────────────────────────────────────────────────────────

with st.container(border=True):
    st.markdown("<div class='me-rowtitle'>Pipeline configuration</div>",
                unsafe_allow_html=True)
    st.caption("The pipeline inherits live data from your macro engine. "
               "Adjust optional parameters below before running.")

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    risk_aversion = c1.number_input("Risk aversion (γ)", 1.0, 5.0, 2.5, 0.1)
    tau_val       = c2.number_input("BL tau", 0.005, 0.10, 0.025, 0.005,
                                     format="%.3f")
    bull_thr      = c3.number_input("Bull threshold", 0.50, 0.75, 0.55, 0.01)
    bear_thr      = c4.number_input("Bear threshold", 0.25, 0.50, 0.45, 0.01)
    w_min_v       = c1.number_input("Min weight", 0.0, 0.10, 0.02, 0.01)
    w_max_v       = c2.number_input("Max weight", 0.10, 0.60, 0.40, 0.05)

    run_btn = st.button("▶  Run CGM Pipeline", type="primary",
                        use_container_width=False)

if not run_btn:
    st.info("Configure parameters above and click **▶ Run CGM Pipeline** to execute.")
    st.stop()

# ── Execute pipeline ──────────────────────────────────────────────────────────

st.markdown("")
progress = st.progress(0, text="Initialising pipeline...")

try:
    from cgm.bridge import (build_nor_models_from_regime,
                             build_pipeline_config,
                             build_cmab_contexts_from_regime,
                             result_to_etf_weights,
                             nor_probs_summary,
                             CGM_ASSETS, PROXY_MAP, ASSET_LABELS)
    from cgm.pipeline import (ComputationalGlobalMacroPipeline,
                               PipelineConfig, PipelineResult)
    from cgm.engines.cmab import CMABSimulation, UCBAgent
    from cgm.engines.qlearning import (TwoPlayerSimulation, QLearningAgent,
                                        SARSAAgent, make_war_of_attrition)
    from cgm.engines.causal import ViewAggregator
    from cgm.portfolio.optimizer import BlackLitterman, OrdinalBL, RobustMVO

except ImportError as e:
    st.error(f"CGM module import failed: {e}. "
             f"Ensure cgm/ is in your project root and scipy is installed.")
    st.stop()

# Layer 1a: CMAB
progress.progress(10, "Layer 1a: CMAB scenario simulation...")
contexts = build_cmab_contexts_from_regime(cur_label, cur_score, components)
cmab_agent = UCBAgent(n_arms=2, c=1.0)
cmab_sim = CMABSimulation(contexts, cmab_agent, n_rounds=400, n_iters=250)
sim_cmab = cmab_sim.run()

# Layer 1b: Q-learning War of Attrition
progress.progress(25, "Layer 1b: Q-learning War of Attrition (Fed vs ECB)...")
spec = make_war_of_attrition()
p1 = QLearningAgent(spec.n_states, spec.n_actions_p1,
                     alpha=0.1, gamma=0.9, epsilon=0.1)
p2 = SARSAAgent(spec.n_states, spec.n_actions_p2,
                 alpha=0.1, gamma=0.9, epsilon=0.1)
ql_sim = TwoPlayerSimulation(spec, p1, p2, n_rounds=24, n_iters=400)
sim_ql = ql_sim.run()

# Layer 2: Noisy-OR
progress.progress(50, "Layer 2: Noisy-OR causal filter...")
nor_models = build_nor_models_from_regime(components, macro, cur_score)
nor_probs  = {a: nor_models[a].net_probability() for a in nor_models}

# Layer 3: View generator
progress.progress(65, "Layer 3: View generator...")
aggregator = ViewAggregator(
    assets=CGM_ASSETS,
    threshold_bull=bull_thr,
    threshold_bear=bear_thr,
)
views = aggregator.from_nor(nor_models)

# Layer 4: Portfolio
progress.progress(80, "Layer 4: Portfolio optimization (BL / OBL / Robust)...")
cfg = build_pipeline_config(
    px=px, cur_score=cur_score, cur_label=cur_label,
    risk_aversion=risk_aversion, tau=tau_val,
    bull_threshold=bull_thr, bear_threshold=bear_thr,
    w_min=w_min_v, w_max=w_max_v,
)

# Build sigma from config
n = len(cfg.assets)
diag = np.array(cfg.sigma_diagonal)
corr = np.array(cfg.correlations)
sigma = diag[:, None] * corr * diag[None, :]

mu       = np.array(cfg.hist_returns)
w_mkt    = np.array(cfg.market_weights)

bl_model  = BlackLitterman(CGM_ASSETS, sigma, w_mkt, risk_aversion, tau_val)

# Run BL with per-asset weight bounds to prevent extreme tilts.
# TLT capped at 35%, HYG at 30% — BL over-allocates to duration/credit
# when real yields are high. Copper (CPER) capped at 15% — a commodity
# position above that is impractical in a macro portfolio.
asset_w_max = []
for a in CGM_ASSETS:
    if a == "Long_Bond":
        asset_w_max.append(min(w_max_v, 0.35))   # cap TLT at 35%
    elif a == "HY_Credit":
        asset_w_max.append(min(w_max_v, 0.30))   # cap HYG at 30%
    elif a == "Copper":
        asset_w_max.append(min(w_max_v, 0.15))   # cap CPER at 15%
    else:
        asset_w_max.append(w_max_v)

from cgm.portfolio.optimizer import _mvo as _cgm_mvo
# Use per-asset caps via posterior → custom MVO
if views:
    bl_mu, bl_sig = bl_model.posterior(views)
else:
    bl_mu, bl_sig = bl_model.pi, sigma

from scipy.optimize import minimize as _sp_minimize
def _bl_opt_capped(mu, sig, gamma, assets, w_min, w_maxes):
    import numpy as np
    n = len(mu)
    def neg_u(w): return -(mu @ w - 0.5*gamma * w @ sig @ w)
    def neg_u_g(w): return -(mu - gamma * sig @ w)
    w0 = np.full(n, 1.0/n)
    bounds = [(w_min, w_maxes[i]) for i in range(n)]
    constraints = [{"type":"eq","fun": lambda w: w.sum()-1.0}]
    res = _sp_minimize(neg_u, w0, jac=neg_u_g, method="SLSQP",
                       bounds=bounds, constraints=constraints,
                       options={"ftol":1e-9,"maxiter":500})
    from cgm.portfolio.optimizer import PortfolioResult
    w = res.x
    ret = float(mu @ w); vol = float(np.sqrt(w @ sig @ w))
    sharpe = ret/vol if vol > 0 else 0.0
    return PortfolioResult(weights=w, expected_return=ret,
                           volatility=vol, sharpe=sharpe,
                           method="Black-Litterman MVO",
                           metadata={"converged": res.success})

bl_result = _bl_opt_capped(bl_mu, bl_sig, risk_aversion,
                            CGM_ASSETS, w_min_v, asset_w_max)

# OBL — ranks from NOR probs
np_arr    = np.array([nor_probs[a] for a in CGM_ASSETS])
var_ranks = 1.0 - diag
obl_model  = OrdinalBL(CGM_ASSETS, mu, sigma, w_mkt, tau_val)
obl_result = obl_model.optimize(np_arr, var_ranks, w_min=w_min_v, w_max=w_max_v)

# Robust MVO — estimation error = vol / sqrt(3y)
bl_mu, _ = bl_model.posterior(views) if views else (mu, sigma)
est_err   = diag / np.sqrt(3.0)
robust    = RobustMVO(CGM_ASSETS, sigma, risk_aversion, est_err)
robust_result = robust.optimize(bl_mu, w_min_v, w_max_v)

progress.progress(100, "Complete.")
st.success("Pipeline complete.")

# ══════════════════════════════════════════════════════════════════════════════
# RESULTS UI
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("")

# ── Layer 1: Scenario engine results ─────────────────────────────────────────

with st.container(border=True):
    st.markdown("<div class='me-rowtitle'>Layer 1 — Scenario Engine</div>",
                unsafe_allow_html=True)
    st.caption("CMAB learns which action (Risk-On vs Risk-Off) maximises "
               "long-run payoff given the current macro type-state probabilities. "
               "Q-learning simulates the Fed (Q-learning) vs ECB (SARSA) "
               "monetary policy War of Attrition.")

    l1a, l1b = st.columns(2, gap="large")

    with l1a:
        st.markdown("<div class='me-rowtitle' style='font-size:11px;'>"
                    "CMAB — UCB exploration</div>", unsafe_allow_html=True)
        best_arm = sim_cmab.get("best_arm_name", "—")
        cum  = sim_cmab["cum_payoffs"]
        names = sim_cmab["arm_names"]

        # Signal counts from contexts
        bull_prior = float(np.clip(cur_score / 100.0, 0.15, 0.85))
        st.markdown(
            f"<div style='padding:10px;border-radius:8px;background:#f8fafc;"
            f"border-left:3px solid {cur_color};margin-bottom:8px;'>"
            f"<div style='font-size:11px;font-weight:800;color:{cur_color};'>"
            f"Optimal action: {best_arm}</div>"
            f"<div style='font-size:10px;color:rgba(0,0,0,0.55);margin-top:2px;'>"
            f"Bullish type-state prior: {bull_prior:.0%} "
            f"· Bearish: {1-bull_prior:.0%}</div>"
            f"</div>", unsafe_allow_html=True)

        fig_cmab = go.Figure()
        colors_cmab = ["#1d4ed8", "#b42318"]
        # Show last 80% of rounds — early exploration noise is not informative
        start = max(0, len(cum) // 5)
        for i, name in enumerate(names):
            fig_cmab.add_trace(go.Scatter(
                y=cum[start:, i], mode="lines", name=name,
                line=dict(color=colors_cmab[i % 2], width=2.2)))
        # y-axis from 0 so relative difference is visible
        y_max = float(cum[start:].max()) * 1.15
        fig_cmab.update_layout(
            height=240, margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor="white", paper_bgcolor="white",
            hovermode="x unified",
            yaxis=dict(title="Cum avg payoff", showgrid=True,
                       gridcolor="#f1f5f9", rangemode="tozero",
                       range=[0, y_max]),
            xaxis=dict(title="Round (last 80%)", showgrid=True,
                       gridcolor="#f1f5f9"),
            legend=dict(orientation="h", y=1.05, font_size=10))
        st.plotly_chart(fig_cmab, width="stretch")

    with l1b:
        st.markdown("<div class='me-rowtitle' style='font-size:11px;'>"
                    "Q-learning — Fed vs ECB War of Attrition</div>",
                    unsafe_allow_html=True)

        fed_final  = float(sim_ql["cum_avg_p1"][-1])
        ecb_final  = float(sim_ql["cum_avg_p2"][-1])
        winner     = "Fed (P1)" if fed_final > ecb_final else "ECB (P2)"
        w_color    = "#1f7a4f" if fed_final > ecb_final else "#b42318"

        st.markdown(
            f"<div style='padding:10px;border-radius:8px;background:#f8fafc;"
            f"border-left:3px solid {w_color};margin-bottom:8px;'>"
            f"<div style='font-size:11px;font-weight:800;color:{w_color};'>"
            f"Higher long-run payoff: {winner}</div>"
            f"<div style='font-size:10px;color:rgba(0,0,0,0.55);margin-top:2px;'>"
            f"Fed avg: {fed_final:.3f} · ECB avg: {ecb_final:.3f}</div>"
            f"</div>", unsafe_allow_html=True)

        fig_ql = go.Figure()
        t_ax = list(range(1, len(sim_ql["cum_avg_p1"]) + 1))
        # Use rolling average (window=5) instead of cumulative — shows
        # convergence clearly rather than the always-declining cumulative avg
        avg1 = sim_ql["avg_rewards_p1"]
        avg2 = sim_ql["avg_rewards_p2"]
        w = max(3, len(avg1) // 5)
        roll1 = pd.Series(avg1).rolling(w, min_periods=1).mean().tolist()
        roll2 = pd.Series(avg2).rolling(w, min_periods=1).mean().tolist()
        fig_ql.add_trace(go.Scatter(
            x=t_ax, y=roll1,
            mode="lines", name="Fed (Q-learning)",
            line=dict(color="#1d4ed8", width=2.2)))
        fig_ql.add_trace(go.Scatter(
            x=t_ax, y=roll2,
            mode="lines", name="ECB (SARSA)",
            line=dict(color="#d97706", width=2.2, dash="dot")))
        y_max_ql = max(max(roll1), max(roll2)) * 1.15
        fig_ql.update_layout(
            height=240, margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor="white", paper_bgcolor="white",
            hovermode="x unified",
            yaxis=dict(title=f"Rolling avg payoff ({w}-round window)",
                       showgrid=True, gridcolor="#f1f5f9",
                       rangemode="tozero", range=[0, y_max_ql]),
            xaxis=dict(title="Round", showgrid=True, gridcolor="#f1f5f9"),
            legend=dict(orientation="h", y=1.05, font_size=10))
        st.plotly_chart(fig_ql, width="stretch")

st.markdown("")

# ── Layer 2 + 3: NOR models + Views ──────────────────────────────────────────

with st.container(border=True):
    st.markdown("<div class='me-rowtitle'>Layer 2 — Causal Filter (Noisy-OR) "
                "· Layer 3 — MacroViews</div>", unsafe_allow_html=True)
    st.caption(
        "Each asset has a Noisy-OR DAG where cause probabilities are derived from "
        "your live FRED z-scores. P(positive return) above the bull threshold "
        "generates a bullish MacroView; below bear threshold → bearish view.")

    l2_summary = nor_probs_summary(nor_probs)

    # NOR probability bar chart
    fig_nor = go.Figure()
    bar_colors = [r["color"] for r in l2_summary]
    labels     = [f"{r['ticker']} ({ASSET_LABELS.get(r['asset'], r['asset'])})" for r in l2_summary]
    probs      = [r["nor_prob"] for r in l2_summary]

    fig_nor.add_trace(go.Bar(
        x=labels, y=probs,
        marker_color=bar_colors,
        text=[f"{p:.3f}" for p in probs],
        textposition="outside",
    ))
    fig_nor.add_hline(y=bull_thr, line_dash="dash", line_color="#1f7a4f",
                      line_width=1,
                      annotation_text=f"Bull threshold {bull_thr:.2f}",
                      annotation_position="right",
                      annotation_font_color="#1f7a4f", annotation_font_size=9)
    fig_nor.add_hline(y=bear_thr, line_dash="dash", line_color="#b42318",
                      line_width=1,
                      annotation_text=f"Bear threshold {bear_thr:.2f}",
                      annotation_position="right",
                      annotation_font_color="#b42318", annotation_font_size=9)
    fig_nor.update_layout(
        height=280, margin=dict(l=10, r=80, t=10, b=10),
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(range=[0, 1], title="P(positive return)",
                   showgrid=True, gridcolor="#f1f5f9"),
        xaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig_nor, width="stretch")

    # Sensitivity tables per asset in expanders
    with st.expander("Sensitivity analysis — what drives each asset's NOR probability"):
        for asset, model in nor_models.items():
            ticker = PROXY_MAP.get(asset, asset)
            st.markdown(f"**{asset} ({ticker})** — P = {nor_probs[asset]:.3f}")
            rows = model.sensitivity_table()
            sense_df = pd.DataFrame(rows)
            st.dataframe(sense_df, hide_index=True, height=140,
                         use_container_width=True)

    # Views summary — show ALL assets, highlight which crossed threshold
    st.markdown("<div class='me-rowtitle' style='margin-top:12px;'>"
                "Generated MacroViews</div>", unsafe_allow_html=True)

    # Build full asset status including neutral
    view_map = {v.asset: v for v in views}
    all_asset_cols = st.columns(len(CGM_ASSETS), gap="small")
    for i, asset in enumerate(CGM_ASSETS):
        col = all_asset_cols[i]
        ticker = PROXY_MAP.get(asset, asset)
        p = nor_probs.get(asset, 0.5)
        if asset in view_map:
            v = view_map[asset]
            vc  = "#1f7a4f" if v.direction > 0 else "#b42318"
            vbg = "#dcfce7" if v.direction > 0 else "#fee2e2"
            arrow = "▲" if v.direction > 0 else "▼"
            label = "BULLISH" if v.direction > 0 else "BEARISH"
            col.markdown(
                f"<div style='padding:8px 10px;border-radius:8px;"
                f"background:{vbg};border:1px solid {vc}44;text-align:center;'>"
                f"<div style='font-size:9px;font-weight:800;color:{vc};'>"
                f"{arrow} {label}</div>"
                f"<div style='font-size:12px;font-weight:900;"
                f"color:rgba(0,0,0,0.85);margin:2px 0;'>{ticker} · {ASSET_LABELS.get(asset, "")}</div>"
                f"<div style='font-size:9px;color:{vc};'>P={p:.3f}</div>"
                f"</div>", unsafe_allow_html=True)
        else:
            col.markdown(
                f"<div style='padding:8px 10px;border-radius:8px;"
                f"background:#f3f4f6;border:1px solid rgba(0,0,0,0.08);"
                f"text-align:center;'>"
                f"<div style='font-size:9px;font-weight:700;color:#6b7280;'>"
                f"→ NEUTRAL</div>"
                f"<div style='font-size:12px;font-weight:900;"
                f"color:rgba(0,0,0,0.60);margin:2px 0;'>{ticker}</div>"
                f"<div style='font-size:9px;color:#6b7280;'>P={p:.3f}</div>"
                f"</div>", unsafe_allow_html=True)

    if not views:
        st.markdown(
            "<div style='padding:12px;border-radius:8px;background:#f3f4f6;"
            "font-size:12px;color:rgba(0,0,0,0.60);margin-top:8px;'>"
            "No asset exceeded the threshold — all NOR probabilities "
            "in the neutral zone. Holding market-cap benchmark weights.</div>",
            unsafe_allow_html=True)

st.markdown("")

# ── Layer 4: Portfolio weights ────────────────────────────────────────────────

with st.container(border=True):
    st.markdown("<div class='me-rowtitle'>Layer 4 — Portfolio Optimization</div>",
                unsafe_allow_html=True)
    st.caption(
        "Black-Litterman blends your market equilibrium prior with the causal "
        "MacroViews. Ordinal BL uses NOR probability ranks instead of precise "
        "view magnitudes. Robust MVO adds estimation-error shrinkage.")

    tab_bl, tab_obl, tab_robust, tab_compare = st.tabs([
        "Black-Litterman", "Ordinal BL", "Robust MVO", "Comparison"
    ])

    def _weight_chart(result, title, assets=CGM_ASSETS, mkt=None):
        fig = go.Figure()
        weights = result.weights.tolist()
        colors  = ["#1f7a4f" if w > (mkt[i] if mkt else 0) else "#b42318"
                   for i, w in enumerate(weights)]
        fig.add_trace(go.Bar(
            x=[f"{PROXY_MAP.get(a,a)}\n({ASSET_LABELS.get(a,a)})" for a in assets],
            y=weights,
            marker_color=colors,
            text=[f"{w:.1%}" for w in weights],
            textposition="outside",
            name="Optimal weight",
        ))
        if mkt is not None:
            fig.add_trace(go.Scatter(
                x=[f"{PROXY_MAP.get(a,a)}\n({ASSET_LABELS.get(a,a)})" for a in assets],
                y=mkt, mode="markers+lines",
                marker=dict(size=8, color="#94a3b8"),
                line=dict(color="#94a3b8", width=1.5, dash="dash"),
                name="Market weight",
            ))
        fig.update_layout(
            height=300, margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(title="Weight", showgrid=True, gridcolor="#f1f5f9"),
            xaxis=dict(showgrid=False),
            legend=dict(orientation="h", y=1.05, font_size=10),
            hovermode="x unified",
        )
        return fig

    def _stats_row(result):
        c1, c2, c3 = st.columns(3)
        c1.metric("Expected Return", f"{result.expected_return:.2%}")
        c2.metric("Volatility",      f"{result.volatility:.2%}")
        c3.metric("Sharpe Ratio",    f"{result.sharpe:.2f}")

    with tab_bl:
        _stats_row(bl_result)
        st.plotly_chart(_weight_chart(bl_result, "BL", mkt=w_mkt.tolist()),
                        width="stretch")
        st.caption("Green = overweight vs market benchmark · "
                   "Red = underweight · Grey dashes = market weights")

    with tab_obl:
        _stats_row(obl_result)
        st.plotly_chart(_weight_chart(obl_result, "OBL", mkt=w_mkt.tolist()),
                        width="stretch")
        st.caption("Ordinal BL uses NOR probability ranks — does not require "
                   "precise return forecasts, only the relative ordering.")

    with tab_robust:
        _stats_row(robust_result)
        st.plotly_chart(_weight_chart(robust_result, "Robust", mkt=w_mkt.tolist()),
                        width="stretch")
        st.caption("Estimation-error shrinkage penalises assets where return "
                   "estimates are most uncertain (high vol / short history).")

    with tab_compare:
        st.markdown("<div class='me-rowtitle' style='margin-bottom:8px;'>"
                    "Weight comparison across methods</div>",
                    unsafe_allow_html=True)
        compare_data = {
            "Asset": [f"{a} ({PROXY_MAP.get(a,a)})" for a in CGM_ASSETS],
            "Market": [f"{w:.1%}" for w in w_mkt],
            "BL":     [f"{w:.1%}" for w in bl_result.weights],
            "OBL":    [f"{w:.1%}" for w in obl_result.weights],
            "Robust": [f"{w:.1%}" for w in robust_result.weights],
        }
        st.dataframe(pd.DataFrame(compare_data), hide_index=True,
                     use_container_width=True)

        st.markdown("")
        st.markdown(
            "<div style='padding:12px 16px;border-radius:10px;"
            "background:#f8fafc;border-left:3px solid #1d4ed8;"
            "font-size:12px;line-height:1.6;color:rgba(0,0,0,0.70);'>"
            "The three methods represent different assumptions about view precision. "
            "BL uses exact view magnitudes from NOR confidence scores. "
            "OBL only needs to know which assets rank higher — more robust to "
            "model mis-specification. Robust MVO adds a penalty for estimation "
            "uncertainty, producing the most conservative weights. "
            "In practice: use BL when you have high confidence in the Noisy-OR "
            "probabilities, OBL when you trust the ranking but not the magnitude, "
            "and Robust when the causal graph is sparse or recently assembled."
            "</div>", unsafe_allow_html=True)

st.markdown("")

# ── ETF weight translation ────────────────────────────────────────────────────

with st.container(border=True):
    st.markdown("<div class='me-rowtitle'>ETF translation — "
                "overlay on your existing signals</div>",
                unsafe_allow_html=True)
    st.caption(
        "BL weights translated back to your engine's ETF tickers. "
        "Use these as a sizing overlay alongside the Regime Playbook confluence signals.")

    etf_weights = result_to_etf_weights(
        type("R", (), {
            "bl_result":     bl_result,
            "obl_result":    obl_result,
            "robust_result": robust_result,
        })(),
        method="bl"
    )

    ew_cols = st.columns(len(etf_weights), gap="small")
    for col, (ticker, w) in zip(ew_cols, etf_weights.items()):
        wc  = "#1f7a4f" if w > 0.15 else "#b42318" if w < 0.05 else "#6b7280"
        wbg = "#dcfce7" if w > 0.15 else "#fee2e2" if w < 0.05 else "#f3f4f6"
        col.markdown(
            f"<div style='padding:8px 10px;border-radius:8px;"
            f"background:{wbg};border:1px solid {wc}22;text-align:center;'>"
            f"<div style='font-size:10px;font-weight:800;color:{wc};'>{ticker}</div>"
                        f"<div style='font-size:8px;color:{wc};opacity:0.7;'>{next((ASSET_LABELS.get(a,"") for a,t in PROXY_MAP.items() if t==ticker), "")}</div>"
            f"<div style='font-size:16px;font-weight:900;color:rgba(0,0,0,0.85);'>"
            f"{w:.1%}</div>"
            f"</div>", unsafe_allow_html=True)

st.markdown("")
st.caption("Simonian (2025) · Computational Global Macro · World Scientific · "
           "Pipeline: CMAB (UCB) + Q-learning + Noisy-OR + Black-Litterman")
st.markdown("<div style='height:48px;'></div>", unsafe_allow_html=True)
