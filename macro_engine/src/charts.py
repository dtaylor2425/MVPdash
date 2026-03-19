# src/charts.py
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── Colour palette ────────────────────────────────────────────────────────────
_BLUE   = "#1d4ed8"
_TEAL   = "#0d9488"
_RED    = "#dc2626"
_ORANGE = "#ea580c"
_PURPLE = "#7c3aed"
_GREY   = "#94a3b8"


def line_chart(df: pd.DataFrame, title: str, y_cols: list, y_axis_title: str = "") -> go.Figure:
    fig = go.Figure()
    colours = [_BLUE, _TEAL, _RED, _ORANGE, _PURPLE]
    for i, c in enumerate(y_cols):
        if c in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[c], mode="lines", name=c,
                line=dict(color=colours[i % len(colours)], width=1.8),
            ))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=y_axis_title,
        height=360,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9")
    fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
    return fig


def single_line(series: pd.Series, title: str, name: str, y_axis_title: str = "",
                color: str = _BLUE) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index, y=series.values, mode="lines", name=name,
        line=dict(color=color, width=1.8),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=y_axis_title,
        height=320,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9")
    fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
    return fig


def ratio_chart(a: pd.Series, b: pd.Series, title: str, name: str) -> go.Figure:
    idx = a.index.intersection(b.index)
    r   = (a.loc[idx] / b.loc[idx]).dropna()
    return single_line(r, title=title, name=name)


def dual_axis_chart(
    left_series: list,   # [(series, name, color), ...]
    right_series: list,  # [(series, name, color), ...]
    title: str,
    left_title: str  = "",
    right_title: str = "",
    zero_line: bool  = True,
    height: int      = 380,
) -> go.Figure:
    """
    Overlay two groups of series on separate y-axes.
    Used for curve-context chart (curve on left, fed funds / CPI / HY OAS on right).
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for s, name, color in left_series:
        s = s.dropna()
        fig.add_trace(
            go.Scatter(x=s.index, y=s.values, mode="lines", name=name,
                       line=dict(color=color, width=2.2)),
            secondary_y=False,
        )

    for s, name, color in right_series:
        s = s.dropna()
        fig.add_trace(
            go.Scatter(x=s.index, y=s.values, mode="lines", name=name,
                       line=dict(color=color, width=1.5, dash="dot")),
            secondary_y=True,
        )

    if zero_line:
        fig.add_hline(y=0, line_dash="dash", line_color="#cbd5e1",
                      line_width=1, secondary_y=False)

    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=20, r=40, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9")
    fig.update_yaxes(title_text=left_title,  showgrid=True,  gridcolor="#f1f5f9", secondary_y=False)
    fig.update_yaxes(title_text=right_title, showgrid=False, secondary_y=True)
    return fig


def vix_term_chart(
    vix:   pd.Series,
    vix3m: pd.Series,
    vix6m: pd.Series | None,
    title: str = "VIX term structure and V-Ratio",
    height: int = 420,
) -> go.Figure:
    """
    Top panel:  VIX spot, VIX3M, VIX6M — y-axis pinned to data range.
    Bottom panel: V-Ratio = VIX / VIX3M — y-axis pinned to data range.

    Rules that make the axes truly tight:
      1. Compute min/max from the SLICED series, not the full history.
      2. hrect / hline are only drawn if their y-value is inside the data window;
         they never expand the axis.
      3. The fill on V-Ratio uses a baseline trace at vr_lo (not tozeroy)
         so the fill never pulls the axis down to 0.
      4. Every yaxis.range is set explicitly with autorange=False.
    """
    vix   = vix.dropna()
    vix3m = vix3m.dropna()

    # Align on common index
    idx = vix.index.intersection(vix3m.index)

    # Handle VIX6M carefully — define v6m before any use
    v6m = pd.Series(dtype=float)
    if vix6m is not None:
        v6m = vix6m.dropna()
        if not v6m.empty:
            idx = idx.intersection(v6m.index)
            v6m = v6m.reindex(idx).dropna()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.62, 0.38],
        vertical_spacing=0.08,
        subplot_titles=["VIX term structure", "V-Ratio (VIX ÷ VIX3M) — above 1 = panic"],
    )

    if len(idx) == 0:
        return fig

    v   = vix.reindex(idx)
    v3m = vix3m.reindex(idx)

    # ── Data bounds: top panel ────────────────────────────────────────────────
    all_top = list(v.dropna()) + list(v3m.dropna())
    if not v6m.empty:
        all_top += list(v6m.dropna())

    top_min = float(min(all_top))
    top_max = float(max(all_top))
    top_pad = max((top_max - top_min) * 0.10, 0.5)
    top_lo  = top_min - top_pad
    top_hi  = top_max + top_pad

    # ── Top panel traces ──────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=v.index, y=v.values, mode="lines", name="VIX (spot)",
        line=dict(color=_RED, width=2.2),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=v3m.index, y=v3m.values, mode="lines", name="VIX3M",
        line=dict(color=_BLUE, width=1.8, dash="dot"),
    ), row=1, col=1)

    if not v6m.empty:
        fig.add_trace(go.Scatter(
            x=v6m.index, y=v6m.values, mode="lines", name="VIX6M",
            line=dict(color=_PURPLE, width=1.5, dash="dash"),
        ), row=1, col=1)

    # Reference lines — only if level is inside the visible window
    for level, lcolor, ldash in [(20, "#94a3b8", "dash"), (30, "#fca5a5", "dash")]:
        if top_lo < level < top_hi:
            fig.add_shape(type="line",
                          x0=idx[0], x1=idx[-1], y0=level, y1=level,
                          xref="x", yref="y",
                          line=dict(color=lcolor, width=1, dash=ldash),
                          row=1, col=1)

    # Fear-zone shading — only between data ceiling and 30 (if 30 is in range)
    if top_hi > 30 and top_lo < 30:
        fig.add_shape(type="rect",
                      x0=idx[0], x1=idx[-1], y0=30, y1=top_hi,
                      xref="x", yref="y",
                      fillcolor="rgba(220,38,38,0.05)", line_width=0,
                      row=1, col=1)
    elif top_lo >= 30:
        # Entire view is above 30 — shade the whole panel lightly
        fig.add_shape(type="rect",
                      x0=idx[0], x1=idx[-1], y0=top_lo, y1=top_hi,
                      xref="x", yref="y",
                      fillcolor="rgba(220,38,38,0.05)", line_width=0,
                      row=1, col=1)

    fig.update_yaxes(
        range=[top_lo, top_hi],
        autorange=False,
        showgrid=True, gridcolor="#f1f5f9",
        title_text="VIX level",
        row=1, col=1,
    )

    # ── Data bounds: V-Ratio panel ────────────────────────────────────────────
    vratio = (v / v3m).dropna()

    vr_min = float(vratio.min())
    vr_max = float(vratio.max())
    vr_pad = max((vr_max - vr_min) * 0.15, 0.01)
    vr_lo  = vr_min - vr_pad
    vr_hi  = vr_max + vr_pad

    # ── V-Ratio panel traces ──────────────────────────────────────────────────

    # Invisible baseline at vr_lo so fill="tonexty" stays within data range
    fig.add_trace(go.Scatter(
        x=vratio.index, y=[vr_lo] * len(vratio),
        mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=vratio.index, y=vratio.values,
        mode="lines", name="V-Ratio",
        line=dict(color=_ORANGE, width=2.2),
        fill="tonexty",
        fillcolor="rgba(234,88,12,0.10)",
    ), row=2, col=1)

    # Panic line at 1.0 — only if visible
    if vr_lo < 1.0 < vr_hi:
        fig.add_shape(type="line",
                      x0=idx[0], x1=idx[-1], y0=1.0, y1=1.0,
                      xref="x2", yref="y2",
                      line=dict(color="#ef4444", width=1.5),
                      row=2, col=1)

    # Complacency line at 0.9 — only if visible
    if vr_lo < 0.9 < vr_hi:
        fig.add_shape(type="line",
                      x0=idx[0], x1=idx[-1], y0=0.9, y1=0.9,
                      xref="x2", yref="y2",
                      line=dict(color="#94a3b8", width=1, dash="dash"),
                      row=2, col=1)

    # Panic zone shading — only if 1.0 is inside the view
    if vr_lo < 1.0 < vr_hi:
        fig.add_shape(type="rect",
                      x0=idx[0], x1=idx[-1], y0=1.0, y1=vr_hi,
                      xref="x2", yref="y2",
                      fillcolor="rgba(220,38,38,0.07)", line_width=0,
                      row=2, col=1)
    elif vr_lo >= 1.0:
        # Entire view is in panic territory
        fig.add_shape(type="rect",
                      x0=idx[0], x1=idx[-1], y0=vr_lo, y1=vr_hi,
                      xref="x2", yref="y2",
                      fillcolor="rgba(220,38,38,0.07)", line_width=0,
                      row=2, col=1)

    fig.update_yaxes(
        range=[vr_lo, vr_hi],
        autorange=False,
        showgrid=True, gridcolor="#f1f5f9",
        title_text="V-Ratio",
        row=2, col=1,
    )

    # ── Global layout ─────────────────────────────────────────────────────────
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9")
    return fig