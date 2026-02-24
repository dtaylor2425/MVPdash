import pandas as pd
import plotly.graph_objects as go

def line_chart(df: pd.DataFrame, title: str, y_cols: list[str], y_axis_title: str = "") -> go.Figure:
    fig = go.Figure()
    for c in y_cols:
        if c in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[c], mode="lines", name=c))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=y_axis_title,
        height=360,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig

def single_line(series: pd.Series, title: str, name: str, y_axis_title: str = "") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", name=name))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=y_axis_title,
        height=320,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig

def ratio_chart(a: pd.Series, b: pd.Series, title: str, name: str) -> go.Figure:
    idx = a.index.intersection(b.index)
    r = (a.loc[idx] / b.loc[idx]).dropna()
    return single_line(r, title=title, name=name)
