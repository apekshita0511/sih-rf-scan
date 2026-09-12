"""Phase 9: Plotly figure builders for the interactive Streamlit dashboard.

Distinct from `experiments/plots.py` (Phase 8's static matplotlib research
figures, saved to disk for the written analysis) -- these are for live,
interactive use inside the dashboard. Every function takes already-computed
arrays/DataFrames and returns a `go.Figure`; nothing here runs a scheduler,
touches the simulator, or computes a metric from scratch (bar/line CI
whiskers reuse `experiments.stats.bootstrap_ci`, the same function Phase 8's
static plots use, so the two never define "confidence interval" differently).

Colour convention matches `experiments/plots.py` so a judge sees the same
strategy colour in a static report figure and in the live dashboard.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from rfscan.experiments.stats import bootstrap_ci

STRATEGY_ORDER = ("sequential", "random", "heuristic", "adaptive")
STRATEGY_COLORS = {
    "sequential": "#7f7f7f",
    "random": "#bcbd22",
    "heuristic": "#1f77b4",
    "adaptive": "#d62728",
}


def _ordered(values: Sequence, order: tuple) -> list:
    present = [v for v in order if v in values]
    rest = sorted(v for v in values if v not in order)
    return present + rest


def spectrum_bar(
    channel_labels: Sequence[str],
    *,
    predicted_proba: np.ndarray | None = None,
    priority: np.ndarray | None = None,
    selected_channel: int | None = None,
    recently_scanned: Sequence[int] = (),
    detected_channels: Sequence[int] = (),
) -> go.Figure:
    """One bar per channel. Bar height is ``priority`` if given, else
    ``predicted_proba``, else a flat 0 (so the chart still renders with only
    channel identity known). Colour marks state: selected (red), recently
    scanned (blue), detected on most recent scan (green outline), else grey.
    Never call this with simulator ground truth -- only observable/predicted
    quantities.
    """
    n = len(channel_labels)
    if priority is not None:
        heights = np.asarray(priority, dtype=float)
        ylabel = "priority"
    elif predicted_proba is not None:
        heights = np.asarray(predicted_proba, dtype=float)
        ylabel = "predicted P(active)"
    else:
        heights = np.zeros(n)
        ylabel = "(no prediction yet)"

    colors = []
    line_colors = []
    for c in range(n):
        if selected_channel is not None and c == selected_channel:
            colors.append(STRATEGY_COLORS["adaptive"])
        elif c in recently_scanned:
            colors.append(STRATEGY_COLORS["heuristic"])
        else:
            colors.append("#cccccc")
        line_colors.append("#2ca02c" if c in detected_channels else "rgba(0,0,0,0)")

    fig = go.Figure(
        go.Bar(
            x=list(channel_labels),
            y=heights,
            marker=dict(color=colors, line=dict(color=line_colors, width=3)),
        )
    )
    fig.update_layout(
        yaxis_title=ylabel,
        xaxis_title="channel",
        template="plotly_white",
        margin=dict(l=40, r=20, t=30, b=40),
        height=320,
    )
    return fig


def priority_breakdown_bar(breakdown: dict) -> go.Figure:
    """Stacked-term bar for the "why did we scan this channel" panel.
    ``breakdown`` is exactly a PriorityBreakdown.to_dict() (pred/explore/
    fresh/trend/redundancy/priority) or an AdaptiveScheduler.explain() dict
    -- no values are invented here."""
    terms = [k for k in ("pred", "explore", "fresh", "trend", "redundancy") if k in breakdown]
    values = [breakdown[t] for t in terms]
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in values]
    fig = go.Figure(go.Bar(x=terms, y=values, marker_color=colors))
    fig.update_layout(
        yaxis_title="contribution to priority",
        template="plotly_white",
        margin=dict(l=40, r=20, t=20, b=40),
        height=280,
    )
    return fig


def scan_raster(scanned_channel: np.ndarray, detected: np.ndarray, n_channels: int) -> go.Figure:
    """Time x channel raster: a dot per slot at the scanned channel, filled
    (detection) or hollow (no detection)."""
    slots = np.arange(len(scanned_channel))
    hit = detected.astype(bool)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=slots[~hit],
            y=scanned_channel[~hit],
            mode="markers",
            marker=dict(size=5, color="#cccccc"),
            name="scan (no detection)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=slots[hit],
            y=scanned_channel[hit],
            mode="markers",
            marker=dict(size=6, color="#2ca02c"),
            name="scan (detection)",
        )
    )
    fig.update_layout(
        xaxis_title="slot",
        yaxis_title="channel",
        yaxis=dict(range=[-0.5, n_channels - 0.5]),
        template="plotly_white",
        margin=dict(l=40, r=20, t=20, b=40),
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def strategy_comparison_bar(
    raw: pd.DataFrame, metric: str, *, group_col: str = "strategy", title: str | None = None
) -> go.Figure:
    """One bar per strategy, mean +/- 95% bootstrap CI (same statistic Phase
    8's static plots and paired tests use)."""
    groups = _ordered(raw[group_col].dropna().unique().tolist(), STRATEGY_ORDER)
    means, los, his = [], [], []
    for g in groups:
        values = raw.loc[raw[group_col] == g, metric].to_numpy(dtype=float)
        ci = bootstrap_ci(values, metric=metric, strategy=str(g))
        means.append(ci.point_estimate)
        los.append(0.0 if np.isnan(ci.lower) else ci.point_estimate - ci.lower)
        his.append(0.0 if np.isnan(ci.upper) else ci.upper - ci.point_estimate)

    colors = [STRATEGY_COLORS.get(g, "#888888") for g in groups]
    fig = go.Figure(
        go.Bar(
            x=groups,
            y=means,
            error_y=dict(type="data", symmetric=False, array=his, arrayminus=los),
            marker_color=colors,
        )
    )
    fig.update_layout(
        yaxis_title=metric,
        title=title or f"{metric} by {group_col} (mean, 95% CI)",
        template="plotly_white",
        margin=dict(l=40, r=20, t=40, b=40),
        height=380,
    )
    return fig


def grouped_bar_by_scenario(
    raw: pd.DataFrame,
    metric: str,
    *,
    strategy_col: str = "strategy",
    scenario_col: str = "scenario",
    title: str | None = None,
) -> go.Figure:
    scenarios = sorted(raw[scenario_col].dropna().unique().tolist())
    strategies = _ordered(raw[strategy_col].dropna().unique().tolist(), STRATEGY_ORDER)
    fig = go.Figure()
    for strat in strategies:
        means = []
        for sc in scenarios:
            values = raw.loc[
                (raw[scenario_col] == sc) & (raw[strategy_col] == strat), metric
            ].to_numpy(dtype=float)
            values = values[~np.isnan(values)]
            means.append(float(np.mean(values)) if values.size else None)
        color = STRATEGY_COLORS.get(strat, "#888")
        fig.add_trace(go.Bar(name=strat, x=scenarios, y=means, marker_color=color))
    fig.update_layout(
        barmode="group",
        yaxis_title=metric,
        title=title or f"{metric} by scenario x strategy",
        template="plotly_white",
        margin=dict(l=40, r=20, t=40, b=80),
        height=420,
        xaxis_tickangle=-30,
    )
    return fig


def line_by_level(
    raw: pd.DataFrame,
    metric: str,
    *,
    level_col: str,
    level_order: tuple[str, ...],
    strategy_col: str = "strategy",
    title: str | None = None,
) -> go.Figure:
    strategies = _ordered(raw[strategy_col].dropna().unique().tolist(), STRATEGY_ORDER)
    fig = go.Figure()
    for strat in strategies:
        means = []
        for level in level_order:
            values = raw.loc[
                (raw[level_col] == level) & (raw[strategy_col] == strat), metric
            ].to_numpy(dtype=float)
            values = values[~np.isnan(values)]
            means.append(float(np.mean(values)) if values.size else None)
        fig.add_trace(
            go.Scatter(
                x=list(level_order),
                y=means,
                mode="lines+markers",
                name=strat,
                line=dict(color=STRATEGY_COLORS.get(strat, "#888")),
            )
        )
    fig.update_layout(
        xaxis_title=level_col,
        yaxis_title=metric,
        title=title or f"{metric} vs {level_col}",
        template="plotly_white",
        margin=dict(l=40, r=20, t=40, b=40),
        height=380,
    )
    return fig


def emerging_timeline(
    slots: np.ndarray,
    priority: np.ndarray,
    belief_mean: np.ndarray,
    *,
    activation_slot: int | None = None,
    discovery_slot: int | None = None,
) -> go.Figure:
    """Two stacked panels: priority and belief mean over time for one
    tracked (emerging) channel, with activation/discovery marked."""
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, subplot_titles=("priority", "belief mean")
    )
    fig.add_trace(
        go.Scatter(x=slots, y=priority, mode="lines", name="priority"), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=slots, y=belief_mean, mode="lines", name="belief mean"), row=2, col=1
    )
    markers = ((activation_slot, "gray"), (discovery_slot, "green"))
    for slot, color in markers:
        if slot is not None:
            fig.add_vline(x=slot, line=dict(color=color, dash="dash"), row=1, col=1)
            fig.add_vline(x=slot, line=dict(color=color, dash="dash"), row=2, col=1)
    fig.update_xaxes(title_text="slot", row=2, col=1)
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=20, t=40, b=40),
        height=480,
        showlegend=False,
    )
    return fig
