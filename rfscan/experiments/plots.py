"""Phase 8: static research/engineering plots (matplotlib).

These are for the written analysis (model cards, architecture notes, a
report) -- distinct from `visualization/plots.py` (Phase 9's *interactive*
Plotly dashboard widgets, untouched here). Every function takes an
already-computed DataFrame/trace and returns a ``Figure``; nothing here runs
an experiment or touches the simulator. Plain bars/lines with error bars --
no decorative styling, per this phase's "should look like research/
engineering evaluation, not fake military graphics" instruction.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this module never opens a GUI window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rfscan.experiments.stats import bootstrap_ci

STRATEGY_ORDER = ("sequential", "random", "heuristic", "adaptive")
_STRATEGY_COLORS = {
    "sequential": "#7f7f7f",
    "random": "#bcbd22",
    "heuristic": "#1f77b4",
    "adaptive": "#d62728",
}


def _ordered(values: list, order: tuple) -> list:
    present = [v for v in order if v in values]
    rest = sorted(v for v in values if v not in order)
    return present + rest


def _clean_top_right(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def bar_with_ci(
    raw: pd.DataFrame,
    metric: str,
    *,
    group_col: str = "strategy",
    title: str | None = None,
    ylabel: str | None = None,
) -> plt.Figure:
    """One bar per distinct value of ``group_col``: mean +/- 95% bootstrap CI
    over ``metric`` (NaN/censored episodes dropped by ``bootstrap_ci``)."""
    groups = _ordered(raw[group_col].dropna().unique().tolist(), STRATEGY_ORDER)
    means, lo_err, hi_err = [], [], []
    for g in groups:
        values = raw.loc[raw[group_col] == g, metric].to_numpy(dtype=float)
        ci = bootstrap_ci(values, metric=metric, strategy=str(g))
        means.append(ci.point_estimate)
        lo_err.append(0.0 if np.isnan(ci.lower) else ci.point_estimate - ci.lower)
        hi_err.append(0.0 if np.isnan(ci.upper) else ci.upper - ci.point_estimate)

    fig, ax = plt.subplots(figsize=(0.9 * len(groups) + 2, 4))
    colors = [_STRATEGY_COLORS.get(g, "#888888") for g in groups]
    ax.bar(range(len(groups)), means, yerr=[lo_err, hi_err], capsize=4, color=colors)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=20, ha="right")
    ax.set_ylabel(ylabel or metric)
    ax.set_title(title or f"{metric} by {group_col} (mean, 95% bootstrap CI)")
    _clean_top_right(ax)
    fig.tight_layout()
    return fig


def grouped_bar_by_scenario(
    raw: pd.DataFrame,
    metric: str,
    *,
    strategy_col: str = "strategy",
    scenario_col: str = "scenario",
    title: str | None = None,
    ylabel: str | None = None,
) -> plt.Figure:
    """Grouped bars: one cluster per scenario, one bar per strategy within it."""
    scenarios = sorted(raw[scenario_col].dropna().unique().tolist())
    strategies = _ordered(raw[strategy_col].dropna().unique().tolist(), STRATEGY_ORDER)
    x = np.arange(len(scenarios))
    width = 0.8 / max(len(strategies), 1)

    fig, ax = plt.subplots(figsize=(1.3 * len(scenarios) + 2, 5))
    for i, strat in enumerate(strategies):
        means = []
        for sc in scenarios:
            values = raw.loc[
                (raw[scenario_col] == sc) & (raw[strategy_col] == strat), metric
            ].to_numpy(dtype=float)
            values = values[~np.isnan(values)]
            means.append(float(np.mean(values)) if values.size else np.nan)
        offset = (i - (len(strategies) - 1) / 2) * width
        ax.bar(x + offset, means, width, label=strat, color=_STRATEGY_COLORS.get(strat, "#888888"))
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=30, ha="right")
    ax.set_ylabel(ylabel or metric)
    ax.set_title(title or f"{metric} by scenario x strategy")
    ax.legend(fontsize=8)
    _clean_top_right(ax)
    fig.tight_layout()
    return fig


def line_by_level(
    raw: pd.DataFrame,
    metric: str,
    *,
    level_col: str,
    level_order: tuple[str, ...],
    strategy_col: str = "strategy",
    title: str | None = None,
    ylabel: str | None = None,
    xlabel: str | None = None,
) -> plt.Figure:
    """One line per strategy across an ordered level axis (noise tier,
    sparse/moderate/dense scenario label, ...)."""
    strategies = _ordered(raw[strategy_col].dropna().unique().tolist(), STRATEGY_ORDER)
    fig, ax = plt.subplots(figsize=(6, 4))
    for strat in strategies:
        means = []
        for level in level_order:
            values = raw.loc[
                (raw[level_col] == level) & (raw[strategy_col] == strat), metric
            ].to_numpy(dtype=float)
            values = values[~np.isnan(values)]
            means.append(float(np.mean(values)) if values.size else np.nan)
        color = _STRATEGY_COLORS.get(strat, "#888888")
        ax.plot(level_order, means, marker="o", label=strat, color=color)
    ax.set_xlabel(xlabel or level_col)
    ax.set_ylabel(ylabel or metric)
    ax.set_title(title or f"{metric} vs {level_col}")
    ax.legend(fontsize=8)
    _clean_top_right(ax)
    fig.tight_layout()
    return fig


def priority_trace_plot(
    trace,
    *,
    activation_slot: int | None = None,
    discovery_slot: int | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Priority and belief-mean trajectories for a
    :class:`~rfscan.experiments.adaptation.PriorityTrace`'s tracked channels,
    with the (optional) activation/discovery slots marked -- the "before/
    after a hit" story as a time series rather than two numbers."""
    slots = np.arange(trace.priority.shape[0])
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    for i, c in enumerate(trace.channels):
        ax1.plot(slots, trace.priority[:, i], label=f"channel {c}")
        ax2.plot(slots, trace.belief_mean[:, i], label=f"channel {c}")
    for slot, color, label in (
        (activation_slot, "gray", "activation"),
        (discovery_slot, "green", "first hit"),
    ):
        if slot is not None:
            ax1.axvline(slot, color=color, linestyle="--", linewidth=1, label=label)
            ax2.axvline(slot, color=color, linestyle="--", linewidth=1)
    ax1.set_ylabel("priority")
    ax2.set_ylabel("belief mean")
    ax2.set_xlabel("slot")
    ax1.legend(fontsize=8)
    ax1.set_title(title or "Priority / belief trace")
    _clean_top_right(ax1)
    _clean_top_right(ax2)
    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure, path: str | Path) -> Path:
    """Write ``fig`` to ``path`` (parent dirs created) and close it -- callers
    generating many figures in a loop should always go through this rather
    than holding figures open, to avoid unbounded matplotlib memory growth."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
