"""experiments/plots.py: static analysis figures (Phase 8). Synthetic data
only -- these tests check the plotting functions run and produce a sane
figure, not that any experiment's real numbers look a certain way."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rfscan.experiments.adaptation import PriorityTrace
from rfscan.experiments.plots import (
    bar_with_ci,
    grouped_bar_by_scenario,
    line_by_level,
    priority_trace_plot,
    save_figure,
)


def _raw(n_seeds=10):
    rng = np.random.default_rng(0)
    rows = []
    for scenario in ("normal", "bursty"):
        for strategy in ("sequential", "random", "adaptive"):
            for seed in range(n_seeds):
                rows.append(
                    {
                        "scenario": scenario,
                        "strategy": strategy,
                        "world_seed": seed,
                        "detection_rate": float(rng.uniform(0.2, 0.8)),
                    }
                )
    return pd.DataFrame(rows)


def test_bar_with_ci_returns_a_figure_with_one_bar_per_group():
    fig = bar_with_ci(_raw(), "detection_rate")
    ax = fig.axes[0]
    assert len(ax.patches) == 3  # sequential, random, adaptive
    plt.close(fig)


def test_bar_with_ci_orders_known_strategies_first():
    raw = _raw()
    fig = bar_with_ci(raw, "detection_rate")
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == ["sequential", "random", "adaptive"]
    plt.close(fig)


def test_grouped_bar_by_scenario_has_one_cluster_per_scenario():
    fig = grouped_bar_by_scenario(_raw(), "detection_rate")
    ax = fig.axes[0]
    # 2 scenarios x 3 strategies = 6 bars
    assert len(ax.patches) == 6
    plt.close(fig)


def test_line_by_level_plots_one_line_per_strategy():
    rng = np.random.default_rng(1)
    rows = []
    for tier in ("low", "elevated", "high"):
        for strategy in ("sequential", "adaptive"):
            for _seed in range(5):
                rows.append(
                    {
                        "noise_tier": tier,
                        "strategy": strategy,
                        "detection_rate": float(rng.uniform(0, 1)),
                    }
                )
    raw = pd.DataFrame(rows)
    fig = line_by_level(
        raw, "detection_rate", level_col="noise_tier", level_order=("low", "elevated", "high")
    )
    ax = fig.axes[0]
    assert len(ax.lines) == 2
    plt.close(fig)


def test_priority_trace_plot_draws_one_line_per_tracked_channel():
    trace = PriorityTrace(
        channels=(2, 5),
        priority=np.random.default_rng(0).uniform(size=(50, 2)),
        belief_mean=np.random.default_rng(1).uniform(size=(50, 2)),
        belief_std=np.random.default_rng(2).uniform(size=(50, 2)),
    )
    fig = priority_trace_plot(trace, activation_slot=10, discovery_slot=15)
    ax1, ax2 = fig.axes
    # 2 channel lines + 2 vlines (activation, discovery) on each axis
    assert len(ax1.lines) == 4
    assert len(ax2.lines) == 4
    plt.close(fig)


def test_save_figure_writes_a_file_and_closes_the_figure(tmp_path):
    fig = bar_with_ci(_raw(), "detection_rate")
    out_path = tmp_path / "sub" / "plot.png"
    result_path = save_figure(fig, out_path)
    assert result_path == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0
    assert not plt.fignum_exists(fig.number)


def test_bar_with_ci_handles_all_nan_metric_gracefully():
    raw = _raw(n_seeds=3)
    raw["detection_rate"] = np.nan
    fig = bar_with_ci(raw, "detection_rate")
    assert fig is not None
    plt.close(fig)
