"""visualization/plots.py: Plotly figure builders for the dashboard (Phase 9).
Synthetic data only -- checks the functions run and produce sane figures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from rfscan.visualization.plots import (
    emerging_timeline,
    grouped_bar_by_scenario,
    line_by_level,
    priority_breakdown_bar,
    scan_raster,
    spectrum_bar,
    strategy_comparison_bar,
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


def test_spectrum_bar_with_no_prediction_yet_still_renders():
    fig = spectrum_bar(["CH0", "CH1", "CH2"])
    assert isinstance(fig, go.Figure)
    assert len(fig.data[0].y) == 3


def test_spectrum_bar_highlights_selected_and_scanned_channels():
    fig = spectrum_bar(
        ["CH0", "CH1", "CH2"],
        predicted_proba=np.array([0.1, 0.9, 0.3]),
        selected_channel=1,
        recently_scanned=[0],
        detected_channels=[1],
    )
    colors = list(fig.data[0].marker.color)
    assert colors[1] != colors[0] != colors[2] or colors[1] != "#cccccc"


def test_priority_breakdown_bar_uses_only_given_terms():
    breakdown = {"pred": 0.4, "explore": 0.1, "fresh": 0.05, "trend": 0.0, "redundancy": -0.02}
    fig = priority_breakdown_bar(breakdown)
    assert list(fig.data[0].x) == ["pred", "explore", "fresh", "trend", "redundancy"]
    assert list(fig.data[0].y) == [0.4, 0.1, 0.05, 0.0, -0.02]


def test_priority_breakdown_bar_handles_partial_dict():
    fig = priority_breakdown_bar({"predicted_proba": 0.5})
    assert len(fig.data[0].x) == 0


def test_scan_raster_splits_hits_and_misses():
    scanned = np.array([0, 1, 2, 0, 1])
    detected = np.array([True, False, True, False, False])
    fig = scan_raster(scanned, detected, n_channels=3)
    assert len(fig.data) == 2
    assert len(fig.data[1].x) == 2  # 2 detections
    assert len(fig.data[0].x) == 3  # 3 non-detections


def test_strategy_comparison_bar_orders_known_strategies_first():
    fig = strategy_comparison_bar(_raw(), "detection_rate")
    assert list(fig.data[0].x) == ["sequential", "random", "adaptive"]


def test_grouped_bar_by_scenario_has_one_trace_per_strategy():
    fig = grouped_bar_by_scenario(_raw(), "detection_rate")
    assert len(fig.data) == 3  # sequential, random, adaptive


def test_line_by_level_plots_one_trace_per_strategy():
    rng = np.random.default_rng(1)
    rows = [
        {"noise_tier": tier, "strategy": strat, "detection_rate": float(rng.uniform(0, 1))}
        for tier in ("low", "elevated", "high")
        for strat in ("sequential", "adaptive")
        for _ in range(5)
    ]
    raw = pd.DataFrame(rows)
    fig = line_by_level(
        raw, "detection_rate", level_col="noise_tier", level_order=("low", "elevated", "high")
    )
    assert len(fig.data) == 2


def test_emerging_timeline_marks_activation_and_discovery():
    slots = np.arange(50)
    priority = np.random.default_rng(0).uniform(size=50)
    belief = np.random.default_rng(1).uniform(size=50)
    fig = emerging_timeline(slots, priority, belief, activation_slot=10, discovery_slot=18)
    assert isinstance(fig, go.Figure)
    assert len(fig.layout.shapes) == 4  # 2 vlines x 2 subplots
