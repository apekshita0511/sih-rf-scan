"""Benchmark grid: schema, reproducibility, fairness, CSV output (Phase 4)."""

from __future__ import annotations

import numpy as np
import pytest

from rfscan.experiments.benchmark import (
    BenchmarkConfig,
    build_scheduler,
    run_benchmark,
    summarize,
    write_benchmark,
)
from rfscan.experiments.runner import run_episode
from rfscan.scheduler.heuristic import HeuristicScheduler
from rfscan.scheduler.sequential import RandomScheduler, SequentialScheduler
from rfscan.simulator.scenarios import make_scenario

_SMALL = BenchmarkConfig(
    scenarios=("normal", "bursty"),
    world_seeds=(0, 1),
    strategies=("sequential", "random", "heuristic"),
    duration_slots=120,
)


def test_build_scheduler_returns_the_right_types():
    assert isinstance(build_scheduler("sequential", 12, agent_seed=0), SequentialScheduler)
    assert isinstance(build_scheduler("random", 12, agent_seed=0), RandomScheduler)
    assert isinstance(build_scheduler("heuristic", 12, agent_seed=0), HeuristicScheduler)
    with pytest.raises(ValueError):
        build_scheduler("nope", 12, agent_seed=0)


def test_config_rejects_unknown_scenario_or_strategy():
    with pytest.raises(ValueError):
        BenchmarkConfig(scenarios=("nope",))
    with pytest.raises(ValueError):
        BenchmarkConfig(strategies=("magic",))
    with pytest.raises(ValueError):
        BenchmarkConfig(world_seeds=())


def test_run_benchmark_produces_one_row_per_episode():
    raw = run_benchmark(_SMALL)
    assert len(raw) == 2 * 2 * 3
    for col in (
        "scenario",
        "strategy",
        "world_seed",
        "detection_rate",
        "mean_detection_delay_slots",
        "scan_efficiency",
        "redundant_scan_rate",
        "channel_coverage",
        "coverage_time_slots",
    ):
        assert col in raw.columns
    assert set(raw["scenario"]) == {"normal", "bursty"}
    assert set(raw["strategy"]) == {"sequential", "random", "heuristic"}


def test_benchmark_is_reproducible_except_for_timing():
    a = run_benchmark(_SMALL).drop(columns=["mean_decision_latency_ms"])
    b = run_benchmark(_SMALL).drop(columns=["mean_decision_latency_ms"])
    assert a.equals(b)


def test_all_strategies_face_an_identical_world():
    scenario = make_scenario("dynamic", 4, duration_slots=200)
    occupancies = []
    for strat in ("sequential", "random", "heuristic"):
        env = scenario.build_environment(4)
        sched = build_scheduler(strat, env.n_channels, agent_seed=4)
        occupancies.append(run_episode(env, sched, budget=200).occupancy)
    assert np.array_equal(occupancies[0], occupancies[1])
    assert np.array_equal(occupancies[0], occupancies[2])


def test_agent_seed_changes_pattern_not_world():
    scenario = make_scenario("normal", 9, duration_slots=200)
    a = run_episode(scenario.build_environment(9), RandomScheduler(12, seed=1), budget=200)
    b = run_episode(scenario.build_environment(9), RandomScheduler(12, seed=2), budget=200)
    assert not np.array_equal(a.scanned_channel, b.scanned_channel)
    assert np.array_equal(a.occupancy, b.occupancy)


def test_summary_schema():
    summary = summarize(run_benchmark(_SMALL))
    assert {"scenario", "strategy"} <= set(summary.columns)
    assert "detection_rate_mean" in summary.columns
    assert "detection_rate_std" in summary.columns
    assert len(summary) == 2 * 3  # scenarios x strategies


def test_write_benchmark_creates_both_csvs(tmp_path):
    config = BenchmarkConfig(
        scenarios=("normal",),
        world_seeds=(0, 1),
        duration_slots=120,
        results_dir=str(tmp_path / "results"),
    )
    raw = run_benchmark(config)
    raw_path, summary_path = write_benchmark(config, raw)
    assert raw_path.exists() and summary_path.exists()
    import pandas as pd

    reloaded = pd.read_csv(raw_path)
    assert len(reloaded) == len(raw) == 2 * 3


def test_full_seven_by_two_grid_executes_cleanly():
    config = BenchmarkConfig(world_seeds=(0, 1), duration_slots=150)
    raw = run_benchmark(config)
    assert len(raw) == 7 * 2 * 3
    rates = raw["detection_rate"].dropna()
    assert ((rates >= 0.0) & (rates <= 1.0)).all()
    # heuristic should never be worse than sequential at landing scans on activity,
    # averaged across scenarios/seeds
    by_strat = raw.groupby("strategy")["on_target_scan_rate"].mean()
    assert by_strat["heuristic"] >= by_strat["sequential"]
