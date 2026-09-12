"""ExperimentRunner: episode lifecycle, determinism, no ground-truth leakage (Phase 4)."""

from __future__ import annotations

import numpy as np
import pytest

from rfscan.experiments.runner import run_episode
from rfscan.scheduler.base import Scheduler
from rfscan.scheduler.sequential import RandomScheduler, SequentialScheduler
from rfscan.simulator.scenarios import make_scenario


def _env(name="normal", seed=0, duration=150):
    return make_scenario(name, seed, duration_slots=duration).build_environment(seed)


def test_episode_result_shapes_and_ranges():
    result = run_episode(_env(), SequentialScheduler(12), budget=150)
    assert result.n_slots == 150
    assert result.scanned_channel.shape == (150,)
    assert result.observed_detection.shape == (150,)
    assert result.occupancy.shape == (150, 12)
    assert result.decision_latencies_s.shape == (150,)
    assert result.scanned_channel.min() >= 0 and result.scanned_channel.max() < 12
    assert result.scenario == "normal"
    assert result.strategy == "sequential"


def test_sequential_scan_pattern_is_round_robin():
    result = run_episode(_env(), SequentialScheduler(12), budget=150)
    assert list(result.scanned_channel[:13]) == list(range(12)) + [0]


def test_runner_is_repeatable_with_the_same_instances():
    env = _env("dynamic", 7)
    sched = RandomScheduler(12, seed=7)
    a = run_episode(env, sched, budget=200)
    b = run_episode(env, sched, budget=200)  # runner resets env + scheduler
    assert np.array_equal(a.scanned_channel, b.scanned_channel)
    assert np.array_equal(a.occupancy, b.occupancy)
    assert np.array_equal(a.observed_detection, b.observed_detection)


def test_budget_must_be_positive():
    with pytest.raises(ValueError):
        run_episode(_env(), SequentialScheduler(12), budget=0)


def test_out_of_range_channel_choice_is_rejected():
    class BadScheduler:
        name = "bad"

        def select_next(self, store, slot):
            return 999

        def update(self, record):
            pass

        def explain(self):
            return {}

        def reset(self):
            pass

    with pytest.raises(ValueError):
        run_episode(_env(), BadScheduler(), budget=10)


def test_scheduler_receives_only_a_readable_store_never_ground_truth():
    seen = {}

    class SpyScheduler:
        name = "spy"

        def select_next(self, store, slot):
            seen["store"] = store
            return 0

        def update(self, record):
            pass

        def explain(self):
            return {}

        def reset(self):
            pass

    run_episode(_env(), SpyScheduler(), budget=5)
    store = seen["store"]
    for forbidden in ("ground_truth", "truth_snapshot", "occupancy_snapshot", "observe", "step"):
        assert not hasattr(store, forbidden)
    assert hasattr(store, "detection_rate")  # it really is the observation store


def test_emerging_metadata_is_extracted():
    result = run_episode(
        make_scenario("emerging_signal", 0).build_environment(0),
        SequentialScheduler(12),
        budget=350,
    )
    assert result.emerging_channels == (5,)
    assert result.emerging_activation_slots == (300,)


def test_agent_seed_recorded_and_world_unchanged():
    env_a = _env("normal", 3)
    env_b = _env("normal", 3)
    a = run_episode(env_a, RandomScheduler(12, seed=1), budget=200, agent_seed=1)
    b = run_episode(env_b, RandomScheduler(12, seed=2), budget=200, agent_seed=2)
    assert a.agent_seed == 1 and b.agent_seed == 2
    assert not np.array_equal(a.scanned_channel, b.scanned_channel)
    assert np.array_equal(a.occupancy, b.occupancy)  # same world seed -> same world


def test_baseline_schedulers_satisfy_the_protocol():
    assert isinstance(SequentialScheduler(4), Scheduler)
    assert isinstance(RandomScheduler(4), Scheduler)
