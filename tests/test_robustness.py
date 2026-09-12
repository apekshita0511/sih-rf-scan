"""experiments/robustness.py: noise sweep + non-stationarity experiments (Phase 8)."""

from __future__ import annotations

import numpy as np
import pytest

from rfscan.experiments.robustness import (
    CHANGING_DISTRIBUTION_FLIP_SLOT,
    CHANGING_DISTRIBUTION_HOT_THEN_QUIET,
    channel_group_split,
    noise_tier_scenario,
    run_noise_sweep,
    run_non_stationarity_experiment,
)
from rfscan.experiments.runner import run_episode
from rfscan.scheduler.sequential import SequentialScheduler
from rfscan.simulator.scenarios import make_scenario


# -- noise sweep ---------------------------------------------------------
def test_noise_tier_scenario_holds_emitters_constant_varies_only_noise():
    low = noise_tier_scenario("low", 0, duration_slots=100)
    high = noise_tier_scenario("high", 0, duration_slots=100)
    assert low.emitters == high.emitters  # same signal_dbm etc. -- only noise differs
    assert low.noise != high.noise


def test_noise_tier_high_matches_the_high_noise_scenarios_own_config():
    high = noise_tier_scenario("high", 0, duration_slots=100)
    from rfscan.simulator.scenarios import make_scenario as _make

    high_noise_scenario = _make("high_noise", 0, duration_slots=100)
    assert high.noise == high_noise_scenario.noise


def test_noise_tier_low_matches_default_noise_config():
    low = noise_tier_scenario("low", 0, duration_slots=100)
    from rfscan.simulator.noise import NoiseModelConfig

    assert low.noise == NoiseModelConfig()


def test_elevated_tier_is_strictly_between_low_and_high():
    low = noise_tier_scenario("low", 0, duration_slots=100)
    mid = noise_tier_scenario("elevated", 0, duration_slots=100)
    high = noise_tier_scenario("high", 0, duration_slots=100)
    assert low.noise.awgn_std_db < mid.noise.awgn_std_db < high.noise.awgn_std_db


def test_unknown_tier_raises():
    with pytest.raises(ValueError):
        noise_tier_scenario("extreme", 0)


def test_noise_sweep_produces_the_expected_grid_shape():
    raw = run_noise_sweep(
        strategies=("sequential", "random"),
        tiers=("low", "high"),
        world_seeds=(0, 1),
        duration_slots=100,
    )
    assert len(raw) == 2 * 2 * 2
    assert set(raw["noise_tier"]) == {"low", "high"}
    assert set(raw["strategy"]) == {"sequential", "random"}


def test_noise_sweep_all_strategies_face_the_same_world_per_tier_and_seed():
    scenario = noise_tier_scenario("high", 2, duration_slots=120)
    env_a = scenario.build_environment(2)
    env_b = scenario.build_environment(2)
    from rfscan.scheduler.sequential import RandomScheduler

    result_a = run_episode(env_a, SequentialScheduler(env_a.n_channels), budget=120)
    result_b = run_episode(env_b, RandomScheduler(env_b.n_channels, seed=2), budget=120)
    assert np.array_equal(result_a.occupancy, result_b.occupancy)


# -- non-stationarity ------------------------------------------------------
def test_channel_group_split_sums_across_channels():
    scenario = make_scenario("changing_distribution", 0, duration_slots=500)
    env = scenario.build_environment(0)
    result = run_episode(env, SequentialScheduler(env.n_channels), budget=500)
    group = channel_group_split(
        result, CHANGING_DISTRIBUTION_HOT_THEN_QUIET, CHANGING_DISTRIBUTION_FLIP_SLOT, "test"
    )
    manual_scans_before = sum(
        int(np.count_nonzero(result.scanned_channel[:400] == c))
        for c in CHANGING_DISTRIBUTION_HOT_THEN_QUIET
    )
    assert group.scans_before == manual_scans_before


def test_non_stationarity_experiment_produces_expected_columns_and_rows():
    raw = run_non_stationarity_experiment(
        strategies=("sequential", "random"), world_seeds=(0, 1), duration_slots=500
    )
    assert len(raw) == 2 * 2
    for col in (
        "hot_then_quiet_rate_before",
        "hot_then_quiet_rate_after",
        "quiet_then_hot_rate_before",
        "quiet_then_hot_rate_after",
    ):
        assert col in raw.columns


def test_sequential_detection_rate_on_hot_then_quiet_group_drops_after_flip():
    """Sanity check on the ground-truth-adjacent scenario mechanics: with
    sequential (uniform, unbiased) coverage, the hot-then-quiet group's own
    *true* activity really does drop after the flip, so its detection rate
    (bounded by activity, not by any scheduler cleverness) should fall."""
    raw = run_non_stationarity_experiment(strategies=("sequential",), world_seeds=(0, 1, 2))
    for _, row in raw.iterrows():
        before, after = row["hot_then_quiet_rate_before"], row["hot_then_quiet_rate_after"]
        if before is not None and after is not None:
            assert after <= before + 0.05
