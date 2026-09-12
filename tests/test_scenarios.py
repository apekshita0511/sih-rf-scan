"""The seven benchmark scenarios and their registry (Phase 4)."""

from __future__ import annotations

import numpy as np
import pytest

from rfscan.simulator.emitters import Behavior
from rfscan.simulator.scenarios import (
    DURATION_SLOTS,
    N_CHANNELS,
    SCENARIOS,
    list_scenarios,
    make_scenario,
)

_EXPECTED = {
    "normal",
    "high_activity",
    "bursty",
    "emerging_signal",
    "high_noise",
    "dynamic",
    "changing_distribution",
}


def _occupancy_matrix(name: str, seed: int, n_slots: int) -> np.ndarray:
    env = make_scenario(name, seed, duration_slots=n_slots).build_environment(seed)
    rows = np.empty((n_slots, env.n_channels), dtype=bool)
    for t in range(n_slots):
        rows[t] = env.occupancy_snapshot()
        env.step()
    return rows


def test_registry_has_the_seven_named_scenarios():
    assert set(list_scenarios()) == _EXPECTED
    assert set(SCENARIOS) == _EXPECTED
    assert list_scenarios() == list(SCENARIOS)  # canonical order preserved


def test_unknown_scenario_is_rejected():
    with pytest.raises(KeyError):
        make_scenario("not_a_scenario")


def test_make_scenario_is_deterministic():
    a = make_scenario("dynamic", 3)
    b = make_scenario("dynamic", 3)
    assert a == b


def test_duration_override():
    sc = make_scenario("normal", 0, duration_slots=200)
    assert sc.duration_slots == 200
    assert make_scenario("normal", 0).duration_slots == DURATION_SLOTS


@pytest.mark.parametrize("name", sorted(_EXPECTED))
def test_every_scenario_builds_and_runs(name):
    env = make_scenario(name, 0, duration_slots=100).build_environment(0)
    assert env.n_channels == N_CHANNELS
    for _ in range(100):
        env.observe(0)
        env.step()


@pytest.mark.parametrize("name", sorted(_EXPECTED))
def test_every_scenario_is_reproducible(name):
    a = _occupancy_matrix(name, 5, 150)
    b = _occupancy_matrix(name, 5, 150)
    assert np.array_equal(a, b)


def test_different_seeds_give_different_worlds():
    a = _occupancy_matrix("normal", 1, 300)
    b = _occupancy_matrix("normal", 2, 300)
    assert not np.array_equal(a, b)


def test_high_activity_is_busier_than_normal():
    normal = _occupancy_matrix("normal", 0, 500).mean()
    high = _occupancy_matrix("high_activity", 0, 500).mean()
    assert high > normal
    assert high > 0.25


def test_bursty_activity_is_short_and_frequent():
    occ = _occupancy_matrix("bursty", 0, 800)
    # each active channel: many short runs
    col = occ[:, 1]
    rising = np.sum(col[1:] & ~col[:-1])
    assert rising > 15
    assert col.mean() < 0.25


def test_emerging_signal_channel_is_silent_until_activation():
    occ = _occupancy_matrix("emerging_signal", 0, 500)
    assert not occ[:300, 5].any()
    assert occ[300:, 5].any()


def test_changing_distribution_shifts_band_at_slot_400():
    occ = _occupancy_matrix("changing_distribution", 0, 800)
    low_first = occ[:400, [0, 1, 2]].mean()
    low_second = occ[400:, [0, 1, 2]].mean()
    high_first = occ[:400, [9, 10, 11]].mean()
    high_second = occ[400:, [9, 10, 11]].mean()
    assert low_first > 0.3 and low_second < 0.1
    assert high_first < 0.1 and high_second > 0.3


def test_dynamic_scenario_applies_its_events():
    occ = _occupancy_matrix("dynamic", 0, 800)
    # CH1 is quietened by the event at slot 200, then revived at slot 550
    assert occ[220:540, 1].mean() < occ[:180, 1].mean()
    assert occ[560:, 1].mean() > occ[300:540, 1].mean()


def test_high_noise_scenario_carries_the_noisy_config():
    sc = make_scenario("high_noise", 0)
    assert sc.noise.awgn_std_db > 3.0
    assert sc.noise.missed_detection_rate > 0.1


def test_emerging_emitter_metadata_present():
    sc = make_scenario("emerging_signal", 0)
    emerging = [e for e in sc.emitters if e.behavior is Behavior.EMERGING]
    assert len(emerging) == 1
    assert emerging[0].channel_index == 5
    assert emerging[0].activation_slot == 300
