"""RFEnvironment: lifecycle, fairness/reproducibility, measurement consistency,
non-stationary events, and ground-truth isolation (Phase 2)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from rfscan.perception.schema import Observation
from rfscan.simulator.emitters import Behavior, EmitterSpec
from rfscan.simulator.environment import RFEnvironment, build_environment
from rfscan.simulator.scenario import (
    NonStationaryEvent,
    ScenarioConfig,
    example_emerging,
    example_normal,
)


def _truth_matrix(env: RFEnvironment, n_slots: int, scan) -> np.ndarray:
    """Roll the env forward, calling ``scan(env)`` each slot; return an
    (n_slots x n_channels) bool array of true occupancy."""
    out = np.empty((n_slots, env.n_channels), dtype=bool)
    for t in range(n_slots):
        scan(env)
        out[t] = [env.ground_truth(c).occupied for c in range(env.n_channels)]
        env.step()
    return out


# -- lifecycle --------------------------------------------------------
def test_reset_step_observe_lifecycle():
    env = example_normal().build_environment(0)
    assert env.slot == 0
    assert isinstance(env.observe(0), Observation)
    env.step()
    env.observe(3)
    env.step()
    assert env.slot == 2
    env.reset(0)
    assert env.slot == 0


def test_observe_rejects_out_of_range_channel():
    env = example_normal().build_environment(0)
    with pytest.raises(IndexError):
        env.observe(99)


# -- reproducibility & fairness -------------------------------------
def test_same_seed_gives_identical_streams():
    a = RFEnvironment(example_normal(), seed=7)
    b = RFEnvironment(example_normal(), seed=7)
    for _ in range(250):
        for c in range(a.n_channels):
            assert a.observe(c) == b.observe(c)
            assert a.ground_truth(c) == b.ground_truth(c)
        a.step()
        b.step()


def test_observation_order_does_not_perturb_the_world():
    fwd = RFEnvironment(example_emerging(), seed=3)
    rev = RFEnvironment(example_emerging(), seed=3)
    n = fwd.n_channels
    tf = _truth_matrix(fwd, 400, lambda e: [e.observe(c) for c in range(n)])
    tr = _truth_matrix(rev, 400, lambda e: [e.observe(c) for c in reversed(range(n))])
    assert np.array_equal(tf, tr)


def test_partial_observation_does_not_perturb_the_world():
    only_first = RFEnvironment(example_emerging(), seed=11)
    only_last = RFEnvironment(example_emerging(), seed=11)
    n = only_first.n_channels
    for _ in range(400):
        only_first.observe(0)
        only_last.observe(n - 1)
        for c in range(n):
            assert only_first.ground_truth(c) == only_last.ground_truth(c)
        only_first.step()
        only_last.step()


def test_different_seed_diverges():
    a = _truth_matrix(RFEnvironment(example_normal(), 1), 300, lambda e: None)
    b = _truth_matrix(RFEnvironment(example_normal(), 2), 300, lambda e: None)
    # persistent emitter on channel 0: trajectories differ on a real fraction of slots
    assert (a[:, 0] != b[:, 0]).mean() > 0.1


def test_build_environment_function_matches_method():
    sc = example_normal()
    a = build_environment(sc, 5)
    b = sc.build_environment(5)
    for _ in range(50):
        for c in range(sc.n_channels):
            assert a.observe(c) == b.observe(c)
        a.step()
        b.step()


# -- occupancy statistics -----------------------------------------
def test_occupancy_statistics_match_configured_rates():
    env = RFEnvironment(example_normal(), seed=0)  # ch0: PERSISTENT p01=0.05 p10=0.02
    truth = _truth_matrix(env, 8_000, lambda e: None)
    assert truth[:, 0].mean() == pytest.approx(0.05 / 0.07, abs=0.08)
    assert not truth[:, 1].any()  # ch1 has no emitter -> never occupied


# -- measurement model -------------------------------------------
def test_snr_is_exactly_rssi_minus_noise():
    env = RFEnvironment(example_normal(), seed=0)
    for _ in range(200):
        for c in range(env.n_channels):
            o = env.observe(c)
            assert o.snr_db == pytest.approx(o.rssi_dbm - o.noise_dbm)
        env.step()


def test_noise_floor_reading_stays_bounded_around_the_configured_floor():
    env = RFEnvironment(example_normal(), seed=0)
    cfg = env.scenario.noise
    readings = []
    for _ in range(3_000):
        readings.extend(env.observe(c).noise_dbm for c in range(env.n_channels))
        env.step()
    arr = np.array(readings)
    assert abs(arr.mean() - cfg.floor_dbm) < 1.0
    assert np.abs(arr - cfg.floor_dbm).max() < 6 * cfg.floor_drift_std_db + 6 * cfg.awgn_std_db


def test_occupied_channel_reads_hotter_than_empty_channel():
    env = RFEnvironment(example_normal(), seed=0)
    hot, cold = [], []
    for _ in range(1_500):
        if env.ground_truth(0).occupied:  # strong persistent emitter, signal -55 dBm
            hot.append(env.observe(0).snr_db)
        cold.append(env.observe(1).snr_db)  # no emitter
        env.step()
    assert np.mean(hot) > 20
    assert abs(np.mean(cold)) < 5


def test_detector_false_positive_and_negative_rates_are_sane():
    env = RFEnvironment(example_normal(), seed=0)
    occ_hits = occ_n = emp_hits = emp_n = 0
    for _ in range(3_000):
        for c in range(env.n_channels):
            occupied = env.ground_truth(c).occupied
            detected = env.observe(c).observed_detection
            if occupied:
                occ_n += 1
                occ_hits += detected
            else:
                emp_n += 1
                emp_hits += detected
        env.step()
    assert occ_hits / occ_n > 0.85
    assert emp_hits / emp_n < 0.10


# -- ground-truth isolation --------------------------------------
def test_observation_exposes_no_ground_truth():
    env = RFEnvironment(example_normal(), seed=0)
    o = env.observe(0)
    names = {f.name for f in dataclasses.fields(o)}
    assert names == {
        "slot",
        "channel_index",
        "rssi_dbm",
        "noise_dbm",
        "snr_db",
        "observed_detection",
    }
    for banned in ("occupied", "true_signal_dbm", "truth", "is_active"):
        assert not hasattr(o, banned)


# -- non-stationary behaviour ----------------------------------
def test_emerging_signal_appears_at_its_activation_slot():
    env = example_emerging().build_environment(0)  # EMERGING emitter on ch4 @ slot 200
    detections_after = 0
    for slot in range(400):
        occupied = env.ground_truth(4).occupied
        if slot < 200:
            assert not occupied
        else:
            detections_after += occupied
        env.step()
    assert detections_after > 10


def test_nonstationary_event_flips_the_world_at_the_configured_slot():
    sc = ScenarioConfig(
        name="ns",
        seed=0,
        n_channels=3,
        duration_slots=400,
        emitters=[
            EmitterSpec(1, Behavior.PERSISTENT, {"p01": 0.0, "p10": 1.0, "signal_dbm": -50.0}),
        ],
        nonstationarity=[
            NonStationaryEvent(slot=150, channel_index=1, param_overrides={"p01": 1.0, "p10": 0.0}),
        ],
    )
    env = sc.build_environment(0)
    for slot in range(400):
        occupied = env.ground_truth(1).occupied
        if slot < 150:
            assert not occupied
        elif slot > 152:
            assert occupied
        env.step()


def test_environment_rejects_event_on_out_of_range_channel():
    sc = ScenarioConfig(
        name="bad",
        seed=0,
        n_channels=2,
        duration_slots=50,
        emitters=[EmitterSpec(0, Behavior.PERSISTENT)],
        nonstationarity=[NonStationaryEvent(slot=10, channel_index=5, param_overrides={})],
    )
    with pytest.raises(ValueError):
        sc.build_environment(0)


# -- channel plans -----------------------------------------------
def test_generic_channel_plan_is_supported():
    sc = ScenarioConfig(
        name="gen",
        seed=0,
        n_channels=20,
        duration_slots=100,
        channel_plan="generic",
        channel_plan_params={"start_hz": 1e9, "spacing_hz": 2e6, "bandwidth_hz": 500e3},
        emitters=[EmitterSpec(7, Behavior.PERSISTENT)],
    )
    env = sc.build_environment(0)
    assert env.n_channels == 20
    assert len(env.channels) == 20
    assert env.channels[1].center_freq_hz == pytest.approx(1e9 + 2e6)
    for _ in range(20):
        env.observe(19)
        env.step()


def test_wifi_channel_plan_is_supported():
    env = example_normal().build_environment(0)
    assert [c.label for c in env.channels][:2] == ["Wi-Fi ch 1", "Wi-Fi ch 2"]
