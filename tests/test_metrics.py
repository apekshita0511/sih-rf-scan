"""Episode metrics on small hand-constructed episodes (Phase 4).

Each test builds an EpisodeResult directly from an occupancy grid + scan plan,
so the expected metric value can be reasoned about by hand.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rfscan.experiments.metrics import (
    compute_episode_metrics,
    extract_activity_intervals,
)
from rfscan.experiments.runner import EpisodeResult


def _result(occ, scanned, detected, *, emerging=(), activation=(), slot_s=0.1):
    occ = np.array(occ, dtype=bool)
    n_slots, n_ch = occ.shape
    return EpisodeResult(
        scenario="hand",
        world_seed=0,
        strategy="x",
        agent_seed=None,
        n_channels=n_ch,
        n_slots=n_slots,
        slot_duration_s=slot_s,
        scanned_channel=np.array(scanned, dtype=np.int64),
        observed_detection=np.array(detected, dtype=bool),
        occupancy=occ,
        emerging_channels=tuple(emerging),
        emerging_activation_slots=tuple(activation),
        decision_latencies_s=np.zeros(n_slots),
    )


def test_no_activity_gives_nan_rate_and_no_delays():
    r = _result(
        occ=[[False, False]] * 5,
        scanned=[0, 1, 0, 1, 0],
        detected=[False] * 5,
    )
    m = compute_episode_metrics(r)
    assert m.n_activity_intervals == 0
    assert math.isnan(m.detection_rate)
    assert m.mean_detection_delay_slots is None
    assert m.time_to_first_detection_slots is None


def test_one_interval_never_detected():
    # channel 0 occupied slots 1..3; scanned but detector never fires
    r = _result(
        occ=[[False], [True], [True], [True], [False]],
        scanned=[0, 0, 0, 0, 0],
        detected=[False, False, False, False, False],
    )
    m = compute_episode_metrics(r)
    assert m.n_activity_intervals == 1
    assert m.n_detected_intervals == 0
    assert m.n_missed_intervals == 1
    assert m.n_missed_detector == 1
    assert m.n_missed_never_scanned == 0
    assert m.detection_rate == 0.0
    assert m.missed_detection_rate == 1.0
    assert m.mean_detection_delay_slots is None


def test_immediate_detection_zero_delay():
    r = _result(
        occ=[[True], [True], [True]],
        scanned=[0, 0, 0],
        detected=[True, False, False],
    )
    m = compute_episode_metrics(r)
    assert m.detection_rate == 1.0
    assert m.mean_detection_delay_slots == 0.0
    assert m.time_to_first_detection_slots == 0
    assert m.true_positive_scans == 1


def test_delayed_detection_measures_from_interval_start():
    # ch0 interval [2, 10); scans on ch0 at slots 5 (miss) and 7 (hit), else ch1
    occ = [[False, False]] * 2 + [[True, False]] * 8
    scanned = [1, 1, 1, 1, 1, 0, 1, 0, 1, 1]
    detected = [False, False, False, False, False, False, False, True, False, False]
    m = compute_episode_metrics(_result(occ, scanned, detected))
    intervals = extract_activity_intervals(_result(occ, scanned, detected))
    assert intervals[0].start_slot == 2 and intervals[0].end_slot == 10
    assert intervals[0].detection_slot == 7
    assert m.mean_detection_delay_slots == 5.0
    assert m.time_to_first_detection_slots == 7


def test_multiple_signals_partial_detection():
    # ch0 interval detected, ch1 interval missed
    occ = [[True, True], [True, True], [True, True]]
    scanned = [0, 0, 1]
    detected = [True, False, False]
    m = compute_episode_metrics(_result(occ, scanned, detected))
    assert m.n_activity_intervals == 2
    assert m.detection_rate == 0.5


def test_censored_interval_flagged():
    # ch0 active through the final slot, never detected -> censored
    occ = [[False], [True], [True], [True]]
    scanned = [0, 0, 0, 0]
    detected = [False, False, False, False]
    r = _result(occ, scanned, detected)
    m = compute_episode_metrics(r)
    assert m.n_censored_intervals == 1
    assert extract_activity_intervals(r)[0].censored is True


def test_ended_miss_is_not_censored():
    occ = [[False], [True], [True], [False], [False]]
    scanned = [0, 0, 0, 0, 0]
    detected = [False] * 5
    r = _result(occ, scanned, detected)
    assert extract_activity_intervals(r)[0].censored is False
    assert compute_episode_metrics(r).n_censored_intervals == 0


def test_false_positive_excluded_from_efficiency():
    # detection on an empty channel
    occ = [[False], [False], [False]]
    scanned = [0, 0, 0]
    detected = [True, False, False]
    m = compute_episode_metrics(_result(occ, scanned, detected))
    assert m.false_positive_scans == 1
    assert m.true_positive_scans == 0
    assert m.scan_efficiency == 0.0
    assert m.total_detections == 1


def test_coverage_full_and_partial():
    occ = [[False, False, False]] * 4
    full = compute_episode_metrics(_result(occ, [0, 1, 2, 0], [False] * 4))
    assert full.channels_covered == 3
    assert full.channel_coverage == 1.0
    assert full.coverage_time_slots == 2
    partial = compute_episode_metrics(_result(occ, [0, 0, 1, 1], [False] * 4))
    assert partial.channels_covered == 2
    assert partial.coverage_time_slots is None


def test_redundant_scans_counted_with_window():
    # scan empty ch0 four slots in a row -> slots 1,2,3 are redundant (window=5)
    occ = [[False, False]] * 5
    scanned = [0, 0, 0, 0, 1]
    detected = [False] * 5
    m = compute_episode_metrics(_result(occ, scanned, detected), redundancy_window=5)
    assert m.redundant_scans == 3
    assert m.redundant_scan_rate == pytest.approx(3 / 5)


def test_a_detection_in_the_window_clears_redundancy():
    occ = [[True], [True], [True], [True]]
    scanned = [0, 0, 0, 0]
    detected = [False, True, False, False]  # slot 1 detects
    m = compute_episode_metrics(_result(occ, scanned, detected), redundancy_window=5)
    # slot1: prior scan slot0, detection at slot1 -> not redundant
    # slot2: window has slot1 detection -> not redundant
    # slot3: window [0..3] contains slot1 detection -> not redundant
    assert m.redundant_scans == 0


def test_scan_efficiency_is_true_positives_over_scans():
    occ = [[True], [True], [False], [True]]
    scanned = [0, 0, 0, 0]
    detected = [True, True, True, False]  # 2 TP, 1 FP, 1 miss
    m = compute_episode_metrics(_result(occ, scanned, detected))
    assert m.true_positive_scans == 2
    assert m.scan_efficiency == pytest.approx(2 / 4)
    assert m.on_target_scan_rate == pytest.approx(3 / 4)


def test_emerging_discovery_delay_measured_from_activation():
    # emerging emitter ch1, activation slot 3; occupied 3.., detected at slot 6
    occ = [[False, False]] * 3 + [[False, True]] * 5
    scanned = [1] * 8
    detected = [False, False, False, False, False, False, True, False]
    r = _result(occ, scanned, detected, emerging=(1,), activation=(3,))
    m = compute_episode_metrics(r)
    assert m.emerging_discovery_delay_slots == 3  # slot 6 - activation 3
    assert m.emerging_discovery_censored is False
    assert m.to_row()["emerging_discovery_delay_s"] == pytest.approx(0.3)


def test_emerging_discovery_censored_when_never_found():
    occ = [[False, False]] * 3 + [[False, True]] * 5
    scanned = [0] * 8  # never scans ch1
    detected = [False] * 8
    r = _result(occ, scanned, detected, emerging=(1,), activation=(3,))
    m = compute_episode_metrics(r)
    assert m.emerging_discovery_delay_slots is None
    assert m.emerging_discovery_censored is True


def test_emerging_metrics_none_without_emerging_emitter():
    occ = [[True]] * 4
    m = compute_episode_metrics(_result(occ, [0] * 4, [True] * 4))
    assert m.emerging_discovery_delay_slots is None
    assert m.emerging_discovery_censored is False


def test_to_row_has_seconds_columns():
    occ = [[True], [True], [True]]
    row = compute_episode_metrics(_result(occ, [0, 0, 0], [False, True, False])).to_row()
    assert "mean_detection_delay_s" in row
    assert "time_to_first_detection_s" in row
    assert row["mean_detection_delay_s"] == pytest.approx(0.1)  # 1 slot * 0.1 s
