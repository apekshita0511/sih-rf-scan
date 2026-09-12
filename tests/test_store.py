"""ObservationStore: aggregates, views, and DataFrame export (Phase 3)."""

from __future__ import annotations

import math

import pytest

from rfscan.perception.schema import Observation, ScanRecord
from rfscan.perception.store import ObservationStore, ReadableStore


def _rec(slot: int, channel: int, detected: bool, decision=None) -> ScanRecord:
    obs = Observation(
        slot=slot,
        channel_index=channel,
        rssi_dbm=-80.0,
        noise_dbm=-95.0,
        snr_db=15.0,
        observed_detection=detected,
    )
    return ScanRecord(observation=obs, decision_info=decision)


def test_empty_store():
    store = ObservationStore(4)
    assert len(store) == 0
    assert store.total_scans == 0
    assert store.coverage() == 0
    assert store.scan_count(2) == 0
    assert store.detection_rate(2, prior=0.5) == 0.5
    assert store.last_scan_slot(2) is None
    assert store.slots_since_last_scan(2, 10) is None
    assert store.last_record(2) is None
    assert store.records_for(2) == []


def test_construction_validates():
    with pytest.raises(ValueError):
        ObservationStore(0)


def test_append_and_counts():
    store = ObservationStore(3)
    store.append(_rec(0, 0, True))
    store.append(_rec(1, 0, False))
    store.append(_rec(2, 0, True))
    store.append(_rec(3, 1, False))
    assert len(store) == 4
    assert store.scan_count(0) == 3
    assert store.scan_count(1) == 1
    assert store.scan_count(2) == 0
    assert store.detection_count(0) == 2
    assert store.detection_rate(0) == pytest.approx(2 / 3)
    assert store.detection_rate(1) == 0.0
    assert store.coverage() == 2


def test_detection_rate_prior_only_applies_when_unscanned():
    store = ObservationStore(2)
    store.append(_rec(0, 0, False))
    assert store.detection_rate(0, prior=1.0) == 0.0  # scanned -> real rate
    assert store.detection_rate(1, prior=1.0) == 1.0  # unscanned -> prior


def test_last_scan_slot_and_staleness():
    store = ObservationStore(2)
    store.append(_rec(5, 0, True))
    store.append(_rec(12, 0, False))
    assert store.last_scan_slot(0) == 12
    assert store.slots_since_last_scan(0, 20) == 8
    assert store.slots_since_last_scan(1, 20) is None


def test_records_for_and_last_record():
    store = ObservationStore(2)
    r0, r1 = _rec(0, 1, True), _rec(3, 1, False)
    store.append(r0)
    store.append(r1)
    assert store.records_for(1) == [r0, r1]
    assert store.last_record(1) is r1
    # returned list is a copy -- mutating it must not affect the store
    store.records_for(1).append(_rec(9, 1, True))
    assert store.scan_count(1) == 2


def test_append_rejects_out_of_range_channel():
    store = ObservationStore(3)
    with pytest.raises(IndexError):
        store.append(_rec(0, 5, True))


def test_read_accessors_reject_out_of_range_channel():
    store = ObservationStore(3)
    for call in (
        lambda: store.scan_count(3),
        lambda: store.detection_rate(3),
        lambda: store.records_for(-1),
        lambda: store.slots_since_last_scan(3, 0),
    ):
        with pytest.raises(IndexError):
            call()


def test_to_frame_shape_and_columns():
    store = ObservationStore(2)
    store.append(_rec(0, 0, True))
    store.append(_rec(1, 1, False))
    frame = store.to_frame()
    assert list(frame.columns) == [
        "slot",
        "channel_index",
        "rssi_dbm",
        "noise_dbm",
        "snr_db",
        "observed_detection",
    ]
    assert len(frame) == 2
    assert frame.loc[0, "observed_detection"] is True or frame.loc[0, "observed_detection"] == True  # noqa: E712


def test_to_frame_empty_has_columns_no_rows():
    frame = ObservationStore(2).to_frame()
    assert len(frame) == 0
    assert "channel_index" in frame.columns


def test_to_frame_flattens_decision_info():
    store = ObservationStore(2)
    store.append(_rec(0, 0, True, decision={"priority": 0.9, "explore": 0.3}))
    store.append(_rec(1, 1, False))  # no decision_info
    frame = store.to_frame()
    assert "decision_priority" in frame.columns
    assert frame.loc[0, "decision_priority"] == pytest.approx(0.9)
    assert math.isnan(frame.loc[1, "decision_priority"])


def test_observation_store_satisfies_readable_store_protocol():
    assert isinstance(ObservationStore(3), ReadableStore)
