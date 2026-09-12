"""Schema contracts, including the anti-leakage guard (architecture.md S16)."""

from __future__ import annotations

import dataclasses

import pytest

from rfscan.perception.schema import (
    FORBIDDEN_OBSERVATION_FIELDS,
    GroundTruth,
    Observation,
    ScanRecord,
)


def _obs(**overrides) -> Observation:
    base = dict(
        slot=3,
        channel_index=6,
        rssi_dbm=-52.0,
        noise_dbm=-95.0,
        snr_db=43.0,
        observed_detection=True,
    )
    base.update(overrides)
    return Observation(**base)


def test_observation_constructs_and_is_frozen():
    obs = _obs()
    assert obs.channel_index == 6
    assert obs.snr_db == 43.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        obs.snr_db = 0.0  # type: ignore[misc]


def test_observation_carries_no_ground_truth():
    """The scheduler / model only ever see an Observation - it must not expose
    the true channel state under any name."""
    field_names = {f.name for f in dataclasses.fields(Observation)}
    assert field_names == {
        "slot",
        "channel_index",
        "rssi_dbm",
        "noise_dbm",
        "snr_db",
        "observed_detection",
    }
    for forbidden in FORBIDDEN_OBSERVATION_FIELDS:
        assert forbidden not in field_names
    lowered = " ".join(field_names)
    for token in ("occup", "truth", "ground"):
        assert token not in lowered


def test_ground_truth_is_separate_type():
    gt = GroundTruth(slot=3, channel_index=6, occupied=True, true_signal_dbm=-52.0)
    assert gt.occupied is True
    assert not isinstance(gt, Observation)
    gt_none = GroundTruth(slot=4, channel_index=1, occupied=False, true_signal_dbm=None)
    assert gt_none.true_signal_dbm is None


def test_scan_record_derives_fields_from_observation():
    rec = ScanRecord(observation=_obs(slot=7, channel_index=2, observed_detection=False))
    assert rec.slot == 7
    assert rec.channel_index == 2
    assert rec.detected is False
    assert rec.decision_info is None

    rec2 = ScanRecord(observation=_obs(), decision_info={"priority": 0.91, "w_pred": 0.42})
    assert rec2.decision_info["priority"] == 0.91
