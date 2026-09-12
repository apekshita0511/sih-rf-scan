"""FeatureBuilder: correctness, availability, empty/partial history, no
future information (Phase 5)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rfscan.perception.features import (
    FEATURE_DOCS,
    FEATURE_NAMES,
    MAX_STALENESS,
    MISSING_VALUE,
    FeatureBuilder,
    decaying_beta_posterior,
)
from rfscan.perception.schema import Observation, ScanRecord
from rfscan.perception.store import ObservationStore


def _rec(slot: int, channel: int, detected: bool, snr: float = 10.0) -> ScanRecord:
    obs = Observation(
        slot=slot,
        channel_index=channel,
        rssi_dbm=-70.0 + snr,
        noise_dbm=-70.0,
        snr_db=snr,
        observed_detection=detected,
    )
    return ScanRecord(observation=obs)


def _store(n=4) -> ObservationStore:
    return ObservationStore(n)


# -- feature names / docs -------------------------------------------------


def test_feature_names_are_documented():
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))
    for name in FEATURE_NAMES:
        doc = FEATURE_DOCS[name]
        assert doc["represents"]
        assert doc["why_useful"]
        assert doc["available"]
        assert doc["leakage"]


def test_channel_index_and_absolute_slot_excluded():
    assert "channel_index" not in FEATURE_NAMES
    assert "slot" not in FEATURE_NAMES
    assert "absolute_slot" not in FEATURE_NAMES


# -- empty-history behavior -------------------------------------------------


def test_empty_history_gives_sentinel_defaults():
    store = _store()
    fb = FeatureBuilder()
    x = fb.build_channel(store, 0, slot=0)
    row = dict(zip(FEATURE_NAMES, x, strict=True))

    assert row["last_rssi_dbm"] == MISSING_VALUE
    assert row["last_noise_dbm"] == MISSING_VALUE
    assert row["last_snr_db"] == MISSING_VALUE
    assert row["prev_snr_db"] == MISSING_VALUE
    assert row["delta_snr_db"] == 0.0
    assert row["last_detection"] == 0.0
    assert row["detection_count_recent"] == 0.0
    assert row["detection_rate_recent"] == 0.0
    assert row["detection_rate_alltime"] == 0.0
    assert row["slots_since_last_scan"] == MAX_STALENESS
    assert row["slots_since_last_detection"] == MAX_STALENESS
    assert row["rolling_snr_mean"] == MISSING_VALUE
    assert row["rolling_snr_std"] == 0.0
    assert row["scan_count"] == 0.0
    assert row["activity_trend"] == 0.0
    # Beta posterior falls back to the prior with no evidence.
    assert row["beta_posterior_mean"] == pytest.approx(0.5)
    assert row["beta_posterior_std"] > 0.0


def test_build_returns_one_row_per_channel():
    store = _store(5)
    fb = FeatureBuilder()
    x = fb.build(store, slot=0)
    assert x.shape == (5, len(FEATURE_NAMES))


# -- partial-history behavior -------------------------------------------------


def test_single_scan_has_no_prev_snr_but_has_last_snr():
    store = _store()
    store.append(_rec(0, 0, detected=True, snr=12.0))
    fb = FeatureBuilder()
    row = dict(zip(FEATURE_NAMES, fb.build_channel(store, 0, slot=1), strict=True))

    assert row["last_snr_db"] == pytest.approx(12.0)
    assert row["prev_snr_db"] == MISSING_VALUE
    assert row["delta_snr_db"] == 0.0
    assert row["last_detection"] == 1.0
    assert row["scan_count"] == 1.0
    assert row["slots_since_last_scan"] == pytest.approx(1.0)
    assert row["slots_since_last_detection"] == pytest.approx(1.0)


def test_two_scans_populate_prev_and_delta():
    store = _store()
    store.append(_rec(0, 0, detected=False, snr=5.0))
    store.append(_rec(1, 0, detected=True, snr=9.0))
    fb = FeatureBuilder()
    row = dict(zip(FEATURE_NAMES, fb.build_channel(store, 0, slot=2), strict=True))

    assert row["last_snr_db"] == pytest.approx(9.0)
    assert row["prev_snr_db"] == pytest.approx(5.0)
    assert row["delta_snr_db"] == pytest.approx(4.0)


def test_staleness_grows_after_last_scan():
    store = _store()
    store.append(_rec(0, 0, detected=True, snr=10.0))
    fb = FeatureBuilder()

    row_soon = dict(zip(FEATURE_NAMES, fb.build_channel(store, 0, slot=1), strict=True))
    row_later = dict(zip(FEATURE_NAMES, fb.build_channel(store, 0, slot=50), strict=True))

    assert row_soon["slots_since_last_scan"] == pytest.approx(1.0)
    assert row_later["slots_since_last_scan"] == pytest.approx(50.0)
    # Uncertainty about a stale channel should be at least as high as a fresh one.
    assert row_later["beta_posterior_std"] >= row_soon["beta_posterior_std"]


def test_detection_count_recent_respects_short_window():
    store = _store()
    for slot in range(7):
        store.append(_rec(slot, 0, detected=True, snr=10.0))
    fb = FeatureBuilder(short_window=5)
    row = dict(zip(FEATURE_NAMES, fb.build_channel(store, 0, slot=7), strict=True))
    # Only the last 5 of 7 detections should count.
    assert row["detection_count_recent"] == 5.0
    assert row["scan_count"] == 7.0


def test_activity_trend_is_nonnegative_and_reacts_to_recent_burst():
    store = _store()
    # Long quiet history, then a burst of recent detections.
    for slot in range(20):
        store.append(_rec(slot, 0, detected=False, snr=0.0))
    for slot in range(20, 25):
        store.append(_rec(slot, 0, detected=True, snr=15.0))
    fb = FeatureBuilder(long_window=20)
    row = dict(zip(FEATURE_NAMES, fb.build_channel(store, 0, slot=25), strict=True))
    assert row["activity_trend"] >= 0.0
    assert row["activity_trend"] > 0.0  # recent rate > all-time rate


# -- no future information (leakage) -------------------------------------------------


def test_build_raises_if_store_has_a_record_at_or_after_query_slot():
    store = _store()
    store.append(_rec(5, 0, detected=True))
    fb = FeatureBuilder()
    with pytest.raises(ValueError, match="future"):
        fb.build_channel(store, 0, slot=5)
    with pytest.raises(ValueError, match="future"):
        fb.build_channel(store, 0, slot=3)
    # slot strictly after the last record is fine.
    fb.build_channel(store, 0, slot=6)


def test_features_module_imports_nothing_from_simulator():
    import ast
    import inspect

    import rfscan.perception.features as features_mod

    tree = ast.parse(inspect.getsource(features_mod))
    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)
    assert not any(m.startswith("rfscan.simulator") for m in imported_modules)


def test_truncating_history_never_changes_earlier_features():
    """Replaying the same episode further (more scans, later slots) must
    never change the feature vector already built at an earlier slot -- proves
    no lookahead into scans that happen later in the same episode."""
    fb = FeatureBuilder()

    def _replay(n_slots: int) -> list[np.ndarray]:
        store = _store()
        snapshots = []
        for slot in range(n_slots):
            snapshots.append(fb.build_channel(store, 0, slot=slot))
            store.append(_rec(slot, 0, detected=(slot % 3 == 0), snr=float(slot)))
        return snapshots

    full_snapshots = _replay(10)
    prefix_snapshots = _replay(6)

    for probe_slot in range(6):
        np.testing.assert_array_equal(full_snapshots[probe_slot], prefix_snapshots[probe_slot])


# -- decaying_beta_posterior -------------------------------------------------


def test_decaying_beta_prior_with_no_evidence():
    mean, std = decaying_beta_posterior([], slot=0, prior_alpha=1.0, prior_beta=1.0)
    assert mean == pytest.approx(0.5)
    assert std == pytest.approx(math.sqrt(1 / 12))  # Beta(1,1) variance = 1/12


def test_decaying_beta_positive_update_raises_mean():
    records = [_rec(0, 0, detected=True)]
    mean, _ = decaying_beta_posterior(records, slot=1, prior_alpha=1.0, prior_beta=1.0)
    assert mean > 0.5


def test_decaying_beta_negative_update_lowers_mean():
    records = [_rec(0, 0, detected=False)]
    mean, _ = decaying_beta_posterior(records, slot=1, prior_alpha=1.0, prior_beta=1.0)
    assert mean < 0.5


def test_decaying_beta_decays_toward_prior_over_time():
    records = [_rec(0, 0, detected=True)]
    mean_soon, _ = decaying_beta_posterior(records, slot=1, decay_lambda=0.9)
    mean_later, _ = decaying_beta_posterior(records, slot=100, decay_lambda=0.9)
    assert mean_later < mean_soon
    assert mean_later == pytest.approx(0.5, abs=0.05)


def test_decaying_beta_is_reproducible():
    records = [_rec(s, 0, detected=(s % 2 == 0)) for s in range(5)]
    a = decaying_beta_posterior(records, slot=10)
    b = decaying_beta_posterior(records, slot=10)
    assert a == b
