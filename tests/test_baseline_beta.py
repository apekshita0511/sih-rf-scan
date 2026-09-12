"""DecayingBetaPredictor: prior, updates, decay, bounds, reproducibility (Phase 5)."""

from __future__ import annotations

import numpy as np
import pytest

from rfscan.models.baseline_beta import DecayingBetaPredictor
from rfscan.perception.features import FEATURE_NAMES, FeatureBuilder
from rfscan.perception.schema import Observation, ScanRecord
from rfscan.perception.store import ObservationStore

_MEAN_COL = FEATURE_NAMES.index("beta_posterior_mean")
_STD_COL = FEATURE_NAMES.index("beta_posterior_std")


def _rec(slot: int, channel: int, detected: bool) -> ScanRecord:
    obs = Observation(
        slot=slot,
        channel_index=channel,
        rssi_dbm=-70.0,
        noise_dbm=-90.0,
        snr_db=20.0,
        observed_detection=detected,
    )
    return ScanRecord(observation=obs)


def test_name_and_protocol_shape():
    predictor = DecayingBetaPredictor()
    assert predictor.name == "decaying_beta"
    assert hasattr(predictor, "predict_proba")
    assert hasattr(predictor, "explain")


def test_prior_with_no_evidence_is_maximally_uncertain():
    store = ObservationStore(3)
    fb = FeatureBuilder()
    x = fb.build(store, slot=0)
    predictor = DecayingBetaPredictor()
    proba = predictor.predict_proba(x)
    assert proba.shape == (3,)
    assert np.allclose(proba, 0.5)
    assert (predictor.uncertainty(x) > 0.0).all()


def test_positive_update_raises_probability():
    store = ObservationStore(1)
    fb = FeatureBuilder()
    predictor = DecayingBetaPredictor()

    before = predictor.predict_proba(fb.build(store, slot=0))[0]
    store.append(_rec(0, 0, detected=True))
    after = predictor.predict_proba(fb.build(store, slot=1))[0]
    assert after > before


def test_negative_update_lowers_probability():
    store = ObservationStore(1)
    fb = FeatureBuilder()
    predictor = DecayingBetaPredictor()

    before = predictor.predict_proba(fb.build(store, slot=0))[0]
    store.append(_rec(0, 0, detected=False))
    after = predictor.predict_proba(fb.build(store, slot=1))[0]
    assert after < before


def test_decay_pulls_prediction_back_toward_prior_over_time():
    store = ObservationStore(1)
    store.append(_rec(0, 0, detected=True))
    fb = FeatureBuilder(beta_decay_lambda=0.9)
    predictor = DecayingBetaPredictor()

    soon = predictor.predict_proba(fb.build(store, slot=1))[0]
    much_later = predictor.predict_proba(fb.build(store, slot=200))[0]
    assert much_later < soon
    assert much_later == pytest.approx(0.5, abs=0.05)


def test_probabilities_are_always_in_unit_interval():
    store = ObservationStore(1)
    fb = FeatureBuilder()
    predictor = DecayingBetaPredictor()
    for slot in range(30):
        store.append(_rec(slot, 0, detected=(slot % 2 == 0)))
    proba = predictor.predict_proba(fb.build(store, slot=30))
    assert (proba >= 0.0).all()
    assert (proba <= 1.0).all()


def test_reproducible_given_same_history():
    store = ObservationStore(1)
    for slot in range(10):
        store.append(_rec(slot, 0, detected=(slot % 3 == 0)))
    fb = FeatureBuilder()
    predictor = DecayingBetaPredictor()
    a = predictor.predict_proba(fb.build(store, slot=10))
    b = predictor.predict_proba(fb.build(store, slot=10))
    assert np.array_equal(a, b)


def test_explain_returns_the_same_numbers_as_the_feature_columns():
    store = ObservationStore(1)
    store.append(_rec(0, 0, detected=True))
    fb = FeatureBuilder()
    predictor = DecayingBetaPredictor()
    x = fb.build_channel(store, 0, slot=1)
    terms = predictor.explain(x)
    assert terms["beta_posterior_mean"] == pytest.approx(x[_MEAN_COL])
    assert terms["beta_posterior_std"] == pytest.approx(x[_STD_COL])
    assert terms["beta_posterior_mean"] == pytest.approx(predictor.predict_proba(x)[0])
