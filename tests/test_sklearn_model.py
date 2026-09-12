"""SklearnPredictor: fit, predict, probability bounds, determinism,
calibration wiring, explain() (Phase 5)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from rfscan.models.sklearn_model import SklearnPredictor
from rfscan.perception.features import FEATURE_NAMES


def _synthetic_xy(n=400, n_features=None, seed=0):
    if n_features is None:
        n_features = len(FEATURE_NAMES)
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, n_features)).astype(np.float32)
    # label driven by a couple of columns + noise, so LR has something real to learn
    logits = 1.5 * x[:, 0] - 0.8 * x[:, 1] + rng.normal(scale=0.5, size=n)
    y = (logits > 0).astype(np.int8)
    return x, y


def _lr_predictor(calibration="isotonic") -> SklearnPredictor:
    return SklearnPredictor(
        name="logistic_regression",
        estimator=Pipeline(
            [("scaler", StandardScaler()), ("lr", LogisticRegression(random_state=0))]
        ),
        calibration=calibration,
    )


def test_predict_before_fit_raises():
    predictor = _lr_predictor()
    with pytest.raises(RuntimeError):
        predictor.predict_proba(np.zeros((1, len(FEATURE_NAMES))))


def test_calibration_without_val_set_raises():
    x, y = _synthetic_xy()
    predictor = _lr_predictor(calibration="isotonic")
    with pytest.raises(ValueError):
        predictor.fit(x, y)


def test_fit_and_predict_shapes_and_bounds():
    x, y = _synthetic_xy()
    x_train, y_train = x[:300], y[:300]
    x_val, y_val = x[300:], y[300:]
    predictor = _lr_predictor().fit(x_train, y_train, x_val, y_val)

    proba = predictor.predict_proba(x_val)
    assert proba.shape == (len(x_val),)
    assert (proba >= 0.0).all()
    assert (proba <= 1.0).all()
    assert predictor.is_fitted


def test_single_row_prediction():
    x, y = _synthetic_xy()
    predictor = _lr_predictor().fit(x[:300], y[:300], x[300:], y[300:])
    single = predictor.predict_proba(x[0])
    assert single.shape == (1,)
    assert 0.0 <= single[0] <= 1.0


def test_no_calibration_mode_skips_calibrator():
    x, y = _synthetic_xy()
    predictor = _lr_predictor(calibration="none").fit(x[:300], y[:300])
    assert predictor._calibrator is None  # type: ignore[attr-defined]
    proba = predictor.predict_proba(x[:5])
    assert proba.shape == (5,)


def test_deterministic_given_fixed_random_state():
    x, y = _synthetic_xy()
    x_train, y_train = x[:300], y[:300]
    x_val, y_val = x[300:], y[300:]

    a = _lr_predictor().fit(x_train, y_train, x_val, y_val).predict_proba(x_val)
    b = _lr_predictor().fit(x_train, y_train, x_val, y_val).predict_proba(x_val)
    np.testing.assert_array_equal(a, b)


def test_explain_gives_linear_contributions_for_logistic_regression():
    x, y = _synthetic_xy()
    predictor = _lr_predictor().fit(x[:300], y[:300], x[300:], y[300:])
    terms = predictor.explain(x[0])
    assert "predicted_proba" in terms
    assert "intercept" in terms
    for name in FEATURE_NAMES:
        assert name in terms
    assert terms["predicted_proba"] == pytest.approx(predictor.predict_proba(x[0])[0])


def test_explain_falls_back_for_non_linear_estimator():
    from sklearn.ensemble import HistGradientBoostingClassifier

    x, y = _synthetic_xy()
    predictor = SklearnPredictor(
        name="hist_gradient_boosting",
        estimator=HistGradientBoostingClassifier(random_state=0, max_iter=20),
        calibration="none",
    ).fit(x[:300], y[:300])
    terms = predictor.explain(x[0])
    assert set(terms) == {"predicted_proba"}


def test_hgb_deterministic_with_fixed_random_state():
    from sklearn.ensemble import HistGradientBoostingClassifier

    x, y = _synthetic_xy()
    x_train, y_train = x[:300], y[:300]
    x_val, y_val = x[300:], y[300:]

    def _build():
        return SklearnPredictor(
            name="hist_gradient_boosting",
            estimator=HistGradientBoostingClassifier(random_state=0, max_iter=20),
            calibration="sigmoid",
        ).fit(x_train, y_train, x_val, y_val)

    a = _build().predict_proba(x_val)
    b = _build().predict_proba(x_val)
    np.testing.assert_array_equal(a, b)
