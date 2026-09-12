"""Clean predictor abstraction around a scikit-learn classifier + calibrator.

Wraps any sklearn ``predict_proba``-capable classifier (Logistic Regression,
HistGradientBoosting, ...) behind the :class:`~rfscan.models.base.Predictor`
protocol, with optional post-hoc probability calibration fit on a *held-out*
validation set -- never on the training fold and never on the test fold
(docs/architecture.md S3.2/S9). Calibration matters here because the Phase 6
scheduler consumes the predicted probability directly as a priority term.

``cv="prefit"`` was removed from scikit-learn's ``CalibratedClassifierCV`` in
favour of wrapping the already-fitted estimator in ``FrozenEstimator`` -- this
module uses that pattern (see ``requirements.txt`` for the pinned sklearn
version this was written against).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from sklearn.base import ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from rfscan.perception.features import FEATURE_NAMES

Calibration = Literal["none", "sigmoid", "isotonic"]


@dataclass(slots=True)
class SklearnPredictor:
    """Fits ``estimator`` on a training fold, then (optionally) calibrates its
    probabilities on a separate validation fold."""

    name: str
    estimator: ClassifierMixin
    calibration: Calibration = "isotonic"
    _fitted: ClassifierMixin | None = field(default=None, init=False, repr=False)
    _calibrator: CalibratedClassifierCV | None = field(default=None, init=False, repr=False)

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> SklearnPredictor:
        """Fit the base estimator on the train fold. If ``calibration`` is not
        ``"none"``, ``x_val``/``y_val`` (a fold the estimator never saw) are
        required and calibration is fit on them."""
        self._fitted = clone(self.estimator).fit(x_train, y_train)
        if self.calibration == "none":
            self._calibrator = None
            return self
        if x_val is None or y_val is None:
            raise ValueError(
                f"calibration={self.calibration!r} requires x_val/y_val "
                "(a held-out fold, never the training data)"
            )
        method = "isotonic" if self.calibration == "isotonic" else "sigmoid"
        self._calibrator = CalibratedClassifierCV(
            FrozenEstimator(self._fitted), method=method
        ).fit(x_val, y_val)
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self._fitted is None:
            raise RuntimeError(f"SklearnPredictor {self.name!r} has not been fit yet")
        x = np.atleast_2d(features)
        model = self._calibrator if self._calibrator is not None else self._fitted
        return model.predict_proba(x)[:, 1]

    def explain(self, x: np.ndarray) -> Mapping[str, float]:
        """Per-feature linear contribution (``coef * scaled_value``) when the
        fitted estimator is a ``Pipeline`` with a ``"lr"`` linear step (the
        ``logistic_regression`` candidate); otherwise falls back to just the
        predicted probability, since a tree ensemble's per-instance
        contribution is not a cheap linear decomposition (docs/architecture.md
        S6/S15 "black-box prediction" -- a documented limitation, not silently
        hidden)."""
        row = np.asarray(x).reshape(1, -1)
        proba = float(self.predict_proba(row)[0])
        steps = getattr(self.fitted_estimator, "named_steps", None)
        if steps is not None and "lr" in steps and hasattr(steps["lr"], "coef_"):
            scaler = steps.get("scaler")
            scaled = scaler.transform(row) if scaler is not None else row
            contributions = steps["lr"].coef_[0] * scaled[0]
            terms = {
                name: float(c) for name, c in zip(FEATURE_NAMES, contributions, strict=True)
            }
            terms["intercept"] = float(steps["lr"].intercept_[0])
            terms["predicted_proba"] = proba
            return terms
        return {"predicted_proba": proba}

    @property
    def is_fitted(self) -> bool:
        return self._fitted is not None

    @property
    def fitted_estimator(self) -> ClassifierMixin:
        """The base estimator after :meth:`fit` (pre-calibration). Read-only
        introspection for reporting (coefficients, tree count, ...)."""
        if self._fitted is None:
            raise RuntimeError(f"SklearnPredictor {self.name!r} has not been fit yet")
        return self._fitted
