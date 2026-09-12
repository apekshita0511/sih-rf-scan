"""Predictor contract.

A predictor turns a :class:`~rfscan.perception.features.FeatureBuilder` output
(``(n_channels, n_features)`` or a single ``(n_features,)`` row) into
``P(active at the next decision opportunity)`` per channel. Structural typing
(Protocol), matching the ``Scheduler`` contract in ``scheduler/base.py``, so
candidates are trivially swappable. This is exactly
docs/architecture.md S4's ``models/base.py PredictorProtocol { predict_proba(X),
partial_fit?(X,y), explain(x) }`` -- ``partial_fit`` is intentionally not part
of the structural contract (few candidates support true online updates; a
Phase 6+ online-learning predictor can add it without breaking this Protocol).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Predictor(Protocol):
    """Structural contract every activity predictor satisfies."""

    name: str

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """``P(active)`` for each row of ``features``.

        Accepts either a single feature vector, shape ``(n_features,)``
        (returns a length-1 array), or a batch, shape
        ``(n_rows, n_features)`` (returns shape ``(n_rows,)``).
        """
        ...

    def explain(self, x: np.ndarray) -> Mapping[str, float]:
        """Named-term rationale for the prediction on a single feature
        vector ``x`` (shape ``(n_features,)``) -- the per-decision
        explanation the S1.2 closed loop attaches to a scan's
        ``decision_info``. What the terms mean is predictor-specific (e.g.
        Beta posterior mean/std, or a linear model's per-feature
        contributions); a predictor with no natural decomposition may return
        just ``{"predicted_proba": p}``.
        """
        ...
