"""Offline predictor evaluation: discrimination, calibration, cost.

Every function here takes plain numpy arrays (no simulator/predictor
dependency beyond ``Predictor.predict_proba``), so it can score the non-ML
:class:`~rfscan.models.baseline_beta.DecayingBetaPredictor` and any
:class:`~rfscan.models.sklearn_model.SklearnPredictor` identically -- the bake
-off compares them on exactly the same numbers (docs/architecture.md S6).

Model accuracy is a diagnostic here, not the project's success metric (S14) --
the closed-loop scheduler (Phase 6) is what actually gets judged. These
functions exist so that diagnostic is honest and reproducible.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    precision: float
    recall: float
    f1: float
    pr_auc: float
    roc_auc: float | None  # None if y_true is single-class (undefined)
    brier: float
    n_positive: int
    n_total: int


def classification_metrics(
    y_true: np.ndarray, y_score: np.ndarray, *, threshold: float = 0.5
) -> ClassificationMetrics:
    """Precision/recall/F1 at ``threshold``, plus threshold-free PR-AUC,
    ROC-AUC (when both classes are present), and Brier score."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    roc_auc: float | None
    if len(np.unique(y_true)) < 2:
        roc_auc = None
    else:
        roc_auc = float(roc_auc_score(y_true, y_score))

    return ClassificationMetrics(
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        pr_auc=float(average_precision_score(y_true, y_score)),
        roc_auc=roc_auc,
        brier=float(brier_score_loss(y_true, y_score)),
        n_positive=int(y_true.sum()),
        n_total=int(len(y_true)),
    )


def recall_at_precision(
    y_true: np.ndarray, y_score: np.ndarray, target_precision: float
) -> float | None:
    """Best recall achievable among *real* thresholds whose precision is >=
    ``target_precision``. ``None`` if no threshold reaches it.

    ``precision_recall_curve`` appends a synthetic trailing point
    (precision=1, recall=0, no associated threshold) for the "classify
    nothing as positive" edge case; that point is excluded here so a target
    precision is only ever satisfied by an actual, usable threshold.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if len(thresholds) == 0:
        return None
    precision, recall = precision[: len(thresholds)], recall[: len(thresholds)]
    ok = precision >= target_precision
    if not ok.any():
        return None
    return float(recall[ok].max())


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    """Reliability-diagram data plus a scalar summary (expected calibration
    error). Both are measured, never asserted -- there is no "calibration
    improved" claim without this."""

    bin_edges: tuple[float, ...]
    bin_mean_predicted: tuple[float | None, ...]  # None for empty bins
    bin_mean_actual: tuple[float | None, ...]
    bin_counts: tuple[int, ...]
    expected_calibration_error: float


def calibration_report(
    y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10
) -> CalibrationReport:
    """Equal-width reliability diagram over ``[0, 1]`` and the expected
    calibration error: ``sum_bin (count_bin / n) * |mean_predicted - mean_actual|``.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_score, edges[1:-1], right=True), 0, n_bins - 1)

    mean_pred: list[float | None] = []
    mean_actual: list[float | None] = []
    counts: list[int] = []
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = bin_idx == b
        count = int(mask.sum())
        counts.append(count)
        if count == 0:
            mean_pred.append(None)
            mean_actual.append(None)
            continue
        mp = float(y_score[mask].mean())
        ma = float(y_true[mask].mean())
        mean_pred.append(mp)
        mean_actual.append(ma)
        ece += (count / n) * abs(mp - ma)

    return CalibrationReport(
        bin_edges=tuple(float(e) for e in edges),
        bin_mean_predicted=tuple(mean_pred),
        bin_mean_actual=tuple(mean_actual),
        bin_counts=tuple(counts),
        expected_calibration_error=float(ece),
    )


def measure_inference_latency(
    predict_fn, features: np.ndarray, *, n_repeats: int = 20
) -> float:
    """Mean wall-clock seconds per **row** of ``predict_fn(features)``,
    averaged over ``n_repeats`` calls on the whole batch. Nondeterministic by
    nature (wall-clock); reported as a diagnostic, not asserted exactly."""
    n_rows = max(1, len(features))
    # one warm-up call, excluded from timing (first-call JIT/allocation cost)
    predict_fn(features)
    start = time.perf_counter()
    for _ in range(n_repeats):
        predict_fn(features)
    elapsed = time.perf_counter() - start
    return elapsed / (n_repeats * n_rows)
