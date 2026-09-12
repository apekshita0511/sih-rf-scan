"""Offline evaluation metrics: classification, calibration, latency (Phase 5)."""

from __future__ import annotations

import numpy as np
import pytest

from rfscan.models.evaluation import (
    calibration_report,
    classification_metrics,
    measure_inference_latency,
    recall_at_precision,
)


def test_perfect_predictions_score_perfectly():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.0, 0.1, 0.9, 1.0])
    m = classification_metrics(y_true, y_score)
    assert m.precision == pytest.approx(1.0)
    assert m.recall == pytest.approx(1.0)
    assert m.f1 == pytest.approx(1.0)
    assert m.pr_auc == pytest.approx(1.0)
    assert m.roc_auc == pytest.approx(1.0)
    assert m.brier == pytest.approx(0.0, abs=1e-9) or m.brier < 0.05


def test_single_class_roc_auc_is_none_not_an_error():
    y_true = np.array([0, 0, 0, 0])
    y_score = np.array([0.1, 0.2, 0.3, 0.4])
    m = classification_metrics(y_true, y_score)
    assert m.roc_auc is None
    assert m.n_positive == 0
    assert m.n_total == 4


def test_threshold_changes_precision_recall_tradeoff():
    y_true = np.array([0, 0, 1, 1, 1])
    y_score = np.array([0.2, 0.6, 0.4, 0.7, 0.9])
    low = classification_metrics(y_true, y_score, threshold=0.3)
    high = classification_metrics(y_true, y_score, threshold=0.8)
    assert low.recall >= high.recall


def test_recall_at_precision_returns_none_when_unreachable():
    y_true = np.array([0, 1, 0, 1])
    y_score = np.array([0.5, 0.5, 0.5, 0.5])  # no threshold separates classes
    assert recall_at_precision(y_true, y_score, target_precision=0.99) is None


def test_recall_at_precision_is_achievable_for_separable_data():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.0, 0.1, 0.2, 0.8, 0.9, 1.0])
    recall = recall_at_precision(y_true, y_score, target_precision=1.0)
    assert recall == pytest.approx(1.0)


def test_calibration_report_bins_and_ece_for_perfectly_calibrated_scores():
    rng = np.random.default_rng(0)
    y_score = rng.uniform(0, 1, size=5000)
    y_true = (rng.uniform(0, 1, size=5000) < y_score).astype(int)
    report = calibration_report(y_true, y_score, n_bins=10)
    assert len(report.bin_counts) == 10
    assert sum(report.bin_counts) == 5000
    # well-calibrated synthetic scores -> low expected calibration error
    assert report.expected_calibration_error < 0.05


def test_calibration_report_handles_empty_bins():
    y_true = np.array([0, 1])
    y_score = np.array([0.05, 0.06])  # everything falls in one low bin
    report = calibration_report(y_true, y_score, n_bins=10)
    assert report.bin_mean_predicted[0] is not None
    assert report.bin_mean_predicted[-1] is None
    assert report.bin_counts[-1] == 0


def test_measure_inference_latency_is_positive_and_reasonable():
    def predict_fn(x):
        return x.sum(axis=1)

    x = np.random.default_rng(0).normal(size=(200, 17))
    latency = measure_inference_latency(predict_fn, x, n_repeats=5)
    assert latency > 0.0
    assert latency < 1.0  # seconds/row -- a trivial sum should be far below this
