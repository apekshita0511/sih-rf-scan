"""load_predictor: wiring an AppConfig ModelConfig to a live Predictor (Phase 6)."""

from __future__ import annotations

import joblib
import pytest

from rfscan.config import ModelConfig
from rfscan.models.baseline_beta import DecayingBetaPredictor
from rfscan.models.loader import load_predictor


class _Dummy:
    name = "dummy"

    def predict_proba(self, features):
        return features[:, 0]

    def explain(self, x):
        return {}


class _OtherDummy:
    name = "logistic_regression"

    def predict_proba(self, features):
        return features[:, 0]

    def explain(self, x):
        return {}


def test_beta_kind_needs_no_file_and_returns_the_reference_predictor(tmp_path):
    config = ModelConfig(kind="beta", artifact_path=str(tmp_path / "does_not_exist.joblib"))
    predictor = load_predictor(config)
    assert isinstance(predictor, DecayingBetaPredictor)


def test_missing_artifact_raises_a_helpful_error(tmp_path):
    config = ModelConfig(
        kind="hist_gradient_boosting", artifact_path=str(tmp_path / "model.joblib")
    )
    with pytest.raises(FileNotFoundError, match="rfscan train"):
        load_predictor(config)


def test_falls_back_to_artifact_path_when_no_per_candidate_file_exists(tmp_path):
    generic_path = tmp_path / "model.joblib"
    joblib.dump(_Dummy(), generic_path)
    config = ModelConfig(kind="hist_gradient_boosting", artifact_path=str(generic_path))
    predictor = load_predictor(config)
    assert predictor.name == "dummy"


def test_prefers_the_named_candidate_file_over_the_generic_artifact(tmp_path):
    generic_path = tmp_path / "model.joblib"
    joblib.dump(_Dummy(), generic_path)

    candidate_path = tmp_path / "model_logistic_regression.joblib"
    joblib.dump(_OtherDummy(), candidate_path)

    config = ModelConfig(kind="logistic_regression", artifact_path=str(generic_path))
    predictor = load_predictor(config)
    assert predictor.name == "logistic_regression"
