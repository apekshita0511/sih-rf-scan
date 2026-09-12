"""Construct a :class:`~rfscan.models.base.Predictor` from an ``AppConfig``.

Separated from ``sklearn_model.py``/``baseline_beta.py`` (pure model code, no
I/O) so callers that need a ready-to-use predictor -- the Phase 6
``AdaptiveScheduler`` wiring in particular -- don't have to duplicate the
"which kind, load from where" decision.
"""

from __future__ import annotations

from pathlib import Path

import joblib

from rfscan.config import ModelConfig
from rfscan.models.base import Predictor
from rfscan.models.baseline_beta import DecayingBetaPredictor


def load_predictor(model_config: ModelConfig) -> Predictor:
    """Return the predictor named by ``model_config``.

    ``kind="beta"`` returns the non-ML :class:`DecayingBetaPredictor` directly
    (no I/O). For any other ``kind``, ``rfscan train`` (S16.5/S16.6) persists
    *every* bake-off candidate, not just the reporting winner, as
    ``model_<name>.joblib`` next to ``model_config.artifact_path`` -- so
    ``kind="logistic_regression"`` loads that candidate specifically. If no
    such per-candidate file exists (e.g. an older run, or a ``kind`` that
    isn't one of the bake-off's own candidate names), falls back to
    ``model_config.artifact_path`` itself -- whichever candidate
    ``choose_final_model`` picked as the offline reporting winner. The two
    can legitimately differ: the bake-off chooses by offline PR-AUC/
    calibration; the live scheduler may need a specific candidate for a
    closed-loop operational reason (S16.6 -- HistGradientBoosting's per-call
    latency at AdaptiveScheduler's tiny per-slot batch size).
    """
    if model_config.kind == "beta":
        return DecayingBetaPredictor()

    path = Path(model_config.artifact_path)
    candidate_path = path.parent / f"model_{model_config.kind}.joblib"
    if candidate_path.exists():
        return joblib.load(candidate_path)
    if not path.exists():
        raise FileNotFoundError(
            f"no trained model at {path} or {candidate_path} -- run `rfscan train` "
            "first, or set model.kind: beta to use the non-ML reference predictor instead."
        )
    return joblib.load(path)
