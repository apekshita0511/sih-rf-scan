"""ML activity-prediction engine (Phase 5).

Predicts P(channel active next slot | observation history). Contains the
predictor protocol, a non-ML decaying Beta-Bernoulli reference, scikit-learn
model wrappers, the training pipeline, and offline evaluation.
"""

from rfscan.models.base import Predictor
from rfscan.models.baseline_beta import DecayingBetaPredictor
from rfscan.models.loader import load_predictor
from rfscan.models.sklearn_model import SklearnPredictor

__all__ = [
    "Predictor",
    "DecayingBetaPredictor",
    "SklearnPredictor",
    "load_predictor",
]
