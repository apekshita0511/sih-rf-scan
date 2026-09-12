"""Non-ML reference predictor: decaying Beta-Bernoulli posterior mean.

This is the "honesty benchmark" from docs/architecture.md S6/S7 -- zero
training, fully explainable, native staleness/uncertainty handling. Every
sklearn candidate in the bake-off must beat it to justify its extra
complexity and calibration effort.

Design: :class:`~rfscan.perception.features.FeatureBuilder` already computes
the decaying Beta posterior mean/std as two of its columns (``beta_
posterior_mean`` / ``beta_posterior_std``, via
``rfscan.perception.features.decaying_beta_posterior``), because the Phase 6
scheduler's belief layer will need the same statistic. Rather than
re-derive the same math from raw history, this predictor simply reads that
column back out -- one array indexing operation, genuinely O(1) per
prediction on top of feature-building.

    prior:    Beta(prior_alpha, prior_beta) per channel (default Beta(1,1),
              i.e. uniform / maximally uncertain).
    evidence: each scan updates a += 1 (detection) or b += 1 (no detection).
    decay:    between scans (and from the last scan to "now"), the posterior
              relaxes geometrically toward the prior:
              a <- lambda**gap * a + (1 - lambda**gap) * prior_alpha
              (same for b). Staler evidence counts for less.
    prediction: posterior mean a / (a + b).
    uncertainty: posterior std, sqrt(a*b / ((a+b)**2 * (a+b+1))) -- large
              when a channel has been scanned rarely or not recently, small
              when it has strong, fresh, one-sided evidence.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from rfscan.perception.features import FEATURE_NAMES

_MEAN_COL = FEATURE_NAMES.index("beta_posterior_mean")
_STD_COL = FEATURE_NAMES.index("beta_posterior_std")


class DecayingBetaPredictor:
    """Reads P(active) straight off the FeatureBuilder's Beta posterior mean."""

    name = "decaying_beta"

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(features)
        return x[:, _MEAN_COL]

    def uncertainty(self, features: np.ndarray) -> np.ndarray:
        """Posterior std per row -- the uncertainty half of the S1.2 closed
        loop's "Probability + uncertainty" output."""
        x = np.atleast_2d(features)
        return x[:, _STD_COL]

    def explain(self, x: np.ndarray) -> Mapping[str, float]:
        """The posterior mean *is* the prediction here, so the explanation is
        just its two defining numbers."""
        row = np.asarray(x).reshape(-1)
        return {
            "beta_posterior_mean": float(row[_MEAN_COL]),
            "beta_posterior_std": float(row[_STD_COL]),
        }
