"""Weighted multi-objective priority score (docs/architecture.md S7, S8).

    priority(c) = w_pred      * p(c)
                + w_explore   * uncertainty(c)
                + w_fresh     * (1 - exp(-staleness(c) / tau))
                + w_trend     * max(0, activity_trend(c))
                - w_redundancy * redundancy(c)

``p(c)`` is the predictor's P(active); ``uncertainty(c)`` is the belief
layer's posterior std; ``staleness(c)`` is slots since last scan;
``activity_trend(c)`` is recent-rate minus all-time rate (already >= 0 by
construction, see :mod:`rfscan.perception.features`, but clamped again here
defensively). ``redundancy(c)`` is not specified numerically in S7 beyond "recently
scanned & confidently empty" -- implemented here as
``exp(-staleness(c) / redundancy_tau) * (1 - p(c))``: large exactly when a
channel was scanned a moment ago (small staleness) *and* the predictor is
confident it is empty (``p(c)`` near 0), decaying smoothly as either staleness
grows or predicted activity rises. This keeps every term a continuous
function of quantities the scheduler already has, with no new state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rfscan.config import SchedulerWeights


@dataclass(frozen=True, slots=True)
class PriorityBreakdown:
    """Per-channel term values for the argmax (or sampled) channel, for the
    explainability panel (docs/architecture.md S10)."""

    pred: float
    explore: float
    fresh: float
    trend: float
    redundancy: float
    priority: float

    def to_dict(self) -> dict[str, float]:
        return {
            "pred": self.pred,
            "explore": self.explore,
            "fresh": self.fresh,
            "trend": self.trend,
            "redundancy": self.redundancy,
            "priority": self.priority,
        }


class PriorityPolicy:
    """Stateless: turns per-channel term arrays into a priority vector."""

    def __init__(self, weights: SchedulerWeights, *, redundancy_tau_slots: float = 5.0) -> None:
        if redundancy_tau_slots <= 0.0:
            raise ValueError(f"redundancy_tau_slots must be > 0, got {redundancy_tau_slots}")
        self._w = weights
        self._redundancy_tau = float(redundancy_tau_slots)

    def score(
        self,
        *,
        predicted_proba: np.ndarray,
        uncertainty: np.ndarray,
        staleness: np.ndarray,
        activity_trend: np.ndarray,
    ) -> tuple[np.ndarray, list[PriorityBreakdown]]:
        """Priority vector (shape ``(n_channels,)``) plus a per-channel term
        breakdown, from the four per-channel input arrays (each shape
        ``(n_channels,)``)."""
        w = self._w
        freshness = 1.0 - np.exp(-staleness / w.freshness_tau_slots)
        trend = np.maximum(0.0, activity_trend)
        redundancy = np.exp(-staleness / self._redundancy_tau) * (1.0 - predicted_proba)

        pred_term = w.w_pred * predicted_proba
        explore_term = w.w_explore * uncertainty
        fresh_term = w.w_fresh * freshness
        trend_term = w.w_trend * trend
        redundancy_term = w.w_redundancy * redundancy

        priority = pred_term + explore_term + fresh_term + trend_term - redundancy_term

        breakdowns = [
            PriorityBreakdown(
                pred=float(pred_term[c]),
                explore=float(explore_term[c]),
                fresh=float(fresh_term[c]),
                trend=float(trend_term[c]),
                redundancy=float(-redundancy_term[c]),
                priority=float(priority[c]),
            )
            for c in range(len(priority))
        ]
        return priority, breakdowns
