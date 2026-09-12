"""AdaptiveScheduler v1 -- the closed loop of docs/architecture.md S1.2, S7, S8.

Wires :class:`~rfscan.scheduler.belief.BeliefState` (exploration term) +
a :class:`~rfscan.models.base.Predictor` (activity prediction term) +
:class:`~rfscan.scheduler.policy.PriorityPolicy` (weighted score) into one
``Scheduler``. No online learning in v1: the predictor is injected pre-fit
(``Predictor`` deliberately has no ``partial_fit`` in its structural contract,
docs/architecture.md S16.5); only the belief layer adapts within an episode.

Hard freshness guarantee (S7): if any channel's staleness reaches
``max_revisit_slots``, it is force-selected regardless of priority, so no
channel -- including a future emerging one -- can be starved forever. Default
``3 * n_channels`` bounds the worst-case revisit interval to about three
full round-robin sweeps' worth of slots, well inside what the emerging-signal
scenarios (S9) run for.

``freeze_belief`` (Phase 8, S16.8): when True, ``update()`` never touches
``BeliefState`` -- the belief stays at the prior forever, isolating "does
online feedback matter" for the ablation study without a second scheduler
implementation. Default False changes nothing about Phase 6/7 behaviour.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from rfscan.config import BeliefConfig, SchedulerWeights
from rfscan.models.base import Predictor
from rfscan.perception.features import FEATURE_NAMES, FeatureBuilder
from rfscan.perception.schema import ScanRecord
from rfscan.perception.store import ReadableStore
from rfscan.scheduler.belief import BeliefState
from rfscan.scheduler.policy import PriorityBreakdown, PriorityPolicy

_STALENESS_COL = FEATURE_NAMES.index("slots_since_last_scan")
_TREND_COL = FEATURE_NAMES.index("activity_trend")


class AdaptiveScheduler:
    name = "adaptive"

    def __init__(
        self,
        n_channels: int,
        predictor: Predictor,
        *,
        weights: SchedulerWeights | None = None,
        belief_config: BeliefConfig | None = None,
        feature_builder: FeatureBuilder | None = None,
        max_revisit_slots: int | None = None,
        redundancy_tau_slots: float = 5.0,
        seed: int = 0,
        freeze_belief: bool = False,
    ) -> None:
        if n_channels < 1:
            raise ValueError(f"n_channels must be >= 1, got {n_channels}")
        self._n = n_channels
        self._predictor = predictor
        self._freeze_belief = freeze_belief
        self._weights = weights or SchedulerWeights()
        self._belief_config = belief_config or BeliefConfig()
        self._seed = int(seed)
        self._feature_builder = feature_builder or FeatureBuilder(
            beta_prior_alpha=self._belief_config.prior_alpha,
            beta_prior_beta=self._belief_config.prior_beta,
            beta_decay_lambda=self._belief_config.decay_lambda,
        )
        self._belief = BeliefState(
            n_channels,
            prior_alpha=self._belief_config.prior_alpha,
            prior_beta=self._belief_config.prior_beta,
            decay_lambda=self._belief_config.decay_lambda,
            seed=self._seed,
        )
        self._policy = PriorityPolicy(self._weights, redundancy_tau_slots=redundancy_tau_slots)
        self._max_revisit = (
            max_revisit_slots if max_revisit_slots is not None else 3 * n_channels
        )
        self._rng = np.random.default_rng(self._seed)
        self._last: dict[str, float] = {}
        self._last_breakdowns: list[PriorityBreakdown] = []
        self._last_predicted: np.ndarray = np.array([])

    def select_next(self, store: ReadableStore, slot: int) -> int:
        self._belief.decay()
        features = self._feature_builder.build(store, slot)
        predicted = np.asarray(self._predictor.predict_proba(features), dtype=np.float64)
        self._last_predicted = predicted
        uncertainty = self._belief.std()
        staleness = features[:, _STALENESS_COL]
        trend = features[:, _TREND_COL]

        priority, breakdowns = self._policy.score(
            predicted_proba=predicted,
            uncertainty=uncertainty,
            staleness=staleness,
            activity_trend=trend,
        )
        self._last_breakdowns = breakdowns

        forced = np.flatnonzero(staleness >= self._max_revisit)
        if forced.size > 0:
            order = np.lexsort((forced, -staleness[forced]))
            choice = int(forced[order[0]])
            forced_flag = 1.0
        else:
            temperature = self._weights.softmax_temperature
            if temperature is None:
                order = np.lexsort((np.arange(self._n), -staleness, -priority))
                choice = int(order[0])
            else:
                logits = priority / temperature
                logits = logits - logits.max()
                probs = np.exp(logits)
                probs /= probs.sum()
                choice = int(self._rng.choice(self._n, p=probs))
            forced_flag = 0.0

        info = breakdowns[choice].to_dict()
        info["channel"] = float(choice)
        info["forced_freshness"] = forced_flag
        self._last = info
        return choice

    def update(self, record: ScanRecord) -> None:
        if not self._freeze_belief:
            self._belief.update(record.channel_index, record.detected)

    def explain(self) -> Mapping[str, float]:
        return dict(self._last)

    def all_breakdowns(self) -> list[PriorityBreakdown]:
        """Every channel's :class:`PriorityBreakdown` from the most recent
        :meth:`select_next` call, not just the one that was chosen.

        ``explain()`` (the ``Scheduler`` protocol) only ever reports the
        chosen channel, by design (S10's stacked-bar panel shows one
        decision). This is Phase 7's read-only introspection extension for
        answering "what was channel *c*'s priority right now, even though it
        wasn't picked" -- e.g. tracking a quiet channel's priority before it
        has enough evidence to win argmax. Zero extra cost: ``PriorityPolicy.
        score`` already computes every channel's breakdown on the hot path;
        this just keeps the reference instead of discarding it.
        """
        return list(self._last_breakdowns)

    def belief_snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        """``(mean, std)`` arrays, shape ``(n_channels,)``, of the belief
        layer's current state (after the last :meth:`select_next`'s
        ``decay()`` and any prior :meth:`update`). Read-only introspection,
        same rationale as :meth:`all_breakdowns`."""
        return self._belief.mean(), self._belief.std()

    def predicted_proba_snapshot(self) -> np.ndarray:
        """The raw ``Predictor.predict_proba`` output from the most recent
        :meth:`select_next` call, shape ``(n_channels,)`` -- e.g. for a "Predicted
        activity: NN%" display. Deliberately separate from
        ``PriorityBreakdown.pred`` (``w_pred * predicted_proba``, S7): reading
        it back out here is correct regardless of what ``w_pred`` happens to
        be, rather than relying on today's default of 1.0. Read-only
        introspection, same rationale as :meth:`all_breakdowns`; changes
        nothing about the decision itself."""
        return self._last_predicted.copy()

    def reset(self) -> None:
        self._belief.reset()
        self._rng = np.random.default_rng(self._seed)
        self._last = {}
        self._last_breakdowns = []
        self._last_predicted = np.array([])
