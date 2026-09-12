"""AdaptiveScheduler v1: belief + predictor + priority policy, wired (Phase 6)."""

from __future__ import annotations

import numpy as np
import pytest

from rfscan.config import SchedulerWeights
from rfscan.perception.schema import Observation, ScanRecord
from rfscan.perception.store import ObservationStore
from rfscan.scheduler.adaptive import AdaptiveScheduler
from rfscan.scheduler.base import Scheduler


class _ConstantPredictor:
    name = "const"

    def __init__(self, value: float = 0.0) -> None:
        self._value = value

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        n = np.atleast_2d(features).shape[0]
        return np.full(n, self._value, dtype=np.float64)

    def explain(self, x: np.ndarray) -> dict:
        return {"predicted_proba": self._value}


class _PerChannelPredictor:
    name = "per_channel"

    def __init__(self, values: list[float]) -> None:
        self._values = np.asarray(values, dtype=np.float64)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return self._values.copy()

    def explain(self, x: np.ndarray) -> dict:
        return {"predicted_proba": 0.0}


def _rec(slot: int, channel: int, detected: bool) -> ScanRecord:
    return ScanRecord(Observation(slot, channel, -80.0, -95.0, 15.0, detected))


def test_satisfies_scheduler_protocol():
    sched = AdaptiveScheduler(4, _ConstantPredictor(0.0))
    assert isinstance(sched, Scheduler)
    assert isinstance(sched.name, str)
    store = ObservationStore(4)
    sched.select_next(store, 0)
    assert isinstance(sched.explain(), dict) or hasattr(sched.explain(), "keys")


def test_sweeps_every_channel_before_repeating_from_a_uniform_start():
    sched = AdaptiveScheduler(6, _ConstantPredictor(0.0), seed=0)
    store = ObservationStore(6)
    picks = []
    for slot in range(6):
        c = sched.select_next(store, slot)
        picks.append(c)
        store.append(_rec(slot, c, detected=False))
    assert sorted(picks) == [0, 1, 2, 3, 4, 5]


def test_fresh_store_prefers_forced_coverage_over_predictor_bias():
    # On a truly empty store every channel is "never scanned" (MAX_STALENESS),
    # which always exceeds the default hard-freshness cap -- by design, initial
    # coverage always wins over a possibly-uninformed predictor at cold start.
    predictor = _PerChannelPredictor([0.1, 0.1, 0.9, 0.1])
    sched = AdaptiveScheduler(4, predictor, seed=0)
    store = ObservationStore(4)
    assert sched.select_next(store, 0) == 0
    assert sched.explain()["forced_freshness"] == 1.0


def test_predictor_bias_breaks_ties_once_every_channel_has_been_scanned_once():
    store = ObservationStore(4)
    for c in range(4):
        store.append(_rec(0, c, detected=False))
    predictor = _PerChannelPredictor([0.1, 0.1, 0.9, 0.1])
    sched = AdaptiveScheduler(4, predictor, seed=0)
    assert sched.select_next(store, 5) == 2
    assert sched.explain()["forced_freshness"] == 0.0


def test_hard_freshness_guarantee_forces_a_never_scanned_channel():
    store = ObservationStore(3)
    for slot in range(10):
        ch = slot % 2  # channels 0/1 alternately scanned; channel 2 never touched
        store.append(_rec(slot, ch, detected=True))
    predictor = _PerChannelPredictor([0.9, 0.9, 0.0])
    sched = AdaptiveScheduler(3, predictor, max_revisit_slots=50, seed=0)
    choice = sched.select_next(store, 10)
    assert choice == 2
    assert sched.explain()["forced_freshness"] == 1.0


def test_without_forcing_predictor_argmax_governs_selection():
    store = ObservationStore(3)
    for slot in range(6):
        ch = slot % 3  # every channel recently scanned -> staleness stays low
        store.append(_rec(slot, ch, detected=False))
    predictor = _PerChannelPredictor([0.1, 0.1, 0.9])
    sched = AdaptiveScheduler(3, predictor, max_revisit_slots=1000, seed=0)
    choice = sched.select_next(store, 6)
    assert choice == 2
    assert sched.explain()["forced_freshness"] == 0.0


def test_explain_reports_the_selected_channel_and_priority_terms():
    predictor = _PerChannelPredictor([0.2, 0.7, 0.1])
    sched = AdaptiveScheduler(3, predictor, seed=0)
    store = ObservationStore(3)
    choice = sched.select_next(store, 0)
    info = sched.explain()
    assert info["channel"] == float(choice)
    expected_keys = {
        "pred", "explore", "fresh", "trend", "redundancy", "priority", "forced_freshness"
    }
    assert expected_keys <= set(info.keys())


def test_update_does_not_raise_and_feeds_the_belief_layer():
    sched = AdaptiveScheduler(2, _ConstantPredictor(0.0), seed=0)
    store = ObservationStore(2)
    for slot in range(10):
        c = sched.select_next(store, slot)
        rec = _rec(slot, c, detected=(c == 0))
        store.append(rec)
        sched.update(rec)  # must not raise


def test_reset_replays_the_same_stream_including_softmax_selection():
    weights = SchedulerWeights(softmax_temperature=0.05)
    predictor = _ConstantPredictor(0.5)

    def _drive(sched, store):
        picks = []
        for slot in range(20):
            c = sched.select_next(store, slot)
            picks.append(c)
            rec = _rec(slot, c, detected=False)
            store.append(rec)
            sched.update(rec)
        return picks

    sched = AdaptiveScheduler(4, predictor, weights=weights, seed=3)
    first = _drive(sched, ObservationStore(4))
    sched.reset()
    second = _drive(sched, ObservationStore(4))
    assert first == second


def test_validates_n_channels():
    with pytest.raises(ValueError):
        AdaptiveScheduler(0, _ConstantPredictor(0.0))
