"""AdaptiveScheduler v1: belief + predictor + priority policy, wired (Phase 6)."""

from __future__ import annotations

import numpy as np
import pytest

from rfscan.config import SchedulerWeights
from rfscan.models.baseline_beta import DecayingBetaPredictor
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


# -- Phase 7: online feedback (all_breakdowns / belief_snapshot) ----
def test_all_breakdowns_covers_every_channel_not_just_the_chosen_one():
    predictor = _PerChannelPredictor([0.2, 0.7, 0.1])
    sched = AdaptiveScheduler(3, predictor, seed=0)
    store = ObservationStore(3)
    for c in range(3):
        store.append(_rec(0, c, detected=False))
    choice = sched.select_next(store, 5)
    breakdowns = sched.all_breakdowns()
    assert len(breakdowns) == 3
    assert breakdowns[choice].priority == pytest.approx(sched.explain()["priority"])


def test_all_breakdowns_is_empty_before_the_first_decision():
    sched = AdaptiveScheduler(3, _ConstantPredictor(0.0), seed=0)
    assert sched.all_breakdowns() == []


def test_belief_snapshot_shapes_and_prior_start():
    sched = AdaptiveScheduler(4, _ConstantPredictor(0.0), seed=0)
    mean, std = sched.belief_snapshot()
    assert mean.shape == (4,) and std.shape == (4,)
    assert np.allclose(mean, 0.5)  # Beta(1,1) prior


def test_a_quiet_channels_priority_rises_after_a_hit():
    """The central Phase 7 story: a channel with no evidence sits at a low
    priority; one HIT on it, with nothing else changing, raises both its
    belief mean and its priority for the very next decision -- proving the
    scheduler's next choice can change because of new evidence, not just that
    BeliefState's math can (already covered in test_belief.py).

    Needs a predictor whose ``pred`` term actually reacts to the hit (through
    FeatureBuilder's fresh per-slot features) -- DecayingBetaPredictor reads
    exactly that. A predictor that ignores its input (e.g. a fixed constant)
    defeats this by design: with pred/trend/fresh/redundancy all pinned equal,
    the only term left to move is ``explore`` (belief std), and more evidence
    lowers uncertainty rather than raising it -- so priority can go *down*
    after a hit under a predictor that cannot see the hit. That's a real,
    non-obvious property of the priority formula (S7), not a bug; documented
    in docs/architecture.md S16.7.
    """
    predictor = DecayingBetaPredictor()
    sched = AdaptiveScheduler(3, predictor, seed=0)
    store = ObservationStore(3)

    # Warm up: every channel scanned once, all empty, so none is "never
    # scanned" (which would trigger the hard freshness guarantee and swamp
    # the effect we're isolating).
    for c in range(3):
        rec = _rec(c, c, detected=False)
        store.append(rec)
        sched.update(rec)

    tracked = 2
    sched.select_next(store, 3)  # settle belief/priority state post-warm-up
    before_mean = sched.belief_snapshot()[0][tracked]
    before_priority = sched.all_breakdowns()[tracked].priority

    hit = _rec(4, tracked, detected=True)
    store.append(hit)
    sched.update(hit)

    sched.select_next(store, 5)
    after_mean = sched.belief_snapshot()[0][tracked]
    after_priority = sched.all_breakdowns()[tracked].priority

    assert after_mean > before_mean
    assert after_priority > before_priority


def test_scheduler_decisions_respond_to_online_feedback():
    """Two schedulers, identical everything, diverge only in whether channel 2
    got a HIT. With that HIT, its priority should be higher than in the no-hit
    twin -- the difference comes purely from the new evidence, not from
    ground truth or different config. Uses DecayingBetaPredictor, same
    reasoning as the test above (needs a predictor whose pred term can
    actually see the hit)."""
    predictor = DecayingBetaPredictor()

    def _build(hit_channel_2: bool) -> tuple[AdaptiveScheduler, ObservationStore]:
        sched = AdaptiveScheduler(3, predictor, seed=0)
        store = ObservationStore(3)
        for c in range(3):
            rec = _rec(c, c, detected=(hit_channel_2 and c == 2))
            store.append(rec)
            sched.update(rec)
        return sched, store

    sched_no_hit, store_no_hit = _build(hit_channel_2=False)
    sched_hit, store_hit = _build(hit_channel_2=True)

    sched_no_hit.select_next(store_no_hit, 3)
    sched_hit.select_next(store_hit, 3)

    priority_no_hit = sched_no_hit.all_breakdowns()[2].priority
    priority_hit = sched_hit.all_breakdowns()[2].priority
    assert priority_hit > priority_no_hit
