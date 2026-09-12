"""Baseline schedulers: sequential, random, epsilon-greedy heuristic (Phase 3)."""

from __future__ import annotations

import numpy as np
import pytest

from rfscan.perception.schema import Observation, ScanRecord
from rfscan.perception.store import ObservationStore
from rfscan.scheduler.base import Scheduler
from rfscan.scheduler.heuristic import HeuristicScheduler
from rfscan.scheduler.sequential import RandomScheduler, SequentialScheduler


def _rec(slot: int, channel: int, detected: bool) -> ScanRecord:
    return ScanRecord(
        Observation(slot, channel, -80.0, -95.0, 15.0, detected)
    )


def _drive(scheduler, store: ObservationStore, n: int, start_slot: int = 0) -> list[int]:
    picks = []
    for slot in range(start_slot, start_slot + n):
        picks.append(scheduler.select_next(store, slot))
    return picks


# -- sequential ------------------------------------------------------
def test_sequential_visits_every_channel_in_order_and_wraps():
    sched = SequentialScheduler(5)
    store = ObservationStore(5)
    assert _drive(sched, store, 11) == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0]


def test_sequential_reset_returns_to_start():
    sched = SequentialScheduler(4)
    store = ObservationStore(4)
    _drive(sched, store, 3)
    sched.reset()
    assert _drive(sched, store, 4) == [0, 1, 2, 3]


def test_sequential_custom_start():
    sched = SequentialScheduler(4, start=2)
    store = ObservationStore(4)
    assert _drive(sched, store, 5) == [2, 3, 0, 1, 2]


def test_sequential_validates_args():
    with pytest.raises(ValueError):
        SequentialScheduler(0)
    with pytest.raises(ValueError):
        SequentialScheduler(3, start=3)


# -- random ---------------------------------------------------------
def test_random_is_reproducible_for_a_seed():
    store = ObservationStore(7)
    a = _drive(RandomScheduler(7, seed=123), store, 200)
    b = _drive(RandomScheduler(7, seed=123), store, 200)
    assert a == b


def test_random_differs_across_seeds():
    store = ObservationStore(7)
    a = _drive(RandomScheduler(7, seed=1), store, 200)
    b = _drive(RandomScheduler(7, seed=2), store, 200)
    assert a != b


def test_random_stays_in_range_and_is_roughly_uniform():
    store = ObservationStore(6)
    picks = np.array(_drive(RandomScheduler(6, seed=0), store, 6000))
    assert picks.min() >= 0 and picks.max() < 6
    counts = np.bincount(picks, minlength=6)
    assert counts.min() > 850 and counts.max() < 1150


def test_random_reset_replays_the_same_stream():
    store = ObservationStore(5)
    sched = RandomScheduler(5, seed=42)
    first = _drive(sched, store, 50)
    sched.reset()
    assert _drive(sched, store, 50) == first


# -- heuristic -----------------------------------------------------
def test_heuristic_sweeps_all_channels_before_repeating():
    sched = HeuristicScheduler(6, epsilon=0.0, seed=0)
    store = ObservationStore(6)
    picks = []
    for slot in range(6):
        c = sched.select_next(store, slot)
        picks.append(c)
        store.append(_rec(slot, c, detected=False))
    assert sorted(picks) == [0, 1, 2, 3, 4, 5]


def test_heuristic_exploits_the_best_channel():
    sched = HeuristicScheduler(4, epsilon=0.0, seed=0)
    store = ObservationStore(4)
    # seed history: channel 2 detects often, others never
    for slot in range(4):
        store.append(_rec(slot, slot, detected=(slot == 2)))
    for slot in range(4, 20):
        store.append(_rec(slot, 2, detected=True))
    assert sched.select_next(store, 21) == 2
    info = sched.explain()
    assert info["mode"] == 0.0 and info["channel"] == 2.0


def test_heuristic_epsilon_one_is_pure_exploration_and_reproducible():
    store = ObservationStore(5)
    a = _drive(HeuristicScheduler(5, epsilon=1.0, seed=7), store, 100)
    b = _drive(HeuristicScheduler(5, epsilon=1.0, seed=7), store, 100)
    assert a == b
    assert len(set(a)) == 5  # visits every channel by chance
    assert min(a) >= 0 and max(a) < 5


def test_heuristic_validates_epsilon():
    with pytest.raises(ValueError):
        HeuristicScheduler(4, epsilon=1.5)


def test_heuristic_reset_replays():
    store = ObservationStore(5)
    sched = HeuristicScheduler(5, epsilon=0.5, seed=3)
    first = _drive(sched, store, 40)
    sched.reset()
    assert _drive(sched, store, 40) == first


# -- protocol -----------------------------------------------------
@pytest.mark.parametrize(
    "sched",
    [SequentialScheduler(4), RandomScheduler(4), HeuristicScheduler(4)],
)
def test_schedulers_satisfy_protocol(sched):
    assert isinstance(sched, Scheduler)
    assert isinstance(sched.name, str)
    assert isinstance(sched.explain(), dict) or hasattr(sched.explain(), "keys")
