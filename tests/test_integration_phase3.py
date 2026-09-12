"""End-to-end wiring for Phase 3: environment + scanner + store + scheduler.

The formal ExperimentRunner is Phase 4; this is the minimal loop it will use,
exercised here to prove the pieces compose and the fairness invariant holds at
the scheduler level.
"""

from __future__ import annotations

import numpy as np

from rfscan.perception.scanner import Scanner
from rfscan.scheduler.heuristic import HeuristicScheduler
from rfscan.scheduler.sequential import RandomScheduler, SequentialScheduler
from rfscan.simulator.emitters import Behavior, EmitterSpec
from rfscan.simulator.scenario import ScenarioConfig, example_normal


def _run(scenario, world_seed, scheduler, n_slots):
    env = scenario.build_environment(world_seed)
    scanner = Scanner(env)
    truth = np.empty((n_slots, env.n_channels), dtype=bool)
    picks = []
    for slot in range(n_slots):
        c = scheduler.select_next(scanner.store, slot)
        scanner.scan(c, scheduler.explain())
        scheduler.update(scanner.store.records[-1])
        truth[slot] = [env.ground_truth(k).occupied for k in range(env.n_channels)]
        env.step()
        picks.append(c)
    return env, scanner, truth, picks


def test_sequential_gives_full_even_coverage():
    sc = example_normal()  # 6 channels
    _, scanner, _, _ = _run(sc, 0, SequentialScheduler(sc.n_channels), 240)
    counts = [scanner.store.scan_count(c) for c in range(sc.n_channels)]
    assert scanner.store.coverage() == sc.n_channels
    assert counts == [40] * sc.n_channels  # 240 / 6, exactly


def test_scheduler_choices_do_not_change_the_rf_world():
    sc = example_normal()
    _, _, truth_seq, _ = _run(sc, 99, SequentialScheduler(sc.n_channels), 300)
    _, _, truth_rand, _ = _run(sc, 99, RandomScheduler(sc.n_channels, seed=5), 300)
    _, _, truth_heur, _ = _run(sc, 99, HeuristicScheduler(sc.n_channels, seed=5), 300)
    assert np.array_equal(truth_seq, truth_rand)
    assert np.array_equal(truth_seq, truth_heur)


def test_agent_seed_changes_scan_pattern_not_world():
    sc = example_normal()
    _, _, truth_a, picks_a = _run(sc, 7, RandomScheduler(sc.n_channels, seed=1), 300)
    _, _, truth_b, picks_b = _run(sc, 7, RandomScheduler(sc.n_channels, seed=2), 300)
    assert picks_a != picks_b
    assert np.array_equal(truth_a, truth_b)


def test_heuristic_concentrates_scans_on_the_active_channel():
    # one strong persistent emitter on ch3, everything else quiet
    sc = ScenarioConfig(
        name="one_hot",
        seed=0,
        n_channels=8,
        duration_slots=600,
        emitters=[
            EmitterSpec(3, Behavior.PERSISTENT, {"p01": 0.15, "p10": 0.02, "signal_dbm": -55.0}),
        ],
    )
    _, scanner, _, _ = _run(sc, 0, HeuristicScheduler(sc.n_channels, epsilon=0.1, seed=0), 500)
    counts = [scanner.store.scan_count(c) for c in range(sc.n_channels)]
    uniform_share = 500 / sc.n_channels
    assert counts[3] > 3 * uniform_share
    assert np.argmax(counts) == 3
    assert np.argmax([scanner.store.detection_count(c) for c in range(sc.n_channels)]) == 3


def test_heuristic_beats_uniform_scan_at_finding_the_active_channel():
    sc = ScenarioConfig(
        name="one_hot",
        seed=0,
        n_channels=10,
        duration_slots=800,
        emitters=[
            EmitterSpec(6, Behavior.PERSISTENT, {"p01": 0.12, "p10": 0.02, "signal_dbm": -57.0}),
        ],
    )
    _, heur, _, _ = _run(sc, 1, HeuristicScheduler(sc.n_channels, epsilon=0.1, seed=1), 600)
    _, rand, _, _ = _run(sc, 1, RandomScheduler(sc.n_channels, seed=1), 600)
    # useful detections per scan
    heur_eff = sum(heur.store.detection_count(c) for c in range(sc.n_channels)) / len(heur.store)
    rand_eff = sum(rand.store.detection_count(c) for c in range(sc.n_channels)) / len(rand.store)
    assert heur_eff > rand_eff
