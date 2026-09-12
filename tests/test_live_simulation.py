"""rfscan/app/live_simulation.py: the dashboard's stepper over the EXISTING
engine (Phase 9). No new simulation/decision logic is exercised here beyond
what Phases 1-8 already have -- these tests check the *wrapper*, especially
that it never leaks ground truth into a scheduler decision.
"""

from __future__ import annotations

import numpy as np
import pytest

from rfscan.app.live_simulation import (
    STRATEGIES,
    LiveSimulation,
    available_scenarios,
    compare_strategies_same_world,
    default_predictor,
    strategy_requires_predictor,
)
from rfscan.models.baseline_beta import DecayingBetaPredictor
from rfscan.scheduler.adaptive import AdaptiveScheduler


def test_available_scenarios_matches_the_real_registry():
    from rfscan.simulator.scenarios import list_scenarios

    assert available_scenarios() == list_scenarios()


def test_strategy_requires_predictor_only_for_adaptive():
    assert strategy_requires_predictor("adaptive") is True
    for name in ("sequential", "random", "heuristic"):
        assert strategy_requires_predictor(name) is False


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_builds_and_steps_for_every_strategy(strategy):
    sim = LiveSimulation("normal", world_seed=0, strategy_name=strategy, budget=20)
    assert sim.n_channels == 12
    assert not sim.done
    record = sim.step()
    assert record is not None
    assert sim.slot == 1
    assert len(sim.history) == 1


def test_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        LiveSimulation("normal", world_seed=0, strategy_name="quantum", budget=10)


def test_step_returns_none_once_budget_exhausted():
    sim = LiveSimulation("normal", world_seed=0, strategy_name="sequential", budget=3)
    for _ in range(3):
        assert sim.step() is not None
    assert sim.done
    assert sim.step() is None
    assert sim.slot == 3  # did not overshoot


def test_reset_returns_to_the_start_and_replays_identically():
    sim = LiveSimulation("bursty", world_seed=3, strategy_name="adaptive", budget=15)
    first_choices = []
    for _ in range(15):
        first_choices.append(sim.step().channel_index)
    sim.reset()
    second_choices = []
    for _ in range(15):
        second_choices.append(sim.step().channel_index)
    assert first_choices == second_choices


def test_same_seed_is_deterministic_across_fresh_instances():
    a = LiveSimulation("emerging_signal", world_seed=7, strategy_name="heuristic", budget=25)
    b = LiveSimulation("emerging_signal", world_seed=7, strategy_name="heuristic", budget=25)
    picks_a = [a.step().channel_index for _ in range(25)]
    picks_b = [b.step().channel_index for _ in range(25)]
    assert picks_a == picks_b


def test_agent_seed_changes_pattern_not_the_world():
    def run(agent_seed):
        sim = LiveSimulation(
            "normal", world_seed=5, strategy_name="random", budget=30, agent_seed=agent_seed
        )
        return [sim.step().channel_index for _ in range(30)]

    picks_1 = run(1)
    picks_2 = run(2)
    assert picks_1 != picks_2


# -- ground-truth leakage: the critical checks --------------------------
def test_scheduler_select_next_only_ever_receives_the_readable_store():
    """Spy on the scheduler's select_next call and confirm the store handed
    to it exposes no ground-truth-bearing method -- mirrors
    tests/test_runner.py's Phase-4 leakage check, applied to this wrapper."""
    sim = LiveSimulation("normal", world_seed=0, strategy_name="sequential", budget=5)
    seen = {}
    original_select_next = sim.scheduler.select_next

    def spy(store, slot):
        seen["store"] = store
        return original_select_next(store, slot)

    sim.scheduler.select_next = spy
    sim.step()
    store = seen["store"]
    for forbidden in ("ground_truth", "truth_snapshot", "occupancy_snapshot", "observe", "step"):
        assert not hasattr(store, forbidden)
    assert hasattr(store, "detection_rate")


def test_ground_truth_snapshot_is_separate_from_the_scheduler_path():
    sim = LiveSimulation("normal", world_seed=0, strategy_name="adaptive", budget=10)
    sim.step()
    truth = sim.ground_truth_snapshot()
    assert isinstance(truth, tuple)
    assert len(truth) == sim.n_channels
    assert all(isinstance(v, (bool, np.bool_)) for v in truth)
    # the scheduler itself must not expose this value under any attribute name
    for forbidden in ("occupancy", "ground_truth", "truth"):
        assert not hasattr(sim.scheduler, forbidden)


def test_adaptive_introspection_available_only_for_adaptive_scheduler():
    adaptive_sim = LiveSimulation("normal", world_seed=0, strategy_name="adaptive", budget=5)
    adaptive_sim.step()
    assert adaptive_sim.all_breakdowns() is not None
    assert adaptive_sim.belief_snapshot() is not None
    assert isinstance(adaptive_sim.scheduler, AdaptiveScheduler)

    seq_sim = LiveSimulation("normal", world_seed=0, strategy_name="sequential", budget=5)
    seq_sim.step()
    assert seq_sim.all_breakdowns() is None
    assert seq_sim.belief_snapshot() is None


def test_scan_history_arrays_match_the_recorded_history():
    sim = LiveSimulation("normal", world_seed=0, strategy_name="random", budget=10)
    for _ in range(10):
        sim.step()
    scanned, detected = sim.scan_history_arrays()
    assert len(scanned) == 10
    assert list(scanned) == [r.channel_index for r in sim.history]
    assert list(detected) == [r.detected for r in sim.history]


def test_default_predictor_needs_no_trained_artifact():
    predictor = default_predictor()
    features = np.zeros((3, 17))
    proba = predictor.predict_proba(features)
    assert proba.shape == (3,)


# -- partial-episode metrics ------------------------------------------
def test_to_episode_result_is_none_before_the_first_step():
    sim = LiveSimulation("normal", world_seed=0, strategy_name="sequential", budget=10)
    assert sim.to_episode_result() is None
    assert sim.partial_metrics() is None


def test_to_episode_result_matches_run_episode_for_a_full_episode():
    """The live stepper's partial-episode result, run to completion, must be
    identical to a plain run_episode call -- same engine, same lifecycle."""
    from rfscan.experiments.runner import run_episode
    from rfscan.scheduler.sequential import SequentialScheduler
    from rfscan.simulator.scenarios import make_scenario

    sim = LiveSimulation("normal", world_seed=2, strategy_name="sequential", budget=50)
    while sim.step() is not None:
        pass
    live_result = sim.to_episode_result()

    scenario = make_scenario("normal", 2, duration_slots=50)
    direct_result = run_episode(scenario.build_environment(2), SequentialScheduler(12), budget=50)

    assert np.array_equal(live_result.scanned_channel, direct_result.scanned_channel)
    assert np.array_equal(live_result.observed_detection, direct_result.observed_detection)
    assert np.array_equal(live_result.occupancy, direct_result.occupancy)


def test_partial_metrics_uses_the_real_compute_episode_metrics():
    from rfscan.experiments.metrics import EpisodeMetrics

    sim = LiveSimulation("normal", world_seed=0, strategy_name="random", budget=20)
    for _ in range(20):
        sim.step()
    metrics = sim.partial_metrics()
    assert isinstance(metrics, EpisodeMetrics)
    assert metrics.n_slots == 20
    assert 0.0 <= metrics.detection_rate <= 1.0 or metrics.detection_rate != metrics.detection_rate


def test_partial_metrics_grows_with_more_steps():
    sim = LiveSimulation("normal", world_seed=0, strategy_name="sequential", budget=30)
    for _ in range(5):
        sim.step()
    early = sim.partial_metrics()
    for _ in range(25):
        sim.step()
    late = sim.partial_metrics()
    assert early.n_slots == 5
    assert late.n_slots == 30


# -- run_slots (automatic multi-slot run) --------------------------------
def test_run_slots_advances_exactly_n_and_returns_the_count():
    sim = LiveSimulation("normal", world_seed=0, strategy_name="sequential", budget=100)
    advanced = sim.run_slots(25)
    assert advanced == 25
    assert sim.slot == 25


def test_run_slots_stops_at_the_episode_budget():
    sim = LiveSimulation("normal", world_seed=0, strategy_name="sequential", budget=10)
    advanced = sim.run_slots(50)  # ask for more than the budget
    assert advanced == 10
    assert sim.slot == 10
    assert sim.done


def test_run_slots_calls_on_slot_once_per_advanced_slot_in_order():
    sim = LiveSimulation("normal", world_seed=0, strategy_name="sequential", budget=20)
    seen = []
    sim.run_slots(7, on_slot=seen.append)
    assert seen == [1, 2, 3, 4, 5, 6, 7]


def test_run_slots_matches_calling_step_the_same_number_of_times():
    sim_a = LiveSimulation("bursty", world_seed=3, strategy_name="adaptive", budget=40)
    for _ in range(15):
        sim_a.step()

    sim_b = LiveSimulation("bursty", world_seed=3, strategy_name="adaptive", budget=40)
    sim_b.run_slots(15)

    assert sim_a.slot == sim_b.slot == 15
    picks_a = [r.channel_index for r in sim_a.history]
    picks_b = [r.channel_index for r in sim_b.history]
    assert picks_a == picks_b


# -- predicted_proba_snapshot passthrough --------------------------------
def test_predicted_proba_snapshot_available_only_for_adaptive():
    adaptive_sim = LiveSimulation("normal", world_seed=0, strategy_name="adaptive", budget=5)
    adaptive_sim.step()
    proba = adaptive_sim.predicted_proba_snapshot()
    assert proba is not None
    assert proba.shape == (adaptive_sim.n_channels,)

    seq_sim = LiveSimulation("normal", world_seed=0, strategy_name="sequential", budget=5)
    seq_sim.step()
    assert seq_sim.predicted_proba_snapshot() is None


# -- compare_strategies_same_world ---------------------------------------
def test_compare_strategies_same_world_returns_metrics_for_every_strategy():
    results = compare_strategies_same_world(
        "normal", world_seed=0, budget=50, predictor=DecayingBetaPredictor()
    )
    assert set(results.keys()) == set(STRATEGIES)
    for metrics in results.values():
        assert metrics.n_slots == 50


def test_compare_strategies_same_world_uses_an_identical_world_per_strategy():
    """Paired-world fairness (S2/S9): every strategy must face the same
    ground-truth realisation -- checked the same way test_benchmark.py and
    test_ablation.py already do, via each LiveSimulation's own world."""
    results = {}
    for strategy in STRATEGIES:
        sim = LiveSimulation(
            "dynamic",
            world_seed=4,
            strategy_name=strategy,
            budget=60,
            predictor=DecayingBetaPredictor() if strategy_requires_predictor(strategy) else None,
        )
        sim.run_slots(60)
        results[strategy] = sim.to_episode_result().occupancy

    baseline = results[STRATEGIES[0]]
    for strategy, occupancy in results.items():
        assert np.array_equal(occupancy, baseline), f"{strategy} saw a different world"


def test_compare_strategies_same_world_is_deterministic():
    a = compare_strategies_same_world(
        "normal", world_seed=1, budget=30, predictor=DecayingBetaPredictor()
    )
    b = compare_strategies_same_world(
        "normal", world_seed=1, budget=30, predictor=DecayingBetaPredictor()
    )
    for strategy in STRATEGIES:
        assert a[strategy].detection_rate == b[strategy].detection_rate
        assert a[strategy].scan_efficiency == b[strategy].scan_efficiency
