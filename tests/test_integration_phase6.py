"""Phase 6 exit criteria (docs/architecture.md S11): the AdaptiveScheduler runs
the full closed loop end-to-end, meets the latency budget, and beats random
scanning on detection delay.

Uses :class:`DecayingBetaPredictor` (no trained-artifact dependency) so this
test is self-contained and does not require ``rfscan train`` to have been run
first -- exactly the same reason the Phase 5 tests avoid needing
``artifacts/model.joblib``.
"""

from __future__ import annotations

from rfscan.experiments.metrics import compute_episode_metrics
from rfscan.experiments.runner import run_episode
from rfscan.models.baseline_beta import DecayingBetaPredictor
from rfscan.scheduler.adaptive import AdaptiveScheduler
from rfscan.scheduler.sequential import RandomScheduler
from rfscan.simulator.emitters import Behavior, EmitterSpec
from rfscan.simulator.scenario import ScenarioConfig
from rfscan.simulator.scenarios import make_scenario


def test_adaptive_full_loop_runs_end_to_end_and_meets_latency_budget():
    scenario = make_scenario("normal", 0, duration_slots=400)
    env = scenario.build_environment(0)
    sched = AdaptiveScheduler(env.n_channels, DecayingBetaPredictor(), seed=0)
    result = run_episode(env, sched, budget=400, agent_seed=0)
    assert result.scanned_channel.shape == (400,)
    metrics = compute_episode_metrics(result)
    assert metrics.mean_decision_latency_ms < 5.0


def _one_hot_scenario(seed: int) -> ScenarioConfig:
    return ScenarioConfig(
        name="one_hot_adaptive",
        seed=seed,
        n_channels=10,
        duration_slots=800,
        emitters=[
            EmitterSpec(6, Behavior.PERSISTENT, {"p01": 0.12, "p10": 0.02, "signal_dbm": -57.0}),
        ],
    )


def test_adaptive_beats_random_on_detection_delay_and_scan_efficiency():
    scenario = _one_hot_scenario(1)
    predictor = DecayingBetaPredictor()

    env_a = scenario.build_environment(1)
    adaptive = AdaptiveScheduler(env_a.n_channels, predictor, seed=1)
    result_a = run_episode(env_a, adaptive, budget=scenario.duration_slots, agent_seed=1)
    metrics_a = compute_episode_metrics(result_a)

    env_r = scenario.build_environment(1)
    random_sched = RandomScheduler(env_r.n_channels, seed=1)
    result_r = run_episode(env_r, random_sched, budget=scenario.duration_slots, agent_seed=1)
    metrics_r = compute_episode_metrics(result_r)

    assert metrics_a.mean_detection_delay_slots is not None
    assert metrics_r.mean_detection_delay_slots is not None
    assert metrics_a.mean_detection_delay_slots <= metrics_r.mean_detection_delay_slots
    assert metrics_a.scan_efficiency > metrics_r.scan_efficiency
    assert metrics_a.on_target_scan_rate > metrics_r.on_target_scan_rate


def test_adaptive_scan_pattern_does_not_perturb_the_rf_world():
    scenario = make_scenario("bursty", 5, duration_slots=200)
    env_a = scenario.build_environment(5)
    env_r = scenario.build_environment(5)
    result_a = run_episode(
        env_a, AdaptiveScheduler(env_a.n_channels, DecayingBetaPredictor(), seed=5), budget=200
    )
    result_r = run_episode(env_r, RandomScheduler(env_r.n_channels, seed=9), budget=200)
    assert (result_a.occupancy == result_r.occupancy).all()
