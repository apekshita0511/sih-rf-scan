"""experiments/adaptation.py: emerging-signal pre/post metrics + priority
tracing (Phase 7)."""

from __future__ import annotations

import numpy as np

from rfscan.experiments.adaptation import (
    emerging_adaptation_metrics,
    trace_adaptive_episode,
)
from rfscan.experiments.metrics import compute_episode_metrics
from rfscan.experiments.runner import run_episode
from rfscan.models.baseline_beta import DecayingBetaPredictor
from rfscan.scheduler.adaptive import AdaptiveScheduler
from rfscan.scheduler.sequential import SequentialScheduler
from rfscan.simulator.scenarios import make_scenario


def test_returns_none_for_a_scenario_with_no_emerging_channel():
    scenario = make_scenario("normal", 0, duration_slots=200)
    result = run_episode(scenario.build_environment(0), SequentialScheduler(12), budget=200)
    assert emerging_adaptation_metrics(result) is None


def test_matches_the_known_emerging_signal_scenario_metadata():
    scenario = make_scenario("emerging_signal", 0, duration_slots=350)
    result = run_episode(scenario.build_environment(0), SequentialScheduler(12), budget=350)
    metrics = emerging_adaptation_metrics(result)
    assert metrics is not None
    assert metrics.channel == 5
    assert metrics.activation_slot == 300


def test_reuses_compute_episode_metrics_discovery_figures_exactly():
    scenario = make_scenario("emerging_signal", 1, duration_slots=350)
    result = run_episode(scenario.build_environment(1), SequentialScheduler(12), budget=350)
    episode_metrics = compute_episode_metrics(result)
    adaptation_metrics = emerging_adaptation_metrics(result, episode_metrics)
    expected_delay = episode_metrics.emerging_discovery_delay_slots
    assert adaptation_metrics.discovered == (expected_delay is not None)
    assert adaptation_metrics.discovery_delay_slots == expected_delay


def test_before_after_counts_are_internally_consistent():
    scenario = make_scenario("emerging_signal", 2, duration_slots=350)
    result = run_episode(scenario.build_environment(2), SequentialScheduler(12), budget=350)
    metrics = emerging_adaptation_metrics(result)
    channel = metrics.channel
    on_channel_scans = int(np.count_nonzero(result.scanned_channel == channel))
    assert metrics.scans_before_activation + metrics.scans_after_activation == on_channel_scans
    if metrics.scans_before_activation:
        assert metrics.detection_rate_before_activation == (
            metrics.detections_before_activation / metrics.scans_before_activation
        )
    else:
        assert metrics.detection_rate_before_activation is None
    if metrics.scans_after_activation:
        assert metrics.detection_rate_after_activation == (
            metrics.detections_after_activation / metrics.scans_after_activation
        )
    else:
        assert metrics.detection_rate_after_activation is None


def test_before_activation_scans_never_hit_since_the_emitter_is_forced_off():
    # EMERGING behaviour is forced OFF before activation_slot (S16.2) -- a
    # scan of that channel pre-activation can only be a false alarm, never a
    # true positive, so detections_before_activation should be rare/zero
    # across several seeds, never systematically high.
    for seed in range(3):
        scenario = make_scenario("emerging_signal", seed, duration_slots=350)
        result = run_episode(scenario.build_environment(seed), SequentialScheduler(12), budget=350)
        metrics = emerging_adaptation_metrics(result)
        if metrics.scans_before_activation:
            assert metrics.detections_before_activation <= metrics.scans_before_activation


def test_trace_shapes_and_no_gaps():
    scenario = make_scenario("normal", 0, duration_slots=100)
    env = scenario.build_environment(0)
    sched = AdaptiveScheduler(env.n_channels, DecayingBetaPredictor(), seed=0)
    result, trace = trace_adaptive_episode(env, sched, budget=100, tracked_channels=(2, 5))
    assert result.n_slots == 100
    assert trace.channels == (2, 5)
    for arr in (trace.priority, trace.belief_mean, trace.belief_std):
        assert arr.shape == (100, 2)
        assert not np.isnan(arr).any()  # on_slot fires every slot -- no gaps


def test_trace_final_row_matches_the_schedulers_final_state():
    scenario = make_scenario("bursty", 3, duration_slots=80)
    env = scenario.build_environment(3)
    sched = AdaptiveScheduler(env.n_channels, DecayingBetaPredictor(), seed=3)
    _, trace = trace_adaptive_episode(env, sched, budget=80, tracked_channels=(0, 1, 2))
    final_mean, final_std = sched.belief_snapshot()
    for i, c in enumerate((0, 1, 2)):
        assert trace.belief_mean[-1, i] == final_mean[c]
        assert trace.belief_std[-1, i] == final_std[c]


def test_trace_does_not_change_the_episode_versus_a_plain_run():
    scenario = make_scenario("normal", 5, duration_slots=120)

    env_plain = scenario.build_environment(5)
    sched_plain = AdaptiveScheduler(env_plain.n_channels, DecayingBetaPredictor(), seed=5)
    plain = run_episode(env_plain, sched_plain, budget=120)

    env_traced = scenario.build_environment(5)
    sched_traced = AdaptiveScheduler(env_traced.n_channels, DecayingBetaPredictor(), seed=5)
    traced, _ = trace_adaptive_episode(env_traced, sched_traced, budget=120, tracked_channels=(0,))

    assert np.array_equal(plain.scanned_channel, traced.scanned_channel)
    assert np.array_equal(plain.occupancy, traced.occupancy)
