"""experiments/ablation.py: component-isolation ablation grid (Phase 8)."""

from __future__ import annotations

import numpy as np

from rfscan.experiments.ablation import (
    ABLATION_VARIANTS,
    build_ablation_scheduler,
    run_ablation,
    summarize_ablation,
)
from rfscan.models.baseline_beta import DecayingBetaPredictor
from rfscan.scheduler.adaptive import AdaptiveScheduler


def test_variants_progressively_enable_one_more_term():
    by_name = {v.name: v for v in ABLATION_VARIANTS}
    a = by_name["A_prediction_only"]
    b = by_name["B_prediction_exploration"]
    c = by_name["C_prediction_exploration_freshness"]
    d = by_name["D_prediction_exploration_freshness_trend"]
    e = by_name["E_full_policy"]

    assert a.weights.w_pred > 0 and a.weights.w_explore == 0
    assert a.weights.w_fresh == 0 and a.weights.w_trend == 0 and a.weights.w_redundancy == 0

    assert b.weights.w_explore > 0
    assert b.weights.w_fresh == 0 and b.weights.w_trend == 0

    assert c.weights.w_fresh > 0
    assert c.weights.w_trend == 0

    assert d.weights.w_trend > 0
    assert d.weights.w_redundancy == 0

    assert e.weights.w_redundancy > 0
    # E is the actual default configuration
    from rfscan.config import SchedulerWeights

    assert e.weights == SchedulerWeights()


def test_variant_f_shares_full_weights_but_freezes_belief():
    by_name = {v.name: v for v in ABLATION_VARIANTS}
    e, f = by_name["E_full_policy"], by_name["F_full_policy_no_online_feedback"]
    assert e.weights == f.weights
    assert e.freeze_belief is False
    assert f.freeze_belief is True


def test_build_ablation_scheduler_wires_freeze_belief_through():
    by_name = {v.name: v for v in ABLATION_VARIANTS}
    f = by_name["F_full_policy_no_online_feedback"]
    sched = build_ablation_scheduler(f, 4, DecayingBetaPredictor(), seed=0)
    assert isinstance(sched, AdaptiveScheduler)
    assert sched._freeze_belief is True  # internal, but this is exactly what we're testing


def test_freeze_belief_variant_never_moves_belief_away_from_prior():
    """Direct proof that variant F's belief truly never learns, across a
    real closed-loop run (not just at construction time)."""
    from rfscan.experiments.adaptation import trace_adaptive_episode
    from rfscan.simulator.scenarios import make_scenario

    by_name = {v.name: v for v in ABLATION_VARIANTS}
    f = by_name["F_full_policy_no_online_feedback"]
    scenario = make_scenario("normal", 0, duration_slots=200)
    env = scenario.build_environment(0)
    sched = build_ablation_scheduler(f, env.n_channels, DecayingBetaPredictor(), seed=0)
    _, trace = trace_adaptive_episode(env, sched, budget=200, tracked_channels=(0, 1, 2))
    assert np.allclose(trace.belief_mean, 0.5)  # Beta(1,1) prior, forever


def test_run_ablation_produces_one_row_per_cell_with_variant_column():
    predictor = DecayingBetaPredictor()
    raw = run_ablation(
        predictor,
        scenarios=("normal", "bursty"),
        world_seeds=(0, 1),
        duration_slots=100,
    )
    assert len(raw) == 2 * 2 * len(ABLATION_VARIANTS)
    assert set(raw["variant"]) == {v.name for v in ABLATION_VARIANTS}
    assert set(raw["strategy"]) == {"adaptive"}
    assert set(raw["scenario"]) == {"normal", "bursty"}


def test_run_ablation_is_reproducible():
    predictor = DecayingBetaPredictor()
    kwargs = dict(scenarios=("normal",), world_seeds=(0,), duration_slots=100)
    a = run_ablation(predictor, **kwargs).drop(columns=["mean_decision_latency_ms"])
    b = run_ablation(predictor, **kwargs).drop(columns=["mean_decision_latency_ms"])
    assert a.equals(b)


def test_all_variants_face_an_identical_world_per_seed():
    """The ablation's fairness invariant: only the scheduler's weights (and
    freeze_belief) change across variants -- the RF world must not."""
    from rfscan.experiments.runner import run_episode
    from rfscan.simulator.scenarios import make_scenario

    predictor = DecayingBetaPredictor()
    scenario = make_scenario("dynamic", 4, duration_slots=150)
    occupancies = []
    for variant in ABLATION_VARIANTS:
        env = scenario.build_environment(4)
        sched = build_ablation_scheduler(variant, env.n_channels, predictor, seed=4)
        occupancies.append(run_episode(env, sched, budget=150).occupancy)
    for occ in occupancies[1:]:
        assert np.array_equal(occupancies[0], occ)


def test_summarize_ablation_groups_by_scenario_and_variant():
    predictor = DecayingBetaPredictor()
    raw = run_ablation(predictor, scenarios=("normal",), world_seeds=(0, 1), duration_slots=100)
    summary = summarize_ablation(raw)
    assert set(summary.columns) >= {"scenario", "variant", "detection_rate_mean"}
    assert len(summary) == len(ABLATION_VARIANTS)
