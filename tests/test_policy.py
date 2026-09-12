"""PriorityPolicy: weighted multi-objective priority score (Phase 6)."""

from __future__ import annotations

import numpy as np
import pytest

from rfscan.config import SchedulerWeights
from rfscan.scheduler.policy import PriorityBreakdown, PriorityPolicy


def _score(policy, **kwargs):
    arrays = {k: np.asarray(v, dtype=np.float64) for k, v in kwargs.items()}
    return policy.score(**arrays)


def test_zero_pred_uncertainty_and_trend_at_zero_staleness_is_pure_redundancy_penalty():
    # staleness=0 + predicted_proba=0 is exactly "just scanned, confidently empty" --
    # the case the redundancy term is designed to penalise (see module docstring).
    weights = SchedulerWeights()
    policy = PriorityPolicy(weights)
    priority, breakdowns = _score(
        policy,
        predicted_proba=[0.0, 0.0],
        uncertainty=[0.0, 0.0],
        staleness=[0.0, 0.0],
        activity_trend=[0.0, 0.0],
    )
    assert np.allclose(priority, -weights.w_redundancy)
    assert all(isinstance(b, PriorityBreakdown) for b in breakdowns)


def test_zero_inputs_at_large_staleness_gives_zero_priority():
    # Large staleness saturates freshness at w_fresh and collapses redundancy to ~0
    # (nothing to be "redundant" about for a channel that hasn't been scanned in ages).
    weights = SchedulerWeights()
    policy = PriorityPolicy(weights)
    priority, _ = _score(
        policy,
        predicted_proba=[0.0, 0.0],
        uncertainty=[0.0, 0.0],
        staleness=[1000.0, 1000.0],
        activity_trend=[0.0, 0.0],
    )
    assert np.allclose(priority, weights.w_fresh)


def test_higher_predicted_proba_gives_higher_priority_all_else_equal():
    policy = PriorityPolicy(SchedulerWeights())
    priority, _ = _score(
        policy,
        predicted_proba=[0.1, 0.9],
        uncertainty=[0.2, 0.2],
        staleness=[10.0, 10.0],
        activity_trend=[0.0, 0.0],
    )
    assert priority[1] > priority[0]


def test_higher_uncertainty_gives_higher_priority():
    policy = PriorityPolicy(SchedulerWeights())
    priority, _ = _score(
        policy,
        predicted_proba=[0.3, 0.3],
        uncertainty=[0.0, 0.5],
        staleness=[10.0, 10.0],
        activity_trend=[0.0, 0.0],
    )
    assert priority[1] > priority[0]


def test_staleness_increases_freshness_term_monotonically():
    policy = PriorityPolicy(SchedulerWeights())
    _, breakdowns = _score(
        policy,
        predicted_proba=[0.0, 0.0, 0.0],
        uncertainty=[0.0, 0.0, 0.0],
        staleness=[0.0, 5.0, 500.0],
        activity_trend=[0.0, 0.0, 0.0],
    )
    fresh = [b.fresh for b in breakdowns]
    assert fresh[0] < fresh[1] < fresh[2]
    # freshness term is bounded by w_fresh (1 - exp(-inf) -> 1)
    assert fresh[2] == pytest.approx(SchedulerWeights().w_fresh, abs=1e-6)


def test_negative_trend_is_clamped_to_zero_contribution():
    policy = PriorityPolicy(SchedulerWeights())
    _, breakdowns = _score(
        policy,
        predicted_proba=[0.0],
        uncertainty=[0.0],
        staleness=[0.0],
        activity_trend=[-0.5],
    )
    assert breakdowns[0].trend == 0.0


def test_redundancy_is_largest_for_just_scanned_confidently_empty_channel():
    policy = PriorityPolicy(SchedulerWeights(), redundancy_tau_slots=5.0)
    priority, breakdowns = _score(
        policy,
        # channel 0: just scanned (staleness 0), predicted empty (p~0)
        # channel 1: just scanned, predicted active (p~1)
        # channel 2: stale (staleness large), predicted empty
        predicted_proba=[0.0, 1.0, 0.0],
        uncertainty=[0.0, 0.0, 0.0],
        staleness=[0.0, 0.0, 500.0],
        activity_trend=[0.0, 0.0, 0.0],
    )
    redundancy_penalty = [-b.redundancy for b in breakdowns]  # stored negated in breakdown
    assert redundancy_penalty[0] > redundancy_penalty[1]
    assert redundancy_penalty[0] > redundancy_penalty[2]
    # the confidently-empty just-scanned channel should score lowest overall
    assert priority[0] < priority[1]
    assert priority[0] < priority[2]


def test_weights_scale_their_own_term_only():
    base = SchedulerWeights(w_pred=1.0, w_explore=0.0, w_fresh=0.0, w_trend=0.0, w_redundancy=0.0)
    boosted = SchedulerWeights(
        w_pred=5.0, w_explore=0.0, w_fresh=0.0, w_trend=0.0, w_redundancy=0.0
    )
    kwargs = dict(
        predicted_proba=[0.4],
        uncertainty=[0.3],
        staleness=[10.0],
        activity_trend=[0.1],
    )
    p_base, _ = _score(PriorityPolicy(base), **kwargs)
    p_boosted, _ = _score(PriorityPolicy(boosted), **kwargs)
    assert p_boosted[0] == pytest.approx(5.0 * p_base[0])


def test_breakdown_to_dict_has_expected_keys():
    policy = PriorityPolicy(SchedulerWeights())
    _, breakdowns = _score(
        policy,
        predicted_proba=[0.5],
        uncertainty=[0.1],
        staleness=[3.0],
        activity_trend=[0.05],
    )
    d = breakdowns[0].to_dict()
    assert set(d.keys()) == {"pred", "explore", "fresh", "trend", "redundancy", "priority"}


def test_rejects_nonpositive_redundancy_tau():
    with pytest.raises(ValueError):
        PriorityPolicy(SchedulerWeights(), redundancy_tau_slots=0.0)
