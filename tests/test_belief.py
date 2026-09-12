"""BeliefState: decaying Beta-Bernoulli occupancy belief (Phase 6)."""

from __future__ import annotations

import numpy as np
import pytest

from rfscan.scheduler.belief import BeliefState


def test_starts_at_prior_mean_and_std():
    belief = BeliefState(4, prior_alpha=1.0, prior_beta=1.0)
    assert np.allclose(belief.mean(), 0.5)
    # Beta(1,1) std = sqrt(1*1 / (2^2 * 3)) = sqrt(1/12)
    assert np.allclose(belief.std(), np.sqrt(1.0 / 12.0))


def test_update_shifts_mean_toward_the_observation():
    belief = BeliefState(3, prior_alpha=1.0, prior_beta=1.0)
    for _ in range(20):
        belief.update(0, True)
        belief.update(1, False)
    assert belief.mean()[0] > 0.9
    assert belief.mean()[1] < 0.1
    assert belief.mean()[2] == pytest.approx(0.5)  # untouched channel


def test_decay_relaxes_unscanned_channels_toward_prior():
    belief = BeliefState(2, prior_alpha=1.0, prior_beta=1.0, decay_lambda=0.9)
    for _ in range(30):
        belief.update(0, True)
    mean_before = belief.mean()[0]
    for _ in range(200):
        belief.decay()
    assert belief.mean()[0] < mean_before
    assert belief.mean()[0] == pytest.approx(0.5, abs=1e-6)  # relaxed back near prior


def test_decay_raises_uncertainty_for_a_confidently_scanned_channel():
    belief = BeliefState(1, decay_lambda=0.95)
    for _ in range(50):
        belief.update(0, True)
    std_confident = belief.std()[0]
    for _ in range(200):
        belief.decay()
    assert belief.std()[0] > std_confident


def test_lambda_one_means_no_decay():
    belief = BeliefState(2, decay_lambda=1.0)
    belief.update(0, True)
    mean_before = belief.mean().copy()
    for _ in range(100):
        belief.decay()
    assert np.allclose(belief.mean(), mean_before)


def test_sample_is_reproducible_for_a_seed_and_in_unit_interval():
    a = BeliefState(5, seed=7)
    b = BeliefState(5, seed=7)
    a.update(2, True)
    b.update(2, True)
    draws_a = a.sample()
    draws_b = b.sample()
    assert np.array_equal(draws_a, draws_b)
    assert ((draws_a >= 0.0) & (draws_a <= 1.0)).all()


def test_reset_returns_to_prior_and_replays_sample_stream():
    belief = BeliefState(3, seed=1)
    belief.update(0, True)
    belief.decay()
    first_sample = belief.sample()
    belief.reset()
    assert np.allclose(belief.mean(), 0.5)
    belief.update(0, True)
    belief.decay()
    assert np.array_equal(belief.sample(), first_sample)


def test_update_out_of_range_channel_raises():
    belief = BeliefState(2)
    with pytest.raises(IndexError):
        belief.update(5, True)


def test_validates_constructor_args():
    with pytest.raises(ValueError):
        BeliefState(0)
    with pytest.raises(ValueError):
        BeliefState(3, decay_lambda=0.0)
    with pytest.raises(ValueError):
        BeliefState(3, decay_lambda=1.5)
    with pytest.raises(ValueError):
        BeliefState(3, prior_alpha=0.0)
