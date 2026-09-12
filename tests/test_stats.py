"""experiments/stats.py: bootstrap CI + paired comparisons (Phase 8)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rfscan.experiments.stats import bootstrap_ci, paired_comparison


# -- bootstrap_ci ------------------------------------------------------
def test_bootstrap_ci_point_estimate_matches_the_mean():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    result = bootstrap_ci(values, metric="x", strategy="adaptive", seed=0)
    assert result.point_estimate == pytest.approx(5.5)
    assert result.lower < result.point_estimate < result.upper
    assert result.n == 10


def test_bootstrap_ci_drops_nans():
    values = np.array([1.0, 2.0, np.nan, 3.0, np.nan, 4.0])
    result = bootstrap_ci(values, metric="x", strategy="adaptive", seed=0)
    assert result.n == 4
    assert result.point_estimate == pytest.approx(2.5)


def test_bootstrap_ci_is_reproducible_for_a_seed():
    values = np.array([1.0, 5.0, 2.0, 9.0, 3.0, 7.0, 4.0, 6.0])
    a = bootstrap_ci(values, metric="x", strategy="s", seed=1)
    b = bootstrap_ci(values, metric="x", strategy="s", seed=1)
    assert a.lower == b.lower and a.upper == b.upper


def test_bootstrap_ci_narrows_with_a_tighter_confidence_level():
    values = np.array([1.0, 5.0, 2.0, 9.0, 3.0, 7.0, 4.0, 6.0, 8.0, 10.0])
    wide = bootstrap_ci(values, metric="x", strategy="s", confidence=0.95, seed=0)
    narrow = bootstrap_ci(values, metric="x", strategy="s", confidence=0.50, seed=0)
    assert (narrow.upper - narrow.lower) < (wide.upper - wide.lower)


def test_bootstrap_ci_handles_too_few_observations_without_crashing():
    result = bootstrap_ci(np.array([5.0]), metric="x", strategy="s")
    assert result.n == 1
    assert math.isnan(result.lower) and math.isnan(result.upper)
    assert result.note is not None

    empty = bootstrap_ci(np.array([]), metric="x", strategy="s")
    assert empty.n == 0
    assert math.isnan(empty.point_estimate)


# -- paired_comparison ---------------------------------------------------
def test_paired_comparison_detects_a_consistent_improvement():
    rng = np.random.default_rng(0)
    a = rng.normal(0.30, 0.02, size=30)
    b = a + 0.10  # strategy b is consistently 0.10 better, every pair
    result = paired_comparison(
        a, b, metric="detection_rate", strategy_a="random", strategy_b="adaptive"
    )
    assert result.n == 30
    assert result.mean_diff == pytest.approx(0.10, abs=1e-9)
    assert result.t_pvalue is not None and result.t_pvalue < 0.001
    assert result.wilcoxon_pvalue is not None and result.wilcoxon_pvalue < 0.001
    assert result.significant_0_05 is True
    assert result.cohen_d is not None and result.cohen_d > 0


def test_paired_comparison_finds_no_significance_for_identical_arrays():
    rng = np.random.default_rng(1)
    a = rng.normal(0.5, 0.1, size=20)
    result = paired_comparison(a, a.copy(), metric="x", strategy_a="s1", strategy_b="s2")
    assert result.mean_diff == 0.0
    assert result.note == "all paired differences are exactly zero -- no test applies"
    assert result.t_pvalue == 1.0 or result.t_pvalue is None
    assert result.significant_0_05 in (None, False)


def test_paired_comparison_drops_nan_pairs_and_reports_shrunk_n():
    a = np.array([0.1, 0.2, np.nan, 0.4, 0.5])
    b = np.array([0.2, np.nan, 0.3, 0.5, 0.6])
    result = paired_comparison(a, b, metric="x", strategy_a="a", strategy_b="b")
    assert result.n == 3  # indices 0, 3, 4 are the only fully-observed pairs


def test_paired_comparison_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        paired_comparison(
            np.array([1.0, 2.0]), np.array([1.0]), metric="x", strategy_a="a", strategy_b="b"
        )


def test_paired_comparison_too_few_pairs_returns_none_test_results():
    a = np.array([0.1])
    b = np.array([0.2])
    result = paired_comparison(a, b, metric="x", strategy_a="a", strategy_b="b")
    assert result.n == 1
    assert result.t_pvalue is None and result.wilcoxon_pvalue is None
    assert result.significant_0_05 is None
    assert result.note is not None


def test_paired_comparison_random_noise_is_not_falsely_significant():
    rng = np.random.default_rng(42)
    a = rng.normal(0.5, 0.2, size=15)
    b = rng.normal(0.5, 0.2, size=15)  # independent noise, same distribution
    result = paired_comparison(a, b, metric="x", strategy_a="a", strategy_b="b")
    # Not a guarantee for every seed, but for this fixed seed/effect size the
    # test should not claim significance where there is truly no effect.
    assert result.significant_0_05 is not True
