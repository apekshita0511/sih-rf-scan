"""Phase 8: statistical comparison utilities.

Pure functions over already-computed per-(scenario, seed) metric arrays --
nothing here runs a scheduler or touches the simulator. Two tools:

* :func:`bootstrap_ci` -- percentile bootstrap confidence interval for a
  statistic (mean by default) over a set of values. Preferred over a normal-
  theory CI for metrics like ``mean_detection_delay_slots`` that are not
  expected to be normally distributed.
* :func:`paired_comparison` -- paired t-test *and* Wilcoxon signed-rank on
  matched (same scenario, same world seed) observations, since every
  strategy in the benchmark grid faces an identical world realisation
  (docs/architecture.md S2/S9) -- a paired design is the correct, more
  powerful test here, not an unpaired one. Reports both tests rather than
  picking one, and only calls a result "significant" (``significant_0_05``)
  when *both* agree at alpha=0.05 -- a deliberately conservative bar, per
  this project's "do not claim significance unless the test actually
  supports it" rule (docs/architecture.md S9's "no fabrication").

Censoring: some metrics (``mean_detection_delay_slots``,
``emerging_discovery_delay_slots``) are ``None``/NaN for an episode where
that strategy never detected anything. Both functions here drop NaNs -- for
``bootstrap_ci`` from the single input array, for ``paired_comparison`` by
dropping any *pair* where either side is NaN, so the two arrays stay
matched. This means the compared sample can shrink for delay-based metrics;
the resulting ``n`` is always reported so this is visible, never silent.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats as scipy_stats


@dataclass(frozen=True, slots=True)
class BootstrapCI:
    metric: str
    strategy: str
    n: int
    point_estimate: float
    lower: float
    upper: float
    confidence: float
    n_resamples: int
    note: str | None = None

    def to_row(self) -> dict:
        return asdict(self)


def bootstrap_ci(
    values: np.ndarray,
    *,
    metric: str,
    strategy: str,
    statistic: Callable[[np.ndarray], float] = np.mean,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> BootstrapCI:
    """Percentile bootstrap CI for ``statistic`` over ``values`` (NaNs
    dropped first -- censored episodes for delay-based metrics)."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    n = int(arr.size)
    if n < 2:
        point = float(statistic(arr)) if n else math.nan
        return BootstrapCI(
            metric=metric,
            strategy=strategy,
            n=n,
            point_estimate=point,
            lower=math.nan,
            upper=math.nan,
            confidence=confidence,
            n_resamples=0,
            note=f"only {n} non-censored observation(s) -- too few to bootstrap",
        )
    rng = np.random.default_rng(seed)
    result = scipy_stats.bootstrap(
        (arr,),
        statistic,
        confidence_level=confidence,
        n_resamples=n_resamples,
        method="percentile",
        random_state=rng,
    )
    return BootstrapCI(
        metric=metric,
        strategy=strategy,
        n=n,
        point_estimate=float(statistic(arr)),
        lower=float(result.confidence_interval.low),
        upper=float(result.confidence_interval.high),
        confidence=confidence,
        n_resamples=n_resamples,
    )


@dataclass(frozen=True, slots=True)
class PairedComparison:
    metric: str
    strategy_a: str
    strategy_b: str
    n: int
    mean_a: float
    mean_b: float
    mean_diff: float  # b - a
    cohen_d: float | None
    t_statistic: float | None
    t_pvalue: float | None
    wilcoxon_statistic: float | None
    wilcoxon_pvalue: float | None
    significant_0_05: bool | None
    note: str | None = None

    def to_row(self) -> dict:
        return asdict(self)


def paired_comparison(
    a: np.ndarray,
    b: np.ndarray,
    *,
    metric: str,
    strategy_a: str,
    strategy_b: str,
    alpha: float = 0.05,
) -> PairedComparison:
    """Paired comparison of ``b`` against ``a`` (matched by array index --
    the caller is responsible for pairing, e.g. identical (scenario, seed)
    row order for both). ``significant_0_05`` is True only if both the
    paired t-test and the Wilcoxon signed-rank test agree at alpha; either
    can be ``None`` if its assumptions don't hold (e.g. zero variance)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"paired arrays must be the same shape, got {a.shape} vs {b.shape}")

    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    n = int(a.size)

    if n < 2:
        return PairedComparison(
            metric=metric,
            strategy_a=strategy_a,
            strategy_b=strategy_b,
            n=n,
            mean_a=float(a.mean()) if n else math.nan,
            mean_b=float(b.mean()) if n else math.nan,
            mean_diff=math.nan,
            cohen_d=None,
            t_statistic=None,
            t_pvalue=None,
            wilcoxon_statistic=None,
            wilcoxon_pvalue=None,
            significant_0_05=None,
            note=f"only {n} paired, non-censored observation(s) -- too few for a test",
        )

    diff = b - a
    mean_diff = float(diff.mean())
    std_diff = float(diff.std(ddof=1))
    cohen_d = mean_diff / std_diff if std_diff > 0 else None

    t_stat = t_p = None
    if std_diff > 0:
        t_res = scipy_stats.ttest_rel(b, a)
        t_stat, t_p = float(t_res.statistic), float(t_res.pvalue)

    w_stat = w_p = None
    note = None
    if np.all(diff == 0):
        note = "all paired differences are exactly zero -- no test applies"
    else:
        try:
            w_res = scipy_stats.wilcoxon(b, a)
            w_stat, w_p = float(w_res.statistic), float(w_res.pvalue)
        except ValueError as exc:
            note = f"Wilcoxon signed-rank test did not run: {exc}"

    p_values = [p for p in (t_p, w_p) if p is not None]
    significant = all(p < alpha for p in p_values) if p_values else None

    return PairedComparison(
        metric=metric,
        strategy_a=strategy_a,
        strategy_b=strategy_b,
        n=n,
        mean_a=float(a.mean()),
        mean_b=float(b.mean()),
        mean_diff=mean_diff,
        cohen_d=cohen_d,
        t_statistic=t_stat,
        t_pvalue=t_p,
        wilcoxon_statistic=w_stat,
        wilcoxon_pvalue=w_p,
        significant_0_05=significant,
        note=note,
    )
