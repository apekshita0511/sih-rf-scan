"""Phase 7: online-adaptation analysis.

Two independent, additive capabilities, neither touching the scheduler's
decision logic:

* :func:`emerging_adaptation_metrics` -- pre/post-activation scan behaviour on
  a scenario's EMERGING channel. Pure post-hoc analysis of an
  :class:`~rfscan.experiments.runner.EpisodeResult`, reusing
  :func:`~rfscan.experiments.metrics.compute_episode_metrics` for the
  discovery-delay figure rather than re-deriving it -- evaluation-layer only
  (uses ground-truth ``occupancy``/``emerging_channels``), same trust
  boundary as :mod:`rfscan.experiments.metrics`, never fed to a scheduler.
* :func:`trace_adaptive_episode` -- runs an episode via the ordinary
  :func:`~rfscan.experiments.runner.run_episode` (its ``on_slot`` hook, not a
  parallel loop) while also recording specific channels' priority and belief
  mean/std at every slot, straight from
  :meth:`~rfscan.scheduler.adaptive.AdaptiveScheduler.all_breakdowns` /
  :meth:`~rfscan.scheduler.adaptive.AdaptiveScheduler.belief_snapshot` -- a
  read-out of exactly what the scheduler itself computed, when it computed
  it. This is what makes the "a previously quiet channel's priority rises
  after a hit" story checkable rather than asserted.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from rfscan.experiments.metrics import EpisodeMetrics, compute_episode_metrics
from rfscan.experiments.runner import EpisodeResult, run_episode
from rfscan.scheduler.adaptive import AdaptiveScheduler
from rfscan.simulator.environment import RFEnvironment


@dataclass(frozen=True, slots=True)
class EmergingAdaptationMetrics:
    """Scan behaviour on the scenario's first EMERGING channel, split at its
    ``activation_slot``. ``None`` fields mean "no scans in that half"."""

    channel: int
    activation_slot: int
    discovered: bool
    discovery_delay_slots: int | None
    scans_before_activation: int
    scans_after_activation: int
    detections_before_activation: int
    detections_after_activation: int
    detection_rate_before_activation: float | None
    detection_rate_after_activation: float | None


def emerging_adaptation_metrics(
    result: EpisodeResult, metrics: EpisodeMetrics | None = None
) -> EmergingAdaptationMetrics | None:
    """``None`` if the scenario has no EMERGING emitter. ``metrics``, if
    already computed by the caller, is reused instead of recomputed."""
    if not result.emerging_channels:
        return None
    if metrics is None:
        metrics = compute_episode_metrics(result)

    channel = result.emerging_channels[0]
    activation = result.emerging_activation_slots[0]

    scanned = result.scanned_channel
    detected = result.observed_detection
    on_channel = scanned == channel
    slot_index = np.arange(result.n_slots)

    before = on_channel & (slot_index < activation)
    after = on_channel & (slot_index >= activation)

    scans_before = int(np.count_nonzero(before))
    scans_after = int(np.count_nonzero(after))
    det_before = int(np.count_nonzero(before & detected))
    det_after = int(np.count_nonzero(after & detected))

    return EmergingAdaptationMetrics(
        channel=channel,
        activation_slot=activation,
        discovered=metrics.emerging_discovery_delay_slots is not None,
        discovery_delay_slots=metrics.emerging_discovery_delay_slots,
        scans_before_activation=scans_before,
        scans_after_activation=scans_after,
        detections_before_activation=det_before,
        detections_after_activation=det_after,
        detection_rate_before_activation=(det_before / scans_before if scans_before else None),
        detection_rate_after_activation=(det_after / scans_after if scans_after else None),
    )


@dataclass(frozen=True, slots=True)
class PriorityTrace:
    """Per-slot priority/belief for ``channels``, shape ``(n_slots,
    len(channels))`` each. Column order matches ``channels``."""

    channels: tuple[int, ...]
    priority: np.ndarray
    belief_mean: np.ndarray
    belief_std: np.ndarray


def trace_adaptive_episode(
    env: RFEnvironment,
    scheduler: AdaptiveScheduler,
    budget: int,
    tracked_channels: Sequence[int],
    *,
    agent_seed: int | None = None,
) -> tuple[EpisodeResult, PriorityTrace]:
    """Run ``scheduler`` through ``run_episode`` as normal, additionally
    recording ``tracked_channels``' priority and belief mean/std at every
    slot. The trace is a read-out of the scheduler's own already-computed
    state (via ``on_slot``), not a second decision-making pass -- the episode
    itself is unaffected."""
    n = len(tracked_channels)
    priority = np.full((budget, n), np.nan)
    belief_mean = np.full((budget, n), np.nan)
    belief_std = np.full((budget, n), np.nan)

    def _capture(slot: int) -> None:
        breakdowns = scheduler.all_breakdowns()
        mean, std = scheduler.belief_snapshot()
        for i, c in enumerate(tracked_channels):
            priority[slot, i] = breakdowns[c].priority
            belief_mean[slot, i] = mean[c]
            belief_std[slot, i] = std[c]

    result = run_episode(env, scheduler, budget, agent_seed=agent_seed, on_slot=_capture)
    trace = PriorityTrace(
        channels=tuple(tracked_channels),
        priority=priority,
        belief_mean=belief_mean,
        belief_std=belief_std,
    )
    return result, trace
