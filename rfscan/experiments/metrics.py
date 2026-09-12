"""Episode-level evaluation metrics, censored-aware.

All metrics are computed from an :class:`EpisodeResult` -- the ground-truth
``occupancy`` grid plus the per-slot scan record. Nothing here runs a scheduler
or touches the simulator; it is pure post-hoc analysis.

Core unit: an **activity interval** -- a maximal run of consecutive slots in
which a channel is truly occupied. Detection, delay, misses, and censoring are
all defined relative to intervals.

Definitions (slots; multiply by ``slot_duration_s`` for seconds):

* activity interval ``(c, start, end)``  -- occupancy[start:end, c] all True,
  occupancy[start-1, c] and occupancy[end, c] False (or out of range). ``end`` is
  exclusive; length = end - start.
* interval **detected**  -- some slot t in [start, end) scanned channel c AND
  observed_detection[t] is True (a true positive). detection_slot = the first
  such t.
* **detection delay** = detection_slot - start   (only for detected intervals).
* interval **missed**   -- not detected. Split into "never scanned during the
  interval" and "scanned but the detector missed every time".
* interval **censored** -- missed AND still active at the last slot (end ==
  n_slots): a longer episode might have detected it, so it is not a clean miss.
* **detection_rate** = detected intervals / total intervals.  NaN if no activity.
* **missed_detection_rate** = missed intervals / total intervals  (= 1 - rate).
* **mean_detection_delay_slots** = mean delay over detected intervals only
  (reported alongside miss/censor counts -- misses are never folded in as delay 0).
* **time_to_first_detection_slots** = slot index of the episode's first true
  positive, measured from slot 0. None if there is none.
* **emerging_discovery_delay_slots** = for the first EMERGING emitter, the first
  true positive on its channel at slot >= activation_slot, minus activation_slot.
  None if absent or undiscovered; ``emerging_discovery_censored`` flags the
  undiscovered-but-still-active case.
* **channel_coverage** = distinct channels scanned / n_channels.
* **coverage_time_slots** = slot at which the last-covered channel got its first
  scan. None if not every channel was scanned.
* **on_target_scan_rate** = scans landing on a truly-occupied channel / total
  scans. Pure targeting quality (strategy-controlled, not detector-limited).
* **scan_efficiency** = true-positive scans / total scans (useful detections per
  scan).
* **redundant_scan_rate** = redundant scans / total scans, where a scan of c at
  t is redundant iff c was also scanned within the previous ``redundancy_window``
  slots and no scan of c in [t-window, t] yielded a detection (observation-based:
  it models information the agent actually gained, so no ground truth is used).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from rfscan.experiments.runner import EpisodeResult

DEFAULT_REDUNDANCY_WINDOW = 5

# The metric columns worth a mean/std/count summary row (benchmark.py,
# ablation.py, robustness.py all group on these -- one shared list so the set
# of "headline" metrics can't silently drift between them).
SUMMARY_METRICS = (
    "detection_rate",
    "missed_detection_rate",
    "mean_detection_delay_slots",
    "time_to_first_detection_slots",
    "emerging_discovery_delay_slots",
    "on_target_scan_rate",
    "scan_efficiency",
    "redundant_scan_rate",
    "channel_coverage",
    "coverage_time_slots",
)


@dataclass(frozen=True, slots=True)
class ActivityInterval:
    channel: int
    start_slot: int
    end_slot: int  # exclusive
    scanned_during: bool
    detected: bool
    detection_slot: int | None
    censored: bool

    @property
    def length_slots(self) -> int:
        return self.end_slot - self.start_slot

    @property
    def detection_delay_slots(self) -> int | None:
        if self.detection_slot is None:
            return None
        return self.detection_slot - self.start_slot


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    # -- identity
    scenario: str
    world_seed: int
    agent_seed: int | None
    strategy: str
    n_channels: int
    n_slots: int
    slot_duration_s: float
    # -- scan tallies
    total_scans: int
    total_detections: int
    true_positive_scans: int
    false_positive_scans: int
    on_target_scans: int
    redundant_scans: int
    # -- interval tallies
    n_activity_intervals: int
    n_detected_intervals: int
    n_missed_intervals: int
    n_missed_never_scanned: int
    n_missed_detector: int
    n_censored_intervals: int
    # -- rates
    detection_rate: float
    missed_detection_rate: float
    on_target_scan_rate: float
    scan_efficiency: float
    redundant_scan_rate: float
    # -- delays (slots)
    mean_detection_delay_slots: float | None
    median_detection_delay_slots: float | None
    max_detection_delay_slots: int | None
    time_to_first_detection_slots: int | None
    emerging_discovery_delay_slots: int | None
    emerging_discovery_censored: bool
    # -- coverage
    channels_covered: int
    channel_coverage: float
    coverage_time_slots: int | None
    # -- timing (NONDETERMINISTIC -- excluded from reproducibility checks)
    mean_decision_latency_ms: float

    def to_row(self) -> dict:
        """Flat dict for a DataFrame row, with seconds-domain duplicates of the
        key delays."""
        row = asdict(self)
        s = self.slot_duration_s
        for slot_key, sec_key in (
            ("mean_detection_delay_slots", "mean_detection_delay_s"),
            ("time_to_first_detection_slots", "time_to_first_detection_s"),
            ("emerging_discovery_delay_slots", "emerging_discovery_delay_s"),
        ):
            value = getattr(self, slot_key)
            row[sec_key] = None if value is None else value * s
        return row


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs of True in a 1-D bool array as (start, end-exclusive)."""
    padded = np.concatenate(([np.int8(0)], mask.astype(np.int8), [np.int8(0)]))
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return [(int(a), int(b)) for a, b in zip(starts, ends, strict=True)]


def extract_activity_intervals(result: EpisodeResult) -> list[ActivityInterval]:
    intervals: list[ActivityInterval] = []
    n_slots = result.n_slots
    for c in range(result.n_channels):
        scanned_here = result.scanned_channel == c
        for start, end in _true_runs(result.occupancy[:, c]):
            scan_slots = np.flatnonzero(scanned_here[start:end]) + start
            scanned_during = scan_slots.size > 0
            tp_slots = [int(s) for s in scan_slots if result.observed_detection[s]]
            detected = len(tp_slots) > 0
            detection_slot = min(tp_slots) if detected else None
            censored = (end == n_slots) and not detected
            intervals.append(
                ActivityInterval(
                    channel=c,
                    start_slot=start,
                    end_slot=end,
                    scanned_during=scanned_during,
                    detected=detected,
                    detection_slot=detection_slot,
                    censored=censored,
                )
            )
    return intervals


def _count_redundant_scans(
    scanned: np.ndarray, detected: np.ndarray, window: int
) -> int:
    count = 0
    for t in range(len(scanned)):
        c = scanned[t]
        lo = max(0, t - window)
        prior_same = [s for s in range(lo, t) if scanned[s] == c]
        if not prior_same:
            continue
        useful = any(detected[s] for s in range(lo, t + 1) if scanned[s] == c)
        if not useful:
            count += 1
    return count


def _emerging_discovery(result: EpisodeResult) -> tuple[int | None, bool]:
    if not result.emerging_channels:
        return None, False
    channel = result.emerging_channels[0]
    activation = result.emerging_activation_slots[0]
    if activation >= result.n_slots:
        return None, False
    occ_c = result.occupancy[:, channel]
    tp = (result.scanned_channel == channel) & result.observed_detection & occ_c
    hits = np.flatnonzero(tp)
    hits = hits[hits >= activation]
    if hits.size:
        return int(hits[0] - activation), False
    return None, bool(occ_c[-1])


def _coverage_time(scanned: np.ndarray, n_channels: int) -> int | None:
    seen: set[int] = set()
    for t, c in enumerate(scanned.tolist()):
        seen.add(c)
        if len(seen) == n_channels:
            return t
    return None


def compute_episode_metrics(
    result: EpisodeResult, *, redundancy_window: int = DEFAULT_REDUNDANCY_WINDOW
) -> EpisodeMetrics:
    scanned = result.scanned_channel
    detected = result.observed_detection
    n_slots = result.n_slots
    total_scans = n_slots

    scanned_occupied = result.occupancy[np.arange(n_slots), scanned]
    true_positive = int(np.count_nonzero(detected & scanned_occupied))
    false_positive = int(np.count_nonzero(detected & ~scanned_occupied))
    on_target = int(np.count_nonzero(scanned_occupied))
    total_detections = int(np.count_nonzero(detected))

    intervals = extract_activity_intervals(result)
    n_int = len(intervals)
    detected_iv = [iv for iv in intervals if iv.detected]
    missed_iv = [iv for iv in intervals if not iv.detected]
    missed_never = [iv for iv in missed_iv if not iv.scanned_during]
    missed_detector = [iv for iv in missed_iv if iv.scanned_during]
    censored_iv = [iv for iv in missed_iv if iv.censored]

    delays = sorted(iv.detection_delay_slots for iv in detected_iv)
    if delays:
        mean_delay: float | None = float(np.mean(delays))
        median_delay: float | None = float(np.median(delays))
        max_delay: int | None = int(delays[-1])
    else:
        mean_delay = median_delay = max_delay = None

    tp_slots = np.flatnonzero(detected & scanned_occupied)
    ttfd = int(tp_slots[0]) if tp_slots.size else None

    emerging_delay, emerging_censored = _emerging_discovery(result)

    redundant = _count_redundant_scans(scanned, detected, redundancy_window)
    channels_covered = int(np.unique(scanned).size)

    def _rate(num: float, den: float) -> float:
        return math.nan if den == 0 else num / den

    return EpisodeMetrics(
        scenario=result.scenario,
        world_seed=result.world_seed,
        agent_seed=result.agent_seed,
        strategy=result.strategy,
        n_channels=result.n_channels,
        n_slots=n_slots,
        slot_duration_s=result.slot_duration_s,
        total_scans=total_scans,
        total_detections=total_detections,
        true_positive_scans=true_positive,
        false_positive_scans=false_positive,
        on_target_scans=on_target,
        redundant_scans=redundant,
        n_activity_intervals=n_int,
        n_detected_intervals=len(detected_iv),
        n_missed_intervals=len(missed_iv),
        n_missed_never_scanned=len(missed_never),
        n_missed_detector=len(missed_detector),
        n_censored_intervals=len(censored_iv),
        detection_rate=_rate(len(detected_iv), n_int),
        missed_detection_rate=_rate(len(missed_iv), n_int),
        on_target_scan_rate=_rate(on_target, total_scans),
        scan_efficiency=_rate(true_positive, total_scans),
        redundant_scan_rate=_rate(redundant, total_scans),
        mean_detection_delay_slots=mean_delay,
        median_detection_delay_slots=median_delay,
        max_detection_delay_slots=max_delay,
        time_to_first_detection_slots=ttfd,
        emerging_discovery_delay_slots=emerging_delay,
        emerging_discovery_censored=emerging_censored,
        channels_covered=channels_covered,
        channel_coverage=channels_covered / result.n_channels,
        coverage_time_slots=_coverage_time(scanned, result.n_channels),
        mean_decision_latency_ms=float(np.mean(result.decision_latencies_s) * 1e3),
    )
