"""Feature pipeline: turn an agent's observation history into a per-channel
feature matrix, for the Phase 5 predictor.

Every feature is a function of :class:`~rfscan.perception.store.ReadableStore`
history strictly *before* the decision slot, plus the decision slot itself
(used only to compute relative staleness -- never included as an absolute
value). No feature ever touches simulator ground truth: this module imports
nothing from :mod:`rfscan.simulator`. See docs/architecture.md S5 and S16.5.

``FEATURE_NAMES`` gives the canonical column order. ``FeatureBuilder.build``
returns an ``(n_channels, len(FEATURE_NAMES))`` array; row ``c`` is channel
``c``'s feature vector as it would look right before ``slot`` is decided.

Design note: ``channel_index`` and the absolute ``slot`` are deliberately
*excluded* from the feature vector (docs/architecture.md S5) so a trained
model generalises across scenarios instead of memorising "channel 6 is hot" or
"slot 300 is when things happen".
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from rfscan.perception.schema import ScanRecord
from rfscan.perception.store import ReadableStore

# Sentinel for a continuous measurement that has never been taken (no scan yet).
# Chosen far outside any physically plausible dBm/SNR reading (which are always
# within roughly [-140, 40]), so it is trivially separable by any of the Phase 5
# candidate models without needing a separate "is_missing" indicator column.
MISSING_VALUE = -999.0

# Sentinel for "no scan / no detection has ever happened on this channel", used
# for the two staleness features. Larger than any benchmark scenario's
# duration_slots (800, docs/architecture.md S16.4), so it reads unambiguously
# as "maximally stale" rather than colliding with a real elapsed-slot count.
MAX_STALENESS = 1000.0

FEATURE_NAMES: tuple[str, ...] = (
    "last_rssi_dbm",
    "last_noise_dbm",
    "last_snr_db",
    "prev_snr_db",
    "delta_snr_db",
    "last_detection",
    "detection_count_recent",
    "detection_rate_recent",
    "detection_rate_alltime",
    "beta_posterior_mean",
    "beta_posterior_std",
    "slots_since_last_scan",
    "slots_since_last_detection",
    "rolling_snr_mean",
    "rolling_snr_std",
    "scan_count",
    "activity_trend",
)

# One entry per FEATURE_NAMES: (what it represents, why it is useful, when it
# is available, leakage note). Rendered into model cards and the Phase 5
# report; kept next to the code so the two can never drift apart.
FEATURE_DOCS: dict[str, dict[str, str]] = {
    "last_rssi_dbm": {
        "represents": "Received signal strength on the most recent scan of this channel.",
        "why_useful": "Raw signal level; a rising RSSI is the most direct precursor of activity.",
        "available": "After the channel's most recent scan. MISSING_VALUE if never scanned.",
        "leakage": "None -- past observation only.",
    },
    "last_noise_dbm": {
        "represents": "Measured noise floor on the most recent scan of this channel.",
        "why_useful": "Contextualises last_rssi_dbm/last_snr_db; a drifting floor changes what "
        "counts as a strong signal.",
        "available": "After the channel's most recent scan. MISSING_VALUE if never scanned.",
        "leakage": "None -- past observation only.",
    },
    "last_snr_db": {
        "represents": "Signal-to-noise ratio on the most recent scan of this channel.",
        "why_useful": "Directly related to the energy detector's decision rule; the single "
        "strongest per-channel predictor of near-term activity.",
        "available": "After the channel's most recent scan. MISSING_VALUE if never scanned.",
        "leakage": "None -- past observation only.",
    },
    "prev_snr_db": {
        "represents": "SNR from the scan before the most recent one on this channel.",
        "why_useful": "Lets the model see a two-point trajectory instead of one snapshot.",
        "available": "After the channel's second scan. MISSING_VALUE with fewer than 2 scans.",
        "leakage": "None -- past observations only.",
    },
    "delta_snr_db": {
        "represents": "last_snr_db - prev_snr_db.",
        "why_useful": "Short-term rate of change; a rising SNR often precedes a detector flip "
        "from empty to occupied.",
        "available": "After the channel's second scan. 0.0 (neutral) with fewer than 2 scans.",
        "leakage": "None -- computed from two past observations.",
    },
    "last_detection": {
        "represents": "Whether the most recent scan of this channel observed a detection (0/1).",
        "why_useful": "Gilbert-Elliott occupancy is persistent; recent detection is strong "
        "evidence of current activity.",
        "available": "After the channel's most recent scan. 0.0 if never scanned.",
        "leakage": "None -- this is the detector's *observed* (possibly false) outcome, not "
        "ground truth.",
    },
    "detection_count_recent": {
        "represents": "Number of detections in the last `short_window` scans of this channel.",
        "why_useful": "A short-horizon activity count, robust to a single stale/noisy reading.",
        "available": "Immediately (0 with no scans).",
        "leakage": "None -- window is over past scans only.",
    },
    "detection_rate_recent": {
        "represents": "Detections / scans over the last `long_window` scans of this channel.",
        "why_useful": "Recent empirical occupancy rate; reacts to regime change faster than the "
        "all-time rate.",
        "available": "Immediately (0.0 with no scans).",
        "leakage": "None -- window is over past scans only.",
    },
    "detection_rate_alltime": {
        "represents": "Detections / scans over this channel's entire history.",
        "why_useful": "Long-run base rate; stabilises the estimate for channels with little "
        "recent traffic.",
        "available": "Immediately (0.0 with no scans).",
        "leakage": "None -- computed from the full past history only.",
    },
    "beta_posterior_mean": {
        "represents": "Mean of a decaying Beta-Bernoulli posterior fit to this channel's scan "
        "history (see decaying_beta_posterior below).",
        "why_useful": "A smoothed, prior-blended occupancy estimate that relaxes toward the "
        "prior as the channel goes unscanned -- doesn't collapse to a stale point estimate.",
        "available": "Immediately (prior mean with no scans).",
        "leakage": "None -- pure function of past scans and the current slot.",
    },
    "beta_posterior_std": {
        "represents": "Std of the same decaying Beta posterior.",
        "why_useful": "An explicit uncertainty signal: rises for rarely-scanned or long-stale "
        "channels, exactly what a scan scheduler should be told to explore.",
        "available": "Immediately.",
        "leakage": "None -- same as beta_posterior_mean.",
    },
    "slots_since_last_scan": {
        "represents": "Slots elapsed since this channel was last scanned.",
        "why_useful": "Recency / staleness; the raw signal the freshness term of a future "
        "scheduler (Phase 6) would consume.",
        "available": "Immediately (MAX_STALENESS if never scanned).",
        "leakage": "None -- relative to the decision slot, not an absolute slot value.",
    },
    "slots_since_last_detection": {
        "represents": "Slots elapsed since this channel last showed a detection.",
        "why_useful": "Distinguishes 'recently scanned but empty' from 'recently active'; a "
        "short value here is strong evidence of an ongoing activity run.",
        "available": "Immediately (MAX_STALENESS if never detected).",
        "leakage": "None -- relative, from past detections only.",
    },
    "rolling_snr_mean": {
        "represents": "Mean SNR over the last `long_window` scans of this channel.",
        "why_useful": "A denoised signal-level estimate, less jumpy than last_snr_db alone.",
        "available": "Immediately (MISSING_VALUE with no scans).",
        "leakage": "None -- window is over past scans only.",
    },
    "rolling_snr_std": {
        "represents": "Std of SNR over the last `long_window` scans of this channel.",
        "why_useful": "High variance flags bursty/intermittent behaviour that a single SNR "
        "reading cannot distinguish from steady weak activity.",
        "available": "Immediately (0.0 with fewer than 2 scans).",
        "leakage": "None -- window is over past scans only.",
    },
    "scan_count": {
        "represents": "Total number of times this channel has been scanned so far.",
        "why_useful": "An evidence-count feature: tells the model how much to trust the other "
        "empirical features versus the prior-dominated ones.",
        "available": "Immediately (0 with no scans).",
        "leakage": "None -- a count of past scans.",
    },
    "activity_trend": {
        "represents": "max(0, detection_rate_recent - detection_rate_alltime).",
        "why_useful": "Isolates *rising* activity (recent rate above the long-run average); "
        "zero for flat or falling channels so it only fires on the case that matters for "
        "catching an emerging signal.",
        "available": "Immediately (0.0 with no scans).",
        "leakage": "None -- derived from the two rate features above.",
    },
}


def decaying_beta_posterior(
    records: list[ScanRecord],
    slot: int,
    *,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    decay_lambda: float = 0.98,
) -> tuple[float, float]:
    """Mean and std of a decaying Beta-Bernoulli posterior over ``records``
    (one channel's scan history, chronological), evaluated at ``slot``.

    Between consecutive scans (and from the last scan up to ``slot``), the
    posterior relaxes geometrically toward the prior:
    ``a <- lambda**gap * a + (1 - lambda**gap) * prior_alpha`` (same for b).
    At each scan, ``a += 1`` if detected else ``b += 1``. This is the same
    update rule as docs/architecture.md S7's belief layer, applied here as a
    feature (not as the Phase 6 scheduler's stateful ``BeliefState``).

    Shared with :class:`rfscan.models.baseline_beta.DecayingBetaPredictor`,
    which uses this exact posterior mean as its prediction.
    """
    a, b = prior_alpha, prior_beta
    prev_slot: int | None = None
    for record in records:
        if prev_slot is not None:
            gap = record.slot - prev_slot
            decay = decay_lambda**gap
            a = decay * a + (1.0 - decay) * prior_alpha
            b = decay * b + (1.0 - decay) * prior_beta
        if record.detected:
            a += 1.0
        else:
            b += 1.0
        prev_slot = record.slot

    if prev_slot is not None:
        gap = slot - prev_slot
        decay = decay_lambda**gap
        a = decay * a + (1.0 - decay) * prior_alpha
        b = decay * b + (1.0 - decay) * prior_beta

    mean = a / (a + b)
    variance = (a * b) / (((a + b) ** 2) * (a + b + 1.0))
    std = math.sqrt(max(variance, 0.0))
    return mean, std


@dataclass(frozen=True, slots=True)
class FeatureBuilder:
    """Builds the ``FEATURE_NAMES`` feature matrix from a store's history.

    Stateless and pure: two calls with the same ``(store, slot)`` always
    return the same array, and nothing here mutates the store. Window sizes
    and the Beta prior match ``configs/default.yaml``'s ``ModelConfig`` /
    ``BeliefConfig`` defaults.
    """

    short_window: int = 5
    long_window: int = 20
    beta_prior_alpha: float = 1.0
    beta_prior_beta: float = 1.0
    beta_decay_lambda: float = 0.98

    @property
    def n_features(self) -> int:
        return len(FEATURE_NAMES)

    def build(self, store: ReadableStore, slot: int) -> np.ndarray:
        """Feature matrix for every channel, shape ``(n_channels, n_features)``,
        as it would look right before ``slot`` is decided."""
        rows = [self._channel_features(store, c, slot) for c in range(store.n_channels)]
        return np.array(rows, dtype=np.float64)

    def build_channel(self, store: ReadableStore, channel_index: int, slot: int) -> np.ndarray:
        """Feature vector for a single channel, shape ``(n_features,)``."""
        return np.array(self._channel_features(store, channel_index, slot), dtype=np.float64)

    def _channel_features(self, store: ReadableStore, channel_index: int, slot: int) -> list[float]:
        records = store.records_for(channel_index)
        if records and records[-1].slot >= slot:
            raise ValueError(
                f"FeatureBuilder.build called with slot={slot}, but channel {channel_index} "
                f"already has a record at slot {records[-1].slot} -- that would leak future "
                "information. Features must be built from history strictly before the "
                "decision slot."
            )

        n = len(records)
        if n == 0:
            last_rssi = last_noise = last_snr = prev_snr = MISSING_VALUE
            delta_snr = 0.0
            last_detection = 0.0
        else:
            last_obs = records[-1].observation
            last_rssi, last_noise, last_snr = last_obs.rssi_dbm, last_obs.noise_dbm, last_obs.snr_db
            last_detection = 1.0 if records[-1].detected else 0.0
            if n >= 2:
                prev_snr = records[-2].observation.snr_db
                delta_snr = last_snr - prev_snr
            else:
                prev_snr = MISSING_VALUE
                delta_snr = 0.0

        recent_short = records[-self.short_window :]
        detection_count_recent = float(sum(1 for r in recent_short if r.detected))

        recent_long = records[-self.long_window :]
        if recent_long:
            detection_rate_recent = sum(1 for r in recent_long if r.detected) / len(recent_long)
            snr_values = [r.observation.snr_db for r in recent_long]
            rolling_snr_mean = float(np.mean(snr_values))
            rolling_snr_std = float(np.std(snr_values)) if len(snr_values) > 1 else 0.0
        else:
            detection_rate_recent = 0.0
            rolling_snr_mean = MISSING_VALUE
            rolling_snr_std = 0.0

        detection_rate_alltime = store.detection_rate(channel_index, prior=0.0)

        beta_mean, beta_std = decaying_beta_posterior(
            records,
            slot,
            prior_alpha=self.beta_prior_alpha,
            prior_beta=self.beta_prior_beta,
            decay_lambda=self.beta_decay_lambda,
        )

        since_scan = store.slots_since_last_scan(channel_index, slot)
        slots_since_last_scan = float(since_scan) if since_scan is not None else MAX_STALENESS

        detection_slots = [r.slot for r in records if r.detected]
        slots_since_last_detection = (
            float(slot - detection_slots[-1]) if detection_slots else MAX_STALENESS
        )

        activity_trend = max(0.0, detection_rate_recent - detection_rate_alltime)

        return [
            last_rssi,
            last_noise,
            last_snr,
            prev_snr,
            delta_snr,
            last_detection,
            detection_count_recent,
            detection_rate_recent,
            detection_rate_alltime,
            beta_mean,
            beta_std,
            slots_since_last_scan,
            slots_since_last_detection,
            rolling_snr_mean,
            rolling_snr_std,
            float(n),
            activity_trend,
        ]
