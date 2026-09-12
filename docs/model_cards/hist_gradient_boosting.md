# Model card: hist_gradient_boosting

## Purpose
Predicts P(channel active at the next decision opportunity) for the Phase 6
adaptive scheduler's priority score. One of 3 candidates compared
in the Phase 5 bake-off (`artifacts/results/model_bakeoff.csv`).

## Prediction target
y = 1[channel c truly occupied at slot t], features built from
ObservationStore history strictly before slot t. See
`rfscan/models/train.py` module docstring for the exact definition.

## Features
17 features per channel (`rfscan.perception.features.FEATURE_NAMES`):
- `last_rssi_dbm`: Received signal strength on the most recent scan of this channel.
- `last_noise_dbm`: Measured noise floor on the most recent scan of this channel.
- `last_snr_db`: Signal-to-noise ratio on the most recent scan of this channel.
- `prev_snr_db`: SNR from the scan before the most recent one on this channel.
- `delta_snr_db`: last_snr_db - prev_snr_db.
- `last_detection`: Whether the most recent scan of this channel observed a detection (0/1).
- `detection_count_recent`: Number of detections in the last `short_window` scans of this channel.
- `detection_rate_recent`: Detections / scans over the last `long_window` scans of this channel.
- `detection_rate_alltime`: Detections / scans over this channel's entire history.
- `beta_posterior_mean`: Mean of a decaying Beta-Bernoulli posterior fit to this channel's scan history (see decaying_beta_posterior below).
- `beta_posterior_std`: Std of the same decaying Beta posterior.
- `slots_since_last_scan`: Slots elapsed since this channel was last scanned.
- `slots_since_last_detection`: Slots elapsed since this channel last showed a detection.
- `rolling_snr_mean`: Mean SNR over the last `long_window` scans of this channel.
- `rolling_snr_std`: Std of SNR over the last `long_window` scans of this channel.
- `scan_count`: Total number of times this channel has been scanned so far.
- `activity_trend`: max(0, detection_rate_recent - detection_rate_alltime).

## Training data
2016000 rows from 210 episodes (7 scenarios x
seeds (0, 1, 2, 3, 4, 5, 6, 7, 8, 9) x policies ('sequential', 'random', 'heuristic_explore')). Label balance (P(active)):
0.1542. All data comes from the software RF simulator
(`rfscan.simulator`) -- synthetic Gilbert-Elliott channel occupancy, not
recorded or real-world spectrum data. See "Investigated but not integrated"
below.

## Split methodology
Temporal, grouped by (scenario, world_seed) -- whole episodes assigned to
exactly one of train/val/test, seeds assigned in increasing order. Train
seeds (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), validation seeds (10, 11, 12, 13, 14), test seeds (15, 16, 17, 18, 19, 20, 21, 22, 23, 24).
No row-level shuffling across folds.

## Assumptions
- The receiver can scan exactly one channel per decision slot (single-receiver,
  discrete-slot model); multi-receiver is a documented future extension.
- Channel occupancy follows the simulator's Gilbert-Elliott process plus the
  five emitter behaviours (persistent/intermittent/bursty/emerging/fading) --
  an abstraction of decision-relevant dynamics, not an RF propagation model.
- Training data is a *mixture* of scan policies (sequential/random/heuristic),
  matching the mix the eventual scheduler will produce, per the distribution-
  shift mitigation in `rfscan/models/train.py`'s module docstring.

## Known leakage risks
- **Feature/label leakage**: structurally prevented -- `FeatureBuilder.build`
  raises if the store already holds a record at or after the query slot
  (`rfscan/perception/features.py`), and the module imports nothing from
  `rfscan.simulator`. Verified by `tests/test_leakage.py`.
- **Train/test contamination**: rows are grouped by whole episode
  `(scenario, world_seed)`, never split at the row level, so no near-duplicate
  adjacent-slot rows from the same episode can appear in two different folds.
- **Distribution shift**: this model is fit on data from non-adaptive
  baseline policies. Once the Phase 6 adaptive scheduler starts choosing scans
  based on its own predictions, the scan distribution it sees will differ from
  training -- an open risk flagged, not solved, in Phase 5 (see
  docs/architecture.md S3.2/S12 R1).

## Investigated but not integrated
The SIH26055 problem statement references the Hugging Face
`alan-turing-institute/turing-synthetic-radar-dataset`. It was inspected
(dataset card) during Phase 5: each row is a Pulse Descriptor Word (Time of
Arrival, Centre Frequency, Pulse Width, Angle of Arrival, Amplitude) for a
*pulse deinterleaving* task (grouping pulses by unknown emitter), not a
discrete-slot, fixed-channel occupancy series. It has no time slots and no
channel plan compatible with this project's `ChannelPlan`/`ObservationStore`
contracts, and at ~4 billion pulses is far larger than this phase's scope.
Not integrated for Phase 5; the simulator remains the sole data source.

## Metrics (test split, n=2016000)
| metric | value |
|---|---|
| precision | 0.7899 |
| recall | 0.7320 |
| f1 | 0.7599 |
| pr_auc | 0.8117 |
| roc_auc | 0.9541 |
| brier | 0.0540 |
| recall@precision>=0.5 | 0.9352 |
| recall@precision>=0.8 | 0.6943 |

## Calibration (test split)
Expected calibration error: 0.0029.

## Computational characteristics
Inference latency: 3.75 us/row (measured, wall-clock).
Parameters/complexity: 181.

## Limitations
Tree ensemble -- opaque per-decision explanation compared to logistic regression's coefficients (mitigated only by the Beta belief layer's separate, always-explainable uncertainty term). Slower inference and more hyperparameters than logistic regression.

## Known failure modes
Can overfit rare (scenario, policy) combinations in train if not regularised (max_depth capped here); calibration fit on a smaller validation fold is noisier for a higher-capacity model.

## Intended use
Offline research/demo component of the SIH26055 academic prototype: predicts
per-channel activity probability to feed the Phase 6 scan-scheduling priority
score, evaluated entirely inside the synthetic RF simulator (no hardware, no
interception, no jamming/spoofing).

## Non-intended use
This model is trained and evaluated exclusively on simulated Gilbert-Elliott
channel-occupancy data. It is **not** trained or validated to identify,
classify, or characterise real-world radar/RF platforms (drones, aircraft,
missiles, or any physical emitter), and must not be used for real-world
signal intelligence, targeting, jamming, or any operational/classified EW
purpose.
