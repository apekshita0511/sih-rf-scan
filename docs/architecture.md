# AI-Driven Adaptive RF Scan Intelligence Platform — Architecture & Design

**SIH26055 · DRDO · Software · Smart India Hackathon 2026**

## 0. One-line technical thesis

> We reframe electronic-warfare spectrum scanning as a **belief-space restless
> multi-armed bandit** and solve it with a **hybrid learned-occupancy +
> Bayesian-uncertainty scheduler**. The deliverable is not a signal classifier —
> it is a closed-loop *scan policy*, measured by detection latency and
> emerging-emitter discovery against sequential/random baselines on identical,
> seeded synthetic RF scenarios, with an ablation isolating each component's
> contribution.

Everything runs on a synthetic RF simulator. No hardware, no interception, no
jamming/spoofing, nothing operational or classified.

---

## 1. Final architecture

### 1.1 Layered view

```
APP LAYER            Streamlit + Plotly dashboard, session state
EXPERIMENT LAYER     runner · metrics · benchmark · ablation
DECISION LAYER       SchedulerProtocol
                     ├ SequentialScheduler   (baseline)
                     ├ RandomScheduler       (baseline)
                     ├ HeuristicScheduler    (epsilon-greedy on empirical rate)
                     └ AdaptiveScheduler   <-- STAR
                          ├ BeliefState   (decaying Beta-Bernoulli)
                          ├ Predictor     (ML activity model)
                          └ PriorityPolicy (multi-objective score)
PERCEPTION LAYER     FeatureBuilder · ObservationStore
SENSOR LAYER         Scanner  (asks the world "scan channel X")
WORLD LAYER          RFEnvironment
                     ├ ChannelPlan       (configurable frequency map)
                     ├ OccupancyModel     (Gilbert-Elliott Markov)
                     ├ EmitterSpec[]      (persistent/burst/...)
                     ├ NoiseModel         (floor drift + AWGN)
                     └ Scenario           (seeded, replayable)

Ground truth flows ONLY to the EXPERIMENT LAYER (evaluation),
NEVER to PERCEPTION / DECISION.
```

### 1.2 The closed loop (one slot)

```
RFEnvironment.step()          world advances 1 slot (restless: happens regardless
                              of the agent). All channel Markov states evolve.
BeliefState.decay()           every unscanned channel's posterior relaxes toward
                              the prior -> uncertainty rises with staleness.
FeatureBuilder.build()   -> X (n_channels x n_features)
Predictor.predict_proba(X) -> p (per-channel P(active next slot))
PriorityPolicy.score()   -> priority vector + per-term explanation
Scheduler.select_next()  -> channel_index*  (argmax priority)
Scanner.scan(index*)     -> Observation (noisy; may be false positive / negative)
ObservationStore.append(record)
BeliefState.update(index*, observed_detection)
[optional] Predictor.partial_fit(...)
Runner.log(slot, index*, observation, ground_truth)   <- evaluation only
```

### 1.3 Time model

Discrete **slots**. One slot = one scan action = one `select_next` call.
`slot_duration_s` is configurable (default 0.1 s), so every result is reported in
both slots and seconds. Multi-receiver (choose *K* channels/slot) is a Phase-11
extension — top-*K* priority with a correlation penalty. Continuous-time with
variable dwell is explicitly out of scope (future work).

---

## 2. System workflow

| Stage | Actor | Output | Key rule |
|---|---|---|---|
| Scenario build | `Scenario.build_environment(seed)` | fresh `RFEnvironment` | fully deterministic |
| World tick | `RFEnvironment.step()` | next occupancy state (all channels) | independent of agent |
| Belief decay | `BeliefState.decay()` | relaxed posteriors | unseen ⇒ more uncertain |
| Feature build | `FeatureBuilder` | feature matrix | no future, no truth |
| Predict | `Predictor` | activity probability per channel | frozen or online |
| Prioritise | `PriorityPolicy` | priority + term breakdown | exploration baked in |
| Select | `Scheduler` | channel index | argmax (or softmax) |
| Scan | `Scanner` → `RFEnvironment.observe(c)` | `Observation` | energy detector ⇒ FP/FN |
| Feedback | `Store` + `BeliefState` (+ `Predictor`) | updated state | Bayesian count update |
| Log | `ExperimentRunner` | per-slot record | truth used only here |
| Metrics | `metrics.py` | detection delay, coverage, ... | per-episode, censored-aware |
| Aggregate | `benchmark.py` | mean ± 95% CI, paired tests | same seeds across strategies |

**Fairness invariant (enforced in code, tested):** the world's stochastic
realisation is keyed by `hash(world_seed, slot, channel_index)`, not by
RNG-stream position. Scanning CH3 vs CH6 at slot 5 cannot change what CH9 does at
slot 6. Every strategy therefore faces a byte-identical RF world for a given
seed. The agent has its own separate RNG.

---

## 3. Data flow

### 3.1 Runtime (online loop)

World state advances via `step()`; only the selected channel is observed; the
observation feeds the store → feature builder → predictor → priority policy →
next selection. Ground truth is logged to a separate evaluation silo and never
re-enters the decision path.

### 3.2 Offline (ML training)

```
scenarios x seeds -> mixed data-collection policy (round-robin + epsilon-random
                     + exploration-heavy) -> log (features_t, truth_{t+1}) per channel
temporal split:  train = scenario-seeds 0-19,  val = 20-29,  test = 30-49
                 (also hold out whole scenario TYPES)
fit LR / RF / HGB -> calibrate (isotonic/Platt on val)
                  -> evaluate (PR-AUC, Brier, recall@precision, latency)
artifacts: model_<name>.joblib + model_card.md + calibration plots
```

**Distribution-shift note (judges will probe):** the AdaptiveScheduler changes
the data it will later see (it scans active channels more, so "stale channel"
feature regions get sparse). Mitigations, all implemented: (1) training data from
a *mixture* of policies including exploration-heavy ones; (2) the Beta belief
layer gives a policy-independent robustness floor; (3) evaluate the **closed
loop**, not just offline AUC; (4) optional periodic refit on a recent-observation
buffer (ablation toggle).

---

## 4. Module design

```
simulator/
  channel.py       ChannelConfig, ChannelPlan.wifi_24ghz(n) / .generic(...), build_channel_plan
  occupancy.py     OccupancyModel  (Gilbert-Elliott: p01, p10; time-varying hooks)      [Phase 2]
  emitters.py      EmitterSpec, Behavior{PERSISTENT, INTERMITTENT, BURSTY, EMERGING, FADING}
  noise.py         NoiseModelConfig (floor, drift, AWGN, threshold, FP/FN rates)
  environment.py   RFEnvironment { step(), observe(c)->Observation, _draw(seed,slot,c) }  [Phase 2]
  scenarios.py     ScenarioConfig, SCENARIOS: dict[str, factory]  (the 7)                 [Phase 4]

perception/
  schema.py        Observation (no truth), GroundTruth (eval silo), ScanRecord
  store.py         ObservationStore (append, per-channel views, to_frame())               [Phase 3]
  features.py      FeatureBuilder.build(store, slot) -> np.ndarray ; FEATURE_NAMES         [Phase 5]

models/
  base.py          PredictorProtocol { predict_proba(X), partial_fit?(X,y), explain(x) }   [Phase 5]
  baseline_beta.py DecayingBetaPredictor  (non-ML reference)                               [Phase 5]
  sklearn_model.py SklearnPredictor(estimator, calibrator)                                 [Phase 5]
  train.py         data collection, temporal split, fit, calibrate, persist               [Phase 5]
  evaluation.py    pr_auc, roc_auc, brier, recall_at_precision, calibration, latency       [Phase 5]

scheduler/
  base.py          SchedulerProtocol { select_next(store, slot)->int, update(record), explain() }
  sequential.py    RandomScheduler, SequentialScheduler                                    [Phase 3]
  heuristic.py     HeuristicScheduler (epsilon-greedy on empirical detection rate)         [Phase 3]
  belief.py        BeliefState  (Beta(a,b) per channel; decay(); update(c,y); mean/std/sample) [Phase 6]
  policy.py        PriorityPolicy (weights, term functions, score()->PriorityBreakdown)   [Phase 6]
  adaptive.py      AdaptiveScheduler (wires BeliefState + Predictor + PriorityPolicy)      [Phase 6]
  ucb.py, thompson.py   alternative decision rules for the bake-off                        [not yet built]

experiments/
  runner.py        run_episode(env, scheduler, budget) -> RunResult ; same-world fairness  [Phase 4]
  metrics.py       episode-level metric functions (censored-aware)                         [Phase 4]
  benchmark.py     grid(scenarios x seeds x strategies) -> raw_results.csv, summary.csv    [Phase 4/6]
  adaptation.py    emerging-signal pre/post metrics + priority/belief tracing              [Phase 7]
  ablation.py      variants A-F (weight masks + online-feedback toggle)                    [Phase 8]
  stats.py         bootstrap_ci, paired_comparison (t-test + Wilcoxon)                     [Phase 8]
  robustness.py    noise sweep + non-stationarity before/after analysis                    [Phase 8]
  plots.py         static matplotlib research plots (bars/lines + CI whiskers)             [Phase 8]

visualization/plots.py    spectrum_bar, belief_band, priority_stack, scan_raster, ...      [Phase 9]
app/dashboard.py          page shell + control sidebar + 7 sections                        [Phase 9]

configs/default.yaml      n_channels, slot_duration_s, budget, seeds, weights, model, ...
main.py / rfscan.cli      CLI: info | train | benchmark | ablate | demo | dashboard
```

Protocols (structural typing), not inheritance, for `Predictor` and `Scheduler`
→ trivial swap, easy mocking. Dataclasses for all specs/records. `logging`
throughout, no prints. Config loaded once into a frozen dataclass.

---

## 5. Data schema

- `ChannelConfig(index, label, center_freq_hz, bandwidth_hz)` — `index` in
  `[0, N)` is the canonical address; `label`/frequency are display + features.
- `Observation(slot, channel_index, rssi_dbm, noise_dbm, snr_db, observed_detection)`
  — **no ground-truth field** (structural anti-leakage).
- `GroundTruth(slot, channel_index, occupied, true_signal_dbm)` — eval silo only.
- `ScanRecord(observation, decision_info?)` — `decision_info` carries the
  priority-term breakdown from Phase 6 for the explainability panel.
- `EmitterSpec(channel_index, behavior, params, activation_slot, deactivation_slot)`.
- `NoiseModelConfig(floor_dbm, floor_drift_std_db, awgn_std_db,
  detection_threshold_snr_db, false_alarm_rate, missed_detection_rate)`.
- `ScenarioConfig(name, seed, n_channels, duration_slots, slot_duration_s,
  channel_plan, channel_plan_params, emitters, noise, nonstationarity)`.
  *(Refinement vs the brief: the channel plan is stored as a name + params so it
  regenerates deterministically, and non-stationarity is a typed
  `NonStationaryEvent`, not a bare tuple — both for clean YAML.)*

**Feature vector** (~16, per channel, from history ≤ t): last RSSI/noise/SNR,
prev SNR, ΔSNR, last detection, detection count (w5), detection rate (recent w20
& all-time), Beta posterior mean & std, slots since last scan, slots since last
detection, rolling SNR mean/std, scan count, trend = recent-rate − all-time-rate.
Deliberately excluded: raw `channel_index` and absolute `slot` (so the policy
generalises across scenarios instead of memorising "CH6 is hot").

**Label:** `GroundTruth.occupied` at slot *t+1* (training only).

---

## 6. ML candidate comparison

| Candidate | Role | Strengths | Weaknesses | Chosen iff… |
|---|---|---|---|---|
| Decaying Beta-Bernoulli (non-ML) | belief layer + reference | zero training, explainable, native staleness/uncertainty | ignores SNR/noise/trend; lags abrupt change | always present as belief layer + honesty benchmark |
| Logistic Regression (L2, calibrated) | Phase-5 deployed predictor | <50 µs/pred, coefficients = explanation, well-calibrated | linear; misses SNR×trend interactions | coefs sane & PR-AUC within ~0.03 of best |
| Random Forest | bake-off | captures interactions, robust | latency, calibration, memory | beats HGB on closed-loop metrics |
| HistGradientBoosting / XGBoost | likely final | best expected PR-AUC, nonlinear, fast inference | needs calibration + tuning | **significant closed-loop** gain over LR |

Decide on PR-AUC, Brier/calibration (the scheduler consumes probabilities),
recall @ fixed precision, latency — **then re-rank by closed-loop detection
delay**. Ship the simplest model that wins the loop. `Predictor` is a protocol;
swapping is one config key.

---

## 7. Scheduler candidate comparison

| Strategy | Mechanism | Exploration | Explainability | Hypothesis |
|---|---|---|---|---|
| Sequential | round-robin | none | trivial | worst delay; predictable coverage cadence |
| Random | uniform | all exploration | trivial | mediocre everything; a floor |
| Heuristic (ε-greedy on rate) | argmax detection rate | fixed ε | medium | good stable, slow on emerging/non-stationary |
| UCB on Beta occupancy | mean + c·std | optimism | high (two numbers) | strong, principled |
| Thompson on Beta | sample θ_c | posterior sampling | medium (stochastic) | strong, good emerging discovery |
| **Weighted multi-objective priority (argmax)** | pred + U + freshness + trend − redundancy | U-term + freshness | **highest** (stacked bar) | best multi-objective + best demo |
| Weighted priority + softmax | as above, sampled | + temperature | high | smoother; maybe better emerging |

**Theoretical anchor:** the restless-bandit optimum is the **Whittle index**; the
weighted policy is a transparent, cheap, tunable approximation, with UCB/Thompson
as principled cross-checks.

**Recommended final:** weighted multi-objective priority (argmax), Beta belief
layer supplying the uncertainty term *and* a robustness floor, ML refining the
point estimate. Weights tuned on train scenarios, **frozen**, reported on
held-out scenarios with a sensitivity sweep.

Priority score:

```
priority(c) = w_pred      * p(c)
            + w_explore    * uncertainty(c)         # Beta posterior std
            + w_fresh      * (1 - exp(-staleness(c) / tau))
            + w_trend      * max(0, recent_rate(c) - alltime_rate(c))
            - w_redundancy * redundancy(c)          # recently scanned & confidently empty
```

Belief update: each slot, for every **unscanned** channel
`alpha <- lambda*alpha + (1-lambda)*alpha_prior` (same for beta); for the
**scanned** channel `alpha += y`, `beta += (1 - y)` with `y = observed_detection`.
A hard freshness guarantee bounds the maximum revisit interval so every channel —
including a future emerger — is eventually scanned.

---

## 8. Recommended final approach

1. **World:** Gilbert-Elliott occupancy per channel + behaviour overlays,
   noise floor drift + AWGN, energy-detector observation with tunable FP/FN.
   Configurable channel plan. Realisation keyed by `(seed, slot, channel)`.
2. **Belief:** decaying Beta-Bernoulli per channel.
3. **Predictor:** calibrated Logistic Regression → HistGradientBoosting if earned.
4. **Scheduler:** weighted multi-objective priority, argmax, frozen weights,
   hard freshness guarantee.
5. **Baselines:** sequential, random, ε-greedy heuristic — same seeded world.
6. **Evaluation:** 30 seeds × 7 scenarios × all strategies, paired tests, 95%
   CIs, ablation A-E, weight sensitivity, robustness grid.
7. **Dashboard:** live single-strategy step-through + precomputed comparison;
   per-decision stacked-bar explanation; emerging-signal event banner.

---

## 9. Experiment methodology

- **Unit:** one episode = (scenario, seed, strategy) for `duration_slots`.
- **Fairness:** identical world realisation per (scenario, seed) across
  strategies; separate seeded agent RNG.
- **Replication:** ≥ 30 seeds/scenario; mean ± 95% CI (bootstrap, 10k resamples).
- **Comparison:** paired (per seed) t-test **and** Wilcoxon; report effect size.
  Claim improvement only when significant *and* material.
- **Generalisation:** tune on seeds 0-19, freeze, evaluate on 30-49 and on
  held-out scenario types.
- **Metrics:** detection rate, mean detection delay, time-to-first-detection,
  emerging-signal discovery delay, coverage time, scan efficiency (useful
  detections / scans), missed-detection rate, redundant-scan rate, decision
  latency.
- **Censoring:** never-detected episodes are right-censored, reported as miss
  rate — not dropped from delay means.
- **No fabrication:** unfavourable results go in the results + limitations
  sections. If the emerging-signal demo underwhelms, adjust the *scenario or
  algorithm* scientifically and re-run — never the numbers.

**Honest prior on the emerging-signal result:** for a single newly-active channel
among many, raw discovery delay may match round-robin. The decisive adaptive
advantages are (a) **post-discovery tracking** (priority ramps in 2-3 slots) and
(b) **aggregate** detection delay across all channels (no scans wasted on
confidently-empty spectrum). Scenario 4 is designed so confidently-empty channels
are suppressed, which *can* also speed discovery — but the experiment decides.

---

## 10. Dashboard wireframe

```
+----------------------------------------------------------------------------+
|  AI-Driven Adaptive RF Scan Intelligence        [Start] [Pause] [Reset]     |
+------------+---------------------------------------------------------------+
| CONTROLS   |  (1) CURRENT RF ENVIRONMENT   RSSI/noise per channel bars     |
| Scenario v |  (2) AI PREDICTION            p(active) + CI band per channel  |
| Seed  [42] |  (3) SCHEDULER   NOW -> CH6 | NEXT -> CH9  prio 0.91          |
| Speed ---o |         pred +0.42 | explore +0.28 | fresh +0.15 |            |
| Strategy v |         trend +0.09 | redundancy -0.03    (stacked bar)       |
| [Compare]  |  (4) SCAN HISTORY   raster: time x channel, dot = detection   |
|            |  (5) PERFORMANCE   precomputed, 30 seeds, 95% CI              |
|            |         mean detection delay:  Seq 2.9 | Rand 3.4 | Adap 1.3  |
|            |  (6) EVENT   NEW ACTIVITY: CH9 @ t=20.0s                      |
|            |         Adaptive discovered @ 23.4s (d 3.4s)                  |
|            |         Sequential discovered @ 28.1s (d 8.1s)               |
+------------+---------------------------------------------------------------+
```

Live mode runs one lightweight strategy loop (capped). Section (5) loads a
precomputed benchmark file — never recomputes live. Theme: neutral engineering,
single accent, monospace numerics. No gauges, no military skin.

---

## 11. Development roadmap

Each phase: implement → run tests → run mini-demo → inspect → explain → approve →
next. Vertical end-to-end slice by Phase 6; demoable MVP by Phase 9.

| Ph | Deliverable | Exit criteria |
|---|---|---|
| 1 | Repo skeleton, configs, schema dataclasses, logging, CI-ready pytest | pytest green; `rfscan --help` / `info` work |
| 2 | RF simulator: channel plan, Gilbert-Elliott occupancy, 5 behaviours, noise, `observe()` | reproducibility test (same seed ⇒ identical, order-independent stream); occupancy stats match config |
| 3 | Baselines: sequential, random, heuristic + `Scanner` + `ObservationStore` | sequential visits 1..N..1; random reproducible; coverage test |
| 4 | Scenario engine (7) + `ExperimentRunner` + first metrics | each scenario replayable; baseline benchmark CSV |
| 5 | Feature pipeline + ML bake-off + calibration | leakage test; model cards + metric table; predictor chosen |
| 6 | `AdaptiveScheduler` v1 — thin loop end-to-end | full loop runs; latency < 5 ms/slot; beats random on delay |
| 7 | Online feedback + emerging-signal adaptation | Scenario 4: emerger priority ramps post-detection |
| 8 | Full experiment framework + ablation A-E + robustness | raw/summary CSV + plots; paired tests; ablation table |
| 9 | Streamlit dashboard, 7 sections, live + precomputed | judge can run a scenario and see prediction + explanation + comparison |
| 10 | Test hardening | coverage ≥ ~80% on core packages; e2e test |
| 11 | Optimization + frozen weight tuning + optional multi-receiver | benchmark runtime acceptable; sensitivity plot |
| 12 | SIH demo: deterministic emerging-signal script + timed comparison | `rfscan demo` reproduces the story with real measured Δ |
| 13 | README (21 sections) + `docs/sih_pitch.md` (A-M) + model cards | every number traceable to a run artifact |

---

## 12. Risk analysis

| # | Risk | L | I | Mitigation |
|---|---|---|---|---|
| R1 | ML feedback loop / distribution shift | H | M | mixed-policy training data; Beta layer floor; closed-loop eval; optional refit |
| R2 | Emerging-signal discovery ≈ sequential | M | M | honest framing (post-discovery tracking + aggregate); Scenario 4 suppresses empty channels |
| R3 | "Your simulator isn't real EW" | H | M | cite Gilbert-Elliott / occupancy literature; configurable params; explicit realism-limits list |
| R4 | Scheduler weights overfit to 7 scenarios | M | M | tune on train, freeze, evaluate on held-out types; sensitivity sweep |
| R5 | Scope vs time (13 phases) | M | H | Phases 1-9 = demoable MVP; thin vertical slice by Phase 6 |
| R6 | Streamlit live-loop performance | M | L | precompute benchmarks; capped live loop; vectorised predict |
| R7 | Fairness bug — scan-order leaks into world | L | H | `(seed, slot, channel)`-keyed draws; invariant test; separate agent RNG |
| R8 | Accidental cheating — truth in decision path | L | H | `Observation` has no truth field (structural); eval silo; test |
| R9 | Poor model calibration → mis-scaled priority | M | L | isotonic/Platt calibration; Brier tracked; weights tuned post-calibration |

---

## 13. SIH innovation statement

We treat the EW receiver's "what do I scan next" question as a *sequential
decision problem under partial observability* — a **restless multi-armed bandit
over channel-occupancy beliefs** — rather than as signal classification. The
scheduler fuses (1) a **learned near-term activity prediction** from
signal-history features, (2) a **decaying Bayesian occupancy belief** whose
uncertainty rises the longer a channel goes unscanned, and (3) **operational
terms** for freshness and rising-activity trend, into one transparent priority
score. It explicitly trades exploitation against exploration, so it keeps
discovering *new* emitters instead of fixating on known ones. It is **measured**,
not asserted: against sequential/random baselines on identical seeded scenarios,
with multi-seed CIs, paired significance tests, a component ablation, and a
robustness sweep. Every decision is explainable as a stacked bar of contributing
terms. Theoretical anchor: the Whittle index for restless bandits.

---

## 14. How this differs from a basic ML project

| Basic ML classifier | This project |
|---|---|
| One fixed dataset, train/test once | Closed loop — model output drives a controller whose actions change future data |
| Success = accuracy / F1 | Success = operational latency (detection / discovery delay); accuracy is a diagnostic |
| "Baseline" = dummy classifier | Baselines are real competing policies on the same world |
| Full label visibility at inference | Partial observability — no ground truth at decision time |
| No exploration concept | Explicit exploration vs exploitation; argmax(p) deliberately rejected |
| Static world | Non-stationary world; online belief adaptation; emerging-emitter discovery |
| "92% accuracy" | Multi-seed CIs, paired tests, ablation, held-out scenario types, robustness grid |
| Black-box prediction | Per-decision explanation as term contributions |

---

## 15. Expected judge questions & answers

**Why ML at all?** Activity is a nonlinear function of recent SNR, noise,
detection history and trend; a learned model calibrates P(active next slot)
better than a hand rule, and it is the *input* to the scheduler. The ablation
reports how much ML adds over the Bayesian belief layer — honestly.

**Why not sequential scanning?** Equal dwell on empty and active spectrum; fixed
worst-case revisit of N slots. The benchmark quantifies the detection-delay gap
on identical worlds.

**Why not just scan argmax p?** It collapses onto known emitters and never
revisits low-probability channels, missing new activity. We add the Beta
uncertainty term and a hard freshness guarantee; ablation B vs E shows the
failure mode.

**How do you stop the model ignoring rare channels?** Uncertainty term is
largest for rarely-scanned channels; freshness term grows with staleness and is
bounded below; `channel_index` is excluded from features.

**How does exploration work?** Uncertainty + freshness bonuses in the priority
score (optionally softmax sampling) — same family as UCB, which we also
benchmark.

**How do you handle an emerging signal?** Freshness guarantee scans it
periodically before it appears; on first detection its posterior and predicted
probability jump, priority ramps within 2-3 slots, scan share rises. We show
measured discovery Δ and post-discovery miss rate vs sequential.

**How do you avoid data leakage?** `Observation` has no truth field (separate
eval silo); features use only history ≤ t; temporal split by scenario-seed and
within-episode causality; a test asserts the scheduler API cannot access truth.

**How did you compare against baselines?** Same seeded world realisation for
every strategy (draws keyed by `(seed, slot, channel)`); 30 seeds × 7 scenarios;
paired t-test + Wilcoxon; 95% bootstrap CIs; effect sizes.

**How do you measure improvement?** The nine metrics in §9, all with CIs.

**Why this ML model / scheduler?** Bake-offs (§6, §7), re-ranked by closed-loop
detection delay. Ship the simplest that wins the loop.

**How realistic is your simulator?** Gilbert-Elliott occupancy (standard in
spectrum-occupancy modelling) + five behaviour patterns + noise-floor drift +
energy detector with false alarms/misses. An abstraction of the
*decision-relevant* dynamics, not an RF propagation model; limits listed
explicitly; parameters configurable.

**Can it adapt when the environment changes?** Yes — Scenarios 6 & 7 are
non-stationary; the decaying belief and trend term respond; online refit
available; adaptation latency plotted.

**Can it scale to more channels?** Priority is O(N)/slot with vectorised predict;
sub-5 ms for N in the hundreds. Multi-receiver top-K is a built extension.

**What if the ML prediction is wrong?** The Beta belief layer bounds the damage
(policy-independent, self-corrects next scan); the freshness guarantee prevents
permanent starvation. Ablation "belief-only, no ML" quantifies the floor.

**What are your limitations?** Discrete-slot time; synthetic world (no real
propagation/modulation); single-receiver by default; weights tuned on a finite
scenario set; ML subject to closed-loop distribution shift. All in the README.

**Why different from a normal classifier?** See §14.

---

## 16. Anti-leakage enforcement & implementation notes

### 16.1 Structural anti-leakage (referenced from `perception/schema.py`)

The agent/evaluation boundary is enforced by *types*, not discipline:

- `Observation` has fields `{slot, channel_index, rssi_dbm, noise_dbm, snr_db,
  observed_detection}` and **no truth field**. `GroundTruth`
  (`{slot, channel_index, occupied, true_signal_dbm}`) is a separate type.
- `RFEnvironment.observe(c)` returns only an `Observation`;
  `RFEnvironment.ground_truth(c)` / `truth_snapshot()` return `GroundTruth` and
  are called only by the experiment layer.
- `tests/test_schema.py` asserts the `Observation` field set and the absence of
  any `occup*` / `truth` / `ground*` token; `tests/test_environment.py` asserts
  `env.observe(...)` exposes no ground-truth attribute.

### 16.2 Phase 2 simulator — realisation choices

Concrete decisions made when implementing §8.1, with the doc updated to match:

- **Fairness draws.** `simulator/rng.py::draw_rng(seed, slot, channel)` returns a
  `numpy` generator seeded from `SeedSequence((seed, slot, channel))` — not the
  builtin `hash` (which is per-process randomised). Every `(slot, channel)` cell
  consumes its generator in a fixed order (emitter draws → noise-floor
  innovation → RSSI AWGN → noise AWGN → false-alarm u → missed-detection u), so
  observation order and partial observation cannot perturb the world. Tested.
- **Occupancy.** `simulator/occupancy.py::GilbertElliott` (2-state Markov, `p01`,
  `p10`) + `EmitterProcess` overlays: `PERSISTENT`/`EMERGING` = plain Markov
  (EMERGING forced OFF before `activation_slot`); `INTERMITTENT` = periodic
  duty-cycle gate around the Markov chain; `BURSTY` = low `p01`, high `p10`;
  `FADING` = `p01` scaled by `exp(-fade_rate · elapsed)`. Per-behaviour
  parameter presets in `_BEHAVIOR_DEFAULTS`, all overridable per emitter and via
  `NonStationaryEvent.param_overrides`.
- **Noise floor.** Modelled as a **mean-reverting AR(1)** process about
  `floor_dbm` (not an unbounded random walk, which would drift tens of dB over an
  episode). This adds one field to `NoiseModelConfig`: `floor_drift_rho`
  (default 0.9); `floor_drift_std_db` is now the *stationary* std of the drift.
- **Measurement (dB domain, internally consistent, not physically exact).**
  `rssi_dbm = 10·log10(signal_lin + noise_lin) + AWGN`,
  `noise_dbm = noise_floor_dbm + AWGN`, `snr_db = rssi_dbm − noise_dbm`.
  Energy detector fires on `snr_db ≥ detection_threshold_snr_db`, then
  `false_alarm_rate` / `missed_detection_rate` flip a fraction of empty /
  occupied outcomes.
- **Scenario factories.** `simulator/scenario.py` gains
  `ScenarioConfig.build_environment(seed)` and a second worked example,
  `example_emerging`. The seven *tuned* benchmark scenarios + registry remain
  Phase 4 (`simulator/scenarios.py` in §4).

### 16.3 Phase 3 sensor + baselines — realisation choices

- **`Scanner` placement.** §1.1 lists a distinct SENSOR LAYER but §4 gives it no
  module. It lives at `perception/scanner.py` (agent-side data acquisition,
  tightly coupled to the store) rather than in `simulator/` (which is the world
  itself). `Scanner.scan(c)` calls only `RFEnvironment.observe` — never
  `ground_truth` — and never advances time.
- **Read-only store contract.** `perception/store.py` defines a `ReadableStore`
  Protocol (the read surface: counts, rates, staleness, per-channel record
  views, `coverage`); `ObservationStore` also has `append`. Schedulers are typed
  against `ReadableStore`, so a strategy structurally cannot inject records.
  Per-channel aggregates are maintained incrementally on `append` (O(1) lookups).
- **`Scheduler` protocol.** `{select_next(store, slot) -> int, update(record),
  explain() -> Mapping[str,float], reset()}` plus a `name` class attribute.
  `reset()` and `name` are the two additions over §4's three-method sketch —
  needed for episode reuse and benchmark labelling.
- **Baselines.** `SequentialScheduler` (cursor advances in `select_next`, so call
  it once per slot), `RandomScheduler` and `HeuristicScheduler` (ε-greedy on
  empirical detection rate; optimistic prior 1.0 forces one full coverage sweep;
  ties broken by staleness then index). Random and heuristic own a **separate
  agent RNG** (`numpy.default_rng(seed)`) — changing the agent seed changes the
  scan pattern, never the RF world (tested).

### 16.4 Phase 4 scenario engine + runner + metrics — realisation choices

- **World RNG.** `simulator/rng.py` switched from `SeedSequence` hashing to a
  counter-based **Philox** generator (`key = world_seed`, `counter` packed from
  `(slot, channel)`). Same `(seed, slot, channel)` keying and guarantees, ~25%
  cheaper construction on the benchmark hot path. (Deeper optimisation of the
  per-cell generator cost is still Phase 11.)
- **Scenarios.** `simulator/scenarios.py` — seven `() -> ScenarioConfig`
  factories in a `SCENARIOS` registry, `make_scenario(name, seed,
  duration_slots=None)`, `list_scenarios()`. All 12 channels / 800 slots / 0.1 s
  so cross-scenario numbers compare. Uses the Phase 1/2 `ScenarioConfig` +
  `EmitterSpec` + `NonStationaryEvent` — no parallel config system.
- **`run_episode(env, scheduler, budget, *, agent_seed=None) -> EpisodeResult`.**
  Resets env + scheduler, then per slot: `select_next` (timed) → `scan` →
  `update` → record `env.occupancy_snapshot()` → `env.step()`. `EpisodeResult`
  holds numpy arrays (`scanned_channel`, `observed_detection`, `occupancy`
  [ground truth, eval only], `decision_latencies_s` [nondeterministic]).
  `env.occupancy_snapshot()` is a new lightweight method (no `GroundTruth`
  objects) for the hot path.
- **Metrics (`experiments/metrics.py`), censored-aware.** Core unit = *activity
  interval* (maximal true-occupancy run on a channel). An interval is *detected*
  iff some slot in it scanned that channel with `observed_detection=True`;
  *censored* iff undetected and still active at the last slot. `detection_rate`,
  `missed_detection_rate`, `mean_detection_delay_slots` (over detected intervals
  only — misses are reported as counts, never folded in as delay 0),
  `time_to_first_detection_slots`, `emerging_discovery_delay_slots` (from the
  EMERGING emitter's `activation_slot`), `channel_coverage`,
  `coverage_time_slots`, `on_target_scan_rate`, `scan_efficiency` (true positives
  / scans), `redundant_scan_rate` (observation-based: repeat scan of a channel
  within `window` slots with no detection in the window). Full definitions in the
  module docstring.
- **Benchmark (`experiments/benchmark.py`).** Phase-4 form: run the
  `scenarios x world_seeds x strategies` grid, one `EpisodeMetrics` row per
  episode → `raw_results.csv`; `summarize()` → mean/std/count per
  `(scenario, strategy)` → `summary.csv`. Wired to `rfscan benchmark [--seeds N]`.
  Fairness: scenario built once per `(scenario, seed)`, a fresh env at that seed
  per strategy, `agent_seed = world_seed`. **Phase 8** adds bootstrap CIs, paired
  t-test / Wilcoxon, the A–E ablation, and the robustness sweep (this is why §4
  tags `benchmark.py` Phase 8 — the inferential layer, not the grid, is what
  Phase 8 owns).

### 16.5 Phase 5 ML pipeline — realisation choices

- **`FeatureBuilder` (`perception/features.py`).** 17 features per channel
  (§5 calls it "~16" — this is that list exactly, nothing added beyond it),
  built purely from `ReadableStore` history strictly before the query slot
  plus the query slot itself (used only for relative staleness, never as an
  absolute value). Pure/stateless: `build(store, slot)` is safe to call
  repeatedly and never mutates the store. Two sentinels stand in for "no
  evidence yet": `MISSING_VALUE = -999.0` for a continuous reading that was
  never taken, `MAX_STALENESS = 1000.0` for "never scanned" / "never
  detected" (chosen above the 800-slot benchmark episode length so it reads
  unambiguously as "maximally stale"). `decaying_beta_posterior()` implements
  §7's belief-update rule as a plain function of `(records, slot)` — this is
  a *feature*, not the Phase 6 `BeliefState`; the two will share the same
  math but `BeliefState` remains a Phase 6 scheduler-side object with its own
  lifecycle. A hard runtime guard (`FeatureBuilder` raises if the store
  already holds a record at or after the query slot) makes "features can't
  see the future" a structural property, not just a convention — this is the
  leakage test in `tests/test_features.py`/`tests/test_leakage.py`.
- **Prediction target.** `y = 1[channel c occupied at slot t]`, `X` built
  from history strictly before `t` — "P(active at the next decision
  opportunity | observations available so far)". Ground truth
  (`RFEnvironment.occupancy_snapshot()`) is used only to build this label in
  the offline collector (`models/train.py::collect_episode`), never as a
  feature and never visible to a scheduler.
- **`models/base.py` `Predictor` Protocol.** `{name, predict_proba(X),
  explain(x)}`, matching §4's sketch exactly except `partial_fit` is left out
  of the structural contract (few Phase 5 candidates support true online
  updates; nothing here forecloses a Phase 6+ predictor that adds it).
- **`DecayingBetaPredictor` (`models/baseline_beta.py`).** The non-ML
  reference. Reads `beta_posterior_mean`/`_std` straight off the
  `FeatureBuilder` output rather than re-deriving the same math — one array
  index, genuinely O(1) beyond feature-building.
- **`SklearnPredictor` (`models/sklearn_model.py`).** Wraps any
  `predict_proba`-capable sklearn estimator; optional post-hoc calibration
  (`isotonic`/`sigmoid`) fit on a held-out validation fold via
  `CalibratedClassifierCV(FrozenEstimator(fitted), ...)` — scikit-learn 1.6+
  removed `cv="prefit"` in favour of this wrapper. `explain(x)` returns a
  linear per-feature decomposition (`coef * scaled_value`) when the fitted
  estimator is a `Pipeline` with a linear `"lr"` step, else falls back to
  `{"predicted_proba": p}` for non-linear models (documented, not hidden).
- **Data collection (`models/train.py`).** A mix of the three Phase 3/4
  baseline schedulers (sequential, random, and a more exploration-heavy
  `epsilon=0.3` heuristic variant, `heuristic_explore`) generates
  `(features, label)` rows across all 7 scenarios — the §3.2 distribution-
  shift mitigation. Temporal split: rows grouped by whole `(scenario,
  world_seed)` episodes (never split at the row level, to avoid near-
  duplicate adjacent-slot rows leaking across folds), seeds assigned to
  folds in increasing order. This module runs a ratio-matched, scaled-down
  version of §3.2's `train=0-19/val=20-29/test=30-49` plan
  (`TRAIN_SEEDS=0-9`, `VAL_SEEDS=10-14`, `TEST_SEEDS=15-24`) to keep runtime
  and artifact size practical for this phase; Phase 8's closed-loop
  benchmark already runs the full 30-seed × 7-scenario grid separately, for
  operational metrics rather than offline model fitting.
- **Bake-off & model selection.** `choose_final_model()` compares
  `hist_gradient_boosting` against `logistic_regression` on the *test* split
  only for reporting (never for tuning): HGB is preferred only if its test
  PR-AUC beats LR's by more than a 0.02 margin **and** its calibration (ECE)
  is not worse by more than 0.02 — otherwise LR is kept as the simpler,
  faster, more explainable model, per §6/§8's "ship the simplest that wins."
  Real measured numbers are in `artifacts/results/model_bakeoff.csv` and
  `docs/model_cards/*.md` (generated by `rfscan train`), not reproduced here
  since they are a run artifact, not a design decision.
- **Evaluation (`models/evaluation.py`).** `classification_metrics()`
  (precision/recall/F1 at threshold 0.5, PR-AUC, ROC-AUC — `None` when
  `y_true` is single-class rather than raising — and Brier score),
  `recall_at_precision()` (excludes `precision_recall_curve`'s synthetic
  trailing "classify nothing as positive" point so a target is only ever met
  by a real, usable threshold), `calibration_report()` (equal-width
  reliability diagram + expected calibration error), and
  `measure_inference_latency()` (wall-clock, seconds/row, warm-up call
  excluded).
- **Dataset investigation.** The SIH26055 problem statement references the
  Hugging Face `alan-turing-institute/turing-synthetic-radar-dataset`.
  Inspected (dataset card) during Phase 5: rows are Pulse Descriptor Words
  (Time of Arrival, Centre Frequency, Pulse Width, Angle of Arrival,
  Amplitude) for a *pulse-deinterleaving* task (grouping ~4 billion pulses by
  unknown emitter) — no discrete time slots, no fixed channel plan, a
  different ML problem from this project's discrete-slot channel-occupancy
  scan scheduling. Not integrated; the simulator remains the sole data
  source (see each model card's "Investigated but not integrated" section).

### 16.6 Phase 6 adaptive scheduler — realisation choices

- **`BeliefState` (`scheduler/belief.py`).** A live, incremental analogue of
  `perception/features.py::decaying_beta_posterior` (which replays a
  channel's full history from scratch — fine as a training-time *feature*,
  too slow as a per-slot, per-channel live object under the < 5 ms/slot
  budget). `BeliefState` keeps `(alpha, beta)` numpy arrays and updates them
  in O(1) per channel: `decay()` applies one uniform forgetting-factor step
  to *every* channel each slot (`a <- lambda*a + (1-lambda)*prior_a`, same
  for `b`), then `update(c, y)` adds the scanned channel's evidence on top.
  This is a deliberate reading of S7's "every unscanned channel decays" rule:
  at the top of the closed loop the scheduler does not yet know which
  channel it is about to scan, so there is no way to exempt it from decay in
  advance; decaying uniformly first and then layering fresh evidence on the
  scanned channel is the standard exponential-forgetting recursive-Bayes
  filter and produces the same qualitative behaviour (unscanned channels
  relax toward the prior and grow less certain; the scanned channel
  sharpens). Also exposes `sample()` (per-channel Beta draw, own RNG) for the
  Phase 8 Thompson-sampling alternative — unused by AdaptiveScheduler v1's
  argmax rule.
- **`PriorityPolicy` (`scheduler/policy.py`).** Implements S7's weighted sum
  exactly for `pred`/`explore`/`fresh`/`trend`. S7 does not give a numerical
  definition for `redundancy(c)` beyond "recently scanned & confidently
  empty" — implemented as
  `exp(-staleness(c) / redundancy_tau) * (1 - p(c))`: large exactly when a
  channel was scanned a moment ago (small staleness) *and* the predictor is
  confident it is empty (`p(c)` near 0), decaying smoothly as either grows,
  using only quantities the scheduler already has (no new state). Returns a
  `PriorityBreakdown` per channel (the five signed term *contributions*, not
  raw inputs) so the chosen channel's breakdown is exactly the S10 dashboard
  stacked bar with no further transformation.
- **`AdaptiveScheduler` (`scheduler/adaptive.py`).** No online learning in
  v1 — the predictor is injected pre-fit, matching `Predictor`'s structural
  contract deliberately excluding `partial_fit` (S16.5); only `BeliefState`
  adapts within an episode. Reuses `FeatureBuilder`'s `slots_since_last_scan`
  and `activity_trend` columns directly for `staleness`/`trend` rather than
  recomputing them from the store a second time.
  **Hard freshness guarantee:** if any channel's staleness reaches
  `max_revisit_slots` (default `3 * n_channels`), it is force-selected
  regardless of priority — tie-broken toward the stalest, then lowest index.
  Because `FeatureBuilder.MAX_STALENESS` (1000, S16.5) is far larger than any
  reasonable `max_revisit_slots`, every never-scanned channel already exceeds
  the cap from slot 0: **the guarantee subsumes initial coverage, not just
  steady-state revisit**, matching S7's stated intent ("so every channel —
  including a future emerger — is eventually scanned") without any special
  cold-start logic. Verified in `tests/test_adaptive.py`.
  **Selection rule:** `softmax_temperature=None` (the S7/S8 recommended
  final: weighted priority, argmax) is the config default; a numeric
  temperature switches to softmax sampling on an internal agent RNG, for the
  S7 "weighted priority + softmax" variant (Phase 8 bake-off candidate),
  already implemented and tested, not yet the default.
- **Predictor wiring (`models/loader.py::load_predictor`).** `kind="beta"`
  returns `DecayingBetaPredictor` directly (no I/O). For any other `kind`,
  `rfscan train` now persists *every* bake-off candidate as
  `model_<name>.joblib` next to the chosen model (`train.py`'s
  `Phase5Result.candidate_model_paths`, an additive Phase 6 extension —
  `choose_final_model`'s offline-reporting winner and its output files are
  unchanged), so `load_predictor` can return a *specific* candidate by name,
  falling back to the single `model_config.artifact_path` file (the
  reporting winner) if no per-candidate file exists. Kept out of
  `AdaptiveScheduler` itself (constructor takes an already-built `Predictor`)
  so scheduler tests never need a trained artifact on disk, mirroring how
  Phase 5's own tests avoid depending on `artifacts/model.joblib`.
- **Live predictor choice — a real closed-loop finding, not a design guess.**
  Phase 5's bake-off chose `hist_gradient_boosting` on offline PR-AUC/
  calibration (S16.5) — that conclusion is correct and unchanged. But
  running it *inside* `AdaptiveScheduler`'s per-slot loop for the first time
  (this phase) surfaced an operational cost the offline bake-off cannot see:
  `CalibratedClassifierCV`-wrapped `HistGradientBoostingClassifier.
  predict_proba` on a 13-row batch (one row per channel, once per slot) took
  a measured 4.7-4.9 ms/call — profiling traced this to sklearn's per-tree
  Python-level call overhead across 181 boosting iterations, which does not
  amortize at this batch size (confirmed by re-measuring at max_iter
  25/50/100/200: latency scales near-linearly with tree count, ~1.5 ms
  fixed overhead + ~0.015 ms/tree). Logistic Regression's single matrix
  multiply measured 1.2-1.3 ms/call under the same conditions. This is
  *exactly* what S6/S8's "ship the simplest model that wins the loop" is
  for: `configs/default.yaml`'s `model.kind: logistic_regression` (already
  the project default, unchanged) is confirmed as `AdaptiveScheduler`'s live
  predictor for this reason, while `hist_gradient_boosting` remains the
  correctly-documented offline bake-off winner in `docs/model_cards/` and
  `model_bakeoff.csv`. Measured with the real trained model on the full
  7-scenario grid: mean decision latency 1.13 ms/slot (max 1.19 ms) — well
  inside the < 5 ms budget, versus 5.4 ms mean (over budget) with
  `hist_gradient_boosting` live. This offline-vs-live predictor split is a
  direct instance of the batch-latency-vs-single-row-latency distinction:
  `model_bakeoff.csv`'s per-row latency figures were measured over
  2-million-row batches, where per-tree call overhead amortizes away — they
  do not predict small-batch, real-time serving latency, which only shows up
  once a model is actually run inside the closed loop.
- **Benchmark wiring.** `experiments/benchmark.py` accepts `"adaptive"` as a
  named strategy (`VALID_STRATEGIES`) with its own `ModelConfig`/
  `SchedulerWeights`/`BeliefConfig` fields on `BenchmarkConfig`, but the
  default `strategies` tuple is unchanged (still the three Phase 4
  baselines) — the full scenarios × seeds × strategies grid *including*
  adaptive, with bootstrap CIs and the paired significance tests, is Phase
  8's "inferential layer" (S4), not Phase 6's. Phase 6's own exit criterion
  (full loop runs; latency < 5 ms/slot; beats random on delay) is verified
  directly in `tests/test_integration_phase6.py` using
  `DecayingBetaPredictor` (no trained-artifact dependency, same reasoning as
  Phase 5's own tests) and confirmed again as a real run with the actual
  trained, calibrated `logistic_regression` model (`rfscan train`'s output,
  the live default per the finding above) across the full 7-scenario × 3-seed
  grid: latency 1.13 ms/slot mean; redundant-scan rate far below heuristic's
  in every scenario (e.g. `normal`: 0.11 vs 0.38); and, in `emerging_signal`
  — the project's central motivating case — heuristic discovered the
  emerging emitter in 0 of 3 seeds (fully censored) while adaptive discovered
  it in all 3. Raw `mean_detection_delay_slots` is not yet uniformly better
  than sequential/heuristic across every scenario — expected, since the
  scheduler weights are still `configs/default.yaml`'s untuned defaults;
  weight tuning against train scenarios, frozen and evaluated on held-out
  ones, is Phase 11's job (S11), and the full paired-significance comparison
  is Phase 8's. Not reproduced in full here since a benchmark result is a
  run artifact, not a design decision, per S9's "no fabrication" rule.

### 16.7 Phase 7 online feedback + emerging-signal adaptation

**The mechanism already existed; Phase 7 makes it observable and proves it.**
`AdaptiveScheduler` (S16.6) already updates `BeliefState` from every scan's
hit/miss and rebuilds `FeatureBuilder` features fresh from the growing store
every slot — a hit or miss already changes the very next decision, before
this phase. What Phase 7 adds is the ability to *show* that, and a set of
tests that actually exercise it end-to-end rather than asserting it:

- **`AdaptiveScheduler.all_breakdowns()` / `belief_snapshot()`.** `select_next`
  already computes a `PriorityBreakdown` for *every* channel (needed to
  argmax); it only ever kept the chosen one (`explain()`, S10's one-decision
  stacked bar). `all_breakdowns()` retains the full list instead of
  discarding it — zero extra computation, since the values already exist on
  the hot path. `belief_snapshot()` similarly exposes the live `BeliefState`
  mean/std. Both are read-only introspection, additive to the `Scheduler`
  protocol (baselines don't need them), and are what makes "channel *c*'s
  priority right now, even though it wasn't picked" checkable.
- **`run_episode`'s `on_slot` hook.** One optional keyword arg, default
  `None`, called once per slot after `scheduler.update(record)` and before
  `env.step()`. Omitted, it costs nothing and every existing Phase 4 caller
  is untouched (`tests/test_runner.py` proves this explicitly). It's the only
  change made to `experiments/runner.py` this phase — everything else in
  Phase 7 is new, additive modules.
- **`experiments/adaptation.py` (new module).** `trace_adaptive_episode`
  drives an ordinary `run_episode` with the `on_slot` hook wired to
  `all_breakdowns()`/`belief_snapshot()`, producing a `PriorityTrace` for
  chosen channels — a read-out of the scheduler's own state, not a second
  decision pass; the traced episode is byte-identical to a plain run
  (tested). `emerging_adaptation_metrics` computes pre/post-activation scan
  counts and detection rates for a scenario's EMERGING channel, reusing
  `compute_episode_metrics`'s existing discovery-delay figure rather than
  re-deriving it. Both live in the experiment/evaluation layer — same trust
  boundary as `metrics.py`, ground truth (`occupancy`, `emerging_channels`)
  in, never out to a scheduler.

**Predictor: deliberately not touched.** S16.5 left `partial_fit` out of the
`Predictor` protocol on purpose, flagging that "a Phase 6+ predictor that
adds it" could do so later if needed. Phase 7 concludes it is **not**
needed: `BeliefState` already gives genuine online adaptation with O(1)
per-slot updates, and `FeatureBuilder` already rebuilds every feature
(`last_detection`, `detection_rate_recent`, `beta_posterior_mean`, ...) fresh
from the store every call, so even `logistic_regression`'s frozen
coefficients see live evidence on every decision — no retraining, no
`partial_fit`, no risk of the "expensive per-slot retraining" this phase was
explicitly warned against. Confirmed, not assumed: the flagship demo below
runs on the *live default* predictor and shows exactly the required
behaviour without any predictor-side change.

**Hit/miss semantics, explicit.** A "hit" is `ScanRecord.detected` (i.e.
`Observation.observed_detection`) being `True` on a scan of channel *c*;
whatever the energy detector reported, possibly a false positive — the
scheduler never sees whether it was real (S16.1). `BeliefState.update(c, y)`
adds `y` to `alpha[c]` (hit) or `1-y` to `beta[c]` (miss) directly, no
ground truth involved. `decay()` (called once per slot, S16.6) relaxes
every channel's `(alpha, beta)` a step toward the prior regardless of
whether it was scanned, so old evidence's influence shrinks geometrically —
a channel unscanned for many slots ends up back near `Beta(prior_alpha,
prior_beta)`, maximally uncertain again. **A real, non-obvious consequence,
found while writing this phase's tests, not by design:** because `decay()`
runs for every channel every slot and a hit only adds evidence (never
directly boosts `explore`), the `explore` priority term (belief std) can
*decrease* right after a hit — more evidence concentrates the posterior,
which lowers uncertainty. A predictor that cannot see the hit (fixed
output, ignoring features) can therefore show *lower* priority immediately
after a hit, since none of `pred`/`trend`/`fresh`/`redundancy` moved either.
This only matters with a feature-blind predictor; `DecayingBetaPredictor`,
`logistic_regression`, and `hist_gradient_boosting` all read features that
do move with a hit, so `pred` (and hence overall priority) still rises in
practice — proven in `tests/test_adaptive.py`, not merely argued.

**Non-stationarity.** No new machinery — `changing_distribution` (S4,
channels 0-2 active first half / channels 9-11 active second half, flip at
slot 400) already existed as a Phase 4 benchmark scenario.
`tests/test_integration_phase7.py` runs it through `AdaptiveScheduler` and
confirms, via `trace_adaptive_episode`, that belief for the early-active
channel is measurably higher late in the first half than late in the
second, and the reverse for the late-active channel — decay is what makes
this possible; a belief layer with `decay_lambda=1.0` (no forgetting) would
stay pinned to whichever half it learned first.

**The flagship demo, run for real** (`emerging_signal`, channel 5 silent
until slot 300, `AdaptiveScheduler` with `DecayingBetaPredictor`, world seed
0, no ground truth to the scheduler, default untuned weights — not tuned
for this seed):

| slot | event | belief mean (ch 5) | priority (ch 5) | scanned? |
|---|---|---|---|---|
| 299 | (pre-activation) | 0.30 | 0.62 | no |
| 308 | first hit (discovery, delay 8 slots) | 0.30 → **0.49** | 0.68 | **yes** |
| 309 | immediately after | 0.49 | **0.53** (dip — freshness reset) | no |
| 316 | 8 slots later | 0.49 | 0.68 (ramped back) | no |
| 317 | second hit | 0.60 | 0.69 | **yes** |
| 320-337 | sustained lock-on | 0.60 → **0.85** | 0.69 → **0.94** | scanned nearly every slot |

Belief and post-discovery scan share both rise clearly (scan share: ~0%
before discovery → effectively continuous after, per
`tests/test_integration_phase7.py`); priority does **not** rise monotonically
at the instant of the hit — it dips one slot (the real property above) then
ramps over roughly 8 slots to a new, sustained plateau almost double its
pre-discovery level. This is a genuine run, not a fabricated table — the
dip is reported alongside the rise per S9/S12's "no fabrication" rule, and
is itself informative: it is exactly the mechanism (`explore` losing ground
right as `fresh` bottoms out) that a judge-facing explanation should name
rather than paper over.

**Performance.** No change to the per-slot decision cost: `all_breakdowns`/
`belief_snapshot` return already-computed references; `on_slot` costs
nothing when omitted (every existing benchmark/CLI path omits it). Measured
after this phase, real `logistic_regression` live predictor, three
scenarios: mean decision latency 1.05-1.07 ms/slot — unchanged from Phase
6's 1.13 ms within run-to-run noise, comfortably inside the 5 ms budget.

**Scope discipline.** No UCB/Thompson/contextual-bandit code, no ablation,
no bootstrap CIs or paired tests, no dashboard — all Phase 8/9, untouched.
`PriorityPolicy`/`BeliefState`'s formulas (S7, S16.6) are unchanged; Phase 7
added observability and tests around them, not new decision logic.

### 16.9 Phase 8 experimental/scientific validation

The question this phase answers: does the adaptive scan strategy actually
improve scanning performance, under which conditions, and which components
are responsible? Not "does adaptive win everywhere" — it doesn't, and this
phase's job is to say honestly where and why.

**New modules, all additive, no Phase 1-7 rewrites.** `experiments/stats.py`
(`bootstrap_ci`: percentile bootstrap, NaN/censored values dropped;
`paired_comparison`: paired t-test *and* Wilcoxon signed-rank, matched by
world seed — correct given every strategy faces an identical world
realisation per S2/S9 — "significant" only when both tests agree at
alpha=0.05, a deliberately conservative bar). `experiments/ablation.py`
(variants A-F, built from `SchedulerWeights`' existing knobs — no parallel
scheduler implementation — plus `AdaptiveScheduler`'s new `freeze_belief`
flag for variant F, isolating online feedback specifically; the hard
freshness guarantee stays active in every variant, since it is a starvation
safety net, not one of the five scored terms, and disabling it would fail
weak variants for an unrelated reason). `experiments/robustness.py` (a
noise sweep that holds emitters constant and interpolates only
`NoiseModelConfig` between "normal"'s default and "high_noise"'s own values
— comparing the two *scenarios* directly would confound noise with
`high_noise`'s weaker `signal_dbm`, two variables at once; and a
non-stationarity analysis reusing `changing_distribution`'s existing,
documented channel flip). `experiments/plots.py` (static matplotlib
bars/lines with CI whiskers — deliberately separate from the future Phase 9
Plotly dashboard in `visualization/`). `rfscan ablate` is now wired for real
(previously a Phase-8 placeholder stub since Phase 4). `matplotlib` added as
a dependency (pyproject.toml) for these static plots only.

**Experiment design.** Full grid: all 7 scenarios × 30 seeds (0-29) × 4
strategies (840 episodes) — the 30-seed target matches S9's own stated
replication target. Ablation: 7 × 15 seeds × 6 variants (630 episodes) — a
smaller seed count than the main grid, documented as a runtime tradeoff, not
hidden. Noise sweep: 3 tiers × 20 seeds × 4 strategies (240 episodes).
Non-stationarity: 20 seeds × 4 strategies (80 episodes). Predictor
comparison: 3 predictors × 3 scenarios × 10 seeds (90 episodes) — a smaller
scope since its purpose is comparing predictor choice, not re-establishing
statistical power the main grid already has. 1,880 episodes total, run via
`scripts/phase8_run_experiments.py` (not part of the package — a one-off,
reproducible script, mirroring the earlier ad-hoc demo scripts) and analysed
via `scripts/phase8_analyze.py`. **No weight tuning was performed** — per
S11's own phase plan, tuning is Phase 11's job; Phase 8 measures the
*current, untuned* configuration.

**Fairness verified, not assumed.** For every experiment, `n_activity_
intervals` (a pure function of ground-truth occupancy, S16.4, independent
of which channels were actually scanned) was checked identical across every
arm sharing a (scenario, seed) cell. Zero mismatches across all five
experiments and 1,880 episodes.

**Headline result — a real tradeoff, not a clean win.** Pooled across all
scenarios/seeds: adaptive's `scan_efficiency`/`on_target_scan_rate` beat
random and sequential by roughly 3x (e.g. 0.462 vs 0.148/0.146) and its
`redundant_scan_rate` is far below heuristic's and random's (0.037 vs
0.282/0.265) — it uses its scan budget far more efficiently than any
baseline. But its raw `detection_rate` (0.431) is *below* sequential's
(0.554) and its `mean_detection_delay_slots` (6.61) is *worse* than every
baseline (4.5-5.7). 100 of 105 per-scenario paired comparisons were
statistically significant (both tests agreeing) — this split result is a
real, well-powered effect, not noise. Interpretation: a blind uniform sweep
is hard to beat on raw coverage speed precisely because it wastes no time
deciding — adaptive's advantage is in *not wasting scans*, not in *finding
things fastest*.

**Emerging-signal reliability vs. speed.** `emerging_signal`/`dynamic`, 30
seeds each: adaptive and both naive baselines discover the emerging channel
in 30/30 seeds; **heuristic fails to ever discover it in 2/30 seeds** and
averages 130-170 slots when it does, versus adaptive's 17-20 (worse than
sequential's 9-11, better than heuristic's catastrophic average by 6-8x).
Adaptive fixes heuristic's specific failure mode (starvation of unlikely
channels); it does not beat a blind sweep at raw discovery speed. Both facts
reported; neither hidden.

**Non-stationarity** (`changing_distribution`, 20 seeds): every strategy's
detection rate tracks the slot-400 flip (passive for uniform sweeps, belief-
decay-driven for adaptive/heuristic). Adaptive reaches the highest post-flip
rate on the newly-active channel group (0.905) of all four strategies,
confirming Phase 7's decay mechanism generalises beyond the single-episode
demo to a 20-seed aggregate.

**Noise robustness** (controlled 3-tier sweep, 20 seeds): adaptive's ~5x
scan-efficiency edge over random/sequential holds at every tier, degrading
proportionally with noise like every strategy — no collapse.

**Sparse vs. dense — a genuine, reported limitation.** Reusing the main
grid's bursty (sparse) / normal (moderate) / high_activity (dense) results:
adaptive's edge over heuristic on scan_efficiency shrinks from sparse
(0.112 vs 0.095, adaptive ahead) to dense, where it **reverses** (0.713 vs
0.799, heuristic ahead). In a maximally crowded band, almost every channel
is worth scanning, so pure exploitation needs little exploration and
adaptive's exploration/freshness overhead becomes a mild net cost. Not
smoothed over.

**Ablation.** Redundant-scan rate falls monotonically as terms are added
(A "prediction only" 0.117 -> E "full policy" 0.037) — the freshness term
(added at C) does most of that work. Detection rate improves modestly and
monotonically A->E (0.409->0.434). **E vs F (online feedback on/off) shows
no large *aggregate* difference** at 15-seed scale — belief update's effect
is real and decisive at the specific slot of a hit (Phase 7's per-slot
trace proved this directly), but that local effect does not read as a large
shift in episode-level aggregates most of which are far from any discovery
event. Reported as a real, nuanced finding, not oversold as "online feedback
barely matters" or hidden as "it obviously matters."

**Predictor comparison for live scheduling** (3 predictors x 3 scenarios x
10 seeds, `AdaptiveScheduler` otherwise identical): HGB gives the best
downstream `scan_efficiency` (0.573) vs LR (0.556) vs the non-ML
`decaying_beta` (0.517, a surprisingly strong, near-free floor) — but at
~3.8ms latency vs LR's ~1.1ms (S16.6's finding reconfirmed at a different
scenario mix). LR remains the right live-serving choice: the efficiency
cost is modest, the latency margin is not.

**Latency — a finding, not an assumption.** Isolated single-episode checks
after this phase still show ~1.1ms mean, matching Phase 6/7. But the full
840-episode main-grid run (continuous, ~19 minutes) showed adaptive's
`mean_decision_latency_ms` median 2.70ms, **75th percentile 5.58ms — over
the 5 ms budget** for a meaningful tail, max 6.47ms. Separately, the
ablation run had one isolated 1792ms outlier in 630 episodes (re-isolated
and confirmed a one-off system stall — the other 104 episodes for that same
variant were clean at ~1.1-1.2ms); the model-comparison run (same session,
run last) was clean throughout at 1.14ms. Read together, this looks like
sustained background system load during a long batch job, not an
algorithmic regression, but it is a real, reported caveat: the comfortable
margin measured in short controlled tests does not fully hold under
sustained heavy load, and should be re-verified on actual target hardware
rather than assumed from short tests alone.

**Statistical methodology.** Paired (not unpaired) comparisons throughout,
matched on world seed, since the fairness invariant (S2) guarantees a valid
pairing. Both a parametric (paired t-test) and non-parametric (Wilcoxon
signed-rank) test are run and must *both* clear alpha=0.05 for
"significant" — deliberately conservative, per S9's "do not claim
significance unless the test supports it." Bootstrap (not normal-theory)
CIs for plotted point estimates, since detection-delay-type metrics are not
assumed normal. Censored (never-detected) episodes contribute NaN, dropped
pairwise (never treated as zero), with the resulting `n` always reported so
sample-size shrinkage is visible, never silent.

**Scope discipline.** No UCB/Thompson/contextual bandits, no dashboard, no
weight tuning, no changes to `PriorityPolicy`/`BeliefState`'s formulas or to
any completed Phase 1-7 decision logic — this phase measured the existing
system, it did not redesign it.
