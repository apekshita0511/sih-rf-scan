# SIH26055 Judge Demo Script

Smart Scan Strategy for Electronic Warfare — DRDO, Software category.

This is a presentation/demo aid, not a technical spec. For the full design
and every phase's findings, see `docs/architecture.md`. For scope caveats,
see the **Limitations** tab in the dashboard or `LIMITATIONS_MARKDOWN` in
`rfscan/app/components.py`.

Every number quoted below comes from `artifacts/results/phase8_raw_results.csv`
(the real, reproducible 840-episode main grid: 4 strategies × 7 scenarios ×
30 seeds) or the corresponding emerging-signal/nonstationarity CSVs. Nothing
here is invented, rounded favorably, or cherry-picked.

---

## 1. 30-second opening pitch

"Electronic warfare receivers can't scan every channel all the time — scan
time is a scarce resource. We reframed 'which channel do I scan next?' as a
**belief-space restless multi-armed bandit** and built a scheduler that
combines a learned occupancy prediction with Bayesian uncertainty,
freshness, and trend signals to decide, slot by slot, where to look next.
We benchmarked it honestly against three baselines across 840 episodes and
we're going to show you both where it wins and where it doesn't."

---

## 2. Problem explanation in simple language

Imagine 12 walkie-talkie channels and you can only listen to one channel per
time slot. Some channels are usually busy, some are usually quiet, and some
suddenly start being used partway through (an "emerging" signal). If you
just listen to channels in a fixed round-robin order, you're guaranteed to
eventually check everything — but you waste time re-checking channels you
already know are quiet, and you have no way to prioritize channels that seem
newly active.

Our scheduler instead builds a running belief about "how likely is each
channel to be active right now," and picks the next channel using that
belief, how uncertain it still is about that channel, how long it's been
since that channel was last checked, and whether its recent trend is
climbing. It's a decision-making layer, not a detector.

---

## 3. Architecture explanation

The closed loop (see also the **About / Methodology** tab):

```
Simulation -> Observation -> Feature Engineering -> ML Prediction
   -> Priority Calculation -> Next Channel -> Scan -> Feedback -> Adaptation
```

- **Simulation**: a synthetic RF world (Gilbert-Elliott occupancy Markov
  model) generates ground truth. The scheduler never sees it directly.
- **Observation**: scanning a channel returns a noisy measurement — it can
  be wrong.
- **Feature engineering**: 17 features rebuilt fresh every slot from scan
  history only (no channel index, no future information — structurally
  anti-leakage).
- **ML prediction**: a calibrated model estimates P(active) per channel.
- **Priority calculation**: `priority(c) = w_pred*p(c) + w_explore*uncertainty(c)
  + w_fresh*freshness(c) + w_trend*trend(c) - w_redundancy*redundancy(c)`.
- **Next channel**: highest priority wins, unless a channel is force-scanned
  by the hard freshness guarantee (too long unscanned).
- **Scan -> Feedback -> Adaptation**: the outcome updates a decaying
  Bayesian belief, which reshapes every later decision.

---

## 4. Live dashboard demonstration sequence

Recommended tab order for a live demo (matches the dashboard's actual tab
order, left to right):

1. **Overview** — headline pitch + honest pooled Phase 8 numbers.
2. **Live Simulation** — one decision loop, step by step, in real time.
3. **Strategy Comparison** — the full Phase 8 benchmark, per scenario.
4. **Emerging Signal** — the discovery demo (this is the strongest visual
   moment — lead the technical portion here if time is short).
5. (Optional, if there's time) **Ablation**, **Robustness**,
   **Model Comparison**, **Limitations**, **About / Methodology**.

---

## 5. Exact clicks/actions to perform

1. Launch: `streamlit run app.py` (already running before judges arrive, if
   possible).
2. Land on **Overview** (default tab). Let the four metric cards and the
   honest-result warning box be visible for a few seconds before speaking.
3. Click the **Live Simulation** tab.
   - Leave defaults: scenario `normal`, seed `0`, strategy `adaptive`
     (index 3), budget `200`.
   - Click **"Run 25 slots"**. Point at the progress bar while it runs.
   - Point at **"SELECTED CHANNEL: CH_"**, **"Predicted activity"**, and
     **"Decision reason"** immediately under the divider.
   - Point at the **Spectrum** bar chart and the **Priority contribution
     breakdown** chart side by side.
   - Expand **"Manual step-through"** and click **"Step"** once or twice to
     show a single decision in isolation.
   - Scroll to **"Compare same scenario across strategies"**, click **"Run
     comparison (all 4 strategies)"**, and let the resulting table speak for
     the trade-off (detection rate vs scan efficiency vs redundant rate).
4. Click the **Strategy Comparison** tab. Show the five pooled bar charts,
   then pick one scenario from the "Per-scenario breakdown" selectbox and
   show the paired-significance table underneath.
5. Click the **Emerging Signal** tab.
   - Click **"▶ Run Emerging Signal Demo (emerging_signal, seed 0,
     Adaptive)"**. This snaps the selectors to the canonical demo
     configuration and reruns.
   - Read the auto-generated narrative sentence aloud ("Channel CH_ stayed
     silent until slot _. Adaptive discovered it _ slots later...").
   - Point at the **Discovery outcome** metric row.
   - Drag the **"Replay up to slot"** slider slowly through the activation
     point so judges see the scan raster "light up" at discovery.
   - Point at the **priority/belief trace** chart for the emerging channel.
   - Scroll to **"Measured discovery delay across seeds"** and read the
     30/30 discovery counts.
6. If time remains, open **Ablation** (show the freshness-term jump at
   variant C) and **Limitations** (state scope explicitly, unprompted).

---

## 6. What to say while each screen is shown

- **Overview**: "This is the headline, and it's an honest one — we're not
  claiming a clean sweep. Sequential wins raw detection rate and delay.
  Adaptive wins scan efficiency and cuts redundant scanning by roughly a
  factor of four versus random."
- **Live Simulation, after Run 25 slots**: "Every slot, the scheduler looks
  at all 12 channels, scores each one with this priority formula, and picks
  the highest. You're watching the actual decision loop, not a replay of
  precomputed data."
- **Live Simulation, priority breakdown chart**: "This bar chart is the
  literal terms of the priority formula for the channel that was just
  picked — nothing here is dramatized or simplified for the demo."
- **Compare same scenario across strategies**: "This runs all four
  strategies against the *exact same* random world, seed-for-seed — so any
  difference in these numbers is the strategy, not luck."
- **Strategy Comparison**: "This is the full 840-episode Phase 8 grid, not
  a live recomputation — loaded straight from the CSV Phase 8 produced."
- **Emerging Signal, after the demo button**: "This is the part of the
  benchmark that matters most for an EW context: something starts
  transmitting that wasn't there before, and the scheduler doesn't know when
  or where. Watch how quickly it notices."

---

## 7. Adaptive vs Sequential comparison

Pooled across all 7 scenarios, 30 seeds each (`phase8_raw_results.csv`):

| Metric | Sequential | Adaptive | Who leads |
|---|---|---|---|
| Detection rate | **0.554** | 0.431 | Sequential |
| Mean detection delay (slots) | **4.52** | 6.61 | Sequential |
| Scan efficiency | 0.148 | **0.462** | Adaptive |
| On-target scan rate | 0.158 | **0.489** | Adaptive |
| Redundant scan rate | **0.000** | 0.037 | Sequential (structurally) |

Say plainly: "Sequential's blind, systematic sweep is faster at raw
coverage. Adaptive is far more efficient about *how* it scans, and stays
close to zero redundant scanning without the structural advantage
Sequential gets for free."

---

## 8. Emerging Signal demonstration

`emerging_signal` scenario, 30 seeds each:

| Strategy | Discovered | Mean delay (slots) |
|---|---|---|
| Sequential | 30/30 | 8.6 |
| Random | 30/30 | 14.8 |
| Heuristic | 28/30 | 173.6 |
| Adaptive | 30/30 | 17.2 |

Talking point: "Sequential is fastest here too, because a fixed sweep will
always eventually hit a newly active channel. The number that matters most
is Heuristic's: it misses the new signal entirely in 2 of 30 runs, and even
when it finds it, it takes almost 10x longer than everyone else — that's
tunnel vision, a real failure mode of naive score-only heuristics. Adaptive
never fails to discover it, and stays within the same order of magnitude as
the two fastest strategies."

Also mention non-stationarity (`changing_distribution`, activity flips
between channel groups at slot 400): "Adaptive's belief decays over time by
design, so after the flip it retargets the newly active group and reaches
the best post-flip detection rate of all four strategies — this is the same
underlying mechanism as the emerging-signal result, just triggered by a
distribution shift instead of a first activation."

---

## 9. How to explain the decision contribution chart

"This is the **Priority contribution breakdown** chart on the Live
Simulation tab. Each bar is one term of the priority formula — prediction,
exploration, freshness, trend, and (negative) redundancy — for the channel
that was just selected. It's a direct visualization of `explain()`/
`all_breakdowns()` on the real scheduler object; nothing is recomputed or
simplified for display. If a channel wins mostly on the freshness bar, it
means the model's confidence was moderate but it hadn't been checked in a
while. If it wins mostly on the prediction bar, the model itself is
confident that channel is active right now."

---

## 10. How to explain freshness

"Freshness measures how long it's been since a channel was last scanned.
Even a channel the model currently thinks is quiet accumulates freshness
priority the longer it goes unchecked, so it eventually gets scanned again
regardless of prediction. There's also a **hard freshness guarantee**: if a
channel goes unscanned past a maximum revisit window, it gets force-scanned
outright, overriding the priority score. That's what 'Decision reason:
Freshness guarantee — channel had not been scanned recently' means when it
appears."

---

## 11. How to explain exploration

"Exploration is a Bayesian uncertainty term — it's high when the model
doesn't yet have enough evidence about a channel to be confident either way,
regardless of whether its current best guess is high or low. This is what
stops the scheduler from fixating only on channels it already believes are
active and ignoring channels it simply hasn't learned about yet."

---

## 12. How to explain prediction

"Prediction is the ML model's estimate of P(channel is active right now),
built from 17 features — recent hit rate, time since last scan, streaks,
trend direction, and so on — none of which include the channel's identity
or any future information. It answers one narrow question: *is this channel
probably active?* It does not decide what to scan next by itself — that's
the scheduler's job (see section 9 below on Prediction vs Scheduling)."

---

## 13. How to explain why CH2 (or any single channel) may be selected repeatedly

"If a channel keeps winning, it usually means the model has learned that
channel is persistently or intermittently active — its predicted-activity
term stays high scan after scan, and each hit reinforces that belief a
little further. That's expected and correct behavior for a persistent
emitter: repeatedly checking a channel that's actually busy is the right
call, not a bug. You can verify this by watching the 'Predicted activity'
metric stay high across consecutive selections of the same channel."

---

## 14. How to explain why another channel may be force-selected by freshness

"Occasionally you'll see a channel get picked even though its predicted
activity is low. That's the freshness or hard-freshness-guarantee term
overriding prediction — the scheduler is enforcing a minimum revisit rate so
that no channel is neglected indefinitely just because the model currently
believes it's quiet. Beliefs decay over time specifically so this can
happen: confidence in 'this channel is quiet' erodes the longer it goes
unchecked, until it's worth spending a scan to confirm or refute it."

---

## 15. Exact Phase 8 results to quote

Pooled, all 7 scenarios, 30 seeds each (`phase8_raw_results.csv`, 840
episodes):

| Strategy | Detection rate | Mean detection delay | Scan efficiency | On-target rate | Redundant rate |
|---|---|---|---|---|---|
| Sequential | 0.554 | 4.52 | 0.148 | 0.158 | 0.000 |
| Random | 0.461 | 5.73 | 0.146 | 0.155 | 0.265 |
| Heuristic | 0.290 | 4.89 | 0.427 | 0.460 | 0.282 |
| Adaptive | 0.431 | 6.61 | 0.462 | 0.489 | 0.037 |

Emerging signal (`emerging_signal`, 30 seeds): Sequential 30/30 @ 8.6,
Random 30/30 @ 14.8, Heuristic 28/30 @ 173.6, Adaptive 30/30 @ 17.2.

Non-stationarity (`changing_distribution`, flip at slot 400): Adaptive
achieves the best post-flip detection rate on the newly-active channel
group of all four strategies.

---

## 16. Questions a technical judge may ask

1. Why is ML needed here at all?
2. Why not just do sequential scanning?
3. Why not simply choose the channel with the highest predicted
   probability?
4. How does the scheduler avoid tunnel vision?
5. What is exploration, precisely?
6. What is freshness, precisely?
7. What does the model actually predict?
8. What features are used?
9. How was the model trained?
10. Why is Logistic Regression used for live serving when
    HistGradientBoosting was the offline bake-off winner?
11. How do you prevent data leakage?
12. How do you evaluate the scheduler?
13. How do you know Adaptive is actually better?
14. Why does Sequential have a higher detection rate?
15. Why does Adaptive have a worse mean detection delay?
16. Why is Sequential's redundant rate exactly 0?
17. What happens when activity changes mid-episode?
18. How do you test an emerging signal?
19. Is this detecting real drones?
20. Is this using real RF?
21. Could this be deployed on real hardware?
22. What are the limitations?
23. What would you improve next?

---

## 17. Strong answers to those questions

1. **Why is ML needed?** A fixed rule can't tell the difference between "a
   channel that's usually quiet" and "a channel that just became active" —
   both look identical to a static heuristic. A learned model tracks
   per-channel dynamics from evidence instead of a hand-tuned rule.
2. **Why not sequential?** Sequential is genuinely strong for raw coverage
   (it wins detection rate and delay in our own numbers) — but it can't
   prioritize; every channel gets equal attention regardless of how likely
   it is to be active. That's exactly the trade-off in section 7.
3. **Why not just take the highest predicted probability?** That's
   essentially the Heuristic baseline, and it's the one that fails worst on
   the emerging-signal test (2 of 30 misses, ~174-slot average delay when it
   does find it) — a pure-prediction policy tunnel-visions onto channels it
   already believes are active and starves channels it hasn't learned about
   yet.
4. **How does it avoid tunnel vision?** The exploration and freshness terms
   actively push priority toward channels the model is uncertain about or
   hasn't checked recently, counterbalancing the prediction term.
5. **What is exploration?** A Bayesian uncertainty measure over the belief
   distribution for a channel — high when there's little evidence either
   way, independent of the point estimate itself.
6. **What is freshness?** Time-since-last-scan, plus a hard revisit-window
   guarantee that force-scans a channel if it's gone unscanned too long,
   regardless of its priority score.
7. **What does the model predict?** P(channel is active in this slot),
   nothing more — see section 9's Prediction vs Scheduling distinction.
8. **What features are used?** 17 features per channel, rebuilt fresh every
   slot from scan history alone — recent hit/miss patterns, time since last
   scan, streak lengths, and trend direction. No channel index and no
   future information (see the anti-leakage answer, #11).
9. **How was the model trained?** Offline, on labeled episodes from the
   same simulator, with a calibrated classifier bake-off (Logistic
   Regression vs HistGradientBoosting) selected by PR-AUC.
10. **Why Logistic Regression live, not HGB?** HGB won the offline bake-off
    on PR-AUC, but under live per-slot latency profiling it measured
    ~3.8-5.4ms against a 5ms/slot budget — its 181-tree ensemble has
    inherent per-call overhead at tiny batch sizes. Logistic Regression
    measured ~1.1ms and is the live-serving default; HGB stays correctly
    documented as the offline winner. See the **Model Comparison** tab.
11. **How is leakage prevented?** Structurally, not just by convention: the
    `Observation` type the scheduler consumes has no ground-truth field at
    all; `GroundTruth` is a separate type kept in an evaluation-only silo;
    and the feature builder raises an exception if the store it's given
    contains any future-dated record.
12. **How is the scheduler evaluated?** Censored-aware episode metrics
    (detection rate, mean detection delay, scan efficiency, on-target rate,
    redundant rate, emerging discovery delay, channel coverage, decision
    latency), computed identically for every strategy against the same
    world per seed, then compared with paired bootstrap CI + paired t-test +
    Wilcoxon (both must agree to call a difference significant).
13. **How do you know Adaptive is actually better?** We don't claim it's
    better on every axis — we show exactly which metrics it wins
    (efficiency, redundancy, robustness to change) and which it loses
    (raw detection rate, delay), with statistical testing behind the
    comparison, not just point estimates.
14. **Why does Sequential have higher detection rate?** Systematic,
    exhaustive coverage is a strong strategy specifically for the "find as
    much as possible" objective — it never skips a channel for long.
15. **Why is Adaptive's mean detection delay worse?** Adaptive spends scans
    on efficiency and exploration rather than guaranteeing short, uniform
    revisit intervals for every channel, which trades away some raw speed.
16. **Why is Sequential's redundant rate exactly 0?**
    > "Because sequential round-robin scanning structurally avoids
    > revisiting a channel within the relevant revisit window. That is a
    > property of the scanning pattern, not evidence that it is making more
    > intelligent decisions."
17. **What happens when activity changes mid-episode?** The belief decays
    over time, so old evidence loses weight and the scheduler retargets
    toward the newly active channels — demonstrated in the
    `changing_distribution` non-stationarity result (Robustness tab).
18. **How do you test an emerging signal?** A dedicated scenario
    (`emerging_signal`) where one channel stays silent until a fixed slot
    and then activates; the scheduler is never told which channel or when —
    discovery delay is measured from actual scan history.
19. **Is this detecting real drones?**
    > "No. This prototype does not identify physical drones, missiles,
    > aircraft, or real emitters. We model the scan-strategy problem using a
    > controlled synthetic RF environment. The research contribution we
    > demonstrate is the intelligent scheduling decision — which virtual
    > channel should be scanned next."
20. **Is this using real RF?** No — no SDR, no real reception/transmission,
    anywhere in the pipeline; see the Limitations tab.
21. **Could this be deployed on real hardware?** Not as-is — it would need
    validation against a real receiver front end, real propagation effects,
    and confirmation that the decision-latency budget holds under real
    system load (Phase 8 already surfaced a latency tail under sustained
    load worth investigating further).
22. **What are the limitations?** See section 18 below and the dashboard's
    Limitations tab in full.
23. **What would you improve next?** See section 18.

---

## 18. Limitations / future work

- Software-only research prototype: no real RF reception/transmission, no
  SDR hardware, no jamming/spoofing, no real emitter detection or
  classification, no operational/classified EW functionality.
- Channel occupancy is a Gilbert-Elliott Markov abstraction of
  decision-relevant dynamics, not an RF propagation model; it does not
  model a single emitter literally frequency-hopping between channels.
- Adaptive's efficiency advantage over the Heuristic baseline shrinks and
  reverses in the densest scenario tested, where most channels are active
  and exploration has little value.
- Decision latency stayed inside budget in short controlled tests but
  showed a meaningful tail over budget in one long, continuous batch run —
  attributed to system load, not an algorithmic regression, but unverified
  on target hardware.
- Future work: validate on real receiver hardware/SDR, revisit the
  HGB-vs-LogReg latency trade-off with a lower-overhead serving path,
  and extend the occupancy model toward literal frequency-agile emitters.

**Prediction vs Scheduling** (central to the project — keep this
distinction sharp under questioning): Prediction answers *"How likely is
this channel to be active?"* Scheduling answers *"Given all channels and
limited scan opportunities, which one should I scan next?"* The ML model
only does the former; the priority formula and freshness guarantee do the
latter. Confusing the two is the single most common misunderstanding a
judge will have going in — the model is not "the algorithm," it's one input
to the algorithm.

---

## 19. 3-minute compressed demo version

1. (30s) Opening pitch (section 1).
2. (60s) **Live Simulation**: Run 25 slots, point at SELECTED CHANNEL /
   Predicted activity / Decision reason, and the priority breakdown chart.
3. (60s) **Emerging Signal**: click the demo button, read the narrative
   sentence, show the discovery-delay table (30/30 vs Heuristic's 28/30 @
   173.6).
4. (30s) Close with the honest headline from section 7 (Sequential wins
   raw detection, Adaptive wins efficiency/robustness) and one limitation
   from section 18.

## 20. 7-minute full demo version

1. (30s) Opening pitch (section 1).
2. (60s) Architecture explanation using the closed-loop diagram (section 3).
3. (2 min) **Live Simulation**: Run 25 slots, explain SELECTED CHANNEL /
   Predicted activity / Decision reason / priority breakdown chart, expand
   Manual step-through for one isolated step, then run "Compare same
   scenario across strategies" and narrate the trade-off table.
4. (30s) **Strategy Comparison**: pooled bar charts + one per-scenario
   significance table.
5. (2 min) **Emerging Signal**: demo button, narrative, discovery outcome
   metrics, slider through the activation point, priority/belief trace,
   discovery-delay table across seeds — the strongest moment, give it the
   most time.
6. (30s) **Limitations** tab, stated unprompted, plus the Prediction vs
   Scheduling distinction (section 18).
7. (30s) Close: invite judge questions, referencing section 17 as needed.
