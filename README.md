# AI-Driven Adaptive RF Scan Intelligence Platform

**Smart India Hackathon 2026 · Problem SIH26055 — Smart Scan Strategy for Electronic Warfare · DRDO · Software**

> **Build status: Phase 8 of 13 complete** — stochastic RF simulator, sensor + memory +
> baseline strategies, the reproducible-evaluation foundation, an ML prediction pipeline
> (17-feature `FeatureBuilder`, calibrated Logistic Regression / HistGradientBoosting, a
> real bake-off via `rfscan train`), the **AdaptiveScheduler** (decaying-Beta `BeliefState`,
> weighted multi-objective `PriorityPolicy`, hard freshness guarantee) with observable,
> proven online feedback and emerging-signal adaptation (Phase 7), and now a **full
> experimental validation** (Phase 8): a 1,880-episode paired benchmark grid (7 scenarios
> × 30 seeds × 4 strategies), a 6-variant priority-component ablation, a controlled noise
> sweep, a non-stationarity study, and a 3-predictor downstream comparison — all with
> paired significance testing (t-test + Wilcoxon, bootstrap CIs) via `rfscan ablate` and
> `scripts/phase8_*.py`. Honest, mixed result, not a clean win: adaptive uses its scan
> budget ~3x more efficiently than naive baselines (scan efficiency, on-target rate) and
> is far less wasteful than the heuristic baseline (redundant-scan rate), but loses to a
> blind uniform sweep on raw detection rate and delay, and its efficiency edge **reverses**
> in the densest scenario. Live serving uses Logistic Regression (~1.1 ms/slot in
> controlled tests; HistGradientBoosting, the offline bake-off winner, exceeds the 5 ms
> budget at ~3.8-5.4 ms) — see `docs/architecture.md` S16.6/S16.9 for the full numbers,
> including a latency caveat found under sustained load, reported rather than hidden.
> Full documentation lands in Phase 13. See [`docs/architecture.md`](docs/architecture.md) for the design.

---

## What this is

A conventional spectrum scanner sweeps channels sequentially or on a fixed
schedule, spending equal effort on empty and active spectrum. This project
reframes *"which channel do I scan next?"* as a **sequential decision problem
under partial observability** — a belief-space restless multi-armed bandit — and
solves it with a hybrid **learned-occupancy + Bayesian-uncertainty scheduler**
that balances exploiting known activity against exploring for emerging signals.

The contribution is the closed-loop **scan policy**, measured by detection
latency and emerging-emitter discovery against sequential / random baselines on
identical, seeded synthetic RF scenarios.

## Scope and safety

This is an academic prototype. It runs **entirely inside a software-simulated
synthetic RF environment**. It does **not** implement — and is not intended for —
real-world RF interception, signal intelligence, jamming, spoofing, weapon
guidance, targeting, unauthorised transmission/reception, or any classified EW
capability. No SDR or physical RF hardware is used or required.

## Requirements

- Python **3.10+** (developed on 3.12)

## Install

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# POSIX:    source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
rfscan info                              # resolved config + channel plan       (available now)
rfscan --config configs/default.yaml info
rfscan benchmark                         # baseline strategy grid -> CSV        (available now)
rfscan benchmark --seeds 10              #   ... over world seeds 0..9
rfscan train                             # ML model bake-off                      (available now)
rfscan ablate                            # Phase 8  — scheduler component ablation  (available now)
rfscan demo                              # Phase 12 — emerging-signal demo
rfscan dashboard                         # Phase 9  — Streamlit dashboard
```

`rfscan benchmark` runs all 7 scenarios x N seeds x {sequential, random, heuristic}
and writes `artifacts/results/raw_results.csv` (one row per episode) and
`summary.csv` (mean / std / count per scenario x strategy).

`rfscan train` collects a labelled dataset from the simulator (a mix of the
sequential/random/heuristic scan policies across all 7 scenarios), splits it
temporally by episode (`TRAIN_SEEDS`/`VAL_SEEDS`/`TEST_SEEDS` in
`rfscan/models/train.py`), runs the bake-off (`decaying_beta` vs calibrated
`logistic_regression` vs `hist_gradient_boosting`), and writes
`artifacts/results/model_bakeoff.csv`, `calibration_curves.csv`,
`docs/model_cards/*.md`, and the chosen model to `artifacts/model.joblib`.

`rfscan ablate` runs the 6-variant priority-component ablation (A: prediction
only, through E: full policy, plus F: full policy with online feedback
frozen) across all 7 scenarios x N seeds, and writes `artifacts/results/
ablation_raw.csv` / `ablation_summary.csv`. The full Phase 8 experiment suite
(paired benchmark grid, ablation, noise sweep, non-stationarity, predictor
comparison, statistics, plots) is reproduced via `scripts/phase8_run_
experiments.py` then `scripts/phase8_analyze.py` — see `docs/architecture.md`
S16.9 for the full results and methodology.

`python main.py <command>` is equivalent to `rfscan <command>`.

## Testing

```bash
pytest
pytest --cov=rfscan --cov-report=term-missing
```

## Project structure

```
rfscan/
  simulator/      synthetic RF world  — channel plan, emitters, noise, scenarios,
                  occupancy (Gilbert-Elliott), RFEnvironment (step/observe/ground_truth)
  perception/     agent-visible data contracts (schema); ObservationStore; Scanner;
                  feature builder (Phase 5)
  models/         ML activity-prediction engine (Phase 5)
  scheduler/      Scheduler protocol + sequential/random/heuristic baselines;
                  adaptive scheduler (Phases 6, 7)
  experiments/    run_episode + EpisodeResult; censored-aware metrics;
                  scenarios x seeds x strategies benchmark grid -> CSV;
                  emerging-signal adaptation metrics + priority tracing (Phase 7);
                  ablation, bootstrap CI + paired significance tests, noise/
                  non-stationarity robustness sweeps, static analysis plots (Phase 8)
  visualization/  Plotly figure builders (Phase 9)
  app/            Streamlit dashboard (Phase 9)
  config.py       AppConfig tree + YAML load/dump
  cli.py          command-line entry point
configs/          default.yaml + per-scenario configs
tests/            pytest suite
docs/             architecture.md, sih_pitch.md (Phase 13)
artifacts/        trained models + experiment results (git-ignored)
```

## Roadmap

| Phase | Deliverable | Status |
|------:|-------------|:------:|
| 1  | Repo skeleton, data contracts, config, CLI, tests | ✅ |
| 2  | RF simulator (occupancy Markov model, noise, observations) | ✅ |
| 3  | Baseline scanners + observation store | ✅ |
| 4  | Scenario engine (7 scenarios) + experiment runner + metrics + baseline benchmark | ✅ |
| 5  | Feature pipeline + ML model bake-off | ✅ |
| 6  | Adaptive scheduler v1 (thin end-to-end loop) | ✅ |
| 7  | Online feedback + emerging-signal adaptation | ✅ |
| 8  | Full benchmark grid + ablation + robustness | ✅ |
| 9  | Streamlit dashboard | ⬜ |
| 10 | Test hardening | ⬜ |
| 11 | Optimization + weight tuning | ⬜ |
| 12 | Deterministic SIH demo | ⬜ |
| 13 | README + pitch + model cards | ⬜ |

## License

MIT (see `pyproject.toml`).
