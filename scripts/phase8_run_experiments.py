"""Phase 8 experiment runner -- NOT a test, NOT part of the package.

Runs the full benchmark grid, ablation study, noise sweep, non-stationarity
experiment, and predictor comparison for real, writing every result to
artifacts/results/. This is a one-off script (mirrors how the Phase 6/7 demo
scripts were run from the scratchpad), kept here under scripts/ so the run is
reproducible without re-typing it, but it is not imported by rfscan itself.

Run: .venv/Scripts/python.exe scripts/phase8_run_experiments.py
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from rfscan.config import ModelConfig
from rfscan.experiments.ablation import run_ablation, summarize_ablation
from rfscan.experiments.benchmark import BenchmarkConfig, run_benchmark, summarize
from rfscan.experiments.robustness import run_non_stationarity_experiment, run_noise_sweep
from rfscan.models.baseline_beta import DecayingBetaPredictor
from rfscan.models.loader import load_predictor
from rfscan.simulator.scenarios import list_scenarios

OUT = Path("artifacts/results")
OUT.mkdir(parents=True, exist_ok=True)

MAIN_SEEDS = tuple(range(30))
ABLATION_SEEDS = tuple(range(15))
NOISE_SEEDS = tuple(range(20))
NONSTATIONARY_SEEDS = tuple(range(20))
MODEL_COMPARISON_SEEDS = tuple(range(10))
MODEL_COMPARISON_SCENARIOS = ("normal", "emerging_signal", "high_activity")

STRATEGIES = ("sequential", "random", "heuristic", "adaptive")


def _timed(label: str, fn):
    t0 = time.time()
    result = fn()
    print(f"[{label}] {time.time() - t0:.1f}s")
    return result


def main() -> None:
    live_predictor = load_predictor(ModelConfig())  # logistic_regression, the live default
    print("live predictor:", live_predictor.name)

    # 1. Main benchmark grid: 7 scenarios x 30 seeds x 4 strategies.
    main_config = BenchmarkConfig(
        scenarios=tuple(list_scenarios()),
        world_seeds=MAIN_SEEDS,
        strategies=STRATEGIES,
        model=ModelConfig(),
        results_dir=str(OUT),
    )
    main_raw = _timed("main grid", lambda: run_benchmark(main_config))
    main_raw.to_csv(OUT / "phase8_raw_results.csv", index=False)
    summarize(main_raw).to_csv(OUT / "phase8_summary.csv", index=False)
    print("main grid rows:", len(main_raw))

    # 2. Ablation: 7 scenarios x 15 seeds x 6 variants, same live predictor.
    ablation_raw = _timed(
        "ablation",
        lambda: run_ablation(live_predictor, world_seeds=ABLATION_SEEDS),
    )
    ablation_raw.to_csv(OUT / "phase8_ablation_raw.csv", index=False)
    summarize_ablation(ablation_raw).to_csv(OUT / "phase8_ablation.csv", index=False)
    print("ablation rows:", len(ablation_raw))

    # 3. Noise sweep: low/elevated/high x 20 seeds x 4 strategies, base "normal".
    noise_raw = _timed(
        "noise sweep",
        lambda: run_noise_sweep(
            strategies=STRATEGIES, world_seeds=NOISE_SEEDS, predictor=live_predictor
        ),
    )
    noise_raw.to_csv(OUT / "phase8_robustness_noise.csv", index=False)
    print("noise sweep rows:", len(noise_raw))

    # 4. Non-stationarity: changing_distribution x 20 seeds x 4 strategies.
    nonstationary_raw = _timed(
        "non-stationarity",
        lambda: run_non_stationarity_experiment(
            strategies=STRATEGIES, world_seeds=NONSTATIONARY_SEEDS, predictor=live_predictor
        ),
    )
    nonstationary_raw.to_csv(OUT / "phase8_robustness_nonstationary.csv", index=False)
    print("non-stationarity rows:", len(nonstationary_raw))

    # 5. Predictor comparison for live scheduling: Beta / LR / HGB, adaptive only.
    hgb_predictor = load_predictor(ModelConfig(kind="hist_gradient_boosting"))
    predictors = {
        "decaying_beta": DecayingBetaPredictor(),
        "logistic_regression": live_predictor,
        "hist_gradient_boosting": hgb_predictor,
    }
    model_rows = []
    for name, predictor in predictors.items():
        cfg = BenchmarkConfig(
            scenarios=MODEL_COMPARISON_SCENARIOS,
            world_seeds=MODEL_COMPARISON_SEEDS,
            strategies=("adaptive",),
            model=ModelConfig(kind=name if name != "decaying_beta" else "beta"),
        )

        def _predictor_run(predictor=predictor, cfg=cfg):
            from rfscan.experiments.benchmark import build_scheduler
            from rfscan.experiments.metrics import compute_episode_metrics
            from rfscan.experiments.runner import run_episode
            from rfscan.simulator.scenarios import make_scenario

            rows = []
            for scenario_name in cfg.scenarios:
                for world_seed in cfg.world_seeds:
                    scenario = make_scenario(scenario_name, world_seed)
                    env = scenario.build_environment(world_seed)
                    sched = build_scheduler(
                        "adaptive", env.n_channels, agent_seed=world_seed, predictor=predictor
                    )
                    result = run_episode(env, sched, budget=scenario.duration_slots, agent_seed=world_seed)
                    row = compute_episode_metrics(result).to_row()
                    rows.append(row)
            return pd.DataFrame(rows)

        raw = _timed(f"model comparison: {name}", _predictor_run)
        raw["predictor"] = name
        model_rows.append(raw)
    model_comparison_raw = pd.concat(model_rows, ignore_index=True)
    model_comparison_raw.to_csv(OUT / "phase8_model_comparison.csv", index=False)
    print("model comparison rows:", len(model_comparison_raw))

    print("\nAll Phase 8 experiments complete. Wrote CSVs to", OUT)


if __name__ == "__main__":
    main()
