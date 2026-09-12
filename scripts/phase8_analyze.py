"""Phase 8 analysis -- consumes the CSVs written by phase8_run_experiments.py,
computes paired statistical comparisons, and generates static plots. Not part
of the rfscan package; a one-off reproducible script, run after the
experiment script.

Run: .venv/Scripts/python.exe scripts/phase8_analyze.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rfscan.experiments.plots import (
    bar_with_ci,
    grouped_bar_by_scenario,
    line_by_level,
    save_figure,
)
from rfscan.experiments.stats import paired_comparison

RESULTS = Path("artifacts/results")
PLOTS = RESULTS / "plots"

STATISTICAL_METRICS = (
    "detection_rate",
    "mean_detection_delay_slots",
    "scan_efficiency",
    "on_target_scan_rate",
    "redundant_scan_rate",
)


def paired_vs_baselines(raw: pd.DataFrame) -> pd.DataFrame:
    """For every scenario and every (baseline, adaptive) pair, a paired
    comparison per metric -- paired on world_seed within that scenario."""
    rows = []
    for scenario, group in raw.groupby("scenario"):
        pivot = {
            strat: sub.set_index("world_seed") for strat, sub in group.groupby("strategy")
        }
        if "adaptive" not in pivot:
            continue
        adaptive = pivot["adaptive"]
        for baseline in ("sequential", "random", "heuristic"):
            if baseline not in pivot:
                continue
            base = pivot[baseline]
            common_seeds = adaptive.index.intersection(base.index)
            for metric in STATISTICAL_METRICS:
                if metric not in adaptive.columns or metric not in base.columns:
                    continue
                a = base.loc[common_seeds, metric].to_numpy(dtype=float)
                b = adaptive.loc[common_seeds, metric].to_numpy(dtype=float)
                result = paired_comparison(
                    a, b, metric=metric, strategy_a=baseline, strategy_b="adaptive"
                )
                row = result.to_row()
                row["scenario"] = scenario
                rows.append(row)
    return pd.DataFrame(rows)


def overall_paired_vs_baselines(raw: pd.DataFrame) -> pd.DataFrame:
    """Same as above but pooled across all scenarios x seeds -- the
    "does adaptive help overall" headline comparison."""
    rows = []
    pivot = {strat: sub for strat, sub in raw.groupby("strategy")}
    if "adaptive" not in pivot:
        return pd.DataFrame()
    adaptive = pivot["adaptive"].set_index(["scenario", "world_seed"])
    for baseline in ("sequential", "random", "heuristic"):
        if baseline not in pivot:
            continue
        base = pivot[baseline].set_index(["scenario", "world_seed"])
        common = adaptive.index.intersection(base.index)
        for metric in STATISTICAL_METRICS:
            a = base.loc[common, metric].to_numpy(dtype=float)
            b = adaptive.loc[common, metric].to_numpy(dtype=float)
            result = paired_comparison(
                a, b, metric=metric, strategy_a=baseline, strategy_b="adaptive"
            )
            rows.append(result.to_row())
    return pd.DataFrame(rows)


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(RESULTS / "phase8_raw_results.csv")

    per_scenario_stats = paired_vs_baselines(raw)
    per_scenario_stats.to_csv(RESULTS / "phase8_statistics_by_scenario.csv", index=False)
    overall_stats = overall_paired_vs_baselines(raw)
    overall_stats.to_csv(RESULTS / "phase8_statistics_overall.csv", index=False)
    print("wrote phase8_statistics_by_scenario.csv,", len(per_scenario_stats), "rows")
    print("wrote phase8_statistics_overall.csv,", len(overall_stats), "rows")
    print("\noverall paired comparison (adaptive vs baselines):")
    print(
        overall_stats[
            ["metric", "strategy_a", "strategy_b", "n", "mean_a", "mean_b", "mean_diff", "significant_0_05"]
        ].to_string(index=False)
    )

    # Plots: overall strategy comparison for key metrics.
    for metric in STATISTICAL_METRICS:
        fig = bar_with_ci(raw, metric, title=f"{metric} by strategy (all scenarios)")
        save_figure(fig, PLOTS / f"overall_{metric}.png")
        fig = grouped_bar_by_scenario(raw, metric, title=f"{metric} by scenario x strategy")
        save_figure(fig, PLOTS / f"by_scenario_{metric}.png")

    # Ablation plot.
    ablation_path = RESULTS / "phase8_ablation_raw.csv"
    if ablation_path.exists():
        ablation_raw = pd.read_csv(ablation_path)
        for metric in ("detection_rate", "scan_efficiency", "redundant_scan_rate"):
            fig = bar_with_ci(
                ablation_raw, metric, group_col="variant", title=f"Ablation: {metric}"
            )
            save_figure(fig, PLOTS / f"ablation_{metric}.png")

    # Noise sweep plot.
    noise_path = RESULTS / "phase8_robustness_noise.csv"
    if noise_path.exists():
        noise_raw = pd.read_csv(noise_path)
        for metric in ("detection_rate", "scan_efficiency"):
            fig = line_by_level(
                noise_raw,
                metric,
                level_col="noise_tier",
                level_order=("low", "elevated", "high"),
                title=f"{metric} vs noise level",
            )
            save_figure(fig, PLOTS / f"noise_sweep_{metric}.png")

    # Model comparison plot.
    model_path = RESULTS / "phase8_model_comparison.csv"
    if model_path.exists():
        model_raw = pd.read_csv(model_path)
        for metric in ("detection_rate", "scan_efficiency", "mean_decision_latency_ms"):
            fig = bar_with_ci(
                model_raw, metric, group_col="predictor", title=f"Predictor comparison: {metric}"
            )
            save_figure(fig, PLOTS / f"model_comparison_{metric}.png")

    print("\nwrote plots to", PLOTS)


if __name__ == "__main__":
    main()
