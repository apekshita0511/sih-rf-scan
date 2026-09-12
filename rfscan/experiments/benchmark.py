"""Strategy comparison grid: scenarios x world seeds x strategies.

Phase 4 form: run every cell, compute censored-aware episode metrics, write
``raw_results.csv`` (one row per episode) and ``summary.csv`` (mean / std / count
per scenario x strategy). Phase 8 extends this with bootstrap CIs, paired
significance tests, the component ablation, and the robustness sweep.

Fairness: for each ``(scenario, world_seed)`` the scenario is built once and a
*fresh* environment at that seed is handed to each strategy -- identical world
realisation (Phase 2 keying), separate agent RNG (``agent_seed = world_seed``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from rfscan.config import BeliefConfig, ModelConfig, SchedulerWeights
from rfscan.experiments.metrics import SUMMARY_METRICS, compute_episode_metrics
from rfscan.experiments.runner import run_episode
from rfscan.logging_config import get_logger
from rfscan.models.base import Predictor
from rfscan.models.loader import load_predictor
from rfscan.scheduler.adaptive import AdaptiveScheduler
from rfscan.scheduler.base import Scheduler
from rfscan.scheduler.heuristic import HeuristicScheduler
from rfscan.scheduler.sequential import RandomScheduler, SequentialScheduler
from rfscan.simulator.scenarios import list_scenarios, make_scenario

log = get_logger("experiments.benchmark")

# The three Phase 4 baselines. "adaptive" (Phase 6) is deliberately not part of
# the default grid -- it needs a trained model artifact (`rfscan train`) or an
# explicit `model.kind: beta` -- but is accepted when named explicitly.
STRATEGIES = ("sequential", "random", "heuristic")
VALID_STRATEGIES = STRATEGIES + ("adaptive",)

_SUMMARY_METRICS = SUMMARY_METRICS  # backward-compat alias; see metrics.py


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    scenarios: tuple[str, ...] = field(default_factory=lambda: tuple(list_scenarios()))
    world_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    strategies: tuple[str, ...] = STRATEGIES
    duration_slots: int | None = None  # None -> each scenario's own default (800)
    redundancy_window: int = 5
    heuristic_epsilon: float = 0.1
    results_dir: str = "artifacts/results"
    model: ModelConfig = field(default_factory=ModelConfig)
    scheduler_weights: SchedulerWeights = field(default_factory=SchedulerWeights)
    belief: BeliefConfig = field(default_factory=BeliefConfig)

    def __post_init__(self) -> None:
        unknown_strategies = set(self.strategies) - set(VALID_STRATEGIES)
        if unknown_strategies:
            raise ValueError(f"unknown strategies: {sorted(unknown_strategies)}")
        unknown_scenarios = set(self.scenarios) - set(list_scenarios())
        if unknown_scenarios:
            raise ValueError(f"unknown scenarios: {sorted(unknown_scenarios)}")
        if not self.world_seeds:
            raise ValueError("world_seeds must be non-empty")

    @property
    def n_episodes(self) -> int:
        return len(self.scenarios) * len(self.world_seeds) * len(self.strategies)


def build_scheduler(
    name: str,
    n_channels: int,
    *,
    agent_seed: int,
    epsilon: float = 0.1,
    predictor: Predictor | None = None,
    scheduler_weights: SchedulerWeights | None = None,
    belief_config: BeliefConfig | None = None,
) -> Scheduler:
    if name == "sequential":
        return SequentialScheduler(n_channels)
    if name == "random":
        return RandomScheduler(n_channels, seed=agent_seed)
    if name == "heuristic":
        return HeuristicScheduler(n_channels, epsilon=epsilon, seed=agent_seed)
    if name == "adaptive":
        if predictor is None:
            raise ValueError("strategy 'adaptive' requires a predictor")
        return AdaptiveScheduler(
            n_channels,
            predictor,
            weights=scheduler_weights,
            belief_config=belief_config,
            seed=agent_seed,
        )
    raise ValueError(f"unknown strategy {name!r}")


def run_benchmark(config: BenchmarkConfig) -> pd.DataFrame:
    """Execute the grid and return one metrics row per episode."""
    predictor = load_predictor(config.model) if "adaptive" in config.strategies else None
    rows: list[dict] = []
    for scenario_name in config.scenarios:
        for world_seed in config.world_seeds:
            scenario = make_scenario(
                scenario_name, world_seed, duration_slots=config.duration_slots
            )
            for strategy in config.strategies:
                env = scenario.build_environment(world_seed)
                scheduler = build_scheduler(
                    strategy,
                    env.n_channels,
                    agent_seed=world_seed,
                    epsilon=config.heuristic_epsilon,
                    predictor=predictor,
                    scheduler_weights=config.scheduler_weights,
                    belief_config=config.belief,
                )
                result = run_episode(
                    env, scheduler, budget=scenario.duration_slots, agent_seed=world_seed
                )
                metrics = compute_episode_metrics(
                    result, redundancy_window=config.redundancy_window
                )
                rows.append(metrics.to_row())
            log.info("benchmark: %s seed=%d done", scenario_name, world_seed)
    return pd.DataFrame(rows)


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    """Mean / std / count per (scenario, strategy) over the metric columns."""
    present = [m for m in _SUMMARY_METRICS if m in raw.columns]
    grouped = raw.groupby(["scenario", "strategy"], sort=True)[present].agg(
        ["mean", "std", "count"]
    )
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    return grouped.reset_index()


def write_benchmark(
    config: BenchmarkConfig, raw: pd.DataFrame, summary: pd.DataFrame | None = None
) -> tuple[Path, Path]:
    out = Path(config.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    if summary is None:
        summary = summarize(raw)
    raw_path = out / "raw_results.csv"
    summary_path = out / "summary.csv"
    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    return raw_path, summary_path
