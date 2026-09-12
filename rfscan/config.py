"""Application configuration.

One frozen :class:`AppConfig` tree, loadable from / dumpable to YAML, with a
default that matches ``configs/default.yaml`` exactly (enforced by
``tests/test_config.py``). Scenario definitions are separate
(:mod:`rfscan.simulator.scenario`); this covers the knobs that are constant
across a benchmark run: channel plan, scheduler weights, belief priors, model
choice, and experiment grid.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import dacite
import yaml


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    n_channels: int = 13
    slot_duration_s: float = 0.1
    scan_budget_slots: int = 2000
    channel_plan: str = "wifi_24ghz"
    channel_plan_params: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SchedulerWeights:
    """Weights for the adaptive scheduler's multi-objective priority score.

    priority(c) = w_pred * p(c)
                + w_explore  * uncertainty(c)
                + w_fresh     * freshness(staleness(c))
                + w_trend     * max(0, activity_trend(c))
                - w_redundancy * redundancy(c)
    """

    w_pred: float = 1.0
    w_explore: float = 0.5
    w_fresh: float = 0.3
    w_trend: float = 0.4
    w_redundancy: float = 0.2
    freshness_tau_slots: float = 20.0
    softmax_temperature: float | None = None  # None -> argmax selection


@dataclass(frozen=True, slots=True)
class BeliefConfig:
    """Decaying Beta-Bernoulli occupancy belief."""

    prior_alpha: float = 1.0
    prior_beta: float = 1.0
    decay_lambda: float = 0.98  # per-slot pull of unscanned channels toward the prior


@dataclass(frozen=True, slots=True)
class ModelConfig:
    kind: str = "logistic_regression"  # beta | logistic_regression | random_forest | hist_gradient_boosting
    calibration: str = "isotonic"  # none | sigmoid | isotonic
    recent_window: int = 20
    artifact_path: str = "artifacts/model.joblib"


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    seeds: list[int] = field(default_factory=lambda: list(range(30)))
    strategies: list[str] = field(
        default_factory=lambda: ["sequential", "random", "heuristic", "adaptive"]
    )
    scenarios: list[str] = field(default_factory=lambda: ["normal"])
    results_dir: str = "artifacts/results"


@dataclass(frozen=True, slots=True)
class AppConfig:
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    scheduler: SchedulerWeights = field(default_factory=SchedulerWeights)
    belief: BeliefConfig = field(default_factory=BeliefConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    random_seed: int = 42
    log_level: str = "INFO"


_DACITE_CONFIG = dacite.Config(strict=True)


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load an :class:`AppConfig` from a YAML file, or return defaults if ``path``
    is ``None``. Unknown keys raise (``strict=True``)."""
    if path is None:
        return AppConfig()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return dacite.from_dict(AppConfig, raw, config=_DACITE_CONFIG)


def dump_config(config: AppConfig, path: str | Path) -> None:
    """Serialise an :class:`AppConfig` to YAML."""
    Path(path).write_text(
        yaml.safe_dump(asdict(config), sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )


def config_to_dict(config: AppConfig) -> dict:
    return asdict(config)
