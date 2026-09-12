"""Phase 8: robustness experiments -- noise sweep and non-stationary
adaptation -- built entirely from existing simulator/config mechanisms
(``NoiseModelConfig``, ``ScenarioConfig.noise``, the ``changing_distribution``
scenario) and the existing ``benchmark.build_scheduler`` wiring. No new
simulator behaviour, no new scenario machinery, no duplicated
scheduler-construction logic.

Sparse-vs-dense and periodic/bursty activity (S8.10/S8.11) need no new code
at all: ``bursty`` (six short bursty emitters), ``normal`` (three moderate
emitters), and ``high_activity`` (~40% of channel-slots occupied, S16.4) are
already a sparse/moderate/dense progression along the existing benchmark
grid, and ``normal``/``emerging_signal`` already mix
``PERSISTENT``/``INTERMITTENT`` (periodic)/``BURSTY`` behaviours (S4). This
is reported by grouping the main benchmark grid's own results, not by a
separate experiment.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from rfscan.config import BeliefConfig, SchedulerWeights
from rfscan.experiments.adaptation import channel_scan_split
from rfscan.experiments.benchmark import build_scheduler
from rfscan.experiments.metrics import compute_episode_metrics
from rfscan.experiments.runner import EpisodeResult, run_episode
from rfscan.logging_config import get_logger
from rfscan.models.base import Predictor
from rfscan.simulator.noise import NoiseModelConfig
from rfscan.simulator.scenario import ScenarioConfig
from rfscan.simulator.scenarios import make_scenario

log = get_logger("experiments.robustness")

# -- noise sweep ---------------------------------------------------------
# The two noise configs already in the benchmark suite: "normal"'s default
# NoiseModelConfig() and "high_noise"'s own values (simulator/scenarios.py),
# reused verbatim -- not re-invented. "elevated" is their midpoint.
_LOW_NOISE = NoiseModelConfig()
_HIGH_NOISE = NoiseModelConfig(
    floor_dbm=-95.0,
    floor_drift_std_db=1.5,
    floor_drift_rho=0.92,
    awgn_std_db=4.0,
    detection_threshold_snr_db=8.0,
    false_alarm_rate=0.06,
    missed_detection_rate=0.15,
)

NOISE_TIERS: dict[str, float] = {"low": 0.0, "elevated": 0.5, "high": 1.0}


def _interpolate_noise(t: float) -> NoiseModelConfig:
    def lerp(a: float, b: float) -> float:
        return a + t * (b - a)

    return NoiseModelConfig(
        floor_dbm=lerp(_LOW_NOISE.floor_dbm, _HIGH_NOISE.floor_dbm),
        floor_drift_std_db=lerp(_LOW_NOISE.floor_drift_std_db, _HIGH_NOISE.floor_drift_std_db),
        floor_drift_rho=lerp(_LOW_NOISE.floor_drift_rho, _HIGH_NOISE.floor_drift_rho),
        awgn_std_db=lerp(_LOW_NOISE.awgn_std_db, _HIGH_NOISE.awgn_std_db),
        detection_threshold_snr_db=lerp(
            _LOW_NOISE.detection_threshold_snr_db, _HIGH_NOISE.detection_threshold_snr_db
        ),
        false_alarm_rate=lerp(_LOW_NOISE.false_alarm_rate, _HIGH_NOISE.false_alarm_rate),
        missed_detection_rate=lerp(
            _LOW_NOISE.missed_detection_rate, _HIGH_NOISE.missed_detection_rate
        ),
    )


def noise_tier_scenario(
    tier: str, seed: int, *, base_scenario: str = "normal", duration_slots: int | None = None
) -> ScenarioConfig:
    """``base_scenario``'s emitters (signal strength held constant) with its
    noise config replaced by ``tier``'s interpolated ``NoiseModelConfig`` --
    isolates noise as the only varying condition. (Comparing the "normal"
    and "high_noise" *scenarios* directly would confound noise with the
    weaker ``signal_dbm`` values ``high_noise`` also uses -- two variables at
    once, not a clean noise-only sweep.)"""
    if tier not in NOISE_TIERS:
        raise ValueError(f"unknown noise tier {tier!r}; available: {sorted(NOISE_TIERS)}")
    scenario = make_scenario(base_scenario, seed, duration_slots=duration_slots)
    noise = _interpolate_noise(NOISE_TIERS[tier])
    return dataclasses.replace(scenario, name=f"{base_scenario}_noise_{tier}", noise=noise)


def run_noise_sweep(
    *,
    strategies: Sequence[str] = ("sequential", "random", "heuristic", "adaptive"),
    tiers: Sequence[str] = tuple(NOISE_TIERS),
    world_seeds: Sequence[int] = tuple(range(20)),
    base_scenario: str = "normal",
    duration_slots: int | None = None,
    redundancy_window: int = 5,
    predictor: Predictor | None = None,
    scheduler_weights: SchedulerWeights | None = None,
    belief_config: BeliefConfig | None = None,
    heuristic_epsilon: float = 0.1,
) -> pd.DataFrame:
    """``scenarios x tiers x seeds x strategies`` grid, one metrics row per
    episode plus a ``noise_tier`` column. Reuses
    ``benchmark.build_scheduler`` -- no separate scheduler-construction path."""
    rows: list[dict] = []
    for tier in tiers:
        for world_seed in world_seeds:
            scenario = noise_tier_scenario(
                tier, world_seed, base_scenario=base_scenario, duration_slots=duration_slots
            )
            for strategy in strategies:
                env = scenario.build_environment(world_seed)
                sched = build_scheduler(
                    strategy,
                    env.n_channels,
                    agent_seed=world_seed,
                    epsilon=heuristic_epsilon,
                    predictor=predictor,
                    scheduler_weights=scheduler_weights,
                    belief_config=belief_config,
                )
                result = run_episode(
                    env, sched, budget=scenario.duration_slots, agent_seed=world_seed
                )
                metrics = compute_episode_metrics(result, redundancy_window=redundancy_window)
                row = metrics.to_row()
                row["noise_tier"] = tier
                rows.append(row)
        log.info("noise sweep: tier=%s done", tier)
    return pd.DataFrame(rows)


# -- non-stationarity ------------------------------------------------------
# changing_distribution's own public, documented behaviour
# (simulator/scenarios.py) -- channels 0-2 active for the first half, then at
# slot 400 fall silent while channels 9-11 switch on. Not hidden ground
# truth: the scenario's own contract, already used by
# tests/test_integration_phase7.py's non-stationarity test.
CHANGING_DISTRIBUTION_FLIP_SLOT = 400
CHANGING_DISTRIBUTION_HOT_THEN_QUIET = (0, 1, 2)
CHANGING_DISTRIBUTION_QUIET_THEN_HOT = (9, 10, 11)


@dataclass(frozen=True, slots=True)
class ChannelGroupSplit:
    label: str
    channels: tuple[int, ...]
    scans_before: int
    scans_after: int
    detections_before: int
    detections_after: int
    detection_rate_before: float | None
    detection_rate_after: float | None


def channel_group_split(
    result: EpisodeResult, channels: Sequence[int], split_slot: int, label: str
) -> ChannelGroupSplit:
    """Like :func:`~rfscan.experiments.adaptation.channel_scan_split`, summed
    over a group of channels (e.g. the three that flip together)."""
    scans_before = scans_after = detections_before = detections_after = 0
    for c in channels:
        sb, sa, db, da = channel_scan_split(result, c, split_slot)
        scans_before += sb
        scans_after += sa
        detections_before += db
        detections_after += da
    return ChannelGroupSplit(
        label=label,
        channels=tuple(channels),
        scans_before=scans_before,
        scans_after=scans_after,
        detections_before=detections_before,
        detections_after=detections_after,
        detection_rate_before=(detections_before / scans_before if scans_before else None),
        detection_rate_after=(detections_after / scans_after if scans_after else None),
    )


def run_non_stationarity_experiment(
    *,
    strategies: Sequence[str] = ("sequential", "random", "heuristic", "adaptive"),
    world_seeds: Sequence[int] = tuple(range(20)),
    duration_slots: int | None = None,
    predictor: Predictor | None = None,
    scheduler_weights: SchedulerWeights | None = None,
    belief_config: BeliefConfig | None = None,
    heuristic_epsilon: float = 0.1,
) -> pd.DataFrame:
    """``changing_distribution`` across strategies x seeds: one row per
    episode with both channel groups' before/after detection rates."""
    rows: list[dict] = []
    for world_seed in world_seeds:
        scenario = make_scenario("changing_distribution", world_seed, duration_slots=duration_slots)
        for strategy in strategies:
            env = scenario.build_environment(world_seed)
            sched = build_scheduler(
                strategy,
                env.n_channels,
                agent_seed=world_seed,
                epsilon=heuristic_epsilon,
                predictor=predictor,
                scheduler_weights=scheduler_weights,
                belief_config=belief_config,
            )
            result = run_episode(
                env, sched, budget=scenario.duration_slots, agent_seed=world_seed
            )
            hot_then_quiet = channel_group_split(
                result,
                CHANGING_DISTRIBUTION_HOT_THEN_QUIET,
                CHANGING_DISTRIBUTION_FLIP_SLOT,
                "hot_then_quiet",
            )
            quiet_then_hot = channel_group_split(
                result,
                CHANGING_DISTRIBUTION_QUIET_THEN_HOT,
                CHANGING_DISTRIBUTION_FLIP_SLOT,
                "quiet_then_hot",
            )
            rows.append(
                {
                    "scenario": "changing_distribution",
                    "world_seed": world_seed,
                    "strategy": strategy,
                    "hot_then_quiet_rate_before": hot_then_quiet.detection_rate_before,
                    "hot_then_quiet_rate_after": hot_then_quiet.detection_rate_after,
                    "hot_then_quiet_scans_before": hot_then_quiet.scans_before,
                    "hot_then_quiet_scans_after": hot_then_quiet.scans_after,
                    "quiet_then_hot_rate_before": quiet_then_hot.detection_rate_before,
                    "quiet_then_hot_rate_after": quiet_then_hot.detection_rate_after,
                    "quiet_then_hot_scans_before": quiet_then_hot.scans_before,
                    "quiet_then_hot_scans_after": quiet_then_hot.scans_after,
                }
            )
    return pd.DataFrame(rows)
