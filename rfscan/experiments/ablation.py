"""Phase 8: component ablation over AdaptiveScheduler's priority policy.

Progressively enables one more priority-score term (docs/architecture.md S7)
at a time, using the SAME ``SchedulerWeights`` knobs the live scheduler
already exposes -- no parallel scheduler implementation, per this phase's
"create clean configuration switches rather than copy-pasting" instruction.
A sixth, orthogonal variant toggles ``AdaptiveScheduler``'s ``freeze_belief``
flag (scheduler/adaptive.py, added this phase) to isolate whether *online
feedback itself* (Phase 7) matters, independent of which weight terms are on.

The hard freshness guarantee (``max_revisit_slots``) is left active in every
variant, including "prediction only" -- it is a starvation safety net, not
one of the five scored terms, and disabling it would make the weakest
variants fail for an unrelated reason (a channel never getting scanned at
all) rather than the reason the ablation is meant to isolate (how good is
this *term combination* at prioritising, given every channel is still
guaranteed eventual coverage). Documented, not hidden.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from rfscan.config import BeliefConfig, SchedulerWeights
from rfscan.experiments.metrics import compute_episode_metrics
from rfscan.experiments.runner import run_episode
from rfscan.logging_config import get_logger
from rfscan.models.base import Predictor
from rfscan.scheduler.adaptive import AdaptiveScheduler
from rfscan.simulator.scenarios import list_scenarios, make_scenario

log = get_logger("experiments.ablation")

_FULL = SchedulerWeights()


@dataclass(frozen=True, slots=True)
class AblationVariant:
    name: str
    weights: SchedulerWeights
    freeze_belief: bool = False
    description: str = ""


def _weights(
    *,
    w_pred: float = 0.0,
    w_explore: float = 0.0,
    w_fresh: float = 0.0,
    w_trend: float = 0.0,
    w_redundancy: float = 0.0,
) -> SchedulerWeights:
    return SchedulerWeights(
        w_pred=w_pred,
        w_explore=w_explore,
        w_fresh=w_fresh,
        w_trend=w_trend,
        w_redundancy=w_redundancy,
        freshness_tau_slots=_FULL.freshness_tau_slots,
        softmax_temperature=_FULL.softmax_temperature,
    )


ABLATION_VARIANTS: tuple[AblationVariant, ...] = (
    AblationVariant(
        "A_prediction_only",
        _weights(w_pred=_FULL.w_pred),
        description="Greedy argmax on predicted P(active) alone.",
    ),
    AblationVariant(
        "B_prediction_exploration",
        _weights(w_pred=_FULL.w_pred, w_explore=_FULL.w_explore),
        description="Adds the belief-uncertainty exploration term.",
    ),
    AblationVariant(
        "C_prediction_exploration_freshness",
        _weights(w_pred=_FULL.w_pred, w_explore=_FULL.w_explore, w_fresh=_FULL.w_fresh),
        description="Adds the freshness/staleness term.",
    ),
    AblationVariant(
        "D_prediction_exploration_freshness_trend",
        _weights(
            w_pred=_FULL.w_pred,
            w_explore=_FULL.w_explore,
            w_fresh=_FULL.w_fresh,
            w_trend=_FULL.w_trend,
        ),
        description="Adds the rising-activity trend term.",
    ),
    AblationVariant(
        "E_full_policy",
        _FULL,
        description="All five terms -- the actual default AdaptiveScheduler configuration.",
    ),
    AblationVariant(
        "F_full_policy_no_online_feedback",
        _FULL,
        freeze_belief=True,
        description=(
            "Full policy, but BeliefState never updates from a hit/miss -- "
            "isolates whether online feedback (Phase 7) matters, independent "
            "of the weight terms."
        ),
    ),
)


def build_ablation_scheduler(
    variant: AblationVariant,
    n_channels: int,
    predictor: Predictor,
    *,
    belief_config: BeliefConfig | None = None,
    seed: int = 0,
) -> AdaptiveScheduler:
    return AdaptiveScheduler(
        n_channels,
        predictor,
        weights=variant.weights,
        belief_config=belief_config,
        seed=seed,
        freeze_belief=variant.freeze_belief,
    )


def run_ablation(
    predictor: Predictor,
    *,
    scenarios: Sequence[str] | None = None,
    world_seeds: Sequence[int] = tuple(range(15)),
    variants: Sequence[AblationVariant] = ABLATION_VARIANTS,
    duration_slots: int | None = None,
    belief_config: BeliefConfig | None = None,
    redundancy_window: int = 5,
) -> pd.DataFrame:
    """Run every (scenario, world_seed, variant) cell and return one metrics
    row per episode, with a ``variant`` column (``strategy`` is always
    ``"adaptive"`` here -- ``variant`` is what distinguishes rows). Same
    predictor instance reused throughout: every fitted ``Predictor`` here is
    stateless after fitting, so this cannot leak state between cells."""
    scenario_names = list(scenarios) if scenarios is not None else list_scenarios()
    rows: list[dict] = []
    for scenario_name in scenario_names:
        for world_seed in world_seeds:
            scenario = make_scenario(scenario_name, world_seed, duration_slots=duration_slots)
            for variant in variants:
                env = scenario.build_environment(world_seed)
                sched = build_ablation_scheduler(
                    variant,
                    env.n_channels,
                    predictor,
                    belief_config=belief_config,
                    seed=world_seed,
                )
                result = run_episode(
                    env, sched, budget=scenario.duration_slots, agent_seed=world_seed
                )
                metrics = compute_episode_metrics(result, redundancy_window=redundancy_window)
                row = metrics.to_row()
                row["variant"] = variant.name
                rows.append(row)
            log.info("ablation: %s seed=%d done", scenario_name, world_seed)
    return pd.DataFrame(rows)


def summarize_ablation(raw: pd.DataFrame) -> pd.DataFrame:
    """Mean / std / count per (scenario, variant) over the summary metrics."""
    from rfscan.experiments.metrics import SUMMARY_METRICS

    present = [m for m in SUMMARY_METRICS if m in raw.columns]
    grouped = raw.groupby(["scenario", "variant"], sort=True)[present].agg(["mean", "std", "count"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    return grouped.reset_index()
