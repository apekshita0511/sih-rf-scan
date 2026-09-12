"""Episode runner: drive one (scenario, seed, strategy) episode to completion.

The runner owns the per-slot lifecycle and is the *only* component that touches
ground truth -- it records `RFEnvironment.truth_snapshot()` into an evaluation
silo (:class:`EpisodeResult.occupancy`) that never reaches the scheduler. The
scheduler receives only ``scanner.store`` (a read-only observation view).

Fairness: the runner does nothing that couples the RF world to the strategy. The
world advances via ``env.step()`` regardless of which channel was scanned, and
the world RNG is keyed by ``(seed, slot, channel)`` (Phase 2), so every strategy
run against the same ``(scenario, world_seed)`` faces a byte-identical world.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from rfscan.logging_config import get_logger
from rfscan.perception.scanner import Scanner
from rfscan.scheduler.base import Scheduler
from rfscan.simulator.emitters import Behavior
from rfscan.simulator.environment import RFEnvironment

log = get_logger("experiments.runner")


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """Raw record of one episode. Arrays are indexed by slot ``0 .. n_slots-1``.

    ``occupancy`` is ground truth and is for evaluation only.
    ``decision_latencies_s`` is wall-clock timing -- nondeterministic, excluded
    from every reproducibility check.
    """

    scenario: str
    world_seed: int
    strategy: str
    agent_seed: int | None
    n_channels: int
    n_slots: int
    slot_duration_s: float
    scanned_channel: np.ndarray  # (n_slots,) int64
    observed_detection: np.ndarray  # (n_slots,) bool
    occupancy: np.ndarray  # (n_slots, n_channels) bool  -- GROUND TRUTH
    emerging_channels: tuple[int, ...]
    emerging_activation_slots: tuple[int, ...]
    decision_latencies_s: np.ndarray  # (n_slots,) float64  -- nondeterministic


def run_episode(
    env: RFEnvironment,
    scheduler: Scheduler,
    budget: int,
    *,
    agent_seed: int | None = None,
) -> EpisodeResult:
    """Run ``scheduler`` against ``env`` for ``budget`` slots (one scan per slot).

    ``env`` and ``scheduler`` are reset first, so the call is self-contained and
    repeatable with the same instances.
    """
    if budget < 1:
        raise ValueError(f"budget must be >= 1, got {budget}")

    env.reset()
    scheduler.reset()
    scanner = Scanner(env)
    n = env.n_channels

    scanned = np.empty(budget, dtype=np.int64)
    detected = np.empty(budget, dtype=bool)
    occupancy = np.empty((budget, n), dtype=bool)
    latency = np.empty(budget, dtype=np.float64)

    for slot in range(budget):
        t0 = time.perf_counter()
        choice = scheduler.select_next(scanner.store, slot)
        latency[slot] = time.perf_counter() - t0
        if not 0 <= choice < n:
            raise ValueError(
                f"scheduler {scheduler.name!r} chose channel {choice}, outside [0, {n})"
            )
        record = scanner.scan(choice, scheduler.explain())
        scheduler.update(record)
        occupancy[slot] = env.occupancy_snapshot()
        scanned[slot] = choice
        detected[slot] = record.detected
        env.step()

    emerging = sorted(
        (
            (spec.channel_index, spec.activation_slot)
            for spec in env.scenario.emitters
            if spec.behavior is Behavior.EMERGING
        ),
        key=lambda pair: pair[1],
    )

    return EpisodeResult(
        scenario=env.scenario.name,
        world_seed=env.seed,
        strategy=scheduler.name,
        agent_seed=agent_seed,
        n_channels=n,
        n_slots=budget,
        slot_duration_s=env.scenario.slot_duration_s,
        scanned_channel=scanned,
        observed_detection=detected,
        occupancy=occupancy,
        emerging_channels=tuple(c for c, _ in emerging),
        emerging_activation_slots=tuple(s for _, s in emerging),
        decision_latencies_s=latency,
    )
