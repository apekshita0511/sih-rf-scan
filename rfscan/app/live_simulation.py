"""Phase 9: thin, stateful wrapper around the EXISTING simulator/scheduler
engine for the dashboard's live simulation tab. No new simulation or
decision logic here -- every :meth:`LiveSimulation.step` call runs exactly
the same ``select_next -> scan -> update -> env.step()`` sequence
:func:`rfscan.experiments.runner.run_episode` already uses, one slot at a
time instead of a full episode, purely so the dashboard can step/pause/
replay interactively.

Ground truth (:meth:`LiveSimulation.ground_truth_snapshot`) is a SEPARATE
method, never passed to ``self.scheduler`` -- the dashboard must label any
use of it "Simulation Truth -- evaluation only" (docs/architecture.md
S16.1's anti-leakage rule is structural in the engine; this wrapper does not
weaken it -- ``select_next`` is only ever called with ``self.scanner.store``,
a ``ReadableStore``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from rfscan.experiments.benchmark import STRATEGIES as BASELINE_STRATEGIES
from rfscan.experiments.benchmark import build_scheduler
from rfscan.experiments.metrics import EpisodeMetrics, compute_episode_metrics
from rfscan.experiments.runner import EpisodeResult
from rfscan.models.base import Predictor
from rfscan.models.baseline_beta import DecayingBetaPredictor
from rfscan.perception.scanner import Scanner
from rfscan.perception.schema import ScanRecord
from rfscan.scheduler.adaptive import AdaptiveScheduler
from rfscan.scheduler.base import Scheduler
from rfscan.simulator.channel import build_channel_plan
from rfscan.simulator.emitters import Behavior
from rfscan.simulator.environment import RFEnvironment
from rfscan.simulator.scenarios import list_scenarios, make_scenario

STRATEGIES = (*BASELINE_STRATEGIES, "adaptive")


def available_scenarios() -> list[str]:
    return list_scenarios()


def strategy_requires_predictor(name: str) -> bool:
    return name == "adaptive"


def default_predictor() -> Predictor:
    """Always-available, zero-I/O predictor for the live demo -- no
    dependency on a trained artifact existing on disk (a fresh clone has
    none, since artifacts/*.joblib is git-ignored)."""
    return DecayingBetaPredictor()


def try_load_live_predictor() -> tuple[Predictor, str]:
    """The real live-serving predictor (logistic_regression,
    docs/architecture.md S16.6), if a trained artifact exists; falls back to
    the beta predictor, with a note, if not."""
    from rfscan.config import ModelConfig
    from rfscan.models.loader import load_predictor

    try:
        return load_predictor(ModelConfig()), "logistic_regression"
    except FileNotFoundError:
        note = "decaying_beta (no trained artifact found -- run `rfscan train`)"
        return DecayingBetaPredictor(), note


@dataclass
class LiveSimulation:
    """One steppable episode. Call :meth:`step` repeatedly (or ``reset`` to
    start over with the same or different scenario/strategy)."""

    scenario_name: str
    world_seed: int
    strategy_name: str
    budget: int
    predictor: Predictor | None = None
    agent_seed: int | None = None
    heuristic_epsilon: float = 0.1

    env: RFEnvironment = field(init=False)
    scanner: Scanner = field(init=False)
    scheduler: Scheduler = field(init=False)
    channel_labels: list[str] = field(init=False)
    slot: int = field(init=False, default=0)
    history: list[ScanRecord] = field(init=False, default_factory=list)
    last_choice: int | None = field(init=False, default=None)
    last_explain: dict = field(init=False, default_factory=dict)
    _occupancy_history: list[tuple[bool, ...]] = field(init=False, default_factory=list)
    _latency_history: list[float] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        if self.strategy_name not in STRATEGIES:
            raise ValueError(f"unknown strategy {self.strategy_name!r}; available: {STRATEGIES}")
        if strategy_requires_predictor(self.strategy_name) and self.predictor is None:
            self.predictor = default_predictor()
        self.reset()

    def reset(self) -> None:
        scenario = make_scenario(self.scenario_name, self.world_seed, duration_slots=self.budget)
        self.env = scenario.build_environment(self.world_seed)
        self.scanner = Scanner(self.env)
        seed = self.agent_seed if self.agent_seed is not None else self.world_seed
        self.scheduler = build_scheduler(
            self.strategy_name,
            self.env.n_channels,
            agent_seed=seed,
            epsilon=self.heuristic_epsilon,
            predictor=self.predictor,
        )
        channels = build_channel_plan(
            scenario.n_channels, scenario.channel_plan, scenario.channel_plan_params
        )
        self.channel_labels = [c.label for c in channels]
        self.slot = 0
        self.history = []
        self.last_choice = None
        self.last_explain = {}
        self._occupancy_history = []
        self._latency_history = []

    @property
    def n_channels(self) -> int:
        return self.env.n_channels

    @property
    def done(self) -> bool:
        return self.slot >= self.budget

    def step(self) -> ScanRecord | None:
        """Advance exactly one slot: select_next -> scan -> update ->
        env.step(), the same lifecycle run_episode uses. Returns the new
        ScanRecord, or None if the episode's budget is already exhausted.

        Also records ``env.occupancy_snapshot()`` (ground truth) into an
        internal evaluation-only history, exactly as
        ``experiments.runner.run_episode`` does -- for computing real
        partial-episode metrics via :meth:`partial_metrics`, never passed to
        ``self.scheduler``."""
        if self.done:
            return None
        t0 = time.perf_counter()
        choice = self.scheduler.select_next(self.scanner.store, self.slot)
        latency = time.perf_counter() - t0
        record = self.scanner.scan(choice, self.scheduler.explain())
        self.scheduler.update(record)
        self.last_choice = choice
        self.last_explain = dict(self.scheduler.explain())
        self.history.append(record)
        self._occupancy_history.append(self.env.occupancy_snapshot())
        self._latency_history.append(latency)
        self.slot += 1
        self.env.step()
        return record

    def all_breakdowns(self):
        """Every channel's PriorityBreakdown (S16.7 introspection), only
        when the live scheduler is an AdaptiveScheduler; None otherwise."""
        if isinstance(self.scheduler, AdaptiveScheduler):
            return self.scheduler.all_breakdowns()
        return None

    def belief_snapshot(self):
        if isinstance(self.scheduler, AdaptiveScheduler):
            return self.scheduler.belief_snapshot()
        return None

    def ground_truth_snapshot(self) -> tuple[bool, ...]:
        """Simulation Truth -- evaluation only. Never pass this to
        ``self.scheduler``; the caller (dashboard) must label any display of
        it accordingly."""
        return self.env.occupancy_snapshot()

    def scan_history_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """``(scanned_channel, detected)`` arrays over the episode so far."""
        scanned = np.array([r.channel_index for r in self.history], dtype=np.int64)
        detected = np.array([r.detected for r in self.history], dtype=bool)
        return scanned, detected

    def to_episode_result(self) -> EpisodeResult | None:
        """An :class:`EpisodeResult` built from the slots run so far -- the
        exact same shape ``experiments.runner.run_episode`` produces for a
        full episode, so :func:`compute_episode_metrics` and
        :func:`emerging_adaptation_metrics` (Phases 4/7) work unmodified on
        a partial live episode. ``None`` before the first :meth:`step`."""
        if not self.history:
            return None
        emerging = sorted(
            (
                (spec.channel_index, spec.activation_slot)
                for spec in self.env.scenario.emitters
                if spec.behavior is Behavior.EMERGING
            ),
            key=lambda pair: pair[1],
        )
        return EpisodeResult(
            scenario=self.env.scenario.name,
            world_seed=self.env.seed,
            strategy=self.scheduler.name,
            agent_seed=self.agent_seed,
            n_channels=self.n_channels,
            n_slots=len(self.history),
            slot_duration_s=self.env.scenario.slot_duration_s,
            scanned_channel=np.array([r.channel_index for r in self.history], dtype=np.int64),
            observed_detection=np.array([r.detected for r in self.history], dtype=bool),
            occupancy=np.array(self._occupancy_history, dtype=bool),
            emerging_channels=tuple(c for c, _ in emerging),
            emerging_activation_slots=tuple(s for _, s in emerging),
            decision_latencies_s=np.array(self._latency_history, dtype=np.float64),
        )

    def partial_metrics(self) -> EpisodeMetrics | None:
        """Real, existing (Phase 4) metric definitions applied to the
        episode so far. ``None`` before the first :meth:`step` -- the
        dashboard should show "Not enough data yet", never a fabricated 0."""
        result = self.to_episode_result()
        if result is None:
            return None
        return compute_episode_metrics(result)
