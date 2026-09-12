"""Reproducible scenario specification.

A :class:`ScenarioConfig` fully determines a synthetic RF episode: given the
scenario and a seed, the environment (Phase 2) produces a byte-identical world,
so every scanner strategy can be benchmarked against the same realisation.

Phase 1 provided the spec and one worked example (``example_normal``). Phase 2
adds :meth:`ScenarioConfig.build_environment` (materialise into a running
:class:`~rfscan.simulator.environment.RFEnvironment`) and a second worked
example, ``example_emerging``, to exercise emitter behaviours and non-stationary
events. The seven *tuned* benchmark scenarios (normal, high-activity, bursty,
emerging, high-noise, dynamic, distribution-shift) and their registry are still
built in Phase 4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rfscan.simulator.emitters import Behavior, EmitterSpec
from rfscan.simulator.noise import NoiseModelConfig

if TYPE_CHECKING:
    from rfscan.simulator.environment import RFEnvironment


@dataclass(frozen=True, slots=True)
class NonStationaryEvent:
    """A scheduled change to a channel's emitter parameters mid-episode."""

    slot: int
    channel_index: int
    param_overrides: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    name: str
    seed: int
    n_channels: int
    duration_slots: int
    slot_duration_s: float = 0.1
    channel_plan: str = "wifi_24ghz"
    channel_plan_params: dict[str, float] = field(default_factory=dict)
    emitters: list[EmitterSpec] = field(default_factory=list)
    noise: NoiseModelConfig = field(default_factory=NoiseModelConfig)
    nonstationarity: list[NonStationaryEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.n_channels < 1:
            raise ValueError(f"n_channels must be >= 1, got {self.n_channels}")
        if self.duration_slots < 1:
            raise ValueError(f"duration_slots must be >= 1, got {self.duration_slots}")
        if self.slot_duration_s <= 0:
            raise ValueError(f"slot_duration_s must be > 0, got {self.slot_duration_s}")
        for emitter in self.emitters:
            if not 0 <= emitter.channel_index < self.n_channels:
                raise ValueError(
                    f"emitter references channel {emitter.channel_index}, "
                    f"outside [0, {self.n_channels})"
                )

    @property
    def duration_s(self) -> float:
        return self.duration_slots * self.slot_duration_s

    def build_environment(self, seed: int | None = None) -> RFEnvironment:
        """Materialise this scenario into a deterministic
        :class:`~rfscan.simulator.environment.RFEnvironment`. ``seed=None`` uses
        the scenario's own ``seed``."""
        from rfscan.simulator.environment import RFEnvironment

        return RFEnvironment(self, self.seed if seed is None else seed)


def example_normal() -> ScenarioConfig:
    """Small hand-written scenario. Not one of the benchmark seven - it exists so
    Phase 1 can exercise construction, validation, and config serialisation."""
    return ScenarioConfig(
        name="example_normal",
        seed=0,
        n_channels=6,
        duration_slots=600,
        slot_duration_s=0.1,
        channel_plan="wifi_24ghz",
        emitters=[
            EmitterSpec(0, Behavior.PERSISTENT, {"p01": 0.05, "p10": 0.02, "signal_dbm": -55.0}),
            EmitterSpec(2, Behavior.INTERMITTENT, {"duty": 0.3, "period": 40.0}),
            EmitterSpec(5, Behavior.BURSTY, {"burst_rate": 0.04, "p10": 0.4}),
        ],
        noise=NoiseModelConfig(),
    )


def example_emerging() -> ScenarioConfig:
    """Worked example with a mid-episode emerging signal and a non-stationary
    event. Exercises the Phase 2 simulator machinery; the tuned benchmark
    ``emerging`` scenario is Phase 4."""
    return ScenarioConfig(
        name="example_emerging",
        seed=0,
        n_channels=8,
        duration_slots=600,
        slot_duration_s=0.1,
        channel_plan="wifi_24ghz",
        emitters=[
            EmitterSpec(1, Behavior.PERSISTENT, {"p01": 0.06, "p10": 0.03, "signal_dbm": -58.0}),
            EmitterSpec(6, Behavior.INTERMITTENT, {"duty": 0.35, "period": 50.0}),
            EmitterSpec(
                4,
                Behavior.EMERGING,
                {"p01": 0.25, "p10": 0.02, "signal_dbm": -62.0},
                activation_slot=200,
            ),
        ],
        nonstationarity=[
            NonStationaryEvent(slot=350, channel_index=6, param_overrides={"duty": 0.8}),
        ],
        noise=NoiseModelConfig(),
    )
