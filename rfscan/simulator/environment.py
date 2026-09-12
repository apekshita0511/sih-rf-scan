"""The synthetic RF world.

:class:`RFEnvironment` turns a :class:`ScenarioConfig` plus an integer seed into
a deterministic, replayable episode.

Each slot, :meth:`RFEnvironment.step` advances *every* channel's emitters and
noise floor -- restless dynamics, independent of what the agent scans -- and
recomputes a per-channel measurement cell. :meth:`RFEnvironment.observe` returns
the agent-visible :class:`Observation` for the current slot;
:meth:`RFEnvironment.ground_truth` returns the hidden :class:`GroundTruth`, which
belongs to the evaluation silo only (see docs/architecture.md S16).

Signal / noise relationship (dB domain, internally consistent, not physically
exact):

    noise floor  ~  floor_dbm + AR(1) drift
    rssi_dbm     =  10*log10(signal_lin + noise_lin) + AWGN
    noise_dbm    =  noise_floor_dbm + AWGN
    snr_db       =  rssi_dbm - noise_dbm
    detection    =  (snr_db >= threshold), then FP/FN flips
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from rfscan.logging_config import get_logger
from rfscan.perception.schema import GroundTruth, Observation
from rfscan.simulator.channel import ChannelConfig, build_channel_plan
from rfscan.simulator.occupancy import EmitterProcess
from rfscan.simulator.rng import draw_rng
from rfscan.simulator.scenario import ScenarioConfig

log = get_logger("simulator.environment")


def _to_lin(power_dbm: float) -> float:
    return 10.0 ** (power_dbm / 10.0)


def _to_dbm(power_lin: float) -> float:
    return 10.0 * math.log10(power_lin)


@dataclass(frozen=True, slots=True)
class _Cell:
    """Everything known about one channel at the current slot."""

    occupied: bool
    true_signal_dbm: float | None
    noise_floor_dbm: float
    rssi_dbm: float
    noise_dbm: float
    snr_db: float
    observed_detection: bool


class RFEnvironment:
    """A seeded, replayable synthetic RF episode."""

    def __init__(self, scenario: ScenarioConfig, seed: int) -> None:
        self.scenario = scenario
        self.channels: list[ChannelConfig] = build_channel_plan(
            scenario.n_channels, scenario.channel_plan, scenario.channel_plan_params
        )
        self._noise = scenario.noise

        self._specs_by_channel: dict[int, list] = {}
        for spec in scenario.emitters:
            self._specs_by_channel.setdefault(spec.channel_index, []).append(spec)

        self._events_by_slot: dict[int, list] = {}
        for event in scenario.nonstationarity:
            if not 0 <= event.channel_index < self.n_channels:
                raise ValueError(
                    f"non-stationary event references channel {event.channel_index}, "
                    f"outside [0, {self.n_channels})"
                )
            if event.slot < 0:
                raise ValueError(f"non-stationary event slot must be >= 0, got {event.slot}")
            if event.channel_index not in self._specs_by_channel:
                log.warning(
                    "non-stationary event at slot %d targets channel %d, which has no emitter",
                    event.slot,
                    event.channel_index,
                )
            self._events_by_slot.setdefault(event.slot, []).append(event)

        self._seed = int(seed)
        self._slot = 0
        self._emitters: dict[int, list[EmitterProcess]] = {}
        self._noise_floor: dict[int, float] = {}
        self._cells: dict[int, _Cell] = {}
        self.reset(seed)

    # -- introspection -------------------------------------------------
    @property
    def n_channels(self) -> int:
        return self.scenario.n_channels

    @property
    def slot(self) -> int:
        return self._slot

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def done(self) -> bool:
        """True once the current slot has reached the scenario's configured span."""
        return self._slot >= self.scenario.duration_slots - 1

    # -- lifecycle ---------------------------------------------------
    def reset(self, seed: int | None = None) -> None:
        """Return to slot 0. ``seed=None`` replays the current seed."""
        if seed is not None:
            self._seed = int(seed)
        self._slot = 0
        self._emitters = {
            c: [EmitterProcess(s) for s in specs] for c, specs in self._specs_by_channel.items()
        }
        self._apply_events(0)
        self._noise_floor = {}
        self._cells = {}
        sigma = self._noise.floor_drift_std_db
        for c in range(self.n_channels):
            rng = draw_rng(self._seed, 0, c)
            for emitter in self._emitters.get(c, []):
                emitter.reset(rng.random())
            self._noise_floor[c] = self._noise.floor_dbm + sigma * rng.standard_normal()
            self._cells[c] = self._measure(c, rng)

    def step(self) -> None:
        """Advance the whole world by one slot."""
        self._slot += 1
        self._apply_events(self._slot)
        noise = self._noise
        rho = noise.floor_drift_rho
        innov_scale = math.sqrt(max(0.0, 1.0 - rho * rho)) * noise.floor_drift_std_db
        for c in range(self.n_channels):
            rng = draw_rng(self._seed, self._slot, c)
            for emitter in self._emitters.get(c, []):
                emitter.advance(self._slot, rng.random())
            centered = self._noise_floor[c] - noise.floor_dbm
            self._noise_floor[c] = (
                noise.floor_dbm + rho * centered + innov_scale * rng.standard_normal()
            )
            self._cells[c] = self._measure(c, rng)

    # -- accessors -------------------------------------------------
    def observe(self, channel_index: int) -> Observation:
        """Agent-visible measurement of ``channel_index`` at the current slot.

        Pure: calling it (in any order, any number of times) never changes the
        world. Repeated calls in the same slot return the identical reading.
        """
        cell = self._cell(channel_index)
        return Observation(
            slot=self._slot,
            channel_index=channel_index,
            rssi_dbm=cell.rssi_dbm,
            noise_dbm=cell.noise_dbm,
            snr_db=cell.snr_db,
            observed_detection=cell.observed_detection,
        )

    def ground_truth(self, channel_index: int) -> GroundTruth:
        """True state of ``channel_index`` at the current slot. Evaluation only."""
        cell = self._cell(channel_index)
        return GroundTruth(
            slot=self._slot,
            channel_index=channel_index,
            occupied=cell.occupied,
            true_signal_dbm=cell.true_signal_dbm,
        )

    def truth_snapshot(self) -> list[GroundTruth]:
        """Ground truth for every channel at the current slot (evaluation only)."""
        return [self.ground_truth(c) for c in range(self.n_channels)]

    def occupancy_snapshot(self) -> tuple[bool, ...]:
        """True occupancy of every channel at the current slot, without building
        `GroundTruth` objects (evaluation only; hot path for the experiment
        runner)."""
        return tuple(self._cells[c].occupied for c in range(self.n_channels))

    # -- internals -------------------------------------------------
    def _cell(self, channel_index: int) -> _Cell:
        if not 0 <= channel_index < self.n_channels:
            raise IndexError(f"channel {channel_index} out of range [0, {self.n_channels})")
        return self._cells[channel_index]

    def _apply_events(self, slot: int) -> None:
        for event in self._events_by_slot.get(slot, []):
            for emitter in self._emitters.get(event.channel_index, []):
                emitter.apply_overrides(event.param_overrides)
            log.debug(
                "non-stationary event applied: channel %d, slot %d, overrides %s",
                event.channel_index,
                slot,
                dict(event.param_overrides),
            )

    def _measure(self, channel_index: int, rng: np.random.Generator) -> _Cell:
        active = [e for e in self._emitters.get(channel_index, []) if e.on]
        floor_dbm = self._noise_floor[channel_index]
        noise_lin = _to_lin(floor_dbm)

        if active:
            signal_lin = sum(_to_lin(e.signal_dbm) for e in active)
            true_signal_dbm: float | None = _to_dbm(signal_lin)
            total_lin = noise_lin + signal_lin
        else:
            true_signal_dbm = None
            total_lin = noise_lin

        awgn = self._noise.awgn_std_db
        rssi_dbm = _to_dbm(total_lin) + awgn * rng.standard_normal()
        noise_dbm = floor_dbm + awgn * rng.standard_normal()
        snr_db = rssi_dbm - noise_dbm

        energy_hit = snr_db >= self._noise.detection_threshold_snr_db
        u_fa = rng.random()
        u_miss = rng.random()
        if active:
            observed = energy_hit and u_miss >= self._noise.missed_detection_rate
        else:
            observed = energy_hit or u_fa < self._noise.false_alarm_rate

        return _Cell(
            occupied=bool(active),
            true_signal_dbm=true_signal_dbm,
            noise_floor_dbm=floor_dbm,
            rssi_dbm=rssi_dbm,
            noise_dbm=noise_dbm,
            snr_db=snr_db,
            observed_detection=observed,
        )


def build_environment(scenario: ScenarioConfig, seed: int) -> RFEnvironment:
    """Construct a fresh :class:`RFEnvironment` for ``scenario`` at ``seed``."""
    return RFEnvironment(scenario, seed)
