"""Stochastic channel-occupancy model.

An emitter's true ON/OFF trajectory is a two-state (Gilbert-Elliott) Markov
process, optionally shaped by its :class:`~rfscan.simulator.emitters.Behavior`:

* ``PERSISTENT``   - low OFF->ON and ON->OFF rates; long dwell in each state.
* ``INTERMITTENT`` - a periodic duty-cycle gate; Markov only while the gate is open.
* ``BURSTY``       - rare onsets, fast decay -> short clustered activations.
* ``EMERGING``     - forced OFF until ``activation_slot`` (see EmitterSpec), then persistent.
* ``FADING``       - onset rate decays (or grows) exponentially over the episode.

Deliberately minimal: one Markov chain plus a few typed overlays. It models the
*decision-relevant* dynamics of channel activity, not RF propagation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from rfscan.simulator.emitters import Behavior, EmitterSpec

# Mean on-state signal power used when an EmitterSpec does not set ``signal_dbm``.
_DEFAULT_SIGNAL_DBM = -60.0

# Per-behaviour parameter presets. Any key here can be overridden per emitter via
# ``EmitterSpec.params`` or mid-episode via ``NonStationaryEvent.param_overrides``.
_BEHAVIOR_DEFAULTS: dict[Behavior, dict[str, float]] = {
    Behavior.PERSISTENT: {"p01": 0.05, "p10": 0.02},
    Behavior.INTERMITTENT: {"p01": 0.70, "p10": 0.05, "duty": 0.30, "period": 40.0, "phase": 0.0},
    Behavior.BURSTY: {"p01": 0.04, "p10": 0.40},
    Behavior.EMERGING: {"p01": 0.12, "p10": 0.03},
    Behavior.FADING: {"p01": 0.15, "p10": 0.03, "fade_rate": 0.003},
}


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


@dataclass(frozen=True, slots=True)
class GilbertElliott:
    """Two-state OFF(0)/ON(1) Markov chain with per-slot transition probabilities."""

    p01: float  # P(OFF -> ON) in one slot
    p10: float  # P(ON  -> OFF) in one slot

    def __post_init__(self) -> None:
        for name in ("p01", "p10"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")

    @property
    def stationary_on_prob(self) -> float:
        """Long-run fraction of time in the ON state."""
        total = self.p01 + self.p10
        return 0.0 if total == 0.0 else self.p01 / total

    def initial_state(self, u: float) -> bool:
        """Draw a state from the stationary distribution given uniform ``u``."""
        return u < self.stationary_on_prob

    def next_state(self, state: bool, u: float) -> bool:
        """Advance one slot given the current ``state`` and uniform ``u``."""
        if state:
            return u >= self.p10
        return u < self.p01


class EmitterProcess:
    """Mutable ON/OFF simulation state for a single :class:`EmitterSpec`.

    One uniform ``u`` in ``[0, 1)`` is consumed per :meth:`advance` call; the
    caller owns the RNG (see the fairness invariant in :mod:`rfscan.simulator.rng`).
    """

    __slots__ = ("spec", "signal_dbm", "_params", "_p01", "_p10", "_on")

    def __init__(self, spec: EmitterSpec) -> None:
        self.spec = spec
        params = dict(_BEHAVIOR_DEFAULTS[spec.behavior])
        params.update(spec.params)
        if spec.behavior is Behavior.BURSTY and "burst_rate" in spec.params:
            params["p01"] = spec.params["burst_rate"]
        self._params = params
        self._p01 = _clip01(float(params["p01"]))
        self._p10 = _clip01(float(params["p10"]))
        self.signal_dbm = float(params.get("signal_dbm", _DEFAULT_SIGNAL_DBM))
        self._on = False

    def apply_overrides(self, overrides: Mapping[str, float]) -> None:
        """Apply a mid-episode parameter change (from a ``NonStationaryEvent``)."""
        self._params.update(overrides)
        if "burst_rate" in overrides:
            self._p01 = _clip01(float(overrides["burst_rate"]))
        if "p01" in overrides:
            self._p01 = _clip01(float(overrides["p01"]))
        if "p10" in overrides:
            self._p10 = _clip01(float(overrides["p10"]))
        if "signal_dbm" in overrides:
            self.signal_dbm = float(overrides["signal_dbm"])

    @property
    def on(self) -> bool:
        return self._on

    def reset(self, u: float) -> None:
        """Reset to a stationary-distribution draw for slot 0 (or OFF if the
        emitter is not active / gated open at slot 0)."""
        if self.spec.is_active_at(0) and self._gate_open(0):
            self._on = self._transitions_at(0).initial_state(u)
        else:
            self._on = False

    def advance(self, slot: int, u: float) -> None:
        """Evolve the ON/OFF state into ``slot``."""
        if not self.spec.is_active_at(slot) or not self._gate_open(slot):
            self._on = False
            return
        self._on = self._transitions_at(slot).next_state(self._on, u)

    # -- overlays -------------------------------------------------------
    def _transitions_at(self, slot: int) -> GilbertElliott:
        p01 = self._p01
        if self.spec.behavior is Behavior.FADING:
            elapsed = max(0, slot - self.spec.activation_slot)
            p01 = _clip01(p01 * math.exp(-float(self._params["fade_rate"]) * elapsed))
        return GilbertElliott(p01=p01, p10=self._p10)

    def _gate_open(self, slot: int) -> bool:
        if self.spec.behavior is not Behavior.INTERMITTENT:
            return True
        period = float(self._params["period"])
        if period <= 0.0:
            return True
        phase = float(self._params["phase"])
        return (((slot + phase) % period) / period) < float(self._params["duty"])
