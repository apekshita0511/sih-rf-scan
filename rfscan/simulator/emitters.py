"""Emitter behaviour specifications.

Each channel may carry zero or more emitters. An emitter's temporal occupancy is
driven by a two-state (ON/OFF) Markov process (Gilbert-Elliott style); the
:class:`Behavior` selects a parameter preset / overlay on top of that process.
The actual stochastic model is implemented in Phase 2; this module only defines
the declarative spec that scenarios are built from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Behavior(str, Enum):
    """Qualitative emitter activity pattern.

    * ``PERSISTENT``   - turns on and largely stays on (low p_10).
    * ``INTERMITTENT`` - periodic on/off duty cycle.
    * ``BURSTY``       - short, clustered activations (high p_01 and high p_10).
    * ``EMERGING``     - silent until ``activation_slot``, then active.
    * ``FADING``       - activity level decays (or grows) over the episode.
    """

    PERSISTENT = "persistent"
    INTERMITTENT = "intermittent"
    BURSTY = "bursty"
    EMERGING = "emerging"
    FADING = "fading"


@dataclass(frozen=True, slots=True)
class EmitterSpec:
    """Declarative description of one emitter on one channel.

    ``params`` holds behaviour-specific knobs, all optional with model defaults:
    ``p01``, ``p10`` (Markov transition probabilities), ``duty``, ``period``,
    ``phase`` (intermittent), ``burst_rate`` (bursty), ``signal_dbm``
    (mean on-state signal power), ``fade_rate`` (fading).
    """

    channel_index: int
    behavior: Behavior
    params: dict[str, float] = field(default_factory=dict)
    activation_slot: int = 0
    deactivation_slot: int | None = None

    def is_active_at(self, slot: int) -> bool:
        """Whether the emitter's activation window contains ``slot`` (ignores the
        stochastic on/off state, which the environment tracks)."""
        if slot < self.activation_slot:
            return False
        if self.deactivation_slot is not None and slot >= self.deactivation_slot:
            return False
        return True
