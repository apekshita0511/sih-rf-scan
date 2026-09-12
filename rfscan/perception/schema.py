"""Core data contracts shared across layers.

Design rule (see docs/architecture.md, sections 5 and 16):

    * :class:`Observation` is the ONLY signal information a scheduler or ML model
      may consume. It deliberately carries no ground-truth field.
    * :class:`GroundTruth` is the true channel state. It lives in the evaluation
      silo (experiment runner + metrics) and must never reach a scheduler.

Keeping these as two separate types makes information leakage a structural
impossibility rather than a convention. ``tests/test_schema.py`` enforces it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

# Field names that must never appear on Observation / ScanRecord. Used both as
# documentation and as an automated leakage guard in the test suite.
FORBIDDEN_OBSERVATION_FIELDS = ("occupied", "truth", "true_signal", "ground_truth", "is_active")


@dataclass(frozen=True, slots=True)
class Observation:
    """Agent-visible result of scanning one channel during one slot.

    Values are what a receiver would *measure* - already corrupted by noise and
    by the energy detector's decision threshold, so ``observed_detection`` may be
    a false positive or a false negative relative to the true channel state.
    """

    slot: int
    channel_index: int
    rssi_dbm: float
    noise_dbm: float
    snr_db: float
    observed_detection: bool


@dataclass(frozen=True, slots=True)
class GroundTruth:
    """True state of one channel at one slot. Evaluation silo only."""

    slot: int
    channel_index: int
    occupied: bool
    true_signal_dbm: float | None


@dataclass(frozen=True, slots=True)
class ScanRecord:
    """One entry in a scanner's history: an observation plus optional decision
    metadata (e.g. the adaptive scheduler's priority-term breakdown, attached
    from Phase 6 for the explainability panel)."""

    observation: Observation
    decision_info: Mapping[str, float] | None = field(default=None)

    @property
    def slot(self) -> int:
        return self.observation.slot

    @property
    def channel_index(self) -> int:
        return self.observation.channel_index

    @property
    def detected(self) -> bool:
        return self.observation.observed_detection
