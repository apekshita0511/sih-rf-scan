"""Scheduler contract.

A scheduler answers one question each slot: *which channel do I scan next?* It
sees only a :class:`~rfscan.perception.store.ReadableStore` -- never simulator
ground truth. Structural typing (Protocol), not inheritance, so strategies are
trivially swappable and mockable (docs/architecture.md S4).

Lifecycle per slot, driven by the experiment runner (Phase 4)::

    channel = scheduler.select_next(store, slot)
    record  = scanner.scan(channel, scheduler.explain())
    scheduler.update(record)
    env.step()
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from rfscan.perception.schema import ScanRecord
from rfscan.perception.store import ReadableStore


@runtime_checkable
class Scheduler(Protocol):
    """Structural contract every scan strategy satisfies."""

    name: str

    def select_next(self, store: ReadableStore, slot: int) -> int:
        """Return the channel index to scan at ``slot``."""
        ...

    def update(self, record: ScanRecord) -> None:
        """Incorporate the result of the scan just performed."""
        ...

    def explain(self) -> Mapping[str, float]:
        """Rationale for the most recent :meth:`select_next` decision, as a flat
        map of named numeric terms (attached to the ScanRecord for the
        explainability panel)."""
        ...

    def reset(self) -> None:
        """Return to the pre-episode state (cursor, agent RNG, counters)."""
        ...
