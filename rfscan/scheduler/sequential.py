"""Non-adaptive baseline schedulers: round-robin and uniform-random.

Both ignore observations entirely -- they are the floor the adaptive scheduler
must beat on the identical seeded world. The random strategy draws from its own
agent RNG, which is separate from the world RNG (docs/architecture.md S2), so
changing the agent seed changes the scan pattern but never the RF world.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from rfscan.perception.schema import ScanRecord
from rfscan.perception.store import ReadableStore


class SequentialScheduler:
    """Round-robin: ``start, start+1, ..., N-1, 0, 1, ...``.

    ``select_next`` returns the current cursor and advances it, so it must be
    called exactly once per slot.
    """

    name = "sequential"

    def __init__(self, n_channels: int, start: int = 0) -> None:
        if n_channels < 1:
            raise ValueError(f"n_channels must be >= 1, got {n_channels}")
        if not 0 <= start < n_channels:
            raise ValueError(f"start must be in [0, {n_channels}), got {start}")
        self._n = n_channels
        self._start = start
        self._cursor = start

    def select_next(self, store: ReadableStore, slot: int) -> int:
        choice = self._cursor
        self._cursor = (self._cursor + 1) % self._n
        return choice

    def update(self, record: ScanRecord) -> None:
        pass

    def explain(self) -> Mapping[str, float]:
        return {"next_cursor": float(self._cursor)}

    def reset(self) -> None:
        self._cursor = self._start


class RandomScheduler:
    """Uniform-random channel selection from a dedicated, seeded agent RNG."""

    name = "random"

    def __init__(self, n_channels: int, seed: int = 0) -> None:
        if n_channels < 1:
            raise ValueError(f"n_channels must be >= 1, got {n_channels}")
        self._n = n_channels
        self._seed = int(seed)
        self._rng = np.random.default_rng(self._seed)
        self._last = -1

    def select_next(self, store: ReadableStore, slot: int) -> int:
        self._last = int(self._rng.integers(self._n))
        return self._last

    def update(self, record: ScanRecord) -> None:
        pass

    def explain(self) -> Mapping[str, float]:
        return {"choice": float(self._last)}

    def reset(self) -> None:
        self._rng = np.random.default_rng(self._seed)
        self._last = -1
