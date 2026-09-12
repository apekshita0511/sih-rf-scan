"""Epsilon-greedy heuristic baseline.

* **Exploit** (prob ``1 - epsilon``): scan the channel with the highest
  *empirical* detection rate so far.
* **Explore** (prob ``epsilon``): scan a uniformly random channel.

An optimistic prior on never-scanned channels forces one full sweep of initial
coverage before the greedy rule takes over. Ties are broken toward the stalest
channel, then the lowest index.

This is a genuine adaptive strategy, but a memoryless one: no belief decay, no
prediction, no activity trend. It therefore lags on non-stationary and emerging
activity -- the gap to the Phase 6 adaptive scheduler is one of the things the
benchmark exists to measure.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from rfscan.perception.schema import ScanRecord
from rfscan.perception.store import ReadableStore


class HeuristicScheduler:
    name = "heuristic"

    def __init__(
        self,
        n_channels: int,
        epsilon: float = 0.1,
        optimistic_prior: float = 1.0,
        seed: int = 0,
    ) -> None:
        if n_channels < 1:
            raise ValueError(f"n_channels must be >= 1, got {n_channels}")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
        self._n = n_channels
        self._epsilon = float(epsilon)
        self._prior = float(optimistic_prior)
        self._seed = int(seed)
        self._rng = np.random.default_rng(self._seed)
        self._last: dict[str, float] = {}

    def select_next(self, store: ReadableStore, slot: int) -> int:
        if self._rng.random() < self._epsilon:
            choice = int(self._rng.integers(self._n))
            self._last = {
                "mode": 1.0,  # explore
                "epsilon": self._epsilon,
                "channel": float(choice),
                "score": math.nan,
            }
            return choice

        best_channel = 0
        best_key: tuple[float, float, int] | None = None
        for c in range(self._n):
            rate = store.detection_rate(c, prior=self._prior)
            since = store.slots_since_last_scan(c, slot)
            staleness = math.inf if since is None else float(since)
            key = (rate, staleness, -c)
            if best_key is None or key > best_key:
                best_key = key
                best_channel = c

        assert best_key is not None
        self._last = {
            "mode": 0.0,  # exploit
            "epsilon": self._epsilon,
            "channel": float(best_channel),
            "score": best_key[0],
        }
        return best_channel

    def update(self, record: ScanRecord) -> None:
        pass

    def explain(self) -> Mapping[str, float]:
        return dict(self._last)

    def reset(self) -> None:
        self._rng = np.random.default_rng(self._seed)
        self._last = {}
