"""The sensor: operate the receiver against the simulated world.

``Scanner.scan(c)`` is the single action that *costs* a scan opportunity. It
reads one channel from the environment at the current slot, wraps the result as
a :class:`ScanRecord` (optionally carrying the scheduler's decision rationale),
and appends it to the :class:`ObservationStore`.

The scanner never advances simulation time -- :meth:`RFEnvironment.step` is the
experiment runner's call (Phase 4). It also never exposes ground truth: it only
ever calls :meth:`RFEnvironment.observe`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from rfscan.logging_config import get_logger
from rfscan.perception.schema import ScanRecord
from rfscan.perception.store import ObservationStore

if TYPE_CHECKING:  # avoids simulator <-> perception import cycle at module load
    from rfscan.simulator.environment import RFEnvironment

log = get_logger("perception.scanner")


class Scanner:
    """Wraps an :class:`RFEnvironment` and logs every scan to an
    :class:`ObservationStore`."""

    def __init__(self, env: RFEnvironment, store: ObservationStore | None = None) -> None:
        self._env = env
        self.store = store if store is not None else ObservationStore(env.n_channels)
        if self.store.n_channels != env.n_channels:
            raise ValueError(
                f"store has {self.store.n_channels} channels, environment has {env.n_channels}"
            )

    def scan(
        self, channel_index: int, decision_info: Mapping[str, float] | None = None
    ) -> ScanRecord:
        """Scan one channel at the environment's current slot and record it."""
        observation = self._env.observe(channel_index)
        record = ScanRecord(observation=observation, decision_info=decision_info)
        self.store.append(record)
        log.debug("scan slot=%d ch=%d det=%s", record.slot, channel_index, record.detected)
        return record

    @property
    def scan_count(self) -> int:
        return len(self.store)

    def history(self) -> tuple[ScanRecord, ...]:
        return self.store.records
