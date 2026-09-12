"""In-memory record of everything a scanner has observed.

The store is the agent's memory. Schedulers read from it and, from Phase 5, the
feature builder does too; nothing here may touch simulator ground truth.
Schedulers are handed the store typed as :class:`ReadableStore` -- the read-only
surface -- so a strategy structurally cannot append fabricated records.

Per-channel aggregates (scan count, detection count, last-scan slot) are
maintained incrementally on :meth:`ObservationStore.append`, so the hot-path
lookups a scheduler makes each slot are O(1).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

import pandas as pd

from rfscan.perception.schema import ScanRecord

_FRAME_COLUMNS = ["slot", "channel_index", "rssi_dbm", "noise_dbm", "snr_db", "observed_detection"]


@runtime_checkable
class ReadableStore(Protocol):
    """The read-only view of an observation history that a scheduler may use."""

    @property
    def n_channels(self) -> int: ...

    def __len__(self) -> int: ...

    def scan_count(self, channel_index: int) -> int: ...

    def detection_count(self, channel_index: int) -> int: ...

    def detection_rate(self, channel_index: int, prior: float = 0.0) -> float: ...

    def last_scan_slot(self, channel_index: int) -> int | None: ...

    def slots_since_last_scan(self, channel_index: int, current_slot: int) -> int | None: ...

    def last_record(self, channel_index: int) -> ScanRecord | None: ...

    def records_for(self, channel_index: int) -> list[ScanRecord]: ...

    def coverage(self) -> int: ...


class ObservationStore:
    """Append-only history of :class:`ScanRecord` with per-channel aggregates."""

    def __init__(self, n_channels: int) -> None:
        if n_channels < 1:
            raise ValueError(f"n_channels must be >= 1, got {n_channels}")
        self._n = n_channels
        self._records: list[ScanRecord] = []
        self._by_channel: list[list[ScanRecord]] = [[] for _ in range(n_channels)]
        self._scan_count = [0] * n_channels
        self._detection_count = [0] * n_channels
        self._last_slot: list[int | None] = [None] * n_channels

    # -- identity / iteration ----------------------------------------
    @property
    def n_channels(self) -> int:
        return self._n

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[ScanRecord]:
        return iter(self._records)

    @property
    def records(self) -> tuple[ScanRecord, ...]:
        return tuple(self._records)

    @property
    def total_scans(self) -> int:
        return len(self._records)

    # -- mutation --------------------------------------------------
    def append(self, record: ScanRecord) -> None:
        c = record.channel_index
        if not 0 <= c < self._n:
            raise IndexError(f"record channel {c} out of range [0, {self._n})")
        self._records.append(record)
        self._by_channel[c].append(record)
        self._scan_count[c] += 1
        if record.detected:
            self._detection_count[c] += 1
        self._last_slot[c] = record.slot

    # -- reads ----------------------------------------------------
    def _check(self, channel_index: int) -> None:
        if not 0 <= channel_index < self._n:
            raise IndexError(f"channel {channel_index} out of range [0, {self._n})")

    def records_for(self, channel_index: int) -> list[ScanRecord]:
        self._check(channel_index)
        return list(self._by_channel[channel_index])

    def last_record(self, channel_index: int) -> ScanRecord | None:
        self._check(channel_index)
        column = self._by_channel[channel_index]
        return column[-1] if column else None

    def scan_count(self, channel_index: int) -> int:
        self._check(channel_index)
        return self._scan_count[channel_index]

    def detection_count(self, channel_index: int) -> int:
        self._check(channel_index)
        return self._detection_count[channel_index]

    def detection_rate(self, channel_index: int, prior: float = 0.0) -> float:
        """Empirical detections / scans for a channel, or ``prior`` if unscanned."""
        self._check(channel_index)
        n = self._scan_count[channel_index]
        if n == 0:
            return prior
        return self._detection_count[channel_index] / n

    def last_scan_slot(self, channel_index: int) -> int | None:
        self._check(channel_index)
        return self._last_slot[channel_index]

    def slots_since_last_scan(self, channel_index: int, current_slot: int) -> int | None:
        """Slots elapsed since this channel was last scanned, or ``None`` if never."""
        self._check(channel_index)
        last = self._last_slot[channel_index]
        return None if last is None else current_slot - last

    def coverage(self) -> int:
        """Number of distinct channels scanned at least once."""
        return sum(1 for n in self._scan_count if n > 0)

    def to_frame(self) -> pd.DataFrame:
        """Flatten the history to a DataFrame. ``decision_info`` keys, when
        present, become ``decision_<key>`` columns (NaN where absent)."""
        if not self._records:
            return pd.DataFrame(columns=_FRAME_COLUMNS)
        rows: list[dict[str, float]] = []
        for record in self._records:
            o = record.observation
            row: dict[str, float] = {
                "slot": o.slot,
                "channel_index": o.channel_index,
                "rssi_dbm": o.rssi_dbm,
                "noise_dbm": o.noise_dbm,
                "snr_db": o.snr_db,
                "observed_detection": o.observed_detection,
            }
            if record.decision_info:
                for key, value in record.decision_info.items():
                    row[f"decision_{key}"] = value
            rows.append(row)
        return pd.DataFrame(rows)
