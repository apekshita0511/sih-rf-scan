"""Noise floor and energy-detector configuration.

The environment (Phase 2, :mod:`rfscan.simulator.environment`) realises the
measured noise floor as a mean-reverting AR(1) drift about ``floor_dbm`` --
``floor_drift_std_db`` is its stationary standard deviation and
``floor_drift_rho`` its per-slot persistence -- plus independent per-observation
AWGN on the RSSI and noise readings. The scanner's energy detector declares
``observed_detection`` when measured SNR exceeds ``detection_threshold_snr_db``;
``false_alarm_rate`` / ``missed_detection_rate`` then flip a tunable fraction of
empty / occupied outcomes so that no scanner - baseline or adaptive - gets a
noise-free view of the world.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NoiseModelConfig:
    floor_dbm: float = -95.0
    floor_drift_std_db: float = 0.4  # stationary std of the AR(1) noise-floor drift
    floor_drift_rho: float = 0.9  # AR(1) persistence of that drift (0 -> white, ->1 -> slow)
    awgn_std_db: float = 1.5
    detection_threshold_snr_db: float = 6.0
    false_alarm_rate: float = 0.02
    missed_detection_rate: float = 0.03

    def __post_init__(self) -> None:
        for name in ("false_alarm_rate", "missed_detection_rate"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.floor_drift_std_db < 0 or self.awgn_std_db < 0:
            raise ValueError("noise std parameters must be non-negative")
        if not 0.0 <= self.floor_drift_rho < 1.0:
            raise ValueError(f"floor_drift_rho must be in [0, 1), got {self.floor_drift_rho}")
