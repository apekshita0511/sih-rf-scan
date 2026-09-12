"""Channel plan: the frequency layout the scanner operates over.

The canonical address of a channel is its integer ``index`` in ``[0, N)``. The
``label`` and centre frequency are for display and for physically-motivated
features only - no code should key behaviour off a specific frequency.

The plan is intentionally not Wi-Fi-locked: :meth:`ChannelPlan.wifi_24ghz` is a
convenient, recognisable preset for demos, while :meth:`ChannelPlan.generic`
builds an arbitrary evenly-spaced band.
"""

from __future__ import annotations

from dataclasses import dataclass

# 802.11 2.4 GHz: channel 1 centred at 2412 MHz, 5 MHz spacing, ~22 MHz occupied bandwidth.
_WIFI_24_BASE_MHZ = 2412.0
_WIFI_24_SPACING_MHZ = 5.0
_WIFI_24_BANDWIDTH_MHZ = 22.0
_WIFI_24_MAX_CHANNELS = 13


@dataclass(frozen=True, slots=True)
class ChannelConfig:
    """One virtual channel / frequency region."""

    index: int
    label: str
    center_freq_hz: float
    bandwidth_hz: float

    @property
    def center_freq_mhz(self) -> float:
        return self.center_freq_hz / 1e6

    @property
    def bandwidth_mhz(self) -> float:
        return self.bandwidth_hz / 1e6


class ChannelPlan:
    """Factories for channel layouts."""

    @staticmethod
    def generic(
        n: int,
        start_hz: float = 100e6,
        spacing_hz: float = 1e6,
        bandwidth_hz: float = 200e3,
    ) -> list[ChannelConfig]:
        """Evenly-spaced band of ``n`` channels starting at ``start_hz``."""
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if spacing_hz <= 0 or bandwidth_hz <= 0:
            raise ValueError("spacing_hz and bandwidth_hz must be positive")
        return [
            ChannelConfig(
                index=i,
                label=f"CH{i}",
                center_freq_hz=start_hz + i * spacing_hz,
                bandwidth_hz=bandwidth_hz,
            )
            for i in range(n)
        ]

    @staticmethod
    def wifi_24ghz(n: int = 13) -> list[ChannelConfig]:
        """First ``n`` channels of the 2.4 GHz Wi-Fi plan (1..13)."""
        if not 1 <= n <= _WIFI_24_MAX_CHANNELS:
            raise ValueError(
                f"Wi-Fi 2.4 GHz defines channels 1..{_WIFI_24_MAX_CHANNELS}, got n={n}"
            )
        return [
            ChannelConfig(
                index=i,
                label=f"Wi-Fi ch {i + 1}",
                center_freq_hz=(_WIFI_24_BASE_MHZ + i * _WIFI_24_SPACING_MHZ) * 1e6,
                bandwidth_hz=_WIFI_24_BANDWIDTH_MHZ * 1e6,
            )
            for i in range(n)
        ]


def build_channel_plan(
    n_channels: int,
    plan: str = "wifi_24ghz",
    params: dict[str, float] | None = None,
) -> list[ChannelConfig]:
    """Resolve a plan name + params into a concrete channel list."""
    params = params or {}
    if plan == "wifi_24ghz":
        return ChannelPlan.wifi_24ghz(n_channels)
    if plan == "generic":
        return ChannelPlan.generic(
            n=n_channels,
            start_hz=params.get("start_hz", 100e6),
            spacing_hz=params.get("spacing_hz", 1e6),
            bandwidth_hz=params.get("bandwidth_hz", 200e3),
        )
    raise ValueError(f"unknown channel plan: {plan!r} (expected 'wifi_24ghz' or 'generic')")
