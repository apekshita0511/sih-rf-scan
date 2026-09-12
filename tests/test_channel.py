"""Channel-plan factories."""

from __future__ import annotations

import pytest

from rfscan.simulator.channel import ChannelPlan, build_channel_plan


def test_generic_plan_layout():
    chans = ChannelPlan.generic(n=5, start_hz=100e6, spacing_hz=1e6, bandwidth_hz=200e3)
    assert [c.index for c in chans] == [0, 1, 2, 3, 4]
    assert chans[0].center_freq_hz == 100e6
    assert chans[4].center_freq_hz == 104e6
    assert all(c.bandwidth_hz == 200e3 for c in chans)
    assert chans[0].label == "CH0"


def test_generic_plan_rejects_bad_args():
    with pytest.raises(ValueError):
        ChannelPlan.generic(n=0)
    with pytest.raises(ValueError):
        ChannelPlan.generic(n=3, spacing_hz=0)


def test_wifi_plan_known_frequencies():
    chans = ChannelPlan.wifi_24ghz(13)
    assert len(chans) == 13
    assert chans[0].index == 0
    assert chans[0].label == "Wi-Fi ch 1"
    assert chans[0].center_freq_mhz == pytest.approx(2412.0)
    assert chans[5].label == "Wi-Fi ch 6"
    assert chans[5].center_freq_mhz == pytest.approx(2437.0)
    assert chans[12].center_freq_mhz == pytest.approx(2472.0)
    assert all(c.bandwidth_mhz == pytest.approx(22.0) for c in chans)


def test_wifi_plan_rejects_out_of_range():
    with pytest.raises(ValueError):
        ChannelPlan.wifi_24ghz(14)
    with pytest.raises(ValueError):
        ChannelPlan.wifi_24ghz(0)


def test_build_channel_plan_dispatch():
    assert len(build_channel_plan(13, "wifi_24ghz")) == 13
    generic = build_channel_plan(8, "generic", {"start_hz": 1e9, "spacing_hz": 2e6})
    assert len(generic) == 8
    assert generic[1].center_freq_hz == 1e9 + 2e6
    with pytest.raises(ValueError):
        build_channel_plan(4, "nonsense")
