"""Scenario spec validation and the worked example."""

from __future__ import annotations

import pytest

from rfscan.simulator.emitters import Behavior, EmitterSpec
from rfscan.simulator.scenario import ScenarioConfig, example_normal


def test_example_normal_builds():
    sc = example_normal()
    assert sc.n_channels == 6
    assert sc.duration_s == pytest.approx(60.0)
    assert len(sc.emitters) == 3
    assert sc.emitters[0].behavior is Behavior.PERSISTENT


def test_scenario_rejects_emitter_on_missing_channel():
    with pytest.raises(ValueError):
        ScenarioConfig(
            name="bad",
            seed=0,
            n_channels=4,
            duration_slots=100,
            emitters=[EmitterSpec(9, Behavior.PERSISTENT)],
        )


def test_scenario_rejects_bad_duration():
    with pytest.raises(ValueError):
        ScenarioConfig(name="bad", seed=0, n_channels=4, duration_slots=0)


def test_emitter_activation_window():
    em = EmitterSpec(0, Behavior.EMERGING, activation_slot=200, deactivation_slot=400)
    assert not em.is_active_at(199)
    assert em.is_active_at(200)
    assert em.is_active_at(399)
    assert not em.is_active_at(400)
