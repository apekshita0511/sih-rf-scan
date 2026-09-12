"""Synthetic RF world.

Configuration data contracts:

    * :mod:`rfscan.simulator.channel`   - channel plan (frequency layout)
    * :mod:`rfscan.simulator.emitters`  - emitter behaviour specs
    * :mod:`rfscan.simulator.noise`     - noise / detector configuration
    * :mod:`rfscan.simulator.scenario`  - reproducible scenario spec

Stochastic engine (Phase 2):

    * :mod:`rfscan.simulator.rng`         - position-independent seeded randomness
    * :mod:`rfscan.simulator.occupancy`   - Gilbert-Elliott occupancy + behaviour overlays
    * :mod:`rfscan.simulator.environment` - RFEnvironment: step / observe / ground_truth

The seven tuned benchmark scenarios and their registry arrive in Phase 4.
"""

from rfscan.simulator.channel import ChannelConfig, ChannelPlan, build_channel_plan
from rfscan.simulator.emitters import Behavior, EmitterSpec
from rfscan.simulator.environment import RFEnvironment, build_environment
from rfscan.simulator.noise import NoiseModelConfig
from rfscan.simulator.occupancy import EmitterProcess, GilbertElliott
from rfscan.simulator.rng import draw_rng
from rfscan.simulator.scenario import (
    NonStationaryEvent,
    ScenarioConfig,
    example_emerging,
    example_normal,
)
from rfscan.simulator.scenarios import SCENARIOS, list_scenarios, make_scenario

__all__ = [
    "ChannelConfig",
    "ChannelPlan",
    "build_channel_plan",
    "Behavior",
    "EmitterSpec",
    "NoiseModelConfig",
    "NonStationaryEvent",
    "ScenarioConfig",
    "example_normal",
    "example_emerging",
    "RFEnvironment",
    "build_environment",
    "EmitterProcess",
    "GilbertElliott",
    "draw_rng",
    "SCENARIOS",
    "list_scenarios",
    "make_scenario",
]
