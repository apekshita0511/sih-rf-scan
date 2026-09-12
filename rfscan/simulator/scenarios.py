"""The seven reproducible benchmark scenarios and their registry.

Each factory returns a :class:`ScenarioConfig` -- the *same* declarative system
Phase 1/2 use, no parallel config. A scenario is fully determined by its name and
a seed: ``make_scenario(name, seed).build_environment(seed)`` yields a
byte-identical world every time, so Sequential / Random / Heuristic all face the
same RF realisation for a given ``(scenario, seed)``.

Common shape: 12 generic channels, 800 slots at 0.1 s (80 s episodes). Only the
RF dynamics differ between scenarios, so cross-scenario numbers stay comparable.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

from rfscan.simulator.emitters import Behavior, EmitterSpec
from rfscan.simulator.noise import NoiseModelConfig
from rfscan.simulator.scenario import NonStationaryEvent, ScenarioConfig

N_CHANNELS = 12
DURATION_SLOTS = 800
SLOT_DURATION_S = 0.1

_PLAN = "generic"
_PLAN_PARAMS = {"start_hz": 2.40e9, "spacing_hz": 5e6, "bandwidth_hz": 4e6}

_PERSISTENT = Behavior.PERSISTENT
_INTERMITTENT = Behavior.INTERMITTENT
_BURSTY = Behavior.BURSTY
_EMERGING = Behavior.EMERGING
_FADING = Behavior.FADING


def _e(channel: int, behavior: Behavior, **params: float) -> EmitterSpec:
    """Terse EmitterSpec builder: ``activation_slot`` is pulled out of kwargs."""
    activation = int(params.pop("activation_slot", 0))
    return EmitterSpec(channel, behavior, dict(params), activation_slot=activation)


def _scenario(
    name: str,
    seed: int,
    emitters: list[EmitterSpec],
    *,
    noise: NoiseModelConfig | None = None,
    events: list[NonStationaryEvent] | None = None,
) -> ScenarioConfig:
    return ScenarioConfig(
        name=name,
        seed=seed,
        n_channels=N_CHANNELS,
        duration_slots=DURATION_SLOTS,
        slot_duration_s=SLOT_DURATION_S,
        channel_plan=_PLAN,
        channel_plan_params=dict(_PLAN_PARAMS),
        emitters=emitters,
        noise=noise or NoiseModelConfig(),
        nonstationarity=events or [],
    )


def _normal(seed: int) -> ScenarioConfig:
    """Moderate activity: one persistent, one intermittent, one bursty emitter;
    the rest of the band quiet. Baseline reference point."""
    return _scenario(
        "normal",
        seed,
        [
            _e(2, _PERSISTENT, p01=0.04, p10=0.04, signal_dbm=-58.0),
            _e(5, _INTERMITTENT, duty=0.30, period=60.0, signal_dbm=-60.0),
            _e(9, _BURSTY, burst_rate=0.03, p10=0.35, signal_dbm=-62.0),
        ],
    )


def _high_activity(seed: int) -> ScenarioConfig:
    """Crowded band: four strong persistent emitters plus intermittent and
    bursty traffic -- roughly 40% of channel-slots occupied."""
    return _scenario(
        "high_activity",
        seed,
        [
            _e(0, _PERSISTENT, p01=0.14, p10=0.03, signal_dbm=-57.0),
            _e(3, _PERSISTENT, p01=0.12, p10=0.03, signal_dbm=-58.0),
            _e(6, _PERSISTENT, p01=0.16, p10=0.02, signal_dbm=-56.0),
            _e(10, _PERSISTENT, p01=0.13, p10=0.03, signal_dbm=-59.0),
            _e(1, _INTERMITTENT, duty=0.60, period=40.0, signal_dbm=-60.0),
            _e(8, _INTERMITTENT, duty=0.55, period=50.0, signal_dbm=-61.0),
            _e(11, _BURSTY, burst_rate=0.09, p10=0.25, signal_dbm=-60.0),
        ],
    )


def _bursty(seed: int) -> ScenarioConfig:
    """Six bursty emitters, no persistent traffic: short, clustered, easy-to-miss
    activations spread across the band."""
    return _scenario(
        "bursty",
        seed,
        [
            _e(c, _BURSTY, burst_rate=0.05, p10=0.45, signal_dbm=-60.0)
            for c in (1, 3, 5, 7, 9, 11)
        ],
    )


def _emerging_signal(seed: int) -> ScenarioConfig:
    """Sparse background (two emitters, ten quiet channels) plus one signal that
    switches on at slot 300 (30 s) on an otherwise-silent channel."""
    return _scenario(
        "emerging_signal",
        seed,
        [
            _e(1, _PERSISTENT, p01=0.05, p10=0.04, signal_dbm=-58.0),
            _e(8, _INTERMITTENT, duty=0.25, period=70.0, signal_dbm=-60.0),
            _e(5, _EMERGING, p01=0.30, p10=0.03, signal_dbm=-59.0, activation_slot=300),
        ],
    )


def _high_noise(seed: int) -> ScenarioConfig:
    """Same emitter structure as ``normal`` but weak signals near the detection
    threshold, a drifting floor, wide AWGN, and elevated false-alarm / missed-
    detection rates -- detection is genuinely lossy."""
    noisy = NoiseModelConfig(
        floor_dbm=-95.0,
        floor_drift_std_db=1.5,
        floor_drift_rho=0.92,
        awgn_std_db=4.0,
        detection_threshold_snr_db=8.0,
        false_alarm_rate=0.06,
        missed_detection_rate=0.15,
    )
    return _scenario(
        "high_noise",
        seed,
        [
            _e(2, _PERSISTENT, p01=0.05, p10=0.04, signal_dbm=-84.0),
            _e(5, _INTERMITTENT, duty=0.30, period=60.0, signal_dbm=-86.0),
            _e(9, _BURSTY, burst_rate=0.03, p10=0.35, signal_dbm=-88.0),
        ],
        noise=noisy,
    )


def _dynamic(seed: int) -> ScenarioConfig:
    """Continuously non-stationary: a fading emitter, an emerging one, and five
    scheduled parameter changes that reshape the activity pattern mid-episode."""
    return _scenario(
        "dynamic",
        seed,
        [
            _e(1, _PERSISTENT, p01=0.10, p10=0.04, signal_dbm=-58.0),
            _e(4, _PERSISTENT, p01=0.08, p10=0.05, signal_dbm=-59.0),
            _e(7, _BURSTY, burst_rate=0.04, p10=0.30, signal_dbm=-60.0),
            _e(10, _EMERGING, p01=0.25, p10=0.04, signal_dbm=-60.0, activation_slot=250),
            _e(2, _FADING, p01=0.30, p10=0.05, fade_rate=0.006, signal_dbm=-61.0),
        ],
        events=[
            NonStationaryEvent(200, 1, {"p01": 0.02, "p10": 0.40}),  # CH1 mostly goes quiet
            NonStationaryEvent(400, 4, {"signal_dbm": -70.0}),  # CH4 weakens
            NonStationaryEvent(400, 7, {"burst_rate": 0.10, "p10": 0.20}),  # CH7 busier
            NonStationaryEvent(550, 1, {"p01": 0.18, "p10": 0.03}),  # CH1 roars back
            NonStationaryEvent(600, 10, {"p01": 0.05, "p10": 0.30}),  # emerger fades out
        ],
    )


def _changing_distribution(seed: int) -> ScenarioConfig:
    """Activity migrates across the band: channels 0-2 active for the first half,
    then at slot 400 they fall silent and channels 9-11 switch on."""
    low = [_e(c, _PERSISTENT, p01=0.15, p10=0.03, signal_dbm=-58.0) for c in (0, 1, 2)]
    high = [_e(c, _PERSISTENT, p01=0.006, p10=0.50, signal_dbm=-58.0) for c in (9, 10, 11)]
    events = [NonStationaryEvent(400, c, {"p01": 0.006, "p10": 0.50}) for c in (0, 1, 2)]
    events += [NonStationaryEvent(400, c, {"p01": 0.15, "p10": 0.03}) for c in (9, 10, 11)]
    return _scenario("changing_distribution", seed, low + high, events=events)


SCENARIOS: dict[str, Callable[[int], ScenarioConfig]] = {
    "normal": _normal,
    "high_activity": _high_activity,
    "bursty": _bursty,
    "emerging_signal": _emerging_signal,
    "high_noise": _high_noise,
    "dynamic": _dynamic,
    "changing_distribution": _changing_distribution,
}


def list_scenarios() -> list[str]:
    """Names of the seven benchmark scenarios, in canonical order."""
    return list(SCENARIOS)


def make_scenario(
    name: str, seed: int = 0, *, duration_slots: int | None = None
) -> ScenarioConfig:
    """Build a benchmark scenario by name. ``duration_slots`` overrides the
    default 800 (used to keep tests fast)."""
    try:
        factory = SCENARIOS[name]
    except KeyError:
        raise KeyError(
            f"unknown scenario {name!r}; available: {sorted(SCENARIOS)}"
        ) from None
    scenario = factory(seed)
    if duration_slots is not None:
        scenario = dataclasses.replace(scenario, duration_slots=duration_slots)
    return scenario
