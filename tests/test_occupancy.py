"""Gilbert-Elliott chain and per-behaviour emitter dynamics (Phase 2)."""

from __future__ import annotations

import numpy as np
import pytest

from rfscan.simulator.emitters import Behavior, EmitterSpec
from rfscan.simulator.occupancy import EmitterProcess, GilbertElliott


def test_stationary_on_prob():
    assert GilbertElliott(p01=0.05, p10=0.02).stationary_on_prob == pytest.approx(0.05 / 0.07)
    assert GilbertElliott(p01=0.0, p10=0.0).stationary_on_prob == 0.0


def test_transitions_are_threshold_rules():
    ge = GilbertElliott(p01=0.1, p10=0.3)
    assert ge.next_state(False, 0.05) is True
    assert ge.next_state(False, 0.50) is False
    assert ge.next_state(True, 0.50) is True
    assert ge.next_state(True, 0.10) is False


def test_gilbert_elliott_validates_probabilities():
    with pytest.raises(ValueError):
        GilbertElliott(p01=1.2, p10=0.1)
    with pytest.raises(ValueError):
        GilbertElliott(p01=0.1, p10=-0.1)


def _run(proc: EmitterProcess, n_slots: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    proc.reset(rng.random())
    trace = np.empty(n_slots, dtype=bool)
    for slot in range(n_slots):
        proc.advance(slot, rng.random())
        trace[slot] = proc.on
    return trace


def test_persistent_occupancy_matches_configured_stationary_rate():
    proc = EmitterProcess(EmitterSpec(0, Behavior.PERSISTENT, {"p01": 0.05, "p10": 0.02}))
    trace = _run(proc, 20_000)
    assert trace.mean() == pytest.approx(0.05 / 0.07, abs=0.05)


def test_persistent_is_temporally_correlated_not_iid():
    proc = EmitterProcess(EmitterSpec(0, Behavior.PERSISTENT, {"p01": 0.05, "p10": 0.02}))
    trace = _run(proc, 20_000).astype(float)
    lag1 = float(np.corrcoef(trace[:-1], trace[1:])[0, 1])
    # A 2-state Markov chain has lag-1 autocorr 1 - p01 - p10 = 0.93; IID would be ~0.
    assert lag1 > 0.6


def test_emerging_is_silent_until_activation_slot():
    proc = EmitterProcess(
        EmitterSpec(0, Behavior.EMERGING, {"p01": 0.3, "p10": 0.02}, activation_slot=200)
    )
    trace = _run(proc, 500)
    assert not trace[:200].any()
    assert trace[200:].any()


def test_intermittent_never_on_while_duty_gate_is_closed():
    proc = EmitterProcess(
        EmitterSpec(0, Behavior.INTERMITTENT, {"duty": 0.3, "period": 40.0, "phase": 0.0})
    )
    trace = _run(proc, 4_000)
    closed = np.array([(slot % 40) / 40 >= 0.3 for slot in range(4_000)])
    assert not trace[closed].any()
    assert 0.10 < trace.mean() < 0.32


def test_bursty_activity_is_sparse_and_clustered():
    proc = EmitterProcess(EmitterSpec(0, Behavior.BURSTY, {"burst_rate": 0.04, "p10": 0.4}))
    trace = _run(proc, 10_000)
    assert trace.mean() < 0.2
    rising_edges = int(np.sum(trace[1:] & ~trace[:-1]))
    mean_run = trace.sum() / max(rising_edges, 1)
    assert rising_edges > 20
    assert mean_run < 6  # short bursts, ~1/p10


def test_fading_activity_decays_over_the_episode():
    proc = EmitterProcess(
        EmitterSpec(0, Behavior.FADING, {"p01": 0.15, "p10": 0.03, "fade_rate": 0.003})
    )
    trace = _run(proc, 4_000)
    assert trace[:1_000].mean() > 0.25
    assert trace[-1_000:].mean() < 0.10


def test_parameter_override_changes_dynamics_live():
    proc = EmitterProcess(EmitterSpec(0, Behavior.PERSISTENT, {"p01": 0.0, "p10": 1.0}))
    proc.reset(0.5)
    for slot in range(1, 50):
        proc.advance(slot, 0.5)
    assert proc.on is False  # p01 = 0 -> can never switch on
    proc.apply_overrides({"p01": 1.0, "p10": 0.0})
    proc.advance(50, 0.5)
    assert proc.on is True
