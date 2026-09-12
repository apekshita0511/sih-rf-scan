"""Phase 7 end-to-end: online feedback, emerging-signal adaptation,
non-stationarity, and the anti-leakage/determinism guarantees that must
survive the new belief/policy/adaptive machinery unchanged.
"""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest

from rfscan.experiments.adaptation import emerging_adaptation_metrics, trace_adaptive_episode
from rfscan.experiments.metrics import compute_episode_metrics
from rfscan.experiments.runner import run_episode
from rfscan.models.baseline_beta import DecayingBetaPredictor
from rfscan.perception.schema import Observation, ScanRecord
from rfscan.perception.store import ObservationStore
from rfscan.scheduler import adaptive, belief, policy
from rfscan.scheduler.adaptive import AdaptiveScheduler
from rfscan.simulator.scenarios import make_scenario


def _rec(slot: int, channel: int, detected: bool) -> ScanRecord:
    return ScanRecord(Observation(slot, channel, -80.0, -95.0, 15.0, detected))


# -- determinism -------------------------------------------------------
def test_emerging_signal_with_adaptive_is_deterministic():
    scenario = make_scenario("emerging_signal", 0, duration_slots=400)

    def _run():
        env = scenario.build_environment(0)
        sched = AdaptiveScheduler(env.n_channels, DecayingBetaPredictor(), seed=0)
        return run_episode(env, sched, budget=400, agent_seed=0)

    a = _run()
    b = _run()
    assert np.array_equal(a.scanned_channel, b.scanned_channel)
    assert np.array_equal(a.occupancy, b.occupancy)
    assert np.array_equal(a.observed_detection, b.observed_detection)


# -- no ground-truth leakage --------------------------------------------
def _imported_modules(mod) -> set[str]:
    tree = ast.parse(inspect.getsource(mod))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


@pytest.mark.parametrize("mod", [belief, policy, adaptive])
def test_phase7_scheduler_modules_never_import_the_simulator(mod):
    modules = _imported_modules(mod)
    assert not any(m.startswith("rfscan.simulator") for m in modules)


def test_adaptive_scheduler_exposes_no_ground_truth_bearing_attribute():
    sched = AdaptiveScheduler(4, DecayingBetaPredictor(), seed=0)
    store = ObservationStore(4)
    sched.select_next(store, 0)
    for forbidden in ("ground_truth", "truth_snapshot", "occupancy", "occupancy_snapshot"):
        assert not hasattr(sched, forbidden)


# -- no future information leaks into online state ----------------------
def test_adaptive_scheduler_rejects_a_store_holding_a_future_record():
    sched = AdaptiveScheduler(3, DecayingBetaPredictor(), seed=0)
    store = ObservationStore(3)
    store.append(_rec(5, 0, detected=False))  # a record already at slot 5
    with pytest.raises(ValueError):
        sched.select_next(store, 5)  # querying slot 5 itself would leak it


def test_belief_only_reflects_scans_up_to_the_current_slot():
    sched = AdaptiveScheduler(2, DecayingBetaPredictor(), seed=0)
    store = ObservationStore(2)
    sched.select_next(store, 0)
    mean_before, _ = sched.belief_snapshot()
    assert np.allclose(mean_before, 0.5)  # nothing scanned yet -> pure prior

    rec = _rec(0, 0, detected=True)
    store.append(rec)
    sched.update(rec)
    mean_after, _ = sched.belief_snapshot()
    assert mean_after[0] > mean_before[0]  # only channel 0 moved
    assert mean_after[1] == pytest.approx(mean_before[1])  # channel 1 untouched


# -- non-stationarity: belief tracks a mid-episode distribution flip -----
def test_belief_adapts_to_a_mid_episode_distribution_change():
    """``changing_distribution``: channels 0-2 are active for the first half
    of the episode, channels 9-11 for the second half (S4.4, docs/architecture
    §16.4). A belief layer that never forgot old evidence would stay pinned
    at whatever it learned early on; decay (S7) must let it track the flip."""
    scenario = make_scenario("changing_distribution", 0)  # default 800 slots
    env = scenario.build_environment(0)
    sched = AdaptiveScheduler(env.n_channels, DecayingBetaPredictor(), seed=0)
    _, trace = trace_adaptive_episode(env, sched, budget=800, tracked_channels=(0, 9))

    ch0_late_first_half = trace.belief_mean[350, 0]
    ch0_late_second_half = trace.belief_mean[750, 0]
    assert ch0_late_first_half > ch0_late_second_half

    ch9_late_first_half = trace.belief_mean[350, 1]
    ch9_late_second_half = trace.belief_mean[750, 1]
    assert ch9_late_second_half > ch9_late_first_half


# -- the flagship story: a quiet channel emerges, gets discovered, adapts --
def test_emerging_channel_priority_and_scan_share_rise_after_discovery():
    """The central Phase 7 demo, run for real (no ground truth to the
    scheduler, no cheating): docs/architecture.md's `emerging_signal`
    scenario, channel 5 silent until slot 300. Track channel 5's priority and
    post-discovery scan share using AdaptiveScheduler's own introspection."""
    scenario = make_scenario("emerging_signal", 0, duration_slots=600)
    env = scenario.build_environment(0)
    sched = AdaptiveScheduler(env.n_channels, DecayingBetaPredictor(), seed=0)
    result, trace = trace_adaptive_episode(env, sched, budget=600, tracked_channels=(5,))

    metrics = emerging_adaptation_metrics(result, compute_episode_metrics(result))
    assert metrics.channel == 5
    assert metrics.activation_slot == 300
    assert metrics.discovered, "channel 5 was never discovered in this run"

    discovery_slot = 300 + metrics.discovery_delay_slots
    priority = trace.priority[:, 0]
    belief_mean = trace.belief_mean[:, 0]

    # Belief mean for channel 5, just before activation vs. well after
    # discovery: the cleanest, most direct "hit raises belief" signal (this
    # is literally BeliefState.mean(), not a derived score).
    belief_pre_activation = belief_mean[299]
    belief_well_after_discovery = belief_mean[min(599, discovery_slot + 30)]
    assert belief_well_after_discovery > belief_pre_activation

    # Priority: right at the scan, the freshness term collapses (staleness
    # resets to ~0), so priority can *dip* for a slot or two before ramping
    # -- exactly docs/architecture.md S9's "priority ramps in 2-3 slots", not
    # a monotonic jump. So compare a window well before activation to a
    # window well after discovery, not adjacent single slots.
    pre_window = priority[max(0, discovery_slot - 40) : discovery_slot - 10]
    post_window = priority[
        min(599, discovery_slot + 20) : min(600, discovery_slot + 60)
    ]
    assert post_window.size > 0 and pre_window.size > 0
    assert float(np.mean(post_window)) > float(np.mean(pre_window))

    # Post-discovery tracking (S9): channel 5's scan share afterward should
    # exceed its scan share beforehand -- the scheduler keeps paying it more
    # attention once it knows the channel is active, not just discovering it
    # once and moving on.
    scanned = result.scanned_channel
    share_before = float(np.mean(scanned[:discovery_slot] == 5))
    share_after = float(np.mean(scanned[discovery_slot:] == 5))
    assert share_after > share_before
