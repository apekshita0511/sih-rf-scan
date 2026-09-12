"""Scanner: env.observe -> ScanRecord -> ObservationStore (Phase 3)."""

from __future__ import annotations

import pytest

from rfscan.perception.scanner import Scanner
from rfscan.perception.schema import ScanRecord
from rfscan.perception.store import ObservationStore
from rfscan.simulator.scenario import example_normal


def test_scan_returns_record_matching_env_observation():
    env = example_normal().build_environment(0)
    scanner = Scanner(env)
    rec = scanner.scan(2)
    assert isinstance(rec, ScanRecord)
    assert rec.channel_index == 2
    assert rec.slot == env.slot == 0
    # same slot + channel -> environment.observe is deterministic
    assert rec.observation == env.observe(2)


def test_scan_appends_to_store_and_counts():
    env = example_normal().build_environment(0)
    scanner = Scanner(env)
    scanner.scan(0)
    scanner.scan(1)
    scanner.scan(0)
    assert scanner.scan_count == 3
    assert len(scanner.store) == 3
    assert scanner.store.scan_count(0) == 2


def test_decision_info_is_passed_through():
    env = example_normal().build_environment(0)
    scanner = Scanner(env)
    rec = scanner.scan(0, decision_info={"priority": 0.42})
    assert rec.decision_info == {"priority": 0.42}
    assert scanner.store.last_record(0).decision_info == {"priority": 0.42}


def test_multiple_scans_in_one_slot_are_all_recorded():
    env = example_normal().build_environment(0)
    scanner = Scanner(env)
    for c in range(env.n_channels):
        scanner.scan(c)
    assert scanner.scan_count == env.n_channels
    assert all(r.slot == 0 for r in scanner.history())


def test_scan_reflects_simulation_time_after_step():
    env = example_normal().build_environment(0)
    scanner = Scanner(env)
    scanner.scan(0)
    env.step()
    rec = scanner.scan(0)
    assert rec.slot == 1
    slots = [r.slot for r in scanner.history()]
    assert slots == [0, 1]


def test_scanner_creates_its_own_store_when_none_given():
    env = example_normal().build_environment(0)
    scanner = Scanner(env)
    assert isinstance(scanner.store, ObservationStore)
    assert scanner.store.n_channels == env.n_channels


def test_scanner_rejects_store_with_wrong_channel_count():
    env = example_normal().build_environment(0)
    with pytest.raises(ValueError):
        Scanner(env, ObservationStore(env.n_channels + 1))
