"""Config defaults, YAML round-trip, and the shipped default.yaml."""

from __future__ import annotations

from pathlib import Path

import dacite
import pytest

from rfscan.config import AppConfig, dump_config, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_YAML = REPO_ROOT / "configs" / "default.yaml"


def test_defaults_are_sane():
    cfg = AppConfig()
    assert cfg.simulation.n_channels == 13
    assert cfg.simulation.channel_plan == "wifi_24ghz"
    assert cfg.belief.decay_lambda == pytest.approx(0.98)
    assert cfg.scheduler.softmax_temperature is None
    assert cfg.experiment.seeds == list(range(30))
    assert cfg.experiment.strategies == ["sequential", "random", "heuristic", "adaptive"]


def test_none_path_returns_defaults():
    assert load_config(None) == AppConfig()


def test_yaml_round_trip(tmp_path):
    original = AppConfig()
    path = tmp_path / "cfg.yaml"
    dump_config(original, path)
    assert load_config(path) == original


def test_shipped_default_yaml_matches_appconfig():
    assert DEFAULT_YAML.exists(), "configs/default.yaml is missing"
    assert load_config(DEFAULT_YAML) == AppConfig()


def test_strict_mode_rejects_unknown_keys(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("random_seed: 7\nnot_a_real_key: 3\n", encoding="utf-8")
    with pytest.raises(dacite.exceptions.DaciteError):
        load_config(path)


def test_partial_override(tmp_path):
    path = tmp_path / "partial.yaml"
    path.write_text("random_seed: 123\nsimulation:\n  n_channels: 8\n", encoding="utf-8")
    cfg = load_config(path)
    assert cfg.random_seed == 123
    assert cfg.simulation.n_channels == 8
    assert cfg.simulation.channel_plan == "wifi_24ghz"  # untouched default
