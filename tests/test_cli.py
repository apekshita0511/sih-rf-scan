"""CLI wiring: --help, info, and not-yet-implemented stubs."""

from __future__ import annotations

import pytest

from rfscan.cli import main


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "Adaptive RF Scan" in capsys.readouterr().out


def test_version_exits_zero():
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0


def test_no_command_prints_help_and_returns_one(capsys):
    assert main([]) == 1
    assert "usage" in capsys.readouterr().out.lower()


def test_info_runs_with_defaults(capsys):
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    assert "channel plan" in out
    assert "Wi-Fi ch 6" in out
    assert "2437.0" in out


def test_info_runs_with_shipped_config(capsys):
    assert main(["--config", "configs/default.yaml", "info"]) == 0
    assert "13 channels" in capsys.readouterr().out


@pytest.mark.parametrize("cmd", ["demo"])
def test_not_ready_commands_return_nonzero(cmd):
    assert main([cmd]) == 3


def test_benchmark_is_wired_as_a_real_subcommand(capsys):
    # benchmark is implemented as of Phase 4 -- it must NOT be a not-ready stub.
    from rfscan.cli import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["benchmark", "--help"])
    assert exc.value.code == 0
    assert "--seeds" in capsys.readouterr().out


def test_benchmark_command_end_to_end(tmp_path, monkeypatch, capsys):
    """Smallest real run: 2 scenarios x 1 seed x 3 strategies, short episodes."""
    monkeypatch.chdir(tmp_path)
    from rfscan.experiments import benchmark as bench_mod

    real_run = bench_mod.run_benchmark
    monkeypatch.setattr(
        bench_mod,
        "run_benchmark",
        lambda cfg: real_run(
            bench_mod.BenchmarkConfig(
                scenarios=("normal", "bursty"),
                world_seeds=cfg.world_seeds,
                duration_slots=120,
                results_dir=cfg.results_dir,
            )
        ),
    )
    assert main(["benchmark", "--seeds", "1"]) == 0
    assert "raw_results.csv" in capsys.readouterr().out
    assert (tmp_path / "artifacts" / "results" / "raw_results.csv").exists()
    assert (tmp_path / "artifacts" / "results" / "summary.csv").exists()


def test_train_is_wired_as_a_real_subcommand(capsys):
    # train is implemented as of Phase 5 -- it must NOT be a not-ready stub.
    from rfscan.cli import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["train", "--help"])
    assert exc.value.code == 0
    assert "--duration-slots" in capsys.readouterr().out


def test_train_command_end_to_end(tmp_path, monkeypatch, capsys):
    """Smallest real run: 2 scenarios, tiny episodes, the real bake-off logic."""
    monkeypatch.chdir(tmp_path)
    from rfscan.models import train as train_mod

    real_run = train_mod.run_phase5_pipeline

    def _tiny_run(*, duration_slots=None, results_dir="artifacts/results", **_kwargs):
        return real_run(
            scenario_names=("normal", "bursty"),
            duration_slots=20,
            results_dir=results_dir,
            model_cards_dir="docs/model_cards",
            model_path="artifacts/model.joblib",
        )

    monkeypatch.setattr(train_mod, "run_phase5_pipeline", _tiny_run)
    assert main(["train"]) == 0
    out = capsys.readouterr().out
    assert "model_bakeoff.csv" in out
    assert "chosen model:" in out
    assert (tmp_path / "artifacts" / "results" / "model_bakeoff.csv").exists()
    assert (tmp_path / "artifacts" / "model.joblib").exists()
    assert (tmp_path / "docs" / "model_cards" / "logistic_regression.md").exists()


def test_ablate_is_wired_as_a_real_subcommand(capsys):
    # ablate is implemented as of Phase 8 -- it must NOT be a not-ready stub.
    from rfscan.cli import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["ablate", "--help"])
    assert exc.value.code == 0
    assert "--seeds" in capsys.readouterr().out


def test_ablate_command_end_to_end(tmp_path, monkeypatch, capsys):
    """Smallest real run: 1 scenario, 1 seed, tiny episodes, the real ablation logic."""
    monkeypatch.chdir(tmp_path)
    from rfscan.experiments import ablation as ablation_mod
    from rfscan.models import loader as loader_mod
    from rfscan.models.baseline_beta import DecayingBetaPredictor

    real_run = ablation_mod.run_ablation

    def _tiny_run(predictor, *, world_seeds=(0,), duration_slots=None, **_kwargs):
        return real_run(predictor, scenarios=("normal",), world_seeds=(0,), duration_slots=20)

    monkeypatch.setattr(ablation_mod, "run_ablation", _tiny_run)
    monkeypatch.setattr(loader_mod, "load_predictor", lambda cfg: DecayingBetaPredictor())

    assert main(["ablate", "--seeds", "1"]) == 0
    out = capsys.readouterr().out
    assert "ablation_raw.csv" in out
    assert (tmp_path / "artifacts" / "results" / "ablation_raw.csv").exists()
    assert (tmp_path / "artifacts" / "results" / "ablation_summary.csv").exists()


def test_dashboard_is_wired_as_a_real_subcommand():
    # dashboard is implemented as of Phase 9 -- it must NOT be a not-ready stub.
    from rfscan.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["dashboard"])
    assert args.func.__name__ == "_cmd_dashboard"


def test_dashboard_command_invokes_streamlit_via_subprocess(monkeypatch):
    import sys

    calls = []
    monkeypatch.setattr(
        "subprocess.call", lambda cmd: calls.append(cmd) or 0
    )
    assert main(["dashboard"]) == 0
    assert calls == [[sys.executable, "-m", "streamlit", "run", "app.py"]]
