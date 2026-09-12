"""rfscan/app/data_loader.py: Phase 8 CSV loading for the dashboard (Phase 9)."""

from __future__ import annotations

import pandas as pd
import pytest

from rfscan.app.data_loader import available_results, load_result


def test_load_result_returns_none_when_file_missing(tmp_path):
    assert load_result("main_grid", results_dir=tmp_path) is None


def test_load_result_loads_an_existing_csv(tmp_path):
    df = pd.DataFrame({"scenario": ["normal"], "strategy": ["adaptive"], "detection_rate": [0.5]})
    df.to_csv(tmp_path / "phase8_raw_results.csv", index=False)
    loaded = load_result("main_grid", results_dir=tmp_path)
    assert loaded is not None
    assert list(loaded["scenario"]) == ["normal"]


def test_load_result_rejects_unknown_name(tmp_path):
    with pytest.raises(ValueError):
        load_result("not_a_real_table", results_dir=tmp_path)


def test_available_results_reports_missing_and_present_files(tmp_path):
    (tmp_path / "phase8_ablation_raw.csv").write_text("scenario,variant\nnormal,A\n")
    status = available_results(results_dir=tmp_path)
    assert status["ablation"] is True
    assert status["main_grid"] is False
    assert set(status.keys()) == {
        "main_grid",
        "main_grid_summary",
        "ablation",
        "ablation_summary",
        "noise_sweep",
        "nonstationarity",
        "model_comparison",
        "statistics_by_scenario",
        "statistics_overall",
    }


def test_load_result_empty_directory_all_none(tmp_path):
    for name in ("main_grid", "ablation", "noise_sweep", "nonstationarity", "model_comparison"):
        assert load_result(name, results_dir=tmp_path) is None
