"""Phase 9: loads Phase 8's precomputed result CSVs for the dashboard.

`artifacts/results/*` is git-ignored (S17's existing policy) and produced by
`scripts/phase8_run_experiments.py` -- it may not exist (a fresh clone, or
before that script has been run). Every loader here returns ``None`` rather
than raising when a file is missing, so the dashboard can render a "no data
yet, run scripts/phase8_run_experiments.py" message instead of crashing.
Nothing here reruns an experiment -- these are read-only CSV loads, wrapped
in ``st.cache_data`` so a Streamlit rerun doesn't reread disk every time.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

RESULTS_DIR = Path("artifacts/results")

_FILES = {
    "main_grid": "phase8_raw_results.csv",
    "main_grid_summary": "phase8_summary.csv",
    "ablation": "phase8_ablation_raw.csv",
    "ablation_summary": "phase8_ablation.csv",
    "noise_sweep": "phase8_robustness_noise.csv",
    "nonstationarity": "phase8_robustness_nonstationary.csv",
    "model_comparison": "phase8_model_comparison.csv",
    "statistics_by_scenario": "phase8_statistics_by_scenario.csv",
    "statistics_overall": "phase8_statistics_overall.csv",
}


@st.cache_data(show_spinner=False)
def _read_csv(path_str: str) -> pd.DataFrame:
    """Cache key is the path string -- cache_data hashes arguments, and a
    Path object hashes fine too, but a plain str keeps this robust across
    Streamlit versions."""
    return pd.read_csv(path_str)


def load_result(name: str, *, results_dir: Path | str = RESULTS_DIR) -> pd.DataFrame | None:
    """Load one named Phase 8 result table (see ``_FILES`` for names).
    Returns ``None`` (never raises) if the file doesn't exist."""
    if name not in _FILES:
        raise ValueError(f"unknown result table {name!r}; available: {sorted(_FILES)}")
    path = Path(results_dir) / _FILES[name]
    if not path.exists():
        return None
    return _read_csv(str(path))


def available_results(*, results_dir: Path | str = RESULTS_DIR) -> dict[str, bool]:
    """Which named result tables actually exist on disk right now -- for a
    dashboard status panel ("Phase 8 results: 9/9 loaded")."""
    root = Path(results_dir)
    return {name: (root / fname).exists() for name, fname in _FILES.items()}
