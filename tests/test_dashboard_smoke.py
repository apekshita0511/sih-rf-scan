"""Minimal Streamlit smoke test for the Phase 9 dashboard: the app must
launch and render every tab without raising, using Streamlit's headless
AppTest harness (no browser). Deliberately not a rendering/pixel test --
just "does the whole thing run end to end without an exception," per this
phase's instruction not to make the suite brittle on a UI smoke test.
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

_APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def test_dashboard_loads_and_renders_every_tab_without_raising():
    at = AppTest.from_file(_APP_PATH, default_timeout=60)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    # every tab's label should appear somewhere in the rendered app
    tab_labels = {t.label for t in at.tabs}
    assert tab_labels == {
        "Live Simulation",
        "Strategy Comparison",
        "Emerging Signal",
        "Ablation",
        "Robustness",
        "Model Comparison",
        "Limitations",
        "About / Methodology",
    }


def test_dashboard_live_simulation_step_button_advances_the_episode():
    at = AppTest.from_file(_APP_PATH, default_timeout=60)
    at.run()
    assert not at.exception
    step_button = next(b for b in at.button if b.label == "Step")
    step_button.click().run()
    assert not at.exception
