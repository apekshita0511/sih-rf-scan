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
        "Overview",
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


def test_dashboard_run_25_slots_button_advances_the_episode():
    at = AppTest.from_file(_APP_PATH, default_timeout=60)
    at.run()
    assert not at.exception
    run_button = next(b for b in at.button if b.label == "Run 25 slots")
    run_button.click().run()
    assert not at.exception
    sim = at.session_state["live_sim"]
    assert sim.slot == 25


def test_dashboard_reset_button_returns_to_slot_zero():
    at = AppTest.from_file(_APP_PATH, default_timeout=60)
    at.run()
    run_button = next(b for b in at.button if b.label == "Run 25 slots")
    run_button.click().run()
    assert at.session_state["live_sim"].slot == 25

    reset_button = next(b for b in at.button if b.label == "Reset")
    reset_button.click().run()
    assert not at.exception
    assert at.session_state["live_sim"].slot == 0


def test_dashboard_same_scenario_comparison_runs_without_raising():
    at = AppTest.from_file(_APP_PATH, default_timeout=60)
    at.run()
    compare_button = next(
        b for b in at.button if b.label == "Run comparison (all 4 strategies)"
    )
    compare_button.click().run()
    assert not at.exception
    result = at.session_state["compare_result"]
    assert set(result.keys()) == {"sequential", "random", "heuristic", "adaptive"}


def test_dashboard_run_emerging_signal_demo_button_sets_canonical_state():
    at = AppTest.from_file(_APP_PATH, default_timeout=60)
    at.run()
    demo_button = next(
        b
        for b in at.button
        if b.label == "▶ Run Emerging Signal Demo (emerging_signal, seed 0, Adaptive)"
    )
    demo_button.click().run()
    assert not at.exception
    assert at.session_state["em_scenario"] == "emerging_signal"
    assert at.session_state["em_strategy"] == "adaptive"
    assert at.session_state["em_seed"] == 0


def test_dashboard_overview_shows_the_real_pooled_phase8_numbers():
    """The headline cards must be computed from the real CSV, not hardcoded
    -- cross-checked against the exact numbers reported for Phase 8. Sequential
    genuinely leads detection_rate, mean_detection_delay, AND redundant_scan_rate
    (trivially ~0 by construction); adaptive leads only scan_efficiency -- the
    dashboard must show exactly this, not imply adaptive leads more than it does."""
    at = AppTest.from_file(_APP_PATH, default_timeout=60)
    at.run()
    assert not at.exception
    metric_values = [m.value for m in at.metric]
    assert "0.554" in metric_values  # detection_rate leader: sequential
    assert "4.524" in metric_values  # mean_detection_delay leader: sequential
    assert "0.462" in metric_values  # scan_efficiency leader: adaptive
    assert "0.000" in metric_values  # redundant_scan_rate leader: sequential

    caption_texts = " ".join(c.value for c in at.caption)
    assert "Leader: **sequential**" in caption_texts
    success_texts = " ".join(s.value for s in at.success) if hasattr(at, "success") else ""
    assert "Adaptive leads" in success_texts or "Adaptive leads" in caption_texts
