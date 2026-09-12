"""rfscan/app/components.py: pure formatting/explanation logic (Phase 9).
Only the Streamlit-free functions are tested directly here -- kpi_row/
ground_truth_expander/limitations_section (thin st.* wrappers) are covered
by the AppTest smoke test instead, so this file needs no browser/script-run
context.
"""

from __future__ import annotations

import math

from rfscan.app.components import LIMITATIONS_MARKDOWN, explanation_text, format_metric


def test_format_metric_handles_none_as_not_enough_data():
    assert format_metric(None) == "Not enough data yet"


def test_format_metric_handles_nan_as_not_enough_data():
    assert format_metric(math.nan) == "Not enough data yet"


def test_format_metric_formats_a_real_value():
    assert format_metric(0.4567, fmt="{:.2f}") == "0.46"


def test_format_metric_applies_unit_suffix():
    assert format_metric(1.5, fmt="{:.1f}", unit=" ms") == "1.5 ms"


def test_explanation_text_handles_missing_breakdown():
    text = explanation_text({}, "CH7")
    assert "CH7" in text
    assert "no decision breakdown" in text


def test_explanation_text_reports_forced_freshness_first():
    breakdown = {"forced_freshness": 1.0, "pred": 0.9, "fresh": 0.9}
    text = explanation_text(breakdown, "CH3")
    assert "force-scanned" in text
    assert "hard freshness guarantee" in text


def test_explanation_text_cites_high_prediction():
    breakdown = {"pred": 0.8, "explore": 0.0, "fresh": 0.0, "trend": 0.0, "redundancy": 0.0}
    text = explanation_text(breakdown, "CH5")
    assert "CH5" in text
    assert "high predicted activity" in text


def test_explanation_text_cites_staleness():
    breakdown = {"pred": 0.0, "explore": 0.0, "fresh": 0.3, "trend": 0.0, "redundancy": 0.0}
    text = explanation_text(breakdown, "CH1")
    assert "not been scanned recently" in text


def test_explanation_text_falls_back_when_nothing_stands_out():
    breakdown = {"pred": 0.01, "explore": 0.01, "fresh": 0.01, "trend": 0.0, "redundancy": 0.0}
    text = explanation_text(breakdown, "CH0")
    assert "balanced combination" in text


def test_explanation_text_never_mentions_ground_truth():
    breakdown = {"pred": 0.9, "explore": 0.9, "fresh": 0.9, "trend": 0.9, "redundancy": -0.9}
    text = explanation_text(breakdown, "CH2")
    for forbidden in ("ground truth", "true state", "actually active", "occupied"):
        assert forbidden not in text.lower()


def test_limitations_markdown_mentions_required_disclosures():
    required = [
        "software-only",
        "SDR",
        "jamming",
        "drone",
        "turing-synthetic-radar-dataset",
        "frequency-agile",
        "reverses",
    ]
    for phrase in required:
        assert phrase.lower() in LIMITATIONS_MARKDOWN.lower()
