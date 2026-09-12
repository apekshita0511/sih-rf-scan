"""Phase 9: small, reusable dashboard-rendering helpers.

Pure formatting/decision-text logic is factored into plain functions with no
Streamlit dependency -- tested directly, no script-run context needed. The
``st.*`` calls around them are thin and only assemble what those functions
already computed; no numbers are invented here that aren't already in a
metric dict or a scheduler's own ``explain()``/``all_breakdowns()`` output.
"""

from __future__ import annotations

import math

import streamlit as st


def format_metric(value: float | None, *, fmt: str = "{:.3f}", unit: str = "") -> str:
    """"Not enough data yet" for ``None``/NaN, per this phase's explicit
    instruction not to invent a number for a metric that can't yet be
    computed reliably (e.g. a partial live episode)."""
    if value is None:
        return "Not enough data yet"
    if isinstance(value, float) and math.isnan(value):
        return "Not enough data yet"
    return fmt.format(value) + unit


def explanation_text(breakdown: dict, channel_label: str) -> str:
    """Human-readable explanation of an AdaptiveScheduler's channel choice,
    built only from the real ``PriorityBreakdown``/``explain()`` values
    (docs/architecture.md S16.6/S16.7) -- no invented numbers, no ground
    truth. ``breakdown`` empty or absent (a baseline strategy) gets a plain
    fallback sentence."""
    if not breakdown:
        return f"Selected {channel_label}. This strategy has no decision breakdown to show."

    if breakdown.get("forced_freshness"):
        return (
            f"{channel_label} was force-scanned by the hard freshness guarantee -- "
            "it had gone unscanned too long, regardless of its predicted priority."
        )

    reasons = []
    pred = breakdown.get("pred")
    explore = breakdown.get("explore")
    fresh = breakdown.get("fresh")
    trend = breakdown.get("trend")
    redundancy = breakdown.get("redundancy")
    if pred is not None and pred > 0.3:
        reasons.append("high predicted activity")
    if fresh is not None and fresh > 0.15:
        reasons.append("has not been scanned recently")
    if explore is not None and explore > 0.15:
        reasons.append("carries high uncertainty, worth exploring")
    if trend is not None and trend > 0.05:
        reasons.append("shows a rising activity trend")
    if redundancy is not None and redundancy < -0.05:
        reasons.append("was recently confirmed empty (penalised, but still chosen)")

    reason_text = ", and ".join(reasons) if reasons else (
        "a balanced combination of prediction, exploration, freshness, and trend"
    )
    return f"{channel_label} was selected because it {reason_text}."


LIMITATIONS_MARKDOWN = """
### Prototype Scope & Limitations

This is a **software-only research prototype** built for SIH26055. Please read
this before drawing conclusions from the demo.

**What this is NOT:**
- No real RF reception or transmission of any kind.
- No SDR or physical RF hardware, anywhere in the pipeline.
- No jamming, spoofing, or interception of real emitters.
- No real drone / missile / aircraft detection or classification.
- No operational or classified electronic-warfare functionality.

**What the simulator does and does not model:**
- Channel occupancy is a Gilbert-Elliott Markov process with configurable
  emitter behaviours (persistent, intermittent, bursty, emerging, fading) --
  an abstraction of decision-relevant dynamics, not an RF propagation model.
- The simulator does **not** model a single emitter literally hopping between
  channels ("frequency-agile" in the strict sense) -- only per-channel
  activity patterns.
- The Hugging Face `alan-turing-institute/turing-synthetic-radar-dataset`
  (referenced in the SIH26055 brief) was investigated but **not integrated**:
  its rows are Pulse Descriptor Words for a pulse-deinterleaving task with no
  discrete time slots or fixed channel plan -- a different problem from this
  project's discrete-slot channel-occupancy scan scheduling.

**Honest, measured findings (Phase 8), not just wins:**
- Adaptive scanning improves scan efficiency and cuts redundant scanning, but
  a simple sequential sweep remains stronger on raw detection speed/rate --
  uniform coverage is hard to beat when the goal is *finding things fast*
  rather than *scanning efficiently*.
- Adaptive's efficiency advantage over the heuristic baseline **shrinks and
  reverses** in the densest scenario tested -- when almost every channel is
  active, exploration has little value and becomes a mild net cost.
- Decision latency stayed comfortably inside budget in short, controlled
  tests, but a long, continuous batch run showed a meaningful tail over
  budget -- most likely sustained system load, not an algorithmic
  regression, but this needs verification on real target hardware before
  being assumed safe.

Every number shown in this dashboard's result pages comes from a real,
reproducible experiment run (`scripts/phase8_run_experiments.py`), not from
a hand-picked or hypothetical scenario.
"""


def kpi_row(metrics: dict[str, float | None], *, formats: dict[str, str] | None = None) -> None:
    """One ``st.metric`` per entry, "Not enough data yet" for anything that
    can't yet be computed."""
    formats = formats or {}
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics.items(), strict=True):
        col.metric(label, format_metric(value, fmt=formats.get(label, "{:.3f}")))


def ground_truth_expander(occupancy: tuple, channel_labels: list[str]) -> None:
    """Collapsed by default -- ground truth is for demonstration only and
    must never be mistaken for what the scheduler can see."""
    with st.expander("Simulation Truth -- evaluation only (never shown to the scheduler)"):
        st.caption(
            "Which channels are truly active right now, for demonstration purposes "
            "only. The scheduler above never receives this -- it only ever sees noisy "
            "scan observations, exactly like a real receiver would."
        )
        cols = st.columns(len(channel_labels))
        for col, label, occ in zip(cols, channel_labels, occupancy, strict=True):
            col.metric(label, "ACTIVE" if occ else "quiet")


def limitations_section() -> None:
    st.markdown(LIMITATIONS_MARKDOWN)
