"""Phase 9: Streamlit dashboard main assembly.

Assembles the UI only -- all simulation/decision logic lives in the core
`rfscan` package (`rfscan.app.live_simulation` wraps the existing engine,
`rfscan.experiments.*` supplies Phase 4/7/8's real metrics/ablation/
robustness/adaptation results, `rfscan.visualization.plots` builds the
figures). Nothing here reruns Phase 8's 1,880-episode experiment suite --
results are loaded from the CSVs it already wrote
(`scripts/phase8_run_experiments.py`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from rfscan.app import components, data_loader, live_simulation
from rfscan.experiments.adaptation import emerging_adaptation_metrics, trace_adaptive_episode
from rfscan.experiments.metrics import compute_episode_metrics
from rfscan.experiments.runner import run_episode
from rfscan.models.baseline_beta import DecayingBetaPredictor
from rfscan.scheduler.adaptive import AdaptiveScheduler
from rfscan.simulator.scenarios import make_scenario
from rfscan.visualization import plots

PAGE_TITLE = "AI-Driven Adaptive RF Scan Intelligence Platform"

_TABS = (
    "Live Simulation",
    "Strategy Comparison",
    "Emerging Signal",
    "Ablation",
    "Robustness",
    "Model Comparison",
    "Limitations",
    "About / Methodology",
)


# -- cached, pure computations (safe to call on every rerun) ------------
@st.cache_data(show_spinner=False)
def _cached_replay(scenario_name: str, seed: int, strategy_name: str, budget: int):
    """A full, deterministic episode replay for the Emerging Signal /
    Strategy Comparison demo pages -- cached so moving a slider doesn't
    rerun the simulator. Always uses DecayingBetaPredictor for "adaptive"
    (zero I/O, so this works without a trained artifact on disk)."""
    from rfscan.experiments.benchmark import build_scheduler

    scenario = make_scenario(scenario_name, seed, duration_slots=budget)
    env = scenario.build_environment(seed)
    if strategy_name == "adaptive":
        scheduler = build_scheduler(
            strategy_name, env.n_channels, agent_seed=seed, predictor=DecayingBetaPredictor()
        )
    else:
        scheduler = build_scheduler(strategy_name, env.n_channels, agent_seed=seed)

    if isinstance(scheduler, AdaptiveScheduler):
        result, trace = trace_adaptive_episode(
            env, scheduler, budget, tracked_channels=tuple(range(env.n_channels))
        )
        return result, trace
    return run_episode(env, scheduler, budget=budget, agent_seed=seed), None


@st.cache_resource(show_spinner=False)
def _load_live_predictor():
    """The expensive part (unpickling a trained model, if one exists) runs
    once server-wide via cache_resource -- not once per viewer session."""
    return live_simulation.try_load_live_predictor()


def _init_state() -> None:
    if "live_predictor_loaded" not in st.session_state:
        predictor, name = _load_live_predictor()
        st.session_state.live_predictor = predictor
        st.session_state.live_predictor_name = name
        st.session_state.live_predictor_loaded = True


def render_header() -> None:
    st.title(PAGE_TITLE)
    st.caption("SIH26055 · Smart Scan Strategy for Electronic Warfare · DRDO · Software category")
    st.info(
        "**Software-only research prototype.** We simulate an RF environment and use machine "
        "learning to predict channel activity, then dynamically decide which *virtual* channel "
        "to scan next. No real RF reception/transmission, no SDR hardware, no jamming or "
        "spoofing, no real emitter interception, no operational EW functionality. "
        "See the **Limitations** tab for the full scope statement."
    )


# -- A. Live Simulation --------------------------------------------------
def render_live_simulation() -> None:
    st.header("Live Simulation")
    st.caption("OBSERVE → PREDICT → PRIORITIZE → SCAN → LEARN → ADAPT")

    c1, c2, c3, c4 = st.columns(4)
    scenario_name = c1.selectbox(
        "Scenario", live_simulation.available_scenarios(), key="live_scenario"
    )
    seed = c2.number_input(
        "World seed", min_value=0, max_value=9999, value=0, step=1, key="live_seed"
    )
    strategy_name = c3.selectbox(
        "Strategy", live_simulation.STRATEGIES, index=3, key="live_strategy"
    )
    budget = c4.number_input(
        "Episode length (slots)", min_value=10, max_value=800, value=200, step=10, key="live_budget"
    )

    sim_key = (scenario_name, int(seed), strategy_name, int(budget))
    if st.session_state.get("live_sim_key") != sim_key:
        predictor = (
            st.session_state.live_predictor
            if live_simulation.strategy_requires_predictor(strategy_name)
            else None
        )
        st.session_state.live_sim = live_simulation.LiveSimulation(
            scenario_name,
            world_seed=int(seed),
            strategy_name=strategy_name,
            budget=int(budget),
            predictor=predictor,
        )
        st.session_state.live_sim_key = sim_key

    sim: live_simulation.LiveSimulation = st.session_state.live_sim

    if strategy_name == "adaptive":
        st.caption(f"Live predictor: **{st.session_state.live_predictor_name}**")

    b1, b2, b3, b4 = st.columns(4)
    if b1.button("Step", width="stretch"):
        sim.step()
    if b2.button("Step x10", width="stretch"):
        for _ in range(10):
            if sim.step() is None:
                break
    if b3.button("Run to end", width="stretch"):
        while sim.step() is not None:
            pass
    if b4.button("Reset", width="stretch"):
        sim.reset()

    st.progress(min(1.0, sim.slot / sim.budget) if sim.budget else 0.0)
    st.caption(f"slot {sim.slot} / {sim.budget}")

    col_spectrum, col_explain = st.columns([2, 1])
    with col_spectrum:
        st.subheader("Spectrum")
        breakdowns = sim.all_breakdowns()
        priority = np.array([b.priority for b in breakdowns]) if breakdowns else None
        recently_scanned = [r.channel_index for r in sim.history[-5:]]
        detected_now = (
            [sim.history[-1].channel_index]
            if sim.history and sim.history[-1].detected
            else []
        )
        st.plotly_chart(
            plots.spectrum_bar(
                sim.channel_labels,
                priority=priority,
                selected_channel=sim.last_choice,
                recently_scanned=recently_scanned,
                detected_channels=detected_now,
            ),
            width="stretch",
        )
        if sim.history:
            scanned, detected = sim.scan_history_arrays()
            st.plotly_chart(
                plots.scan_raster(scanned, detected, sim.n_channels), width="stretch"
            )

    with col_explain:
        st.subheader("Why did we scan this channel?")
        if sim.last_choice is not None:
            label = sim.channel_labels[sim.last_choice]
            st.write(components.explanation_text(sim.last_explain, label))
            if sim.last_explain:
                st.plotly_chart(
                    plots.priority_breakdown_bar(sim.last_explain), width="stretch"
                )
        else:
            st.write("Press **Step** to begin.")

    st.subheader("Real-time metrics")
    metrics = sim.partial_metrics()
    if metrics is None:
        st.info("Not enough data yet -- press Step.")
    else:
        components.kpi_row(
            {
                "Detection rate": metrics.detection_rate,
                "Mean detection delay": metrics.mean_detection_delay_slots,
                "Scan efficiency": metrics.scan_efficiency,
                "On-target rate": metrics.on_target_scan_rate,
                "Redundant rate": metrics.redundant_scan_rate,
                "Channel coverage": metrics.channel_coverage,
            }
        )
        st.caption(f"Scans completed: {len(sim.history)}")

    components.ground_truth_expander(sim.ground_truth_snapshot(), sim.channel_labels)


# -- B. Strategy Comparison (Phase 8 precomputed results) ----------------
def render_strategy_comparison() -> None:
    st.header("Strategy Comparison")
    raw = data_loader.load_result("main_grid")
    if raw is None:
        st.warning(
            "No Phase 8 results found. Run `scripts/phase8_run_experiments.py` to generate "
            "`artifacts/results/phase8_raw_results.csv`."
        )
        return

    st.subheader("Overall (all scenarios, 30 seeds each, pooled)")
    st.warning(
        "**Honest trade-off, not a clean win:** adaptive scanning improves scan efficiency "
        "and reduces redundant scans, but uniform sequential scanning remains stronger on "
        "raw coverage / detection speed."
    )
    overall_metrics = (
        "detection_rate",
        "mean_detection_delay_slots",
        "scan_efficiency",
        "on_target_scan_rate",
        "redundant_scan_rate",
    )
    metric_cols = st.columns(5)
    for col, metric in zip(metric_cols, overall_metrics, strict=True):
        with col:
            fig = plots.strategy_comparison_bar(raw, metric, title=metric)
            st.plotly_chart(fig, width="stretch")

    st.subheader("Per-scenario breakdown")
    scenario_name = st.selectbox("Scenario", sorted(raw["scenario"].unique()), key="cmp_scenario")
    metric = st.selectbox("Metric", overall_metrics, key="cmp_metric")
    subset = raw[raw["scenario"] == scenario_name]
    fig = plots.strategy_comparison_bar(subset, metric, title=f"{metric} -- {scenario_name}")
    st.plotly_chart(fig, width="stretch")

    stats = data_loader.load_result("statistics_by_scenario")
    if stats is not None:
        st.caption("Paired significance (t-test + Wilcoxon, both must agree at alpha=0.05):")
        stat_cols = [
            "metric",
            "strategy_a",
            "strategy_b",
            "n",
            "mean_a",
            "mean_b",
            "mean_diff",
            "significant_0_05",
        ]
        st.dataframe(
            stats[stats["scenario"] == scenario_name][stat_cols],
            width="stretch",
            hide_index=True,
        )


# -- C. Emerging Signal ---------------------------------------------------
def render_emerging_signal() -> None:
    st.header("Emerging Signal")
    st.caption(
        "A channel that starts silent, then becomes active partway through the episode. "
        "The scheduler is never told which channel this is or when it activates -- it must "
        "discover it purely through scan observations."
    )

    c1, c2, c3 = st.columns(3)
    scenario_name = c1.selectbox("Scenario", ("emerging_signal", "dynamic"), key="em_scenario")
    strategy_name = c2.selectbox("Strategy", live_simulation.STRATEGIES, index=3, key="em_strategy")
    seed = c3.number_input("Seed", min_value=0, max_value=29, value=0, step=1, key="em_seed")

    budget = 600 if scenario_name == "emerging_signal" else 800
    result, trace = _cached_replay(scenario_name, int(seed), strategy_name, budget)
    metrics = compute_episode_metrics(result)
    adaptation = emerging_adaptation_metrics(result, metrics)

    if adaptation is None:
        st.warning(f"{scenario_name} has no EMERGING emitter -- unexpected, please report this.")
        return

    st.subheader("Discovery outcome (this run)")
    d1, d2, d3 = st.columns(3)
    d1.metric("Emerging channel", f"CH{adaptation.channel}")
    d2.metric("Activation slot", adaptation.activation_slot)
    d3.metric(
        "Discovered?",
        "Yes" if adaptation.discovered else "No (censored)",
        delta=f"{adaptation.discovery_delay_slots} slots" if adaptation.discovered else None,
    )

    slot_index = st.slider(
        "Replay up to slot", 0, result.n_slots - 1, value=result.n_slots - 1, key="em_slot"
    )
    scanned = result.scanned_channel[: slot_index + 1]
    detected = result.observed_detection[: slot_index + 1]
    raster_fig = plots.scan_raster(scanned, detected, result.n_channels)
    st.plotly_chart(raster_fig, width="stretch")

    if trace is not None:
        st.subheader("Priority / belief trace for the emerging channel")
        ch = adaptation.channel
        discovery_slot = (
            adaptation.activation_slot + adaptation.discovery_delay_slots
            if adaptation.discovered
            else None
        )
        shown_activation = (
            adaptation.activation_slot if adaptation.activation_slot <= slot_index else None
        )
        shown_discovery = (
            discovery_slot
            if discovery_slot is not None and discovery_slot <= slot_index
            else None
        )
        timeline_fig = plots.emerging_timeline(
            np.arange(result.n_slots)[: slot_index + 1],
            trace.priority[: slot_index + 1, ch],
            trace.belief_mean[: slot_index + 1, ch],
            activation_slot=shown_activation,
            discovery_slot=shown_discovery,
        )
        st.plotly_chart(timeline_fig, width="stretch")
    else:
        st.caption(
            "This strategy has no priority/belief state to trace -- only its scan "
            "choices are shown above."
        )

    st.subheader(f"Measured discovery delay across seeds (Phase 8, `{scenario_name}`)")
    main_grid = data_loader.load_result("main_grid")
    if main_grid is None:
        st.warning(
            "No Phase 8 results found -- run `scripts/phase8_run_experiments.py` to see "
            "discovery statistics across all 30 seeds."
        )
    else:
        scenario_rows = main_grid[main_grid["scenario"] == scenario_name]
        n_seeds = scenario_rows["world_seed"].nunique()
        summary = scenario_rows.groupby("strategy").agg(
            discovered=("emerging_discovery_delay_slots", lambda s: int(s.notna().sum())),
            mean_delay=("emerging_discovery_delay_slots", "mean"),
        )
        summary = summary.reindex(live_simulation.STRATEGIES)
        summary["discovered"] = summary["discovered"].map(lambda n: f"{int(n)}/{n_seeds}")
        summary["mean_delay"] = summary["mean_delay"].map(
            lambda v: f"{v:.1f}" if pd.notna(v) else "never discovered"
        )
        summary.columns = ["discovered (n/seeds)", "mean delay (slots)"]
        st.table(summary)
        st.warning(
            "Adaptive does not beat a blind sweep on raw discovery speed, but it avoids the "
            "heuristic's tunnel-vision failure mode -- see the discovery counts above, "
            "computed from the actual Phase 8 run."
        )


# -- D. Ablation -----------------------------------------------------------
def render_ablation() -> None:
    st.header("Ablation: which components of the priority score matter?")
    raw = data_loader.load_result("ablation")
    if raw is None:
        st.warning(
            "No ablation results found. Run `rfscan ablate` or `scripts/phase8_run_experiments.py`."
        )
        return

    variant_labels = {
        "A_prediction_only": "A: prediction only",
        "B_prediction_exploration": "B: + exploration",
        "C_prediction_exploration_freshness": "C: + freshness",
        "D_prediction_exploration_freshness_trend": "D: + trend",
        "E_full_policy": "E: full policy",
        "F_full_policy_no_online_feedback": "F: full policy, no online feedback",
    }
    display = raw.copy()
    display["variant"] = display["variant"].map(variant_labels).fillna(display["variant"])

    metric = st.selectbox(
        "Metric",
        ("detection_rate", "scan_efficiency", "on_target_scan_rate", "redundant_scan_rate"),
        key="ablation_metric",
    )
    fig = plots.strategy_comparison_bar(display, metric, group_col="variant", title=metric)
    st.plotly_chart(fig, width="stretch")
    st.warning(
        "The freshness term (added at variant C) contributes substantially to redundant-scan "
        "reduction. Not every component improves every metric independently -- E vs F (online "
        "feedback on/off) shows no large *aggregate* difference at this scale, even though "
        "online feedback is provably decisive at the specific moment of a hit (see the "
        "Emerging Signal tab)."
    )


# -- E. Robustness ---------------------------------------------------------
def render_robustness() -> None:
    st.header("Robustness")

    st.subheader("Noise sweep (emitters held constant, only noise varied)")
    noise_raw = data_loader.load_result("noise_sweep")
    if noise_raw is None:
        st.warning("No noise-sweep results found.")
    else:
        metric = st.selectbox("Metric", ("detection_rate", "scan_efficiency"), key="noise_metric")
        st.plotly_chart(
            plots.line_by_level(
                noise_raw, metric, level_col="noise_tier", level_order=("low", "elevated", "high")
            ),
            width="stretch",
        )

    st.subheader("Non-stationarity: mid-episode activity flip (`changing_distribution`)")
    ns_raw = data_loader.load_result("nonstationarity")
    if ns_raw is None:
        st.warning("No non-stationarity results found.")
    else:
        summary = ns_raw.groupby("strategy")[
            ["quiet_then_hot_rate_before", "quiet_then_hot_rate_after"]
        ].mean()
        st.bar_chart(summary)
        st.caption(
            "Detection rate on the newly-active channel group, before vs. after the flip at "
            "slot 400. Adaptive's decaying belief lets it retarget after the change, reaching "
            "the highest post-flip rate of all four strategies."
        )


# -- F. Model Comparison ----------------------------------------------------
def render_model_comparison() -> None:
    st.header("Predictor comparison for live scheduling")
    raw = data_loader.load_result("model_comparison")
    if raw is None:
        st.warning("No model-comparison results found.")
        return

    summary = raw.groupby("predictor")[
        ["detection_rate", "scan_efficiency", "mean_decision_latency_ms"]
    ].mean()
    st.dataframe(summary.style.format("{:.3f}"), width="stretch")
    st.warning(
        "HistGradientBoosting gives the best downstream scan efficiency but the least latency "
        "headroom (~3.8 ms vs. a 5 ms/slot budget). **Logistic Regression is the current "
        "live-serving default** -- this dashboard does not switch it automatically."
    )
    latency_fig = plots.strategy_comparison_bar(
        raw, "mean_decision_latency_ms", group_col="predictor", title="Decision latency (ms)"
    )
    st.plotly_chart(latency_fig, width="stretch")


# -- G. Limitations ----------------------------------------------------------
def render_limitations() -> None:
    st.header("Prototype Scope & Limitations")
    components.limitations_section()


# -- H. About / Methodology --------------------------------------------------
def render_about() -> None:
    st.header("About / Methodology")
    st.markdown(
        """
This project reframes *"which channel do I scan next?"* as a **belief-space
restless multi-armed bandit** and solves it with a hybrid learned-occupancy +
Bayesian-uncertainty scheduler.

**The closed loop:**

```
Simulation  ->  Observation  ->  Feature Engineering  ->  ML Prediction
    ->  Priority Calculation  ->  Next Channel  ->  Scan  ->  Feedback  ->  Adaptation
```

| Stage | Role |
|---|---|
| Simulation | A synthetic RF world generates ground truth, never shown to the scheduler. |
| Observation | The Scanner reads one channel/slot, returning a noisy measurement (maybe wrong). |
| Feature Engineering | 17 features rebuilt fresh from scan history every slot (no future info). |
| ML Prediction | A calibrated model estimates P(active) per channel from those features. |
| Priority Calculation | Weighs prediction, exploration, freshness, trend, redundancy together. |
| Next Channel | The highest-priority channel is chosen (or force-scanned if long overdue). |
| Scan | The chosen channel is observed; the outcome (hit or miss) is recorded. |
| Feedback | A decaying Bayesian belief updates from that outcome. |
| Adaptation | The updated belief and fresh features reshape every later decision. |

See `docs/architecture.md` for the full design, every phase's realisation
notes, and every number in this dashboard's result pages.
"""
    )


def render() -> None:
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    _init_state()
    render_header()
    tabs = st.tabs(list(_TABS))
    with tabs[0]:
        render_live_simulation()
    with tabs[1]:
        render_strategy_comparison()
    with tabs[2]:
        render_emerging_signal()
    with tabs[3]:
        render_ablation()
    with tabs[4]:
        render_robustness()
    with tabs[5]:
        render_model_comparison()
    with tabs[6]:
        render_limitations()
    with tabs[7]:
        render_about()


def main() -> None:
    render()
