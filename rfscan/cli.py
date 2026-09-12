"""Command-line entry point.

    rfscan info                 # show resolved config + channel plan (works now)
    rfscan benchmark            # Phase 4  - baseline strategy comparison grid -> CSV (works now)
    rfscan train                # Phase 5  - ML activity-prediction bake-off (works now)
    rfscan ablate               # Phase 8  - adaptive-scheduler component ablation
    rfscan demo                 # Phase 12 - deterministic emerging-signal story
    rfscan dashboard            # Phase 9  - launch the Streamlit dashboard

Subcommands not yet implemented print the phase they land in and exit non-zero,
so scripts fail loudly rather than silently no-op.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

from rfscan import __version__
from rfscan.config import AppConfig, load_config
from rfscan.logging_config import configure_logging, get_logger
from rfscan.simulator.channel import build_channel_plan

log = get_logger("cli")

_NOT_READY = {
    "ablate": 8,
    "demo": 12,
    "dashboard": 9,
}


def _cmd_info(args: argparse.Namespace) -> int:
    config: AppConfig = load_config(args.config)
    sim = config.simulation
    channels = build_channel_plan(sim.n_channels, sim.channel_plan, sim.channel_plan_params)

    print(f"rfscan {__version__}")
    print(f"config          : {args.config or '<built-in defaults>'}")
    print(f"channel plan    : {sim.channel_plan}  ({len(channels)} channels)")
    print(f"slot duration   : {sim.slot_duration_s:g} s")
    print(
        f"scan budget     : {sim.scan_budget_slots} slots "
        f"({sim.scan_budget_slots * sim.slot_duration_s:g} s)"
    )
    print(f"predictor       : {config.model.kind}  (calibration: {config.model.calibration})")
    print(
        "scheduler wts   : "
        f"pred={config.scheduler.w_pred:g} explore={config.scheduler.w_explore:g} "
        f"fresh={config.scheduler.w_fresh:g} trend={config.scheduler.w_trend:g} "
        f"redundancy={config.scheduler.w_redundancy:g}"
    )
    print(f"belief          : Beta({config.belief.prior_alpha:g}, {config.belief.prior_beta:g}), "
          f"decay lambda={config.belief.decay_lambda:g}")
    print()
    print(f"  {'idx':>3}  {'label':<14} {'centre (MHz)':>13} {'bandwidth (MHz)':>16}")
    for ch in channels:
        print(
            f"  {ch.index:>3}  {ch.label:<14} {ch.center_freq_mhz:>13.1f} {ch.bandwidth_mhz:>16.1f}"
        )
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    from rfscan.experiments.benchmark import (
        BenchmarkConfig,
        run_benchmark,
        summarize,
        write_benchmark,
    )
    from rfscan.simulator.scenarios import list_scenarios

    config = load_config(args.config)
    bench = BenchmarkConfig(
        scenarios=tuple(list_scenarios()),
        world_seeds=tuple(range(args.seeds)),
        results_dir=config.experiment.results_dir,
    )
    log.info(
        "benchmark: %d scenarios x %d seeds x %d strategies = %d episodes",
        len(bench.scenarios),
        len(bench.world_seeds),
        len(bench.strategies),
        bench.n_episodes,
    )
    raw = run_benchmark(bench)
    summary = summarize(raw)
    raw_path, summary_path = write_benchmark(bench, raw, summary)
    print(f"wrote {raw_path}  ({len(raw)} episode rows)")
    print(f"wrote {summary_path}")

    pivot = raw.pivot_table(
        index="scenario", columns="strategy", values="detection_rate", aggfunc="mean"
    )
    print("\nmean interval detection rate (per scenario x strategy):")
    print(pivot.to_string(float_format=lambda x: f"{x:.3f}"))
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from rfscan.models.train import run_phase5_pipeline

    config = load_config(args.config)
    result = run_phase5_pipeline(
        duration_slots=args.duration_slots,
        results_dir=config.experiment.results_dir,
    )
    print(f"wrote {result.bakeoff_csv}")
    print(f"wrote {result.calibration_csv}")
    for path in result.model_card_paths:
        print(f"wrote {path}")
    print(f"wrote {result.model_path}")
    print(f"\nchosen model: {result.chosen_name}")
    print(result.chosen_reason)

    by_key = {(r.name, r.split): r for r in result.results}
    print("\ntest-split metrics:")
    cols = ("model", "pr_auc", "roc_auc", "brier", "ece", "latency_us")
    print(f"  {cols[0]:<24} {cols[1]:>8} {cols[2]:>8} {cols[3]:>8} {cols[4]:>8} {cols[5]:>11}")
    for name in result.fitted:
        r = by_key[(name, "test")]
        roc = f"{r.metrics.roc_auc:.4f}" if r.metrics.roc_auc is not None else "n/a"
        print(
            f"  {name:<24} {r.metrics.pr_auc:>8.4f} {roc:>8} {r.metrics.brier:>8.4f} "
            f"{r.calibration.expected_calibration_error:>8.4f} {r.latency_s_per_row * 1e6:>11.2f}"
        )
    return 0


def _make_not_ready(name: str, phase: int) -> Callable[[argparse.Namespace], int]:
    def run(_args: argparse.Namespace) -> int:
        log.warning("`rfscan %s` is implemented in Phase %d - not available yet.", name, phase)
        return 3

    return run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rfscan",
        description="AI-Driven Adaptive RF Scan Intelligence Platform (SIH26055).",
    )
    parser.add_argument("--version", action="version", version=f"rfscan {__version__}")
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="YAML config file (default: built-in defaults / configs/default.yaml)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="logging verbosity (default: INFO)",
    )

    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.add_parser("info", help="show resolved config and channel plan").set_defaults(
        func=_cmd_info
    )
    bench_parser = sub.add_parser(
        "benchmark", help="Phase 4: run the baseline strategy comparison grid and write CSVs"
    )
    bench_parser.add_argument(
        "--seeds",
        type=int,
        default=5,
        metavar="N",
        help="number of world seeds, 0..N-1 (default: 5)",
    )
    bench_parser.set_defaults(func=_cmd_benchmark)

    train_parser = sub.add_parser(
        "train", help="Phase 5: run the ML activity-prediction model bake-off"
    )
    train_parser.add_argument(
        "--duration-slots",
        type=int,
        default=None,
        metavar="N",
        help="override each scenario's episode length (default: each scenario's own, 800)",
    )
    train_parser.set_defaults(func=_cmd_train)

    help_text = {
        "ablate": "Phase 8: run the adaptive-scheduler component ablation",
        "demo": "Phase 12: run the deterministic emerging-signal demo",
        "dashboard": "Phase 9: launch the Streamlit dashboard",
    }
    for name, phase in _NOT_READY.items():
        sub.add_parser(name, help=help_text[name]).set_defaults(func=_make_not_ready(name, phase))

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    func: Callable[[argparse.Namespace], int] | None = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 1
    return func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
