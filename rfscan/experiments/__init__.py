"""Experiment framework (Phases 4, 7, 8).

Phase 4 ships the reproducible-evaluation foundation:

* :mod:`rfscan.experiments.runner`    - ``run_episode`` + ``EpisodeResult``
* :mod:`rfscan.experiments.metrics`   - censored-aware episode-level metrics
* :mod:`rfscan.experiments.benchmark` - scenarios x seeds x strategies grid -> CSV

Phase 7 adds :mod:`rfscan.experiments.adaptation` -- emerging-signal
pre/post-activation detection rates and per-channel priority/belief tracing
for AdaptiveScheduler, both pure post-hoc analysis.

Bootstrap CIs, paired significance tests, the A-E ablation, and the robustness
sweep are Phase 8.
"""

from rfscan.experiments.adaptation import (
    EmergingAdaptationMetrics,
    PriorityTrace,
    emerging_adaptation_metrics,
    trace_adaptive_episode,
)
from rfscan.experiments.benchmark import (
    BenchmarkConfig,
    build_scheduler,
    run_benchmark,
    summarize,
    write_benchmark,
)
from rfscan.experiments.metrics import (
    ActivityInterval,
    EpisodeMetrics,
    compute_episode_metrics,
    extract_activity_intervals,
)
from rfscan.experiments.runner import EpisodeResult, run_episode

__all__ = [
    "run_episode",
    "EpisodeResult",
    "compute_episode_metrics",
    "extract_activity_intervals",
    "EpisodeMetrics",
    "ActivityInterval",
    "BenchmarkConfig",
    "run_benchmark",
    "summarize",
    "write_benchmark",
    "build_scheduler",
    "EmergingAdaptationMetrics",
    "emerging_adaptation_metrics",
    "PriorityTrace",
    "trace_adaptive_episode",
]
