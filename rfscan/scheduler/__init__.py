"""Scan-scheduling strategies (Phases 3, 6, 7).

Phase 3 shipped the contract and three baselines:

* :class:`~rfscan.scheduler.base.Scheduler`        - structural protocol
* :class:`~rfscan.scheduler.sequential.SequentialScheduler` - round-robin
* :class:`~rfscan.scheduler.sequential.RandomScheduler`     - uniform-random (agent RNG)
* :class:`~rfscan.scheduler.heuristic.HeuristicScheduler`   - epsilon-greedy on empirical rate

Phase 6 adds the adaptive scheduler:

* :class:`~rfscan.scheduler.belief.BeliefState`      - decaying Beta-Bernoulli occupancy belief
* :class:`~rfscan.scheduler.policy.PriorityPolicy`   - weighted multi-objective priority score
* :class:`~rfscan.scheduler.adaptive.AdaptiveScheduler` - wires the two above + a Predictor

Online feedback / emerging-signal adaptation refinements and the UCB / Thompson
alternatives arrive in Phases 7-8. Every strategy sees only a read-only
observation store.
"""

from rfscan.scheduler.adaptive import AdaptiveScheduler
from rfscan.scheduler.base import Scheduler
from rfscan.scheduler.belief import BeliefState
from rfscan.scheduler.heuristic import HeuristicScheduler
from rfscan.scheduler.policy import PriorityBreakdown, PriorityPolicy
from rfscan.scheduler.sequential import RandomScheduler, SequentialScheduler

__all__ = [
    "Scheduler",
    "SequentialScheduler",
    "RandomScheduler",
    "HeuristicScheduler",
    "BeliefState",
    "PriorityPolicy",
    "PriorityBreakdown",
    "AdaptiveScheduler",
]
