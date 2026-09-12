"""Scan-scheduling strategies (Phases 3, 6, 7).

Phase 3 ships the contract and the three baselines:

* :class:`~rfscan.scheduler.base.Scheduler`        - structural protocol
* :class:`~rfscan.scheduler.sequential.SequentialScheduler` - round-robin
* :class:`~rfscan.scheduler.sequential.RandomScheduler`     - uniform-random (agent RNG)
* :class:`~rfscan.scheduler.heuristic.HeuristicScheduler`   - epsilon-greedy on empirical rate

The adaptive scheduler (belief state, multi-objective priority policy, and the
UCB / Thompson alternatives) arrives in Phases 6-8. Every strategy sees only a
read-only observation store.
"""

from rfscan.scheduler.base import Scheduler
from rfscan.scheduler.heuristic import HeuristicScheduler
from rfscan.scheduler.sequential import RandomScheduler, SequentialScheduler

__all__ = [
    "Scheduler",
    "SequentialScheduler",
    "RandomScheduler",
    "HeuristicScheduler",
]
