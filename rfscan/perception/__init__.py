"""Perception layer: what the agent is allowed to see.

* :mod:`rfscan.perception.schema`   - agent-visible data contracts (Phase 1)
* :mod:`rfscan.perception.store`    - ObservationStore + ReadableStore (Phase 3)
* :mod:`rfscan.perception.scanner`  - Scanner: env.observe -> ScanRecord -> store (Phase 3)
* :mod:`rfscan.perception.features` - FeatureBuilder: history -> feature matrix (Phase 5)

Nothing here may import simulator ground truth.
"""

from rfscan.perception.features import FEATURE_DOCS, FEATURE_NAMES, FeatureBuilder
from rfscan.perception.scanner import Scanner
from rfscan.perception.schema import (
    FORBIDDEN_OBSERVATION_FIELDS,
    GroundTruth,
    Observation,
    ScanRecord,
)
from rfscan.perception.store import ObservationStore, ReadableStore

__all__ = [
    "FORBIDDEN_OBSERVATION_FIELDS",
    "GroundTruth",
    "Observation",
    "ScanRecord",
    "ObservationStore",
    "ReadableStore",
    "Scanner",
    "FeatureBuilder",
    "FEATURE_NAMES",
    "FEATURE_DOCS",
]
