"""Dedicated anti-leakage tests for the Phase 5 ML pipeline.

Complements the FeatureBuilder-level guard tests in ``tests/test_features.py``
(structural "no future record" check, module-import check, prefix-replay
invariance) with end-to-end proofs on the full offline collection pipeline in
``rfscan.models.train`` -- the layer that actually touches ground truth (to
build labels) and therefore the layer most likely to accidentally leak it.
"""

from __future__ import annotations

import ast
import inspect

import numpy as np

from rfscan.models.train import collect_episode
from rfscan.perception.features import FEATURE_NAMES


def test_channel_index_and_slot_are_never_feature_columns():
    """docs/architecture.md S5: raw channel identity and absolute time are
    deliberately excluded so a model can't just memorise "channel 6 is hot"
    or "slot 300 is when things happen" instead of learning from history."""
    forbidden = {"channel_index", "slot", "absolute_slot", "world_seed", "scenario"}
    assert forbidden.isdisjoint(FEATURE_NAMES)


def test_features_module_imports_no_simulator_ground_truth():
    import rfscan.perception.features as features_mod

    tree = ast.parse(inspect.getsource(features_mod))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    assert not any(m.startswith("rfscan.simulator") for m in modules)


def test_truncating_the_episode_does_not_change_earlier_rows():
    """Collect the same (scenario, seed, policy) episode twice, once cut short
    and once run to completion. Every row belonging to a slot within the
    shorter run's span must be byte-identical (features AND labels) in both
    -- proving the label/feature pipeline for slot t never depends on
    anything that happens at slot > t, all the way through Dataset
    construction, not just inside FeatureBuilder."""
    short = collect_episode("dynamic", seed=2, policy="heuristic_explore", duration_slots=20)
    long = collect_episode("dynamic", seed=2, policy="heuristic_explore", duration_slots=45)

    long_prefix_mask = long.slot < 20
    # Sort both by (slot, channel_index) so row order lines up regardless of
    # any internal ordering choice in collect_episode.
    short_order = np.lexsort((short.channel_index, short.slot))
    long_order = np.lexsort((long.channel_index[long_prefix_mask], long.slot[long_prefix_mask]))

    np.testing.assert_array_equal(
        short.x[short_order], long.x[long_prefix_mask][long_order]
    )
    np.testing.assert_array_equal(
        short.y[short_order], long.y[long_prefix_mask][long_order]
    )


def test_different_policies_after_a_shared_prefix_do_not_change_matching_prefix_features():
    """Even though the *scan choices* differ, the environment itself is
    identical for a given (scenario, seed) regardless of scheduler (Phase 2/3
    fairness invariant). What differs between policies is which channels get
    scanned and thus which features accumulate evidence -- feature *values*
    for a channel that happens to be scanned identically by both policies
    early on must still agree, since both draw from the same world."""
    seq = collect_episode("normal", seed=5, policy="sequential", duration_slots=12)
    rand = collect_episode("normal", seed=5, policy="random", duration_slots=12)

    # Slot 0: neither policy has scanned anything yet, so every channel's
    # feature row must be the identical "empty history" vector under both.
    seq_slot0 = seq.x[(seq.slot == 0)][np.argsort(seq.channel_index[seq.slot == 0])]
    rand_slot0 = rand.x[(rand.slot == 0)][np.argsort(rand.channel_index[rand.slot == 0])]
    np.testing.assert_array_equal(seq_slot0, rand_slot0)

    # Ground truth (label) at slot 0 is world-determined, independent of scheduler.
    seq_y0 = seq.y[(seq.slot == 0)][np.argsort(seq.channel_index[seq.slot == 0])]
    rand_y0 = rand.y[(rand.slot == 0)][np.argsort(rand.channel_index[rand.slot == 0])]
    np.testing.assert_array_equal(seq_y0, rand_y0)
