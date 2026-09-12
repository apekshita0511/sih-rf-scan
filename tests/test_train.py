"""Data collection, label construction, temporal split, and model selection
(Phase 5)."""

from __future__ import annotations

import numpy as np
import pytest

from rfscan.models.evaluation import CalibrationReport, ClassificationMetrics
from rfscan.models.train import (
    TEST_SEEDS,
    TRAIN_SEEDS,
    VAL_SEEDS,
    BakeoffResult,
    Dataset,
    build_collection_scheduler,
    choose_final_model,
    collect_dataset,
    collect_episode,
    temporal_split,
)
from rfscan.perception.features import FeatureBuilder
from rfscan.simulator.environment import RFEnvironment
from rfscan.simulator.scenarios import make_scenario

# -- collect_episode: label correctness + temporal alignment -----------------


def test_collect_episode_row_count_and_shape():
    ds = collect_episode("normal", seed=0, policy="sequential", duration_slots=20)
    scenario = make_scenario("normal", 0, duration_slots=20)
    n_channels = scenario.n_channels
    assert len(ds) == 20 * n_channels
    assert ds.x.shape == (20 * n_channels, FeatureBuilder().n_features)
    assert set(np.unique(ds.y).tolist()) <= {0, 1}


def test_collect_episode_labels_match_independently_replayed_ground_truth():
    """The label for (slot, channel) must equal RFEnvironment.occupancy_snapshot()
    at that same slot in an independently replayed episode -- direct check of
    the S5 temporal alignment ("truth at slot t", not t-1 or t+1)."""
    ds = collect_episode("bursty", seed=3, policy="random", duration_slots=15)
    scenario = make_scenario("bursty", 3, duration_slots=15)
    env = RFEnvironment(scenario, 3)
    n = env.n_channels

    for slot in range(15):
        truth = np.array(env.occupancy_snapshot(), dtype=np.int8)
        row_mask = ds.slot == slot
        # rows for this slot are ordered by channel_index (see collect_episode)
        labels = ds.y[row_mask][np.argsort(ds.channel_index[row_mask])]
        np.testing.assert_array_equal(labels, truth)
        env.step()
    assert n > 0


def test_collect_episode_is_deterministic():
    a = collect_episode("normal", seed=1, policy="heuristic_explore", duration_slots=25)
    b = collect_episode("normal", seed=1, policy="heuristic_explore", duration_slots=25)
    np.testing.assert_array_equal(a.x, b.x)
    np.testing.assert_array_equal(a.y, b.y)


def test_unknown_policy_raises():
    with pytest.raises(ValueError):
        build_collection_scheduler("nonexistent", 4, agent_seed=0)


# -- collect_dataset: multi-episode concatenation -----------------------------


def test_collect_dataset_concatenates_all_episodes():
    ds = collect_dataset(("normal", "bursty"), (0, 1), ("sequential",), duration_slots=10)
    n_channels = make_scenario("normal", 0, duration_slots=10).n_channels
    assert len(ds) == 2 * 2 * 1 * 10 * n_channels  # scenarios x seeds x policies x slots x ch


# -- temporal split: ordering, disjointness, no future leakage --------------


def _toy_dataset(seeds: tuple[int, ...], rows_per_seed: int = 4) -> Dataset:
    n = len(seeds) * rows_per_seed
    seed_col = np.repeat(np.array(seeds, dtype=np.int32), rows_per_seed)
    return Dataset(
        x=np.zeros((n, 3), dtype=np.float32),
        y=np.zeros(n, dtype=np.int8),
        scenario=np.full(n, "normal"),
        world_seed=seed_col,
        policy=np.full(n, "sequential"),
        slot=np.tile(np.arange(rows_per_seed, dtype=np.int32), len(seeds)),
        channel_index=np.zeros(n, dtype=np.int16),
    )


def test_default_split_seed_ranges_are_disjoint_and_ordered():
    train_set, val_set, test_set = set(TRAIN_SEEDS), set(VAL_SEEDS), set(TEST_SEEDS)
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)
    assert max(TRAIN_SEEDS) < min(VAL_SEEDS)
    assert max(VAL_SEEDS) < min(TEST_SEEDS)


def test_temporal_split_assigns_every_row_to_the_right_fold():
    seeds = tuple(range(9))
    dataset = _toy_dataset(seeds)
    train, val, test = temporal_split(
        dataset, train_seeds=(0, 1, 2), val_seeds=(3, 4, 5), test_seeds=(6, 7, 8)
    )
    assert set(train.world_seed.tolist()) == {0, 1, 2}
    assert set(val.world_seed.tolist()) == {3, 4, 5}
    assert set(test.world_seed.tolist()) == {6, 7, 8}
    assert len(train) + len(val) + len(test) == len(dataset)


def test_temporal_split_folds_never_overlap():
    seeds = tuple(range(12))
    dataset = _toy_dataset(seeds)
    train, val, test = temporal_split(
        dataset, train_seeds=tuple(range(0, 5)), val_seeds=tuple(range(5, 8)),
        test_seeds=tuple(range(8, 12)),
    )
    train_idx = set(zip(train.world_seed.tolist(), train.slot.tolist(), strict=True))
    val_idx = set(zip(val.world_seed.tolist(), val.slot.tolist(), strict=True))
    test_idx = set(zip(test.world_seed.tolist(), test.slot.tolist(), strict=True))
    assert train_idx.isdisjoint(val_idx)
    assert train_idx.isdisjoint(test_idx)
    assert val_idx.isdisjoint(test_idx)


def test_temporal_split_on_real_collected_data_keeps_whole_episodes_together():
    """No (scenario, seed) pair may have rows split across two folds."""
    ds = collect_dataset(("normal",), tuple(range(6)), ("sequential",), duration_slots=8)
    train, val, test = temporal_split(
        ds, train_seeds=(0, 1), val_seeds=(2, 3), test_seeds=(4, 5)
    )
    for split in (train, val, test):
        seeds_in_split = set(split.world_seed.tolist())
        for other in (train, val, test):
            if other is split:
                continue
            assert seeds_in_split.isdisjoint(set(other.world_seed.tolist()))


# -- choose_final_model: evidence-based, not manufactured --------------------


def _fake_result(name: str, split: str, pr_auc: float, ece: float) -> BakeoffResult:
    metrics = ClassificationMetrics(
        precision=0.5, recall=0.5, f1=0.5, pr_auc=pr_auc, roc_auc=0.5, brier=0.2,
        n_positive=10, n_total=100,
    )
    calibration = CalibrationReport(
        bin_edges=(0.0, 1.0), bin_mean_predicted=(0.5,), bin_mean_actual=(0.5,),
        bin_counts=(100,), expected_calibration_error=ece,
    )
    return BakeoffResult(
        name=name, split=split, metrics=metrics, calibration=calibration,
        recall_at_p50=None, recall_at_p80=None, latency_s_per_row=1e-6, n_params=10,
    )


def test_choose_final_model_keeps_lr_when_hgb_gain_is_small():
    results = [
        _fake_result("logistic_regression", "test", pr_auc=0.70, ece=0.03),
        _fake_result("hist_gradient_boosting", "test", pr_auc=0.705, ece=0.03),
    ]
    name, reason = choose_final_model(results)
    assert name == "logistic_regression"
    assert "logistic_regression" in reason


def test_choose_final_model_picks_hgb_when_gain_is_meaningful_and_calibration_holds():
    results = [
        _fake_result("logistic_regression", "test", pr_auc=0.60, ece=0.03),
        _fake_result("hist_gradient_boosting", "test", pr_auc=0.70, ece=0.03),
    ]
    name, reason = choose_final_model(results)
    assert name == "hist_gradient_boosting"
    assert "hist_gradient_boosting" in reason


def test_choose_final_model_keeps_lr_when_hgb_calibration_is_much_worse():
    results = [
        _fake_result("logistic_regression", "test", pr_auc=0.60, ece=0.02),
        _fake_result("hist_gradient_boosting", "test", pr_auc=0.70, ece=0.20),
    ]
    name, _ = choose_final_model(results)
    assert name == "logistic_regression"
