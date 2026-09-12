"""End-to-end wiring for Phase 5: environment -> scanner -> store -> feature
builder -> predictor. Mirrors tests/test_integration_phase3.py's shape, one
layer further up the stack (docs/architecture.md S1.1's closed loop, minus the
Phase 6 scheduler/priority policy)."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from rfscan.models.baseline_beta import DecayingBetaPredictor
from rfscan.models.sklearn_model import SklearnPredictor
from rfscan.perception.features import FeatureBuilder
from rfscan.perception.scanner import Scanner
from rfscan.scheduler.heuristic import HeuristicScheduler
from rfscan.simulator.scenario import example_normal


def test_feature_builder_consumes_only_the_store_and_predictor_consumes_only_features():
    """The full chain runs without ever touching env.ground_truth /
    env.truth_snapshot -- the predictor only ever sees Observation-derived
    features, matching docs/architecture.md S1.1's "ground truth flows only
    to the experiment layer" rule."""
    scenario = example_normal()
    env = scenario.build_environment(0)
    scanner = Scanner(env)
    scheduler = HeuristicScheduler(env.n_channels, epsilon=0.2, seed=0)
    fb = FeatureBuilder()
    predictor = DecayingBetaPredictor()

    for slot in range(100):
        x = fb.build(scanner.store, slot)
        assert x.shape == (env.n_channels, fb.n_features)

        proba = predictor.predict_proba(x)
        assert proba.shape == (env.n_channels,)
        assert (proba >= 0.0).all() and (proba <= 1.0).all()

        choice = scheduler.select_next(scanner.store, slot)
        record = scanner.scan(choice, scheduler.explain())
        scheduler.update(record)
        env.step()

    assert len(scanner.store) == 100
    assert scanner.store.coverage() >= 1


def test_predictions_improve_qualitatively_as_evidence_accumulates():
    """Not a claim about scheduler performance (that's Phase 6/8) -- just that
    the wired-up pipeline reacts sensibly: a channel scanned repeatedly with
    detections should end up with a higher predicted probability than a
    channel never scanned at all."""
    scenario = example_normal()
    env = scenario.build_environment(0)
    scanner = Scanner(env)
    fb = FeatureBuilder()
    predictor = DecayingBetaPredictor()

    hot_channel = 0  # persistent emitter in example_normal()
    for _slot in range(150):
        scanner.scan(hot_channel)
        env.step()

    x = fb.build(scanner.store, slot=150)
    proba = predictor.predict_proba(x)
    never_scanned = [c for c in range(env.n_channels) if c != hot_channel]
    assert proba[hot_channel] != proba[never_scanned[0]]
    assert proba[never_scanned[0]] == 0.5  # untouched channels sit at the prior


def test_full_chain_with_a_fitted_sklearn_predictor():
    scenario = example_normal()
    env = scenario.build_environment(1)
    scanner = Scanner(env)
    scheduler = HeuristicScheduler(env.n_channels, epsilon=0.3, seed=1)
    fb = FeatureBuilder()

    rows_x, rows_y = [], []
    for slot in range(200):
        x = fb.build(scanner.store, slot)
        truth = np.array(env.occupancy_snapshot(), dtype=int)
        rows_x.append(x)
        rows_y.append(truth)
        choice = scheduler.select_next(scanner.store, slot)
        record = scanner.scan(choice, scheduler.explain())
        scheduler.update(record)
        env.step()

    x_all = np.concatenate(rows_x)
    y_all = np.concatenate(rows_y)
    split = len(x_all) // 2
    predictor = SklearnPredictor(
        name="logistic_regression",
        estimator=Pipeline(
            [("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=500, random_state=0))]
        ),
        calibration="sigmoid",
    ).fit(x_all[:split], y_all[:split], x_all[split:], y_all[split:])

    x_probe = fb.build(scanner.store, slot=200)
    proba = predictor.predict_proba(x_probe)
    assert proba.shape == (env.n_channels,)
    assert (proba >= 0.0).all() and (proba <= 1.0).all()
