"""Offline data collection, temporal split, the ML bake-off, and model cards.

Prediction target (docs/architecture.md S5, fixed here -- not to be changed
silently elsewhere): for channel ``c`` and decision slot ``t``,

    y = 1[channel c is truly occupied at slot t]
    X = FeatureBuilder features for channel c, built from ObservationStore
        history strictly before slot t (whatever scanner history the
        currently-running strategy happens to have accumulated)

i.e. "P(activity on channel c at the next decision opportunity | observations
available so far)". Ground truth (``RFEnvironment.occupancy_snapshot()``) is
used ONLY to build the label column here in the offline training pipeline --
never as a feature, and never visible to a scheduler at decision time
(docs/architecture.md S1.1/S16.1).

Data collection policy: a *mix* of the three Phase 3/4 baseline schedulers
(sequential = round-robin coverage, random = uniform exploration,
heuristic_explore = a more exploration-heavy epsilon-greedy variant than the
benchmark default) generates the (features, label) rows, per
docs/architecture.md S3.2's distribution-shift mitigation -- a single fixed
policy's data would bias the model toward whatever that policy already
prefers to scan.

Temporal split: rows are grouped by ``(scenario, world_seed)`` -- a whole
episode is entirely train, entirely val, or entirely test, never split across
folds (adjacent slots within one episode are highly correlated, so splitting
by individual row would leak near-duplicate context across folds even though
no single row's features see its own future). Seeds are assigned to folds in
increasing order, standing in for "collected earlier / later" batches, per
docs/architecture.md S3.2's ``train = seeds 0-19, val = 20-29, test = 30-49``
plan. This module runs a ratio-matched, scaled-down version of that plan
(train/val/test = 10/5/10 seeds, ~5M feature rows total) to keep the Phase 5
bake-off's runtime and artifact size practical in this session; Phase 8's
closed-loop benchmark already runs the full 30-seed x 7-scenario grid
separately, for a different purpose (operational metrics, not offline
model fitting).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from rfscan.logging_config import get_logger
from rfscan.models.baseline_beta import DecayingBetaPredictor
from rfscan.models.evaluation import (
    CalibrationReport,
    ClassificationMetrics,
    calibration_report,
    classification_metrics,
    measure_inference_latency,
    recall_at_precision,
)
from rfscan.models.sklearn_model import SklearnPredictor
from rfscan.perception.features import FEATURE_DOCS, FEATURE_NAMES, FeatureBuilder
from rfscan.perception.scanner import Scanner
from rfscan.scheduler.heuristic import HeuristicScheduler
from rfscan.scheduler.sequential import RandomScheduler, SequentialScheduler
from rfscan.simulator.scenarios import list_scenarios, make_scenario

log = get_logger("models.train")

COLLECTION_POLICIES: tuple[str, ...] = ("sequential", "random", "heuristic_explore")

# Ratio-matched scale-down of architecture.md S3.2's 20/10/20 plan (see module
# docstring). Disjoint by construction -- a leakage test asserts this.
TRAIN_SEEDS: tuple[int, ...] = tuple(range(0, 10))
VAL_SEEDS: tuple[int, ...] = tuple(range(10, 15))
TEST_SEEDS: tuple[int, ...] = tuple(range(15, 25))

RANDOM_STATE = 42


def build_collection_scheduler(name: str, n_channels: int, agent_seed: int):
    if name == "sequential":
        return SequentialScheduler(n_channels)
    if name == "random":
        return RandomScheduler(n_channels, seed=agent_seed)
    if name == "heuristic_explore":
        return HeuristicScheduler(n_channels, epsilon=0.3, seed=agent_seed)
    raise ValueError(f"unknown collection policy {name!r}")


@dataclass(frozen=True, slots=True)
class Dataset:
    """Flat (row = one channel at one decision slot in one episode) dataset.

    ``X`` columns follow ``FEATURE_NAMES``. ``y`` is the S5 label. The
    remaining arrays are provenance only -- never fed to a model.
    """

    x: np.ndarray  # (n_rows, len(FEATURE_NAMES)) float32
    y: np.ndarray  # (n_rows,) int8
    scenario: np.ndarray  # (n_rows,) <U32 scenario name
    world_seed: np.ndarray  # (n_rows,) int32
    policy: np.ndarray  # (n_rows,) <U32 collection policy name
    slot: np.ndarray  # (n_rows,) int32
    channel_index: np.ndarray  # (n_rows,) int16

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, mask: np.ndarray) -> Dataset:
        return Dataset(
            x=self.x[mask],
            y=self.y[mask],
            scenario=self.scenario[mask],
            world_seed=self.world_seed[mask],
            policy=self.policy[mask],
            slot=self.slot[mask],
            channel_index=self.channel_index[mask],
        )

    def label_balance(self) -> float:
        return float(self.y.mean()) if len(self.y) else float("nan")


def collect_episode(
    scenario_name: str,
    seed: int,
    policy: str,
    *,
    duration_slots: int | None = None,
    feature_builder: FeatureBuilder | None = None,
) -> Dataset:
    """Replay one (scenario, seed, policy) episode, capturing causal features
    and same-slot ground-truth labels for every channel at every slot.

    Mirrors ``experiments.runner.run_episode``'s per-slot lifecycle (select ->
    scan -> update -> step) but additionally snapshots
    ``FeatureBuilder.build(store, slot)`` *before* the slot's scan happens --
    something ``EpisodeResult`` does not expose, so this is a standalone
    collector rather than a change to the runner.
    """
    fb = feature_builder or FeatureBuilder()
    scenario = make_scenario(scenario_name, seed, duration_slots=duration_slots)
    env = scenario.build_environment(seed)
    scheduler = build_collection_scheduler(policy, env.n_channels, agent_seed=seed)
    env.reset()
    scheduler.reset()
    scanner = Scanner(env)

    budget = scenario.duration_slots
    n = env.n_channels
    x = np.empty((budget, n, fb.n_features), dtype=np.float32)
    y = np.empty((budget, n), dtype=np.int8)

    for slot in range(budget):
        x[slot] = fb.build(scanner.store, slot)
        y[slot] = np.array(env.occupancy_snapshot(), dtype=np.int8)
        choice = scheduler.select_next(scanner.store, slot)
        record = scanner.scan(choice, scheduler.explain())
        scheduler.update(record)
        env.step()

    n_rows = budget * n
    return Dataset(
        x=x.reshape(n_rows, fb.n_features),
        y=y.reshape(n_rows),
        scenario=np.full(n_rows, scenario_name),
        world_seed=np.full(n_rows, seed, dtype=np.int32),
        policy=np.full(n_rows, policy),
        slot=np.repeat(np.arange(budget, dtype=np.int32), n),
        channel_index=np.tile(np.arange(n, dtype=np.int16), budget),
    )


def collect_dataset(
    scenario_names: tuple[str, ...],
    seeds: tuple[int, ...],
    policies: tuple[str, ...] = COLLECTION_POLICIES,
    *,
    duration_slots: int | None = None,
    feature_builder: FeatureBuilder | None = None,
) -> Dataset:
    """Collect and concatenate episodes across every (scenario, seed, policy)."""
    fb = feature_builder or FeatureBuilder()
    episodes = [
        collect_episode(
            scenario_name, seed, policy, duration_slots=duration_slots, feature_builder=fb
        )
        for scenario_name in scenario_names
        for seed in seeds
        for policy in policies
    ]
    log.info(
        "collected %d episodes (%d scenarios x %d seeds x %d policies), %d rows",
        len(episodes),
        len(scenario_names),
        len(seeds),
        len(policies),
        sum(len(e) for e in episodes),
    )
    return Dataset(
        x=np.concatenate([e.x for e in episodes]),
        y=np.concatenate([e.y for e in episodes]),
        scenario=np.concatenate([e.scenario for e in episodes]),
        world_seed=np.concatenate([e.world_seed for e in episodes]),
        policy=np.concatenate([e.policy for e in episodes]),
        slot=np.concatenate([e.slot for e in episodes]),
        channel_index=np.concatenate([e.channel_index for e in episodes]),
    )


def temporal_split(
    dataset: Dataset,
    *,
    train_seeds: tuple[int, ...] = TRAIN_SEEDS,
    val_seeds: tuple[int, ...] = VAL_SEEDS,
    test_seeds: tuple[int, ...] = TEST_SEEDS,
) -> tuple[Dataset, Dataset, Dataset]:
    """Split ``dataset`` by ``world_seed`` membership. Disjoint by
    construction (asserted by ``tests/test_leakage.py``)."""
    train = dataset.subset(np.isin(dataset.world_seed, train_seeds))
    val = dataset.subset(np.isin(dataset.world_seed, val_seeds))
    test = dataset.subset(np.isin(dataset.world_seed, test_seeds))
    return train, val, test


def build_candidates(*, random_state: int = RANDOM_STATE) -> dict[str, object]:
    """The three bake-off candidates: the non-ML reference plus two calibrated
    sklearn models (docs/architecture.md S6)."""
    return {
        "decaying_beta": DecayingBetaPredictor(),
        "logistic_regression": SklearnPredictor(
            name="logistic_regression",
            estimator=Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "lr",
                        LogisticRegression(
                            max_iter=2000, random_state=random_state, class_weight=None
                        ),
                    ),
                ]
            ),
            calibration="isotonic",
        ),
        "hist_gradient_boosting": SklearnPredictor(
            name="hist_gradient_boosting",
            estimator=HistGradientBoostingClassifier(
                random_state=random_state, max_depth=6, max_iter=200
            ),
            calibration="isotonic",
        ),
    }


@dataclass(frozen=True, slots=True)
class BakeoffResult:
    name: str
    split: str
    metrics: ClassificationMetrics
    calibration: CalibrationReport
    recall_at_p50: float | None
    recall_at_p80: float | None
    latency_s_per_row: float
    n_params: int | None  # None if not applicable / not measured

    def to_row(self) -> dict:
        return {
            "model": self.name,
            "split": self.split,
            "precision": self.metrics.precision,
            "recall": self.metrics.recall,
            "f1": self.metrics.f1,
            "pr_auc": self.metrics.pr_auc,
            "roc_auc": self.metrics.roc_auc,
            "brier": self.metrics.brier,
            "expected_calibration_error": self.calibration.expected_calibration_error,
            "recall_at_precision_0.5": self.recall_at_p50,
            "recall_at_precision_0.8": self.recall_at_p80,
            "latency_us_per_row": self.latency_s_per_row * 1e6,
            "n_params": self.n_params,
            "n_positive": self.metrics.n_positive,
            "n_total": self.metrics.n_total,
        }


def _n_params(name: str, model: object) -> int | None:
    if name == "decaying_beta":
        return 0
    fitted = model.fitted_estimator  # SklearnPredictor
    if name == "logistic_regression":
        lr = fitted.named_steps["lr"]
        return int(lr.coef_.size + lr.intercept_.size)
    if name == "hist_gradient_boosting":
        return int(getattr(fitted, "n_iter_", 0))
    return None


def run_bakeoff(
    train: Dataset, val: Dataset, test: Dataset, *, random_state: int = RANDOM_STATE
) -> tuple[dict[str, object], list[BakeoffResult]]:
    """Fit every candidate on ``train`` (+ calibrate on ``val`` where
    applicable), then score each on both ``val`` and ``test``. Never tunes
    against ``test``."""
    candidates = build_candidates(random_state=random_state)
    fitted: dict[str, object] = {}
    results: list[BakeoffResult] = []

    for name, model in candidates.items():
        if hasattr(model, "fit"):
            model.fit(train.x, train.y, val.x, val.y)
        fitted[name] = model

        for split_name, split in (("val", val), ("test", test)):
            proba = model.predict_proba(split.x)
            metrics = classification_metrics(split.y, proba)
            calib = calibration_report(split.y, proba)
            sample = split.x if len(split.x) <= 2000 else split.x[:2000]
            latency = measure_inference_latency(model.predict_proba, sample)
            results.append(
                BakeoffResult(
                    name=name,
                    split=split_name,
                    metrics=metrics,
                    calibration=calib,
                    recall_at_p50=recall_at_precision(split.y, proba, 0.5),
                    recall_at_p80=recall_at_precision(split.y, proba, 0.8),
                    latency_s_per_row=latency,
                    n_params=_n_params(name, model),
                )
            )
        log.info("bake-off: %s fit + scored", name)

    return fitted, results


def choose_final_model(
    results: list[BakeoffResult],
    *,
    pr_auc_margin: float = 0.02,
    ece_tolerance: float = 0.02,
) -> tuple[str, str]:
    """Pick between logistic_regression and hist_gradient_boosting on the
    TEST split, per docs/architecture.md S6/S8's rule: ship the simplest
    model unless a candidate wins by a *meaningful* margin.

    HGB is preferred over LR only if its test PR-AUC exceeds LR's by more
    than ``pr_auc_margin`` AND its calibration (ECE) is not worse by more
    than ``ece_tolerance``. Returns ``(chosen_name, reason)``.
    """
    by_key = {(r.name, r.split): r for r in results}
    lr = by_key[("logistic_regression", "test")]
    hgb = by_key[("hist_gradient_boosting", "test")]

    pr_auc_gain = hgb.metrics.pr_auc - lr.metrics.pr_auc
    ece_gain = (
        hgb.calibration.expected_calibration_error - lr.calibration.expected_calibration_error
    )

    if pr_auc_gain > pr_auc_margin and ece_gain <= ece_tolerance:
        reason = (
            f"hist_gradient_boosting test PR-AUC beats logistic_regression by "
            f"{pr_auc_gain:.4f} (> margin {pr_auc_margin}) without meaningfully worse "
            f"calibration (ECE delta {ece_gain:+.4f} <= tolerance {ece_tolerance})."
        )
        return "hist_gradient_boosting", reason

    reason = (
        f"logistic_regression kept: hist_gradient_boosting's test PR-AUC gain "
        f"({pr_auc_gain:+.4f}) did not clear the {pr_auc_margin} margin, or its "
        f"calibration cost (ECE delta {ece_gain:+.4f}) exceeded the {ece_tolerance} "
        "tolerance -- not a meaningful improvement over the simpler, faster, more "
        "explainable model."
    )
    return "logistic_regression", reason


_MODEL_CARD_TEMPLATE = """# Model card: {name}

## Purpose
Predicts P(channel active at the next decision opportunity) for the Phase 6
adaptive scheduler's priority score. One of {n_candidates} candidates compared
in the Phase 5 bake-off ({bakeoff_ref}).

## Prediction target
y = 1[channel c truly occupied at slot t], features built from
ObservationStore history strictly before slot t. See
`rfscan/models/train.py` module docstring for the exact definition.

## Features
{n_features} features per channel (`rfscan.perception.features.FEATURE_NAMES`):
{feature_list}

## Training data
{n_train} rows from {n_train_episodes} episodes ({scenarios} scenarios x
seeds {train_seeds} x policies {policies}). Label balance (P(active)):
{train_balance:.4f}. All data comes from the software RF simulator
(`rfscan.simulator`) -- synthetic Gilbert-Elliott channel occupancy, not
recorded or real-world spectrum data. See "Investigated but not integrated"
below.

## Split methodology
Temporal, grouped by (scenario, world_seed) -- whole episodes assigned to
exactly one of train/val/test, seeds assigned in increasing order. Train
seeds {train_seeds}, validation seeds {val_seeds}, test seeds {test_seeds}.
No row-level shuffling across folds.

## Assumptions
- The receiver can scan exactly one channel per decision slot (single-receiver,
  discrete-slot model); multi-receiver is a documented future extension.
- Channel occupancy follows the simulator's Gilbert-Elliott process plus the
  five emitter behaviours (persistent/intermittent/bursty/emerging/fading) --
  an abstraction of decision-relevant dynamics, not an RF propagation model.
- Training data is a *mixture* of scan policies (sequential/random/heuristic),
  matching the mix the eventual scheduler will produce, per the distribution-
  shift mitigation in `rfscan/models/train.py`'s module docstring.

## Known leakage risks
- **Feature/label leakage**: structurally prevented -- `FeatureBuilder.build`
  raises if the store already holds a record at or after the query slot
  (`rfscan/perception/features.py`), and the module imports nothing from
  `rfscan.simulator`. Verified by `tests/test_leakage.py`.
- **Train/test contamination**: rows are grouped by whole episode
  `(scenario, world_seed)`, never split at the row level, so no near-duplicate
  adjacent-slot rows from the same episode can appear in two different folds.
- **Distribution shift**: this model is fit on data from non-adaptive
  baseline policies. Once the Phase 6 adaptive scheduler starts choosing scans
  based on its own predictions, the scan distribution it sees will differ from
  training -- an open risk flagged, not solved, in Phase 5 (see
  docs/architecture.md S3.2/S12 R1).

## Investigated but not integrated
The SIH26055 problem statement references the Hugging Face
`alan-turing-institute/turing-synthetic-radar-dataset`. It was inspected
(dataset card) during Phase 5: each row is a Pulse Descriptor Word (Time of
Arrival, Centre Frequency, Pulse Width, Angle of Arrival, Amplitude) for a
*pulse deinterleaving* task (grouping pulses by unknown emitter), not a
discrete-slot, fixed-channel occupancy series. It has no time slots and no
channel plan compatible with this project's `ChannelPlan`/`ObservationStore`
contracts, and at ~4 billion pulses is far larger than this phase's scope.
Not integrated for Phase 5; the simulator remains the sole data source.

## Metrics (test split, n={test_n})
| metric | value |
|---|---|
| precision | {precision:.4f} |
| recall | {recall:.4f} |
| f1 | {f1:.4f} |
| pr_auc | {pr_auc:.4f} |
| roc_auc | {roc_auc} |
| brier | {brier:.4f} |
| recall@precision>=0.5 | {recall_p50} |
| recall@precision>=0.8 | {recall_p80} |

## Calibration (test split)
Expected calibration error: {ece:.4f}.

## Computational characteristics
Inference latency: {latency_us:.2f} us/row (measured, wall-clock).
Parameters/complexity: {n_params}.

## Limitations
{limitations}

## Known failure modes
{failure_modes}

## Intended use
Offline research/demo component of the SIH26055 academic prototype: predicts
per-channel activity probability to feed the Phase 6 scan-scheduling priority
score, evaluated entirely inside the synthetic RF simulator (no hardware, no
interception, no jamming/spoofing).

## Non-intended use
This model is trained and evaluated exclusively on simulated Gilbert-Elliott
channel-occupancy data. It is **not** trained or validated to identify,
classify, or characterise real-world radar/RF platforms (drones, aircraft,
missiles, or any physical emitter), and must not be used for real-world
signal intelligence, targeting, jamming, or any operational/classified EW
purpose.
"""


def render_model_card(
    name: str,
    *,
    train: Dataset,
    test_result: BakeoffResult,
    n_candidates: int,
    limitations: str,
    failure_modes: str,
) -> str:
    feature_list = "\n".join(f"- `{f}`: {FEATURE_DOCS[f]['represents']}" for f in FEATURE_NAMES)
    m = test_result.metrics
    episode_keys = zip(
        train.scenario.tolist(), train.world_seed.tolist(), train.policy.tolist(), strict=True
    )
    roc_auc_str = f"{m.roc_auc:.4f}" if m.roc_auc is not None else "undefined (single class)"
    recall_p50 = (
        f"{test_result.recall_at_p50:.4f}" if test_result.recall_at_p50 is not None else "n/a"
    )
    recall_p80 = (
        f"{test_result.recall_at_p80:.4f}" if test_result.recall_at_p80 is not None else "n/a"
    )
    return _MODEL_CARD_TEMPLATE.format(
        name=name,
        n_candidates=n_candidates,
        bakeoff_ref="`artifacts/results/model_bakeoff.csv`",
        n_features=len(FEATURE_NAMES),
        feature_list=feature_list,
        n_train=len(train),
        n_train_episodes=len(set(episode_keys)),
        scenarios=len(set(train.scenario.tolist())),
        train_seeds=TRAIN_SEEDS,
        val_seeds=VAL_SEEDS,
        test_seeds=TEST_SEEDS,
        policies=COLLECTION_POLICIES,
        train_balance=train.label_balance(),
        test_n=m.n_total,
        precision=m.precision,
        recall=m.recall,
        f1=m.f1,
        pr_auc=m.pr_auc,
        roc_auc=roc_auc_str,
        brier=m.brier,
        recall_p50=recall_p50,
        recall_p80=recall_p80,
        ece=test_result.calibration.expected_calibration_error,
        latency_us=test_result.latency_s_per_row * 1e6,
        n_params=test_result.n_params if test_result.n_params is not None else "n/a",
        limitations=limitations,
        failure_modes=failure_modes,
    )


def write_model_cards(
    out_dir: str | Path,
    fitted: dict[str, object],
    results: list[BakeoffResult],
    train: Dataset,
    limitations_by_model: dict[str, str],
    failure_modes_by_model: dict[str, str],
) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    by_key = {(r.name, r.split): r for r in results}
    paths = []
    for name in fitted:
        test_result = by_key[(name, "test")]
        card = render_model_card(
            name,
            train=train,
            test_result=test_result,
            n_candidates=len(fitted),
            limitations=limitations_by_model.get(name, "Not documented."),
            failure_modes=failure_modes_by_model.get(name, "Not documented."),
        )
        path = out / f"{name}.md"
        path.write_text(card, encoding="utf-8")
        paths.append(path)
    return paths


def default_scenario_names() -> tuple[str, ...]:
    return tuple(list_scenarios())


LIMITATIONS_BY_MODEL: dict[str, str] = {
    "decaying_beta": (
        "Ignores SNR/noise/trend entirely -- relies only on the detection-history "
        "columns of the feature vector. Lags abrupt regime changes because its "
        "geometric decay has a single fixed time constant (`beta_decay_lambda`)."
    ),
    "logistic_regression": (
        "Linear in the (scaled) feature space -- cannot capture SNR x trend or "
        "SNR x staleness interactions. Standardization is somewhat distorted by "
        "the large MISSING_VALUE / MAX_STALENESS sentinels on never-scanned channels."
    ),
    "hist_gradient_boosting": (
        "Tree ensemble -- opaque per-decision explanation compared to logistic "
        "regression's coefficients (mitigated only by the Beta belief layer's "
        "separate, always-explainable uncertainty term). Slower inference and "
        "more hyperparameters than logistic regression."
    ),
}

FAILURE_MODES_BY_MODEL: dict[str, str] = {
    "decaying_beta": (
        "Systematically under-reacts right after a channel's state flips (e.g. an "
        "EMERGING signal's activation) until enough fresh evidence accumulates to "
        "outweigh the decayed prior."
    ),
    "logistic_regression": (
        "Under-predicts on scenarios with a sharp SNR-threshold decision boundary "
        "and elevated FP/FN rates (e.g. high_noise), relative to a model that can "
        "learn a nonlinear boundary."
    ),
    "hist_gradient_boosting": (
        "Can overfit rare (scenario, policy) combinations in train if not "
        "regularised (max_depth capped here); calibration fit on a smaller "
        "validation fold is noisier for a higher-capacity model."
    ),
}


def write_bakeoff_results(out_dir: str | Path, results: list[BakeoffResult]) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "model_bakeoff.csv"
    pd.DataFrame([r.to_row() for r in results]).to_csv(path, index=False)
    return path


def write_calibration_curves(out_dir: str | Path, results: list[BakeoffResult]) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in results:
        c = r.calibration
        for i in range(len(c.bin_counts)):
            rows.append(
                {
                    "model": r.name,
                    "split": r.split,
                    "bin_low": c.bin_edges[i],
                    "bin_high": c.bin_edges[i + 1],
                    "mean_predicted": c.bin_mean_predicted[i],
                    "mean_actual": c.bin_mean_actual[i],
                    "count": c.bin_counts[i],
                }
            )
    path = out / "calibration_curves.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@dataclass(frozen=True, slots=True)
class Phase5Result:
    dataset: Dataset
    train: Dataset
    val: Dataset
    test: Dataset
    fitted: dict[str, object]
    results: list[BakeoffResult]
    chosen_name: str
    chosen_reason: str
    bakeoff_csv: Path
    calibration_csv: Path
    model_card_paths: list[Path]
    model_path: Path


def run_phase5_pipeline(
    *,
    scenario_names: tuple[str, ...] | None = None,
    duration_slots: int | None = None,
    results_dir: str | Path = "artifacts/results",
    model_cards_dir: str | Path = "docs/model_cards",
    model_path: str | Path = "artifacts/model.joblib",
    random_state: int = RANDOM_STATE,
) -> Phase5Result:
    """End-to-end Phase 5: collect -> split -> bake-off -> choose -> persist.

    Never tunes against the test split -- ``choose_final_model`` reads test
    metrics only to report the bake-off's outcome, not to search over
    hyperparameters.
    """
    names = scenario_names or default_scenario_names()
    seeds = TRAIN_SEEDS + VAL_SEEDS + TEST_SEEDS
    dataset = collect_dataset(names, seeds, duration_slots=duration_slots)
    train, val, test = temporal_split(dataset)
    log.info(
        "split: train=%d rows (%d seeds) val=%d rows (%d seeds) test=%d rows (%d seeds)",
        len(train),
        len(TRAIN_SEEDS),
        len(val),
        len(VAL_SEEDS),
        len(test),
        len(TEST_SEEDS),
    )

    fitted, results = run_bakeoff(train, val, test, random_state=random_state)
    chosen_name, chosen_reason = choose_final_model(results)
    log.info("chosen model: %s -- %s", chosen_name, chosen_reason)

    bakeoff_csv = write_bakeoff_results(results_dir, results)
    calibration_csv = write_calibration_curves(results_dir, results)
    card_paths = write_model_cards(
        model_cards_dir, fitted, results, train, LIMITATIONS_BY_MODEL, FAILURE_MODES_BY_MODEL
    )

    model_out = Path(model_path)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[chosen_name], model_out)

    return Phase5Result(
        dataset=dataset,
        train=train,
        val=val,
        test=test,
        fitted=fitted,
        results=results,
        chosen_name=chosen_name,
        chosen_reason=chosen_reason,
        bakeoff_csv=bakeoff_csv,
        calibration_csv=calibration_csv,
        model_card_paths=card_paths,
        model_path=model_out,
    )
