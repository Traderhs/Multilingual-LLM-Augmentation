"""Run the frozen Binary Adaptive policy on the untouched test splits.

This script has two explicit stages:

1. ``--prepare-only`` freezes every training-only OOF threshold required by the
   test evaluation and writes ``PRETEST_THRESHOLD_LOCK.json``. It does not read
   test labels or test embeddings.
2. ``--evaluate`` verifies the frozen pre-test lock, marks test access, and
   performs the one-shot test evaluation without changing any ON/OFF decision,
   ratio, synthetic weight, or threshold.

Confirmatory comparison
-----------------------
* Base is evaluated for every cell.
* Frozen Adaptive is evaluated for every cell. For an OFF cell it is an exact
  alias of Base, not a separately fitted or tuned model.
* Matched All-real and Naive Hybrid are contextual comparators only for ON
  cells, at the ratio already frozen by the adaptive policy. No comparison
  ratio is invented for OFF cells.

Thresholds are selected solely from training data by five-fold out-of-fold
predictions. Development and test labels are never used to choose thresholds.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


EXPERIMENT_NAME = "Binary Frozen-Adaptive One-Shot Test Evaluation"
EXPERIMENT_ID = "binary_frozen_adaptive_one_shot_test"
PROTOCOL_VERSION = "binary-frozen-adaptive-test-v1"
PRETEST_LOCK_VERSION = "binary-frozen-adaptive-pretest-v1"
FINAL_LOCK_VERSION = "binary-frozen-adaptive-one-shot-test-v1"
PROTOCOL_SEED = 20260713
BOOTSTRAP_REPS = 10_000
OOF_FOLDS = 5
EXPECTED_REPEATS = 50
EXPECTED_EMBEDDING_DIMENSION = 4096
EXPECTED_CELLS = {
    "bn_binary_cinexdrama",
    "en_binary_sst2",
    "ha_binary_hausa_movie_review",
    "ko_binary_nsmc",
    "ml_binary_dravidian_codemix",
}

CLASSIFIER_CONFIG = {
    "penalty": "l2",
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 2000,
    "tol": 1e-4,
    "fit_intercept": True,
    "class_weight": None,
}


@dataclass(frozen=True)
class TestDiagnostics:
    macro_f1_at_0_5: float
    macro_f1_at_oof_threshold: float
    auroc: float
    oof_threshold: float
    test_predicted_positive_rate_at_0_5: float
    test_predicted_positive_rate_at_oof_threshold: float
    train_predicted_positive_rate_at_0_5: float
    intercept: float
    coefficient_l2_norm: float
    n_iter: int
    converged: bool
    convergence_warning: bool
    tn_at_0_5: int
    fp_at_0_5: int
    fn_at_0_5: int
    tp_at_0_5: int
    tn_at_oof_threshold: int
    fp_at_oof_threshold: int
    fn_at_oof_threshold: int
    tp_at_oof_threshold: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in (PROTOCOL_SEED, *parts))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "little")


def strict_json_load(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(
            f"Non-standard JSON constant {value!r} in {path}. "
            "Rerun the decision-boundary diagnostic after replacing NaN with None "
            "and setting json.dumps(..., allow_nan=False)."
        )

    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def verify_lock_files(
    root: Path,
    lock_name: str,
    *,
    skip_optional: set[str] | None = None,
) -> dict[str, Any]:
    lock_path = root / lock_name
    if not lock_path.is_file():
        raise FileNotFoundError(f"Missing lock: {lock_path}")
    lock = strict_json_load(lock_path)
    files = lock.get("files")
    if not isinstance(files, dict):
        raise RuntimeError(f"Invalid files map in {lock_path}")
    optional = skip_optional or set()
    for relative, expected in files.items():
        relative = str(relative)
        path = root / relative
        if not path.is_file():
            if relative in optional:
                continue
            raise FileNotFoundError(f"Locked file is missing: {path}")
        actual = sha256_file(path)
        if actual != str(expected):
            raise RuntimeError(
                f"Hash mismatch for {path}: expected={expected}, actual={actual}"
            )
    return lock


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def load_weight_audit(path: Path) -> dict[tuple[str, int, float], dict[str, Any]]:
    result: dict[tuple[str, int, float], dict[str, Any]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["cell_id"]), int(row["repeat_seed"]), float(row["ratio"]))
            if key in result:
                raise RuntimeError(f"Duplicate selection-audit key at line {line_number}: {key}")
            if not bool(row.get("selection_match")):
                raise RuntimeError(f"Selection mismatch in audit at line {line_number}: {key}")
            selected = [int(value) for value in row["selected_global_generation_indices"]]
            if len(selected) != int(row["selected_n"]):
                raise RuntimeError(f"Selected-index count mismatch for {key}")
            result[key] = row
    return result


def bool_value(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no", "", "nan"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def validate_classifier_config(protocol: Mapping[str, Any]) -> None:
    actual = protocol.get("classifier")
    if actual != CLASSIFIER_CONFIG:
        raise RuntimeError(
            "Frozen adaptive protocol classifier differs from the required fixed "
            f"classifier. expected={CLASSIFIER_CONFIG}, actual={actual}"
        )


def load_frozen_policy(
    adaptive_diagnostic_root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, Any], pd.DataFrame]:
    lock = verify_lock_files(adaptive_diagnostic_root, "ADAPTIVE_SELECTION_LOCK.json")
    if lock.get("status") != "adaptive_policy_frozen_ready_for_one_shot_test":
        raise RuntimeError(f"Unexpected adaptive lock status: {lock.get('status')!r}")
    if lock.get("test_data_accessed") is not False:
        raise RuntimeError("Adaptive selection lock does not certify test_data_accessed=false")

    protocol_path = adaptive_diagnostic_root / "final_adaptive_protocol.json"
    protocol = strict_json_load(protocol_path)
    if protocol.get("test_data_accessed") is not False:
        raise RuntimeError("Final adaptive protocol does not certify test_data_accessed=false")
    if protocol.get("development_only") is not True:
        raise RuntimeError("Final adaptive protocol is not marked development_only=true")
    validate_classifier_config(protocol)

    raw_policy = protocol.get("final_adaptive_policy")
    if not isinstance(raw_policy, list):
        raise RuntimeError("final_adaptive_policy must be a list")
    policy: dict[str, dict[str, Any]] = {}
    for item in raw_policy:
        if not isinstance(item, dict):
            raise RuntimeError("Invalid final_adaptive_policy item")
        cell_id = str(item["cell_id"])
        if cell_id in policy:
            raise RuntimeError(f"Duplicate policy cell: {cell_id}")
        if not bool_value(item.get("test_evaluation_allowed")):
            raise RuntimeError(f"Test evaluation is not allowed for {cell_id}")
        augmentation = str(item["augmentation"]).upper()
        ratio = float(item["selected_ratio"])
        weight = float(item["selected_synthetic_weight"])
        arm = str(item["final_training_arm"])
        if augmentation == "ON":
            if ratio <= 0 or weight <= 0 or arm != "adaptive_hybrid":
                raise RuntimeError(f"Invalid ON policy for {cell_id}: {item}")
        elif augmentation == "OFF":
            if ratio != 0.0 or weight != 0.0 or arm != "base":
                raise RuntimeError(f"Invalid OFF policy for {cell_id}: {item}")
        else:
            raise RuntimeError(f"Unknown augmentation state for {cell_id}: {augmentation}")
        normalized = dict(item)
        normalized["augmentation"] = augmentation
        normalized["selected_ratio"] = ratio
        normalized["selected_synthetic_weight"] = weight
        policy[cell_id] = normalized

    if set(policy) != EXPECTED_CELLS:
        raise RuntimeError(
            f"Frozen policy cells differ from the expected five binary cells: "
            f"missing={sorted(EXPECTED_CELLS - set(policy))}, "
            f"extra={sorted(set(policy) - EXPECTED_CELLS)}"
        )

    diagnostic_path = adaptive_diagnostic_root / "decision_boundary_repeat_results.csv"
    diagnostic = pd.read_csv(diagnostic_path)
    if not diagnostic.empty:
        required = {"cell_id", "repeat_seed", "ratio", "synthetic_weight"}
        missing = required - set(diagnostic.columns)
        if missing:
            raise RuntimeError(f"Diagnostic repeat file missing columns: {sorted(missing)}")
        diagnostic["cell_id"] = diagnostic["cell_id"].astype(str)
        diagnostic["repeat_seed"] = pd.to_numeric(
            diagnostic["repeat_seed"], errors="raise"
        ).astype(int)

    return policy, protocol, lock, diagnostic


def load_train_embedding_maps(
    experiment_root: Path,
    cell_id: str,
) -> tuple[np.ndarray, dict[str, int], dict[int, int], Path, Path]:
    root = experiment_root / "Embeddings" / "Qwen3"
    array_path = root / f"{cell_id}.npy"
    manifest_path = root / f"{cell_id}_manifest.csv"
    if not array_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Missing Qwen training embeddings for {cell_id} under {root}")
    array = np.load(array_path, mmap_mode="r")
    if array.ndim != 2 or array.shape[1] != EXPECTED_EMBEDDING_DIMENSION:
        raise RuntimeError(f"Unexpected training embedding shape for {cell_id}: {array.shape}")
    manifest = pd.read_csv(manifest_path)
    required = {"embedding_row_index", "record_type", "real_row_id", "global_generation_index"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"{manifest_path} missing columns: {sorted(missing)}")

    real: dict[str, int] = {}
    synthetic: dict[int, int] = {}
    for row in manifest.itertuples(index=False):
        index = int(getattr(row, "embedding_row_index"))
        if index < 0 or index >= array.shape[0]:
            raise RuntimeError(f"Embedding index out of range in {manifest_path}: {index}")
        record_type = str(getattr(row, "record_type"))
        if record_type == "real":
            key = str(getattr(row, "real_row_id"))
            if key in real:
                raise RuntimeError(f"Duplicate real embedding key {cell_id}/{key}")
            real[key] = index
        elif record_type == "synthetic":
            key = int(float(getattr(row, "global_generation_index")))
            if key in synthetic:
                raise RuntimeError(f"Duplicate synthetic embedding key {cell_id}/{key}")
            synthetic[key] = index
    return array, real, synthetic, array_path, manifest_path


def test_artifact_paths(
    prepared_root: Path,
    eval_embedding_root: Path,
    cell_id: str,
) -> tuple[Path, Path, Path]:
    test_split = prepared_root / cell_id / "test.csv"
    array_path = eval_embedding_root / f"{cell_id}_test.npy"
    manifest_path = eval_embedding_root / f"{cell_id}_test_manifest.csv"
    for path in (test_split, array_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen test artifact: {path}")
    return test_split, array_path, manifest_path


def load_test_embeddings(
    prepared_root: Path,
    eval_embedding_root: Path,
    cell_id: str,
) -> tuple[np.ndarray, np.ndarray, list[str], Path, Path, Path]:
    test_path, array_path, manifest_path = test_artifact_paths(
        prepared_root, eval_embedding_root, cell_id
    )
    test = pd.read_csv(test_path)
    if "label" not in test.columns:
        raise ValueError(f"{test_path} missing required column: label")
    labels = pd.to_numeric(test["label"], errors="raise").astype(int).to_numpy()
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError(f"{cell_id}: test labels are not binary 0/1")

    array = np.load(array_path, mmap_mode="r")
    if array.ndim != 2 or array.shape[1] != EXPECTED_EMBEDDING_DIMENSION:
        raise RuntimeError(f"Unexpected test embedding shape for {cell_id}: {array.shape}")
    manifest = pd.read_csv(manifest_path)
    if "embedding_row_index" not in manifest.columns:
        raise ValueError(f"{manifest_path} must contain embedding_row_index")
    if "split" in manifest.columns:
        manifest = manifest[manifest["split"].astype(str).str.lower() == "test"].copy()

    manifest_id_column = (
        "row_id"
        if "row_id" in manifest.columns
        else "real_row_id"
        if "real_row_id" in manifest.columns
        else None
    )
    test_id_column = next(
        (
            name
            for name in ("row_id", "real_row_id", "id", "example_id", "uid")
            if name in test.columns
        ),
        None,
    )

    if manifest_id_column is not None and test_id_column is not None:
        manifest[manifest_id_column] = manifest[manifest_id_column].astype(str)
        test_ids = test[test_id_column].astype(str).tolist()
        mapping = {
            row_id: int(index)
            for row_id, index in zip(
                manifest[manifest_id_column], manifest["embedding_row_index"], strict=True
            )
        }
        missing_ids = [row_id for row_id in test_ids if row_id not in mapping]
        if missing_ids:
            raise RuntimeError(f"{cell_id}: test embedding rows missing: {missing_ids[:20]}")
        indices = np.asarray([mapping[row_id] for row_id in test_ids], dtype=np.int64)
    else:
        manifest = manifest.sort_values("embedding_row_index", kind="stable").reset_index(drop=True)
        if len(manifest) != len(test):
            raise RuntimeError(
                f"{cell_id}: positional test alignment is impossible because test.csv has "
                f"{len(test)} rows but the test embedding manifest has {len(manifest)} rows"
            )
        indices = pd.to_numeric(
            manifest["embedding_row_index"], errors="raise"
        ).astype(int).to_numpy()
        if len(np.unique(indices)) != len(indices):
            raise RuntimeError(f"{cell_id}: duplicate test embedding indices")
        if "label" in manifest.columns:
            manifest_labels = pd.to_numeric(
                manifest["label"], errors="raise"
            ).astype(int).to_numpy()
            if not np.array_equal(manifest_labels, labels):
                raise RuntimeError(
                    f"{cell_id}: positional test alignment failed label-order validation"
                )
        test_ids = [f"test_position_{index}" for index in range(len(test))]

    if len(indices) == 0 or indices.min() < 0 or indices.max() >= array.shape[0]:
        raise RuntimeError(f"{cell_id}: test embedding index out of range")
    vectors = np.asarray(array[indices], dtype=np.float32)
    return vectors, labels, test_ids, test_path, array_path, manifest_path


def logistic_regression(repeat_seed: int) -> LogisticRegression:
    return LogisticRegression(random_state=repeat_seed, **CLASSIFIER_CONFIG)


def macro_f1_from_counts(tp: int, fp: int, fn: int, tn: int) -> float:
    positive_denominator = 2 * tp + fp + fn
    negative_denominator = 2 * tn + fp + fn
    positive = 0.0 if positive_denominator == 0 else (2.0 * tp) / positive_denominator
    negative = 0.0 if negative_denominator == 0 else (2.0 * tn) / negative_denominator
    return (positive + negative) / 2.0


def choose_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Maximize training OOF Macro-F1 with frozen deterministic tie-breaking."""
    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    order = np.argsort(-probabilities, kind="stable")
    sorted_probabilities = probabilities[order]
    sorted_labels = y_true[order]
    total_positive = int(sorted_labels.sum())
    total_negative = int(sorted_labels.size - total_positive)

    candidates: list[tuple[float, float]] = []
    threshold_none = float(np.nextafter(sorted_probabilities[0], np.inf))
    candidates.append(
        (threshold_none, macro_f1_from_counts(0, 0, total_positive, total_negative))
    )

    tp = 0
    fp = 0
    position = 0
    while position < sorted_probabilities.size:
        value = sorted_probabilities[position]
        end = position
        while end < sorted_probabilities.size and sorted_probabilities[end] == value:
            if sorted_labels[end] == 1:
                tp += 1
            else:
                fp += 1
            end += 1
        fn = total_positive - tp
        tn = total_negative - fp
        candidates.append((float(value), macro_f1_from_counts(tp, fp, fn, tn)))
        position = end

    prediction = probabilities >= 0.5
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    candidates.append((0.5, macro_f1_from_counts(int(tp), int(fp), int(fn), int(tn))))

    best_score = max(score for _, score in candidates)
    tied = [
        threshold
        for threshold, score in candidates
        if math.isclose(score, best_score, rel_tol=0.0, abs_tol=1e-15)
    ]
    return min(tied, key=lambda threshold: (abs(threshold - 0.5), threshold))


def compute_training_only_oof_threshold(
    x_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    repeat_seed: int,
    condition_name: str,
) -> float:
    y_train = np.asarray(y_train, dtype=np.int64)
    sample_weight = np.asarray(sample_weight, dtype=np.float64)
    class_counts = np.bincount(y_train, minlength=2)
    if class_counts.min() < OOF_FOLDS:
        raise RuntimeError(
            f"Cannot run {OOF_FOLDS}-fold stratification; class counts={class_counts.tolist()}"
        )
    splitter = StratifiedKFold(
        n_splits=OOF_FOLDS,
        shuffle=True,
        random_state=stable_seed(repeat_seed, condition_name, "oof") % (2**32),
    )
    oof = np.empty(y_train.size, dtype=np.float64)
    for train_index, validation_index in splitter.split(x_train, y_train):
        model = logistic_regression(repeat_seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(
                x_train[train_index],
                y_train[train_index],
                sample_weight=sample_weight[train_index],
            )
        oof[validation_index] = model.predict_proba(x_train[validation_index])[:, 1]
    return float(choose_threshold(y_train, oof))


def fit_and_evaluate_test(
    x_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    repeat_seed: int,
    oof_threshold: float,
) -> TestDiagnostics:
    model = logistic_regression(repeat_seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x_train, y_train, sample_weight=sample_weight)
    convergence_warning = any(
        issubclass(item.category, ConvergenceWarning) for item in caught
    )

    train_probabilities = model.predict_proba(x_train)[:, 1]
    test_probabilities = model.predict_proba(x_test)[:, 1]
    prediction_0_5 = (test_probabilities >= 0.5).astype(int)
    prediction_oof = (test_probabilities >= oof_threshold).astype(int)
    tn_05, fp_05, fn_05, tp_05 = confusion_matrix(
        y_test, prediction_0_5, labels=[0, 1]
    ).ravel()
    tn_oof, fp_oof, fn_oof, tp_oof = confusion_matrix(
        y_test, prediction_oof, labels=[0, 1]
    ).ravel()
    n_iter = int(np.max(model.n_iter_))
    return TestDiagnostics(
        macro_f1_at_0_5=float(
            f1_score(y_test, prediction_0_5, average="macro", zero_division=0)
        ),
        macro_f1_at_oof_threshold=float(
            f1_score(y_test, prediction_oof, average="macro", zero_division=0)
        ),
        auroc=float(roc_auc_score(y_test, test_probabilities)),
        oof_threshold=float(oof_threshold),
        test_predicted_positive_rate_at_0_5=float(prediction_0_5.mean()),
        test_predicted_positive_rate_at_oof_threshold=float(prediction_oof.mean()),
        train_predicted_positive_rate_at_0_5=float(
            (train_probabilities >= 0.5).mean()
        ),
        intercept=float(np.ravel(model.intercept_)[0]),
        coefficient_l2_norm=float(np.linalg.norm(model.coef_)),
        n_iter=n_iter,
        converged=bool(n_iter < int(CLASSIFIER_CONFIG["max_iter"])),
        convergence_warning=convergence_warning,
        tn_at_0_5=int(tn_05),
        fp_at_0_5=int(fp_05),
        fn_at_0_5=int(fn_05),
        tp_at_0_5=int(tp_05),
        tn_at_oof_threshold=int(tn_oof),
        fp_at_oof_threshold=int(fp_oof),
        fn_at_oof_threshold=int(fn_oof),
        tp_at_oof_threshold=int(tp_oof),
    )


def build_condition(
    condition: str,
    ratio: float,
    synthetic_weight: float,
    repetition: Mapping[str, Any],
    audit_row: Mapping[str, Any] | None,
    bank_labels: Mapping[str, int],
    generation_labels: Mapping[int, int],
    real_index: Mapping[str, int],
    synthetic_index: Mapping[int, int],
    embedding_array: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, str]:
    base_ids = [str(value) for value in repetition["base_row_ids"]]
    if len(base_ids) != int(repetition["base_n"]):
        raise RuntimeError("base_row_ids count differs from base_n")

    if condition == "base":
        real_ids = base_ids
        synthetic_ids: list[int] = []
        synthetic_sample_weight = 0.0
        selection_sha256 = ""
    else:
        if ratio <= 0:
            raise RuntimeError(f"Condition {condition} requires a positive frozen ratio")
        n_add = int(round(ratio * int(repetition["base_n"])))
        if condition == "matched_all_real":
            real_ids = base_ids + [
                str(value) for value in repetition["additional_real_master_order"][:n_add]
            ]
            synthetic_ids = []
            synthetic_sample_weight = 0.0
            selection_sha256 = ""
        elif condition in {"naive_hybrid", "adaptive_hybrid"}:
            if audit_row is None:
                raise RuntimeError(f"{condition} requires a frozen selection-audit row")
            synthetic_ids = [
                int(value) for value in audit_row["selected_global_generation_indices"]
            ]
            if len(synthetic_ids) != n_add:
                raise RuntimeError(
                    f"Selected synthetic count {len(synthetic_ids)} != n_add {n_add}"
                )
            real_ids = base_ids
            synthetic_sample_weight = (
                1.0 if condition == "naive_hybrid" else synthetic_weight
            )
            selection_sha256 = str(audit_row["selection_sha256"])
        else:
            raise ValueError(f"Unknown training condition: {condition}")

    missing_real = [row_id for row_id in real_ids if row_id not in real_index]
    missing_synthetic = [index for index in synthetic_ids if index not in synthetic_index]
    if missing_real or missing_synthetic:
        raise RuntimeError(
            f"Missing embeddings: real={missing_real[:10]}, "
            f"synthetic={missing_synthetic[:10]}"
        )
    vector_indices = [real_index[row_id] for row_id in real_ids] + [
        synthetic_index[index] for index in synthetic_ids
    ]
    labels = [bank_labels[row_id] for row_id in real_ids] + [
        generation_labels[index] for index in synthetic_ids
    ]
    weights = [1.0] * len(real_ids) + [synthetic_sample_weight] * len(synthetic_ids)
    vectors = np.asarray(
        embedding_array[np.asarray(vector_indices, dtype=np.int64)], dtype=np.float32
    )
    return (
        vectors,
        np.asarray(labels, dtype=np.int64),
        np.asarray(weights, dtype=np.float64),
        len(real_ids),
        len(synthetic_ids),
        selection_sha256,
    )


def bootstrap_mean(values: np.ndarray, *key: object) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size != EXPECTED_REPEATS:
        raise ValueError(f"Expected {EXPECTED_REPEATS} values, got {values.size}")
    if not np.isfinite(values).all():
        raise ValueError("Bootstrap input contains non-finite values")
    rng = np.random.default_rng(stable_seed(*key))
    indices = rng.integers(0, values.size, size=(BOOTSTRAP_REPS, values.size))
    means = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "se": float(values.std(ddof=1) / math.sqrt(values.size)),
        "ci_lower": float(np.quantile(means, 0.025)),
        "ci_upper": float(np.quantile(means, 0.975)),
    }


def expected_arms(policy_row: Mapping[str, Any]) -> list[str]:
    if str(policy_row["augmentation"]).upper() == "ON":
        return ["base", "matched_all_real", "naive_hybrid", "frozen_adaptive"]
    return ["base", "frozen_adaptive"]


def diagnostic_threshold_map(
    diagnostic: pd.DataFrame,
) -> dict[tuple[str, int, str], tuple[float, str]]:
    result: dict[tuple[str, int, str], tuple[float, str]] = {}
    if diagnostic.empty:
        return result
    arm_columns = {
        "base": "base_oof_threshold",
        "matched_all_real": "matched_all_real_oof_threshold",
        "naive_hybrid": "naive_hybrid_oof_threshold",
        "frozen_adaptive": "adaptive_hybrid_oof_threshold",
    }
    for arm, column in arm_columns.items():
        if column not in diagnostic.columns:
            raise RuntimeError(f"Diagnostic repeat file missing threshold column: {column}")
    for row in diagnostic.itertuples(index=False):
        cell_id = str(getattr(row, "cell_id"))
        repeat_seed = int(getattr(row, "repeat_seed"))
        for arm, column in arm_columns.items():
            threshold = float(getattr(row, column))
            if not math.isfinite(threshold):
                raise RuntimeError(
                    f"Non-finite frozen threshold for {cell_id}/{repeat_seed}/{arm}"
                )
            key = (cell_id, repeat_seed, arm)
            if key in result:
                raise RuntimeError(f"Duplicate frozen diagnostic threshold: {key}")
            result[key] = (threshold, "locked_decision_boundary_diagnostic")
    return result


def validate_non_test_inputs(
    prepared_root: Path,
    manifest_root: Path,
    experiment_root: Path,
    eval_embedding_root: Path,
    weight_grid_root: Path,
    adaptive_diagnostic_root: Path,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    pd.DataFrame,
    dict[tuple[str, int, float], dict[str, Any]],
    dict[int, int],
    dict[str, str],
]:
    policy, protocol, adaptive_lock, diagnostic = load_frozen_policy(
        adaptive_diagnostic_root
    )

    weight_lock = verify_lock_files(
        weight_grid_root,
        "SYNTHETIC_WEIGHT_GRID_LOCK.json",
        skip_optional={"synthetic_weight_status.sqlite3"},
    )
    qwen_root = experiment_root / "Embeddings" / "Qwen3"
    verify_lock_files(qwen_root, "QWEN_EMBEDDING_LOCK.json")

    audit_path = weight_grid_root / "weight_selection_audit.jsonl.gz"
    audit = load_weight_audit(audit_path)
    generation_plan_path = manifest_root / "stage_a_generation_plan.csv"
    generation_plan = pd.read_csv(generation_plan_path)
    required_generation = {"global_generation_index", "label"}
    missing_generation = required_generation - set(generation_plan.columns)
    if missing_generation:
        raise RuntimeError(
            f"Generation plan missing columns: {sorted(missing_generation)}"
        )
    generation_labels = {
        int(index): int(label)
        for index, label in zip(
            generation_plan["global_generation_index"],
            generation_plan["label"],
            strict=True,
        )
    }

    # Existence only. No test CSV, test manifest, or test array is read here.
    for cell_id in sorted(policy):
        test_artifact_paths(prepared_root, eval_embedding_root, cell_id)

    input_hashes = {
        "adaptive_selection_lock": sha256_file(
            adaptive_diagnostic_root / "ADAPTIVE_SELECTION_LOCK.json"
        ),
        "final_adaptive_protocol": sha256_file(
            adaptive_diagnostic_root / "final_adaptive_protocol.json"
        ),
        "decision_boundary_repeat_results": sha256_file(
            adaptive_diagnostic_root / "decision_boundary_repeat_results.csv"
        ),
        "synthetic_weight_grid_lock": sha256_file(
            weight_grid_root / "SYNTHETIC_WEIGHT_GRID_LOCK.json"
        ),
        "weight_selection_audit": sha256_file(audit_path),
        "qwen_training_embedding_lock": sha256_file(
            qwen_root / "QWEN_EMBEDDING_LOCK.json"
        ),
        "generation_plan": sha256_file(generation_plan_path),
    }
    _ = weight_lock
    return (
        policy,
        protocol,
        adaptive_lock,
        diagnostic,
        audit,
        generation_labels,
        input_hashes,
    )


def prepare_pretest_thresholds(args: argparse.Namespace) -> int:
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        existing_lock = output_root / "PRETEST_THRESHOLD_LOCK.json"
        if existing_lock.is_file():
            verify_lock_files(output_root, existing_lock.name)
            print(f"Pre-test threshold lock already exists and is valid: {existing_lock}")
            return 0
        raise FileExistsError(f"Output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    (
        policy,
        protocol,
        _adaptive_lock,
        diagnostic,
        audit,
        generation_labels,
        input_hashes,
    ) = validate_non_test_inputs(
        args.prepared_root.resolve(),
        args.manifest_root.resolve(),
        args.experiment_root.resolve(),
        args.eval_embedding_root.resolve(),
        args.weight_grid_root.resolve(),
        args.adaptive_diagnostic_root.resolve(),
    )

    diagnostic_thresholds = diagnostic_threshold_map(diagnostic)
    threshold_rows: list[dict[str, Any]] = []
    artifact_hashes = dict(input_hashes)

    for cell_id in sorted(policy):
        policy_row = policy[cell_id]
        ratio = float(policy_row["selected_ratio"])
        synthetic_weight = float(policy_row["selected_synthetic_weight"])
        bank_path = args.manifest_root.resolve() / cell_id / "experiment_bank.csv"
        repetitions_path = args.manifest_root.resolve() / cell_id / "repetitions.jsonl"
        bank = pd.read_csv(bank_path)
        required_bank = {"row_id", "label"}
        missing_bank = required_bank - set(bank.columns)
        if missing_bank:
            raise RuntimeError(f"{bank_path} missing columns: {sorted(missing_bank)}")
        bank["row_id"] = bank["row_id"].astype(str)
        bank_labels = {
            row_id: int(label)
            for row_id, label in zip(bank["row_id"], bank["label"], strict=True)
        }
        repetitions = {
            int(row["repeat_seed"]): row for row in read_jsonl(repetitions_path)
        }
        if len(repetitions) != EXPECTED_REPEATS:
            raise RuntimeError(
                f"{cell_id}: expected {EXPECTED_REPEATS} repetition records, "
                f"found {len(repetitions)}"
            )

        cell_diagnostic = diagnostic[diagnostic["cell_id"] == cell_id].copy()
        if not cell_diagnostic.empty:
            if len(cell_diagnostic) != EXPECTED_REPEATS:
                raise RuntimeError(
                    f"{cell_id}: locked decision-boundary diagnostic has "
                    f"{len(cell_diagnostic)} rows instead of {EXPECTED_REPEATS}"
                )
            diagnostic_seeds = set(cell_diagnostic["repeat_seed"].astype(int))
            if diagnostic_seeds != set(repetitions):
                raise RuntimeError(
                    f"{cell_id}: diagnostic/repetition seed mismatch"
                )
        if policy_row["augmentation"] == "ON":
            if cell_diagnostic.empty:
                raise RuntimeError(
                    f"{cell_id}: ON policy has no locked decision-boundary thresholds"
                )
            diagnostic_ratios = set(
                pd.to_numeric(cell_diagnostic["ratio"], errors="raise").astype(float)
            )
            diagnostic_weights = set(
                pd.to_numeric(
                    cell_diagnostic["synthetic_weight"], errors="raise"
                ).astype(float)
            )
            if diagnostic_ratios != {ratio} or diagnostic_weights != {synthetic_weight}:
                raise RuntimeError(
                    f"{cell_id}: frozen policy ratio/weight does not match the locked "
                    f"diagnostic rows; policy=({ratio}, {synthetic_weight}), "
                    f"diagnostic=({sorted(diagnostic_ratios)}, {sorted(diagnostic_weights)})"
                )

        embedding_array, real_index, synthetic_index, array_path, manifest_path = (
            load_train_embedding_maps(args.experiment_root.resolve(), cell_id)
        )
        artifact_hashes[f"{cell_id}/experiment_bank"] = sha256_file(bank_path)
        artifact_hashes[f"{cell_id}/repetitions"] = sha256_file(repetitions_path)
        artifact_hashes[f"{cell_id}/train_embedding_array"] = sha256_file(array_path)
        artifact_hashes[f"{cell_id}/train_embedding_manifest"] = sha256_file(
            manifest_path
        )

        computed_threshold_cache: dict[tuple[int, str], tuple[float, str]] = {}
        for repeat_seed in sorted(repetitions):
            repetition = repetitions[repeat_seed]
            arms = expected_arms(policy_row)
            for arm in arms:
                if arm == "frozen_adaptive" and policy_row["augmentation"] == "OFF":
                    base_key = (cell_id, repeat_seed, "base")
                    cache_key = (repeat_seed, "base")
                    if base_key in diagnostic_thresholds:
                        threshold, source = diagnostic_thresholds[base_key]
                    elif cache_key in computed_threshold_cache:
                        threshold, source = computed_threshold_cache[cache_key]
                    else:
                        x_train, y_train, sample_weight, *_ = build_condition(
                            "base",
                            0.0,
                            0.0,
                            repetition,
                            None,
                            bank_labels,
                            generation_labels,
                            real_index,
                            synthetic_index,
                            embedding_array,
                        )
                        threshold = compute_training_only_oof_threshold(
                            x_train,
                            y_train,
                            sample_weight,
                            repeat_seed,
                            "base",
                        )
                        source = "computed_pretest_training_only_oof"
                        computed_threshold_cache[cache_key] = (threshold, source)
                    threshold_rows.append(
                        {
                            "cell_id": cell_id,
                            "repeat_seed": repeat_seed,
                            "arm": arm,
                            "training_condition": "base",
                            "augmentation": "OFF",
                            "ratio": 0.0,
                            "synthetic_weight": 0.0,
                            "oof_threshold": threshold,
                            "threshold_source": "base_exact_alias:" + source,
                        }
                    )
                    continue

                frozen_key = (cell_id, repeat_seed, arm)
                cache_condition = (
                    "adaptive_hybrid" if arm == "frozen_adaptive" else arm
                )
                cache_key = (repeat_seed, cache_condition)
                if frozen_key in diagnostic_thresholds:
                    threshold, source = diagnostic_thresholds[frozen_key]
                elif cache_key in computed_threshold_cache:
                    threshold, source = computed_threshold_cache[cache_key]
                else:
                    condition = cache_condition
                    audit_row = None
                    if condition in {"naive_hybrid", "adaptive_hybrid"}:
                        audit_key = (cell_id, repeat_seed, ratio)
                        if audit_key not in audit:
                            raise RuntimeError(f"Missing frozen selection audit: {audit_key}")
                        audit_row = audit[audit_key]
                    x_train, y_train, sample_weight, *_ = build_condition(
                        condition,
                        ratio,
                        synthetic_weight,
                        repetition,
                        audit_row,
                        bank_labels,
                        generation_labels,
                        real_index,
                        synthetic_index,
                        embedding_array,
                    )
                    threshold = compute_training_only_oof_threshold(
                        x_train,
                        y_train,
                        sample_weight,
                        repeat_seed,
                        condition,
                    )
                    source = "computed_pretest_training_only_oof"
                    computed_threshold_cache[cache_key] = (threshold, source)

                threshold_rows.append(
                    {
                        "cell_id": cell_id,
                        "repeat_seed": repeat_seed,
                        "arm": arm,
                        "training_condition": (
                            "adaptive_hybrid" if arm == "frozen_adaptive" else arm
                        ),
                        "augmentation": str(policy_row["augmentation"]),
                        "ratio": ratio if arm != "base" else 0.0,
                        "synthetic_weight": (
                            synthetic_weight if arm == "frozen_adaptive" else 1.0
                            if arm == "naive_hybrid"
                            else 0.0
                        ),
                        "oof_threshold": threshold,
                        "threshold_source": source,
                    }
                )
        del embedding_array

    thresholds_path = output_root / "pretest_thresholds.csv"
    write_csv(thresholds_path, threshold_rows, list(threshold_rows[0].keys()))

    expected_rows = sum(
        len(expected_arms(row)) * EXPECTED_REPEATS for row in policy.values()
    )
    if len(threshold_rows) != expected_rows:
        raise RuntimeError(
            f"Pre-test threshold row count mismatch: {len(threshold_rows)} != {expected_rows}"
        )
    threshold_frame = pd.DataFrame(threshold_rows)
    if threshold_frame.duplicated(["cell_id", "repeat_seed", "arm"]).any():
        raise RuntimeError("Duplicate pre-test threshold keys")
    if not np.isfinite(threshold_frame["oof_threshold"].to_numpy(dtype=float)).all():
        raise RuntimeError("Pre-test threshold table contains non-finite values")

    pretest_protocol = {
        "protocol_version": PRETEST_LOCK_VERSION,
        "created_at": utc_now(),
        "experiment_name": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "test_data_accessed": False,
        "policy_change_allowed": False,
        "threshold_selection": {
            "source": "training-only five-fold out-of-fold probabilities",
            "objective": "Macro-F1",
            "tie_break": "threshold nearest 0.5, then lower threshold",
            "development_labels_used": False,
            "test_labels_used": False,
            "locked_threshold_rows": len(threshold_rows),
        },
        "test_arms": {
            "all_cells": ["base", "frozen_adaptive"],
            "on_cells_only": ["matched_all_real", "naive_hybrid"],
            "off_cell_frozen_adaptive_behavior": "exact_base_alias",
        },
        "classifier": CLASSIFIER_CONFIG,
        "frozen_policy": list(policy.values()),
        "adaptive_protocol_sha256": sha256_file(
            args.adaptive_diagnostic_root.resolve() / "final_adaptive_protocol.json"
        ),
        "input_artifact_sha256": artifact_hashes,
        "prohibited_during_test": protocol.get("prohibited_after_lock", []),
    }
    protocol_path = output_root / "pretest_protocol.json"
    write_json(protocol_path, pretest_protocol)

    lock = {
        "lock_version": PRETEST_LOCK_VERSION,
        "created_at": utc_now(),
        "experiment_name": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "status": "pretest_thresholds_frozen_ready_for_one_shot_evaluation",
        "test_data_accessed": False,
        "files": {
            thresholds_path.name: sha256_file(thresholds_path),
            protocol_path.name: sha256_file(protocol_path),
        },
        "input_artifact_sha256": input_hashes,
    }
    lock_path = output_root / "PRETEST_THRESHOLD_LOCK.json"
    write_json(lock_path, lock)
    print(f"Pre-test thresholds frozen without test access: {lock_path}")
    print("Next command: rerun this script with --evaluate")
    return 0


def load_pretest_thresholds(
    output_root: Path,
    current_input_hashes: Mapping[str, str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    lock = verify_lock_files(output_root, "PRETEST_THRESHOLD_LOCK.json")
    if lock.get("status") != "pretest_thresholds_frozen_ready_for_one_shot_evaluation":
        raise RuntimeError(f"Unexpected pre-test lock status: {lock.get('status')!r}")
    if lock.get("test_data_accessed") is not False:
        raise RuntimeError("Pre-test threshold lock already reports test access")
    locked_inputs = lock.get("input_artifact_sha256")
    if not isinstance(locked_inputs, dict):
        raise RuntimeError("Pre-test lock is missing input_artifact_sha256")
    for name, actual in current_input_hashes.items():
        expected = locked_inputs.get(name)
        if expected != actual:
            raise RuntimeError(
                f"Input changed after pre-test threshold lock: {name}; "
                f"expected={expected}, actual={actual}"
            )

    frame = pd.read_csv(output_root / "pretest_thresholds.csv")
    required = {
        "cell_id",
        "repeat_seed",
        "arm",
        "training_condition",
        "augmentation",
        "ratio",
        "synthetic_weight",
        "oof_threshold",
        "threshold_source",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"pretest_thresholds.csv missing columns: {sorted(missing)}")
    frame["cell_id"] = frame["cell_id"].astype(str)
    frame["repeat_seed"] = pd.to_numeric(frame["repeat_seed"], errors="raise").astype(int)
    frame["oof_threshold"] = pd.to_numeric(
        frame["oof_threshold"], errors="raise"
    ).astype(float)
    if frame.duplicated(["cell_id", "repeat_seed", "arm"]).any():
        raise RuntimeError("Duplicate pre-test threshold keys")
    if not np.isfinite(frame["oof_threshold"].to_numpy()).all():
        raise RuntimeError("Non-finite pre-test threshold")
    return frame, lock


def evaluate_one_shot(args: argparse.Namespace) -> int:
    output_root = args.output_root.resolve()
    final_lock_path = output_root / "ADAPTIVE_ONE_SHOT_TEST_LOCK.json"
    access_marker_path = output_root / "TEST_ACCESS_STARTED.json"
    if final_lock_path.exists():
        raise RuntimeError(
            f"The one-shot test has already been completed: {final_lock_path}"
        )
    if access_marker_path.exists():
        raise RuntimeError(
            "A prior run already marked test access. Do not rerun or overwrite the "
            f"one-shot evaluation. Inspect {access_marker_path} and the failure first."
        )

    (
        policy,
        _protocol,
        _adaptive_lock,
        _diagnostic,
        audit,
        generation_labels,
        current_input_hashes,
    ) = validate_non_test_inputs(
        args.prepared_root.resolve(),
        args.manifest_root.resolve(),
        args.experiment_root.resolve(),
        args.eval_embedding_root.resolve(),
        args.weight_grid_root.resolve(),
        args.adaptive_diagnostic_root.resolve(),
    )
    thresholds, pretest_lock = load_pretest_thresholds(
        output_root, current_input_hashes
    )
    threshold_map = {
        (str(row.cell_id), int(row.repeat_seed), str(row.arm)): float(row.oof_threshold)
        for row in thresholds.itertuples(index=False)
    }

    # This marker is written immediately before any test CSV/manifest/array is read.
    access_marker = {
        "created_at": utc_now(),
        "experiment_name": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "status": "test_access_started",
        "test_data_accessed": True,
        "pretest_threshold_lock_sha256": sha256_file(
            output_root / "PRETEST_THRESHOLD_LOCK.json"
        ),
        "adaptive_selection_lock_sha256": sha256_file(
            args.adaptive_diagnostic_root.resolve() / "ADAPTIVE_SELECTION_LOCK.json"
        ),
        "warning": "The untouched test evaluation has begun. Policy and thresholds must not be changed or rerun based on these results.",
    }
    write_json(access_marker_path, access_marker)

    # Test access starts here.
    eval_lock = verify_lock_files(
        args.eval_embedding_root.resolve(), "QWEN_EVAL_EMBEDDING_LOCK.json"
    )

    repeat_rows: list[dict[str, Any]] = []
    test_artifact_hashes: dict[str, str] = {
        "qwen_eval_embedding_lock": sha256_file(
            args.eval_embedding_root.resolve() / "QWEN_EVAL_EMBEDDING_LOCK.json"
        )
    }

    for cell_id in sorted(policy):
        policy_row = policy[cell_id]
        ratio = float(policy_row["selected_ratio"])
        synthetic_weight = float(policy_row["selected_synthetic_weight"])
        bank_path = args.manifest_root.resolve() / cell_id / "experiment_bank.csv"
        repetitions_path = args.manifest_root.resolve() / cell_id / "repetitions.jsonl"
        bank = pd.read_csv(bank_path)
        bank["row_id"] = bank["row_id"].astype(str)
        bank_labels = {
            row_id: int(label)
            for row_id, label in zip(bank["row_id"], bank["label"], strict=True)
        }
        repetitions = {
            int(row["repeat_seed"]): row for row in read_jsonl(repetitions_path)
        }
        embedding_array, real_index, synthetic_index, _, _ = load_train_embedding_maps(
            args.experiment_root.resolve(), cell_id
        )
        x_test, y_test, _test_ids, test_path, test_array_path, test_manifest_path = (
            load_test_embeddings(
                args.prepared_root.resolve(), args.eval_embedding_root.resolve(), cell_id
            )
        )
        test_artifact_hashes[f"{cell_id}/test_split"] = sha256_file(test_path)
        test_artifact_hashes[f"{cell_id}/test_embedding_array"] = sha256_file(
            test_array_path
        )
        test_artifact_hashes[f"{cell_id}/test_embedding_manifest"] = sha256_file(
            test_manifest_path
        )

        for repeat_seed in sorted(repetitions):
            repetition = repetitions[repeat_seed]
            per_repeat_results: dict[str, dict[str, Any]] = {}
            for arm in expected_arms(policy_row):
                if arm == "frozen_adaptive" and policy_row["augmentation"] == "OFF":
                    if "base" not in per_repeat_results:
                        raise RuntimeError("Base must be evaluated before OFF adaptive alias")
                    copied = dict(per_repeat_results["base"])
                    copied["arm"] = "frozen_adaptive"
                    copied["training_condition"] = "base"
                    copied["result_source"] = "reused_base_exact_alias"
                    copied["augmentation"] = "OFF"
                    copied["ratio"] = 0.0
                    copied["synthetic_weight"] = 0.0
                    copied["threshold_source"] = str(
                        thresholds.loc[
                            (thresholds["cell_id"] == cell_id)
                            & (thresholds["repeat_seed"] == repeat_seed)
                            & (thresholds["arm"] == arm),
                            "threshold_source",
                        ].iloc[0]
                    )
                    per_repeat_results[arm] = copied
                    repeat_rows.append(copied)
                    continue

                condition = "adaptive_hybrid" if arm == "frozen_adaptive" else arm
                audit_row = None
                if condition in {"naive_hybrid", "adaptive_hybrid"}:
                    audit_key = (cell_id, repeat_seed, ratio)
                    if audit_key not in audit:
                        raise RuntimeError(f"Missing frozen selection audit: {audit_key}")
                    audit_row = audit[audit_key]
                x_train, y_train, sample_weight, n_real, n_synthetic, selection_sha = (
                    build_condition(
                        condition,
                        ratio,
                        synthetic_weight,
                        repetition,
                        audit_row,
                        bank_labels,
                        generation_labels,
                        real_index,
                        synthetic_index,
                        embedding_array,
                    )
                )
                threshold_key = (cell_id, repeat_seed, arm)
                if threshold_key not in threshold_map:
                    raise RuntimeError(f"Missing pre-test threshold: {threshold_key}")
                threshold = threshold_map[threshold_key]
                diagnostics = fit_and_evaluate_test(
                    x_train,
                    y_train,
                    sample_weight,
                    x_test,
                    y_test,
                    repeat_seed,
                    threshold,
                )
                threshold_source = str(
                    thresholds.loc[
                        (thresholds["cell_id"] == cell_id)
                        & (thresholds["repeat_seed"] == repeat_seed)
                        & (thresholds["arm"] == arm),
                        "threshold_source",
                    ].iloc[0]
                )
                row: dict[str, Any] = {
                    "experiment_name": EXPERIMENT_NAME,
                    "experiment_id": EXPERIMENT_ID,
                    "protocol_version": PROTOCOL_VERSION,
                    "cell_id": cell_id,
                    "repeat_seed": repeat_seed,
                    "arm": arm,
                    "training_condition": condition,
                    "result_source": "new_test_fit",
                    "augmentation": str(policy_row["augmentation"]),
                    "ratio": ratio if arm != "base" else 0.0,
                    "synthetic_weight": (
                        synthetic_weight if arm == "frozen_adaptive" else 1.0
                        if arm == "naive_hybrid"
                        else 0.0
                    ),
                    "selection_sha256": selection_sha,
                    "n_real": n_real,
                    "n_synthetic": n_synthetic,
                    "sum_sample_weight": float(sample_weight.sum()),
                    "threshold_source": threshold_source,
                }
                row.update(asdict(diagnostics))
                per_repeat_results[arm] = row
                repeat_rows.append(row)

            base = per_repeat_results["base"]
            adaptive = per_repeat_results["frozen_adaptive"]
            print(
                f"{cell_id} repeat={repeat_seed} completed | "
                f"adaptive-base calibrated delta="
                f"{adaptive['macro_f1_at_oof_threshold'] - base['macro_f1_at_oof_threshold']:+.6f}"
            )
        del embedding_array

    repeat_path = output_root / "test_repeat_results.csv"
    write_csv(repeat_path, repeat_rows, list(repeat_rows[0].keys()))
    frame = pd.DataFrame(repeat_rows)

    expected_total_rows = sum(
        len(expected_arms(row)) * EXPECTED_REPEATS for row in policy.values()
    )
    if len(frame) != expected_total_rows:
        raise RuntimeError(f"Test result row count mismatch: {len(frame)} != {expected_total_rows}")
    if frame.duplicated(["cell_id", "repeat_seed", "arm"]).any():
        raise RuntimeError("Duplicate test result keys")

    metric_names = [
        "macro_f1_at_0_5",
        "macro_f1_at_oof_threshold",
        "auroc",
        "test_predicted_positive_rate_at_0_5",
        "test_predicted_positive_rate_at_oof_threshold",
    ]
    arm_summary_rows: list[dict[str, Any]] = []
    for (cell_id, arm), group in frame.groupby(["cell_id", "arm"], sort=True):
        group = group.sort_values("repeat_seed")
        if len(group) != EXPECTED_REPEATS:
            raise RuntimeError(f"{cell_id}/{arm}: expected {EXPECTED_REPEATS} results")
        for metric in metric_names:
            estimate = bootstrap_mean(
                group[metric].to_numpy(dtype=float), cell_id, arm, metric, "test_arm"
            )
            arm_summary_rows.append(
                {
                    "cell_id": cell_id,
                    "arm": arm,
                    "metric": metric,
                    **estimate,
                }
            )
    arm_summary_path = output_root / "test_arm_summary.csv"
    write_csv(
        arm_summary_path,
        arm_summary_rows,
        list(arm_summary_rows[0].keys()),
    )

    comparison_specs = [
        ("frozen_adaptive", "base"),
        ("frozen_adaptive", "naive_hybrid"),
        ("frozen_adaptive", "matched_all_real"),
        ("naive_hybrid", "base"),
        ("matched_all_real", "base"),
    ]
    comparison_metrics = [
        "macro_f1_at_0_5",
        "macro_f1_at_oof_threshold",
        "auroc",
    ]
    comparison_rows: list[dict[str, Any]] = []
    for cell_id in sorted(policy):
        cell = frame[frame["cell_id"] == cell_id]
        by_arm = {
            arm: group.sort_values("repeat_seed").set_index("repeat_seed")
            for arm, group in cell.groupby("arm")
        }
        for left_arm, right_arm in comparison_specs:
            if left_arm not in by_arm or right_arm not in by_arm:
                continue
            left = by_arm[left_arm]
            right = by_arm[right_arm]
            if not left.index.equals(right.index):
                raise RuntimeError(
                    f"Repeat-seed mismatch for {cell_id}: {left_arm} vs {right_arm}"
                )
            for metric in comparison_metrics:
                differences = (
                    left[metric].to_numpy(dtype=float)
                    - right[metric].to_numpy(dtype=float)
                )
                estimate = bootstrap_mean(
                    differences,
                    cell_id,
                    left_arm,
                    right_arm,
                    metric,
                    "test_comparison",
                )
                comparison_rows.append(
                    {
                        "scope": "cell",
                        "cell_id": cell_id,
                        "left_arm": left_arm,
                        "right_arm": right_arm,
                        "comparison": f"{left_arm}_minus_{right_arm}",
                        "metric": metric,
                        **estimate,
                        "ci_excludes_zero_positive": estimate["ci_lower"] > 0.0,
                        "ci_excludes_zero_negative": estimate["ci_upper"] < 0.0,
                    }
                )

    # Equal-cell pooled policy effect. Repeat seeds are aligned before averaging.
    adaptive_deltas_by_cell: dict[str, pd.DataFrame] = {}
    for cell_id in sorted(policy):
        cell = frame[frame["cell_id"] == cell_id]
        base = cell[cell["arm"] == "base"].sort_values("repeat_seed").set_index(
            "repeat_seed"
        )
        adaptive = cell[cell["arm"] == "frozen_adaptive"].sort_values(
            "repeat_seed"
        ).set_index("repeat_seed")
        if not base.index.equals(adaptive.index):
            raise RuntimeError(f"Adaptive/Base repeat mismatch for {cell_id}")
        delta = pd.DataFrame(index=base.index)
        for metric in comparison_metrics:
            delta[metric] = adaptive[metric] - base[metric]
        adaptive_deltas_by_cell[cell_id] = delta

    for scope_name, scope_cells in (
        ("all_five_cells_equal_weight", sorted(policy)),
        (
            "augmentation_on_cells_equal_weight",
            sorted(
                cell_id
                for cell_id, row in policy.items()
                if row["augmentation"] == "ON"
            ),
        ),
    ):
        if not scope_cells:
            continue
        reference_index = adaptive_deltas_by_cell[scope_cells[0]].index
        if any(
            not adaptive_deltas_by_cell[cell_id].index.equals(reference_index)
            for cell_id in scope_cells
        ):
            raise RuntimeError(f"Repeat seeds are not aligned in pooled scope {scope_name}")
        for metric in comparison_metrics:
            matrix = np.vstack(
                [
                    adaptive_deltas_by_cell[cell_id][metric].to_numpy(dtype=float)
                    for cell_id in scope_cells
                ]
            )
            repeat_equal_cell_means = matrix.mean(axis=0)
            estimate = bootstrap_mean(
                repeat_equal_cell_means,
                scope_name,
                "frozen_adaptive",
                "base",
                metric,
                "test_pooled",
            )
            comparison_rows.append(
                {
                    "scope": scope_name,
                    "cell_id": "",
                    "left_arm": "frozen_adaptive",
                    "right_arm": "base",
                    "comparison": "frozen_adaptive_minus_base",
                    "metric": metric,
                    **estimate,
                    "ci_excludes_zero_positive": estimate["ci_lower"] > 0.0,
                    "ci_excludes_zero_negative": estimate["ci_upper"] < 0.0,
                }
            )

    comparison_path = output_root / "test_paired_comparison_summary.csv"
    write_csv(comparison_path, comparison_rows, list(comparison_rows[0].keys()))

    policy_rows: list[dict[str, Any]] = []
    for cell_id in sorted(policy):
        policy_row = policy[cell_id]
        calibrated = next(
            row
            for row in comparison_rows
            if row["scope"] == "cell"
            and row["cell_id"] == cell_id
            and row["comparison"] == "frozen_adaptive_minus_base"
            and row["metric"] == "macro_f1_at_oof_threshold"
        )
        auroc = next(
            row
            for row in comparison_rows
            if row["scope"] == "cell"
            and row["cell_id"] == cell_id
            and row["comparison"] == "frozen_adaptive_minus_base"
            and row["metric"] == "auroc"
        )
        policy_rows.append(
            {
                "cell_id": cell_id,
                "frozen_augmentation": policy_row["augmentation"],
                "frozen_ratio": policy_row["selected_ratio"],
                "frozen_synthetic_weight": policy_row["selected_synthetic_weight"],
                "test_macro_f1_delta_mean": calibrated["mean"],
                "test_macro_f1_delta_ci_lower": calibrated["ci_lower"],
                "test_macro_f1_delta_ci_upper": calibrated["ci_upper"],
                "test_auroc_delta_mean": auroc["mean"],
                "test_auroc_delta_ci_lower": auroc["ci_lower"],
                "test_auroc_delta_ci_upper": auroc["ci_upper"],
                "policy_changed_after_test": False,
            }
        )
    policy_summary_path = output_root / "test_policy_summary.csv"
    write_csv(policy_summary_path, policy_rows, list(policy_rows[0].keys()))

    report = {
        "protocol_version": PROTOCOL_VERSION,
        "created_at": utc_now(),
        "experiment_name": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "status": "completed_one_shot_test_evaluation",
        "test_data_accessed": True,
        "policy_changed_after_test": False,
        "thresholds_changed_after_test": False,
        "classifier": CLASSIFIER_CONFIG,
        "frozen_policy": list(policy.values()),
        "test_arm_rule": {
            "all_cells": ["base", "frozen_adaptive"],
            "on_cells_only": ["matched_all_real", "naive_hybrid"],
            "off_cell_frozen_adaptive_behavior": "exact_base_alias",
            "reason_no_matched_or_naive_for_off_cells": (
                "The frozen policy contains no positive ratio for OFF cells; the test "
                "evaluation does not invent a post-lock comparison ratio."
            ),
        },
        "inference": {
            "bootstrap_repetitions": BOOTSTRAP_REPS,
            "bootstrap_unit": "paired downstream repeat",
            "pooled_weighting": "equal cell weight",
            "seed": PROTOCOL_SEED,
        },
        "input_artifact_sha256": current_input_hashes,
        "test_artifact_sha256": test_artifact_hashes,
        "pretest_threshold_lock_sha256": sha256_file(
            output_root / "PRETEST_THRESHOLD_LOCK.json"
        ),
        "test_access_marker_sha256": sha256_file(access_marker_path),
        "policy_results": policy_rows,
        "prohibited_post_test_actions": [
            "changing ON/OFF decisions based on test results",
            "changing ratio or synthetic weight based on test results",
            "changing any OOF threshold based on test results",
            "searching another condition on the test split",
            "rerunning the one-shot test to choose a preferred result",
        ],
    }
    report_path = output_root / "test_evaluation_report.json"
    write_json(report_path, report)

    output_files = {
        repeat_path.name: sha256_file(repeat_path),
        arm_summary_path.name: sha256_file(arm_summary_path),
        comparison_path.name: sha256_file(comparison_path),
        policy_summary_path.name: sha256_file(policy_summary_path),
        report_path.name: sha256_file(report_path),
        access_marker_path.name: sha256_file(access_marker_path),
        "pretest_thresholds.csv": sha256_file(output_root / "pretest_thresholds.csv"),
        "pretest_protocol.json": sha256_file(output_root / "pretest_protocol.json"),
        "PRETEST_THRESHOLD_LOCK.json": sha256_file(
            output_root / "PRETEST_THRESHOLD_LOCK.json"
        ),
    }
    final_lock = {
        "lock_version": FINAL_LOCK_VERSION,
        "created_at": utc_now(),
        "experiment_name": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "status": "completed_one_shot_test_evaluation",
        "test_data_accessed": True,
        "policy_changed_after_test": False,
        "files": output_files,
        "input_lock_sha256": {
            "adaptive_selection": sha256_file(
                args.adaptive_diagnostic_root.resolve() / "ADAPTIVE_SELECTION_LOCK.json"
            ),
            "pretest_threshold": sha256_file(
                output_root / "PRETEST_THRESHOLD_LOCK.json"
            ),
            "synthetic_weight_grid": sha256_file(
                args.weight_grid_root.resolve() / "SYNTHETIC_WEIGHT_GRID_LOCK.json"
            ),
            "qwen_eval_embedding": sha256_file(
                args.eval_embedding_root.resolve() / "QWEN_EVAL_EMBEDDING_LOCK.json"
            ),
        },
        "eval_lock_status": eval_lock.get("status"),
        "pretest_lock_status": pretest_lock.get("status"),
    }
    write_json(final_lock_path, final_lock)
    print(f"One-shot test evaluation completed and locked: {final_lock_path}")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--prepare-only",
        action="store_true",
        help="Freeze all training-only thresholds without reading test artifacts.",
    )
    mode.add_argument(
        "--evaluate",
        action="store_true",
        help="Run the one-shot test using an existing PRETEST_THRESHOLD_LOCK.json.",
    )
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=project_root / "Data" / "Prepared",
    )
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=project_root / "Data" / "ExperimentManifests" / "v1",
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=project_root / "Results" / "BinaryMatchedSizeExperiment",
    )
    parser.add_argument(
        "--eval-embedding-root",
        type=Path,
        default=(
            project_root
            / "Results"
            / "BinaryMatchedSizeExperiment"
            / "Embeddings"
            / "Qwen3Eval"
        ),
    )
    parser.add_argument(
        "--weight-grid-root",
        type=Path,
        default=(
            project_root
            / "Results"
            / "BinaryMatchedSizeExperiment"
            / "Downstream"
            / "SyntheticWeightGrid"
        ),
    )
    parser.add_argument(
        "--adaptive-diagnostic-root",
        type=Path,
        default=(
            project_root
            / "Results"
            / "BinaryMatchedSizeExperiment"
            / "Downstream"
            / "AdaptivePolicy"
            / "DecisionBoundaryDiagnostic"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            project_root
            / "Results"
            / "BinaryMatchedSizeExperiment"
            / "Downstream"
            / "AdaptivePolicy"
            / "OneShotTest"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.prepare_only:
            return prepare_pretest_thresholds(args)
        return evaluate_one_shot(args)
    except Exception as exc:
        print(f"adaptive one-shot test failed: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
