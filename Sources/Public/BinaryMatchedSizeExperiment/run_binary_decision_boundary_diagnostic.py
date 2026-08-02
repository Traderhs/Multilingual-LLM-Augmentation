"""Run the training-only threshold diagnostic for the frozen adaptive candidate.

The script is development-only. It reconstructs the exact real and synthetic
training rows used by the similarity-OFF weight grid, then compares four arms:
Base, Matched All-real, Naive Hybrid (weight=1), and the frozen Adaptive
candidate. Thresholds are selected exclusively from training data via 5-fold
out-of-fold predictions; development labels are used only for evaluation.

A positive adaptive candidate is finalized as ON only when its Macro-F1 gain
versus the correspondingly calibrated Base retains a positive paired-bootstrap
95% CI. A failed candidate falls directly back to Base. The script never
searches for a second positive candidate and never reads the test split.
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


EXPERIMENT_NAME = "Binary Adaptive Decision-Boundary Diagnostic"
EXPERIMENT_ID = "binary_adaptive_decision_boundary_diagnostic"
LOCK_VERSION = "binary-adaptive-diagnostic-v1"
PROTOCOL_SEED = 20260713
BOOTSTRAP_REPS = 10_000
OOF_FOLDS = 5
EXPECTED_REPEATS = 50

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
class FitDiagnostics:
    macro_f1_at_0_5: float
    macro_f1_at_oof_threshold: float
    auroc: float
    oof_threshold: float
    dev_predicted_positive_rate_at_0_5: float
    dev_predicted_positive_rate_at_oof_threshold: float
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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def verify_simple_lock(root: Path, lock_name: str) -> dict[str, Any]:
    lock_path = root / lock_name
    if not lock_path.is_file():
        raise FileNotFoundError(f"Missing lock: {lock_path}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    files = lock.get("files")
    if not isinstance(files, dict):
        raise RuntimeError(f"Invalid files map in {lock_path}")
    for relative, expected in files.items():
        path = root / relative
        if not path.is_file():
            # The adaptive lock has only portable analysis artifacts. Weight-grid
            # SQLite is verified by the earlier protocol-freezing stage.
            if relative == "synthetic_weight_status.sqlite3":
                continue
            raise FileNotFoundError(f"Locked file is missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"Hash mismatch for {path}: expected={expected}, actual={actual}"
            )
    return lock


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def load_audit(path: Path) -> dict[tuple[str, int, float], dict[str, Any]]:
    result: dict[tuple[str, int, float], dict[str, Any]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["cell_id"]), int(row["repeat_seed"]), float(row["ratio"]))
            if key in result:
                raise RuntimeError(f"Duplicate weight-selection audit key at line {line_number}: {key}")
            if not bool(row.get("selection_match")):
                raise RuntimeError(f"Selection mismatch in audit at line {line_number}: {key}")
            selected = [int(value) for value in row["selected_global_generation_indices"]]
            if len(selected) != int(row["selected_n"]):
                raise RuntimeError(f"Selected-index count mismatch for {key}")
            result[key] = row
    return result


def load_preselection(adaptive_root: Path) -> pd.DataFrame:
    path = adaptive_root / "adaptive_preselection.csv"
    frame = pd.read_csv(path)
    required = {
        "cell_id",
        "selected_arm",
        "selected_ratio",
        "selected_synthetic_weight",
        "provisional_policy_state",
        "guard_positive_macro_f1_ci",
        "guard_local_stability",
        "guard_auroc_noninferiority",
        "diagnostic_required",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"adaptive_preselection.csv missing columns: {sorted(missing)}")
    if frame["cell_id"].duplicated().any():
        raise RuntimeError("adaptive_preselection.csv contains duplicate cell_id values")
    return frame


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


def discover_dev_embedding_files(
    eval_embedding_root: Path,
    cell_id: str,
) -> tuple[Path, Path]:
    arrays = []
    for path in eval_embedding_root.rglob("*.npy"):
        normalized = path.as_posix().lower()
        if cell_id.lower() in normalized and "dev" in normalized:
            arrays.append(path)
    if len(arrays) != 1:
        raise RuntimeError(
            f"{cell_id}: expected exactly one development embedding array under "
            f"{eval_embedding_root}, found {[str(path) for path in arrays]}. "
            "Pass a root whose development artifacts are unambiguous and separate from test."
        )
    array_path = arrays[0]

    manifests = []
    for path in eval_embedding_root.rglob("*.csv"):
        normalized = path.as_posix().lower()
        if cell_id.lower() in normalized and "dev" in normalized and "manifest" in normalized:
            manifests.append(path)
    if len(manifests) != 1:
        # Prefer a same-stem manifest when generic discovery is ambiguous.
        same_stem = [
            path
            for path in eval_embedding_root.rglob("*.csv")
            if path.stem in {
                array_path.stem + "_manifest",
                array_path.stem.replace("_dev", "_dev_manifest"),
                array_path.stem.replace("dev", "dev_manifest"),
            }
        ]
        manifests = same_stem
    if len(manifests) != 1:
        raise RuntimeError(
            f"{cell_id}: expected exactly one development embedding manifest under "
            f"{eval_embedding_root}, found {[str(path) for path in manifests]}"
        )
    return array_path, manifests[0]


def load_dev_embeddings(
    prepared_root: Path,
    eval_embedding_root: Path,
    cell_id: str,
) -> tuple[np.ndarray, np.ndarray, list[str], Path, Path]:
    dev_path = prepared_root / cell_id / "dev.csv"
    if not dev_path.is_file():
        raise FileNotFoundError(f"Missing development split: {dev_path}")
    dev = pd.read_csv(dev_path)
    if "label" not in dev.columns:
        raise ValueError(f"{dev_path} missing required column: label")
    labels = pd.to_numeric(dev["label"], errors="raise").astype(int).to_numpy()
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError(f"{cell_id}: development labels are not binary 0/1")

    array_path, manifest_path = discover_dev_embedding_files(eval_embedding_root, cell_id)
    array = np.load(array_path, mmap_mode="r")
    manifest = pd.read_csv(manifest_path)
    if "embedding_row_index" not in manifest.columns:
        raise ValueError(f"{manifest_path} must contain embedding_row_index")
    if "split" in manifest.columns:
        manifest = manifest[manifest["split"].astype(str).str.lower() == "dev"].copy()

    manifest_id_column = (
        "row_id" if "row_id" in manifest.columns
        else "real_row_id" if "real_row_id" in manifest.columns
        else None
    )
    dev_id_column = next(
        (name for name in ("row_id", "real_row_id", "id", "example_id", "uid") if name in dev.columns),
        None,
    )

    if manifest_id_column is not None and dev_id_column is not None:
        manifest[manifest_id_column] = manifest[manifest_id_column].astype(str)
        dev_ids = dev[dev_id_column].astype(str).tolist()
        mapping = {
            row_id: int(index)
            for row_id, index in zip(
                manifest[manifest_id_column], manifest["embedding_row_index"], strict=True
            )
        }
        missing_ids = [row_id for row_id in dev_ids if row_id not in mapping]
        if missing_ids:
            raise RuntimeError(
                f"{cell_id}: development embedding rows missing: {missing_ids[:20]}"
            )
        indices = np.asarray([mapping[row_id] for row_id in dev_ids], dtype=np.int64)
    else:
        # Prepared dev CSVs in this project do not consistently carry row_id.
        # The dev embedding job writes rows in source CSV order, so use positional
        # alignment only after strict cardinality/index/optional-label checks.
        manifest = manifest.sort_values("embedding_row_index", kind="stable").reset_index(drop=True)
        if len(manifest) != len(dev):
            raise RuntimeError(
                f"{cell_id}: cannot use positional dev alignment because dev.csv has "
                f"{len(dev)} rows but the dev embedding manifest has {len(manifest)} rows"
            )
        indices = pd.to_numeric(
            manifest["embedding_row_index"], errors="raise"
        ).astype(int).to_numpy()
        if len(np.unique(indices)) != len(indices):
            raise RuntimeError(f"{cell_id}: duplicate development embedding indices")
        if "label" in manifest.columns:
            manifest_labels = pd.to_numeric(
                manifest["label"], errors="raise"
            ).astype(int).to_numpy()
            if not np.array_equal(manifest_labels, labels):
                raise RuntimeError(
                    f"{cell_id}: positional dev alignment failed label-order validation"
                )
        dev_ids = [f"dev_position_{index}" for index in range(len(dev))]

    if len(indices) == 0 or indices.min() < 0 or indices.max() >= array.shape[0]:
        raise RuntimeError(f"{cell_id}: development embedding index out of range")
    vectors = np.asarray(array[indices], dtype=np.float32)
    return vectors, labels, dev_ids, array_path, manifest_path


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
    manifest = pd.read_csv(manifest_path)
    required = {"embedding_row_index", "record_type", "real_row_id", "global_generation_index"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"{manifest_path} missing columns: {sorted(missing)}")

    real: dict[str, int] = {}
    synthetic: dict[int, int] = {}
    for row in manifest.itertuples(index=False):
        index = int(getattr(row, "embedding_row_index"))
        record_type = str(getattr(row, "record_type"))
        if record_type == "real":
            key = str(getattr(row, "real_row_id"))
            if key in real:
                raise RuntimeError(f"Duplicate real embedding key {cell_id}/{key}")
            real[key] = index
        elif record_type == "synthetic":
            raw = getattr(row, "global_generation_index")
            key = int(float(raw))
            if key in synthetic:
                raise RuntimeError(f"Duplicate synthetic embedding key {cell_id}/{key}")
            synthetic[key] = index
    return array, real, synthetic, array_path, manifest_path


def logistic_regression(repeat_seed: int) -> LogisticRegression:
    return LogisticRegression(random_state=repeat_seed, **CLASSIFIER_CONFIG)


def macro_f1_from_counts(tp: int, fp: int, fn: int, tn: int) -> float:
    positive_denominator = 2 * tp + fp + fn
    negative_denominator = 2 * tn + fp + fn
    positive = 0.0 if positive_denominator == 0 else (2.0 * tp) / positive_denominator
    negative = 0.0 if negative_denominator == 0 else (2.0 * tn) / negative_denominator
    return (positive + negative) / 2.0


def choose_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Maximize Macro-F1 on OOF predictions with deterministic tie-breaking."""
    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    order = np.argsort(-probabilities, kind="stable")
    sorted_probabilities = probabilities[order]
    sorted_labels = y_true[order]
    total_positive = int(sorted_labels.sum())
    total_negative = int(sorted_labels.size - total_positive)

    candidates: list[tuple[float, float]] = []
    # Predict none as positive.
    threshold_none = float(np.nextafter(sorted_probabilities[0], np.inf))
    candidates.append((threshold_none, macro_f1_from_counts(0, 0, total_positive, total_negative)))

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

    # Always expose the conventional threshold as a candidate.
    prediction = probabilities >= 0.5
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    candidates.append((0.5, macro_f1_from_counts(int(tp), int(fp), int(fn), int(tn))))

    best_score = max(score for _, score in candidates)
    tied = [
        threshold
        for threshold, score in candidates
        if math.isclose(score, best_score, rel_tol=0.0, abs_tol=1e-15)
    ]
    # Prefer the least disruptive threshold, then the lower threshold.
    return min(tied, key=lambda threshold: (abs(threshold - 0.5), threshold))


def fit_with_training_only_threshold(
    x_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    x_dev: np.ndarray,
    y_dev: np.ndarray,
    repeat_seed: int,
    condition_name: str,
) -> FitDiagnostics:
    y_train = np.asarray(y_train, dtype=np.int64)
    sample_weight = np.asarray(sample_weight, dtype=np.float64)
    if x_train.shape[0] != y_train.size or y_train.size != sample_weight.size:
        raise ValueError("Training vectors, labels, and sample weights have different lengths")
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
    any_warning = False
    for train_index, validation_index in splitter.split(x_train, y_train):
        model = logistic_regression(repeat_seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(
                x_train[train_index],
                y_train[train_index],
                sample_weight=sample_weight[train_index],
            )
        any_warning |= any(issubclass(item.category, ConvergenceWarning) for item in caught)
        oof[validation_index] = model.predict_proba(x_train[validation_index])[:, 1]

    threshold = choose_threshold(y_train, oof)
    model = logistic_regression(repeat_seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x_train, y_train, sample_weight=sample_weight)
    final_warning = any(issubclass(item.category, ConvergenceWarning) for item in caught)
    any_warning |= final_warning

    train_probabilities = model.predict_proba(x_train)[:, 1]
    dev_probabilities = model.predict_proba(x_dev)[:, 1]
    dev_prediction_0_5 = (dev_probabilities >= 0.5).astype(int)
    dev_prediction_threshold = (dev_probabilities >= threshold).astype(int)
    tn_05, fp_05, fn_05, tp_05 = confusion_matrix(
        y_dev, dev_prediction_0_5, labels=[0, 1]
    ).ravel()
    tn_thr, fp_thr, fn_thr, tp_thr = confusion_matrix(
        y_dev, dev_prediction_threshold, labels=[0, 1]
    ).ravel()

    n_iter = int(np.max(model.n_iter_))
    return FitDiagnostics(
        macro_f1_at_0_5=float(f1_score(y_dev, dev_prediction_0_5, average="macro", zero_division=0)),
        macro_f1_at_oof_threshold=float(
            f1_score(y_dev, dev_prediction_threshold, average="macro", zero_division=0)
        ),
        auroc=float(roc_auc_score(y_dev, dev_probabilities)),
        oof_threshold=float(threshold),
        dev_predicted_positive_rate_at_0_5=float(dev_prediction_0_5.mean()),
        dev_predicted_positive_rate_at_oof_threshold=float(dev_prediction_threshold.mean()),
        train_predicted_positive_rate_at_0_5=float((train_probabilities >= 0.5).mean()),
        intercept=float(np.ravel(model.intercept_)[0]),
        coefficient_l2_norm=float(np.linalg.norm(model.coef_)),
        n_iter=n_iter,
        converged=bool(n_iter < int(CLASSIFIER_CONFIG["max_iter"])),
        convergence_warning=any_warning,
        tn_at_0_5=int(tn_05),
        fp_at_0_5=int(fp_05),
        fn_at_0_5=int(fn_05),
        tp_at_0_5=int(tp_05),
        tn_at_oof_threshold=int(tn_thr),
        fp_at_oof_threshold=int(fp_thr),
        fn_at_oof_threshold=int(fn_thr),
        tp_at_oof_threshold=int(tp_thr),
    )


def paired_bootstrap(values: np.ndarray, *key: object) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size != EXPECTED_REPEATS:
        raise ValueError(f"Expected {EXPECTED_REPEATS} paired values, got {values.size}")
    rng = np.random.default_rng(stable_seed(*key))
    indices = rng.integers(0, values.size, size=(BOOTSTRAP_REPS, values.size))
    means = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "se": float(values.std(ddof=1) / math.sqrt(values.size)),
        "ci_lower": float(np.quantile(means, 0.025)),
        "ci_upper": float(np.quantile(means, 0.975)),
    }


def build_condition(
    condition: str,
    ratio: float,
    synthetic_weight: float,
    repetition: Mapping[str, Any],
    audit_row: Mapping[str, Any],
    bank_labels: Mapping[str, int],
    generation_labels: Mapping[int, int],
    real_index: Mapping[str, int],
    synthetic_index: Mapping[int, int],
    embedding_array: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    base_ids = [str(value) for value in repetition["base_row_ids"]]
    n_add = int(round(ratio * int(repetition["base_n"])))
    if condition == "base":
        real_ids = base_ids
        synthetic_ids: list[int] = []
        synthetic_sample_weight = 0.0
    elif condition == "matched_all_real":
        real_ids = base_ids + [
            str(value) for value in repetition["additional_real_master_order"][:n_add]
        ]
        synthetic_ids = []
        synthetic_sample_weight = 0.0
    elif condition in {"naive_hybrid", "adaptive_hybrid"}:
        real_ids = base_ids
        synthetic_ids = [int(value) for value in audit_row["selected_global_generation_indices"]]
        if len(synthetic_ids) != n_add:
            raise RuntimeError(
                f"Selected synthetic count {len(synthetic_ids)} != n_add {n_add}"
            )
        synthetic_sample_weight = 1.0 if condition == "naive_hybrid" else synthetic_weight
    else:
        raise ValueError(f"Unknown condition: {condition}")

    missing_real = [row_id for row_id in real_ids if row_id not in real_index]
    missing_synthetic = [index for index in synthetic_ids if index not in synthetic_index]
    if missing_real or missing_synthetic:
        raise RuntimeError(
            f"Missing embeddings: real={missing_real[:10]}, synthetic={missing_synthetic[:10]}"
        )
    vector_indices = [real_index[row_id] for row_id in real_ids] + [
        synthetic_index[index] for index in synthetic_ids
    ]
    labels = [bank_labels[row_id] for row_id in real_ids] + [
        generation_labels[index] for index in synthetic_ids
    ]
    weights = [1.0] * len(real_ids) + [synthetic_sample_weight] * len(synthetic_ids)
    vectors = np.asarray(embedding_array[np.asarray(vector_indices, dtype=np.int64)], dtype=np.float32)
    return (
        vectors,
        np.asarray(labels, dtype=np.int64),
        np.asarray(weights, dtype=np.float64),
        len(real_ids),
        len(synthetic_ids),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        required=True,
        help="BinaryMatchedSizeExperiment root containing Embeddings/Qwen3",
    )
    parser.add_argument("--eval-embedding-root", type=Path, required=True)
    parser.add_argument("--weight-grid-root", type=Path, required=True)
    parser.add_argument("--adaptive-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    prepared_root = args.prepared_root.resolve()
    manifest_root = args.manifest_root.resolve()
    experiment_root = args.experiment_root.resolve()
    eval_embedding_root = args.eval_embedding_root.resolve()
    weight_grid_root = args.weight_grid_root.resolve()
    adaptive_root = args.adaptive_root.resolve()
    output_root = args.output_root.resolve()

    if "test" in eval_embedding_root.as_posix().lower():
        raise RuntimeError("Development diagnostic refuses an eval root whose path contains 'test'")
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    adaptive_lock = verify_simple_lock(adaptive_root, "BINARY_ADAPTIVE_PROTOCOL_LOCK.json")
    if adaptive_lock.get("test_data_accessed") is not False:
        raise RuntimeError("Adaptive protocol lock does not certify test_data_accessed=false")
    weight_lock = verify_simple_lock(weight_grid_root, "SYNTHETIC_WEIGHT_GRID_LOCK.json")
    preselection = load_preselection(adaptive_root)
    audit_path = weight_grid_root / "weight_selection_audit.jsonl.gz"
    audit = load_audit(audit_path)

    generation_plan_path = manifest_root / "stage_a_generation_plan.csv"
    generation_plan = pd.read_csv(generation_plan_path)
    generation_labels = {
        int(index): int(label)
        for index, label in zip(
            generation_plan["global_generation_index"],
            generation_plan["label"],
            strict=True,
        )
    }

    repeat_rows: list[dict[str, Any]] = []
    artifact_hashes: dict[str, str] = {
        "adaptive_protocol_lock": sha256_file(
            adaptive_root / "BINARY_ADAPTIVE_PROTOCOL_LOCK.json"
        ),
        "weight_grid_lock": sha256_file(
            weight_grid_root / "SYNTHETIC_WEIGHT_GRID_LOCK.json"
        ),
        "weight_selection_audit": sha256_file(audit_path),
        "generation_plan": sha256_file(generation_plan_path),
    }

    pending_cells = preselection[
        preselection["diagnostic_required"].map(bool_value)
    ].copy()

    for selected in pending_cells.itertuples(index=False):
        cell_id = str(selected.cell_id)
        ratio = float(selected.selected_ratio)
        synthetic_weight = float(selected.selected_synthetic_weight)
        if ratio <= 0 or synthetic_weight <= 0:
            raise RuntimeError(f"{cell_id}: diagnostic candidate must have positive ratio and weight")

        bank_path = manifest_root / cell_id / "experiment_bank.csv"
        repetitions_path = manifest_root / cell_id / "repetitions.jsonl"
        bank = pd.read_csv(bank_path)
        bank["row_id"] = bank["row_id"].astype(str)
        bank_labels = {
            row_id: int(label)
            for row_id, label in zip(bank["row_id"], bank["label"], strict=True)
        }
        repetitions = {
            int(row["repeat_seed"]): row for row in read_jsonl(repetitions_path)
        }
        if len(repetitions) != EXPECTED_REPEATS:
            raise RuntimeError(f"{cell_id}: expected {EXPECTED_REPEATS} repetition records")

        embedding_array, real_index, synthetic_index, train_array_path, train_manifest_path = (
            load_train_embedding_maps(experiment_root, cell_id)
        )
        x_dev, y_dev, _dev_ids, dev_array_path, dev_manifest_path = load_dev_embeddings(
            prepared_root, eval_embedding_root, cell_id
        )
        artifact_hashes[f"{cell_id}/experiment_bank"] = sha256_file(bank_path)
        artifact_hashes[f"{cell_id}/repetitions"] = sha256_file(repetitions_path)
        artifact_hashes[f"{cell_id}/train_embedding_array"] = sha256_file(train_array_path)
        artifact_hashes[f"{cell_id}/train_embedding_manifest"] = sha256_file(train_manifest_path)
        artifact_hashes[f"{cell_id}/dev_embedding_array"] = sha256_file(dev_array_path)
        artifact_hashes[f"{cell_id}/dev_embedding_manifest"] = sha256_file(dev_manifest_path)
        artifact_hashes[f"{cell_id}/dev_split"] = sha256_file(
            prepared_root / cell_id / "dev.csv"
        )

        for repeat_seed in sorted(repetitions):
            repetition = repetitions[repeat_seed]
            audit_key = (cell_id, repeat_seed, ratio)
            if audit_key not in audit:
                raise RuntimeError(f"Missing selection audit row: {audit_key}")
            audit_row = audit[audit_key]
            condition_diagnostics: dict[str, FitDiagnostics] = {}
            condition_sizes: dict[str, tuple[int, int, float]] = {}
            for condition in (
                "base",
                "matched_all_real",
                "naive_hybrid",
                "adaptive_hybrid",
            ):
                x_train, y_train, sample_weight, n_real, n_synthetic = build_condition(
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
                diagnostics = fit_with_training_only_threshold(
                    x_train,
                    y_train,
                    sample_weight,
                    x_dev,
                    y_dev,
                    repeat_seed,
                    condition,
                )
                condition_diagnostics[condition] = diagnostics
                condition_sizes[condition] = (
                    n_real,
                    n_synthetic,
                    float(sample_weight.sum()),
                )

            base = condition_diagnostics["base"]
            matched = condition_diagnostics["matched_all_real"]
            naive = condition_diagnostics["naive_hybrid"]
            adaptive = condition_diagnostics["adaptive_hybrid"]
            row: dict[str, Any] = {
                "experiment_name": EXPERIMENT_NAME,
                "experiment_id": EXPERIMENT_ID,
                "protocol_version": LOCK_VERSION,
                "cell_id": cell_id,
                "repeat_seed": repeat_seed,
                "ratio": ratio,
                "synthetic_weight": synthetic_weight,
                "selection_sha256": str(audit_row["selection_sha256"]),
            }
            for condition, diagnostics in condition_diagnostics.items():
                prefix = condition
                row.update({f"{prefix}_{key}": value for key, value in asdict(diagnostics).items()})
                n_real, n_synthetic, sum_weight = condition_sizes[condition]
                row[f"{prefix}_n_real"] = n_real
                row[f"{prefix}_n_synthetic"] = n_synthetic
                row[f"{prefix}_sum_sample_weight"] = sum_weight

            row.update(
                {
                    "adaptive_minus_base_macro_f1_at_0_5": adaptive.macro_f1_at_0_5
                    - base.macro_f1_at_0_5,
                    "adaptive_minus_base_macro_f1_at_oof_threshold": adaptive.macro_f1_at_oof_threshold
                    - base.macro_f1_at_oof_threshold,
                    "adaptive_minus_base_auroc": adaptive.auroc - base.auroc,
                    "adaptive_minus_naive_macro_f1_at_oof_threshold": adaptive.macro_f1_at_oof_threshold
                    - naive.macro_f1_at_oof_threshold,
                    "adaptive_minus_matched_macro_f1_at_oof_threshold": adaptive.macro_f1_at_oof_threshold
                    - matched.macro_f1_at_oof_threshold,
                    "adaptive_minus_base_intercept": adaptive.intercept - base.intercept,
                    "adaptive_minus_base_dev_positive_rate_at_0_5": adaptive.dev_predicted_positive_rate_at_0_5
                    - base.dev_predicted_positive_rate_at_0_5,
                }
            )
            repeat_rows.append(row)
            print(
                f"{cell_id} repeat={repeat_seed} completed | "
                f"calibrated delta={row['adaptive_minus_base_macro_f1_at_oof_threshold']:+.6f}"
            )

        del embedding_array

    repeat_results_path = output_root / "decision_boundary_repeat_results.csv"
    if repeat_rows:
        write_csv(repeat_results_path, repeat_rows, list(repeat_rows[0].keys()))
    else:
        write_csv(
            repeat_results_path,
            [],
            [
                "experiment_name",
                "experiment_id",
                "protocol_version",
                "cell_id",
                "repeat_seed",
                "ratio",
                "synthetic_weight",
            ],
        )

    repeat_frame = pd.DataFrame(repeat_rows)
    summary_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    pending_by_cell = {str(row.cell_id): row for row in pending_cells.itertuples(index=False)}

    for selected in preselection.itertuples(index=False):
        cell_id = str(selected.cell_id)
        diagnostic_required = bool_value(selected.diagnostic_required)
        if not diagnostic_required:
            final_rows.append(
                {
                    "cell_id": cell_id,
                    "augmentation": "OFF",
                    "selected_ratio": 0.0,
                    "selected_synthetic_weight": 0.0,
                    "final_training_arm": "base",
                    "decision_reason": str(selected.provisional_policy_state),
                    "diagnostic_macro_f1_ci_lower": None,
                    "diagnostic_macro_f1_ci_upper": None,
                    "diagnostic_auroc_ci_lower": None,
                    "test_evaluation_allowed": True,
                }
            )
            continue

        cell = repeat_frame[repeat_frame["cell_id"] == cell_id].sort_values("repeat_seed")
        if len(cell) != EXPECTED_REPEATS:
            raise RuntimeError(f"{cell_id}: diagnostic repeat count is {len(cell)}")
        default_macro = paired_bootstrap(
            cell["adaptive_minus_base_macro_f1_at_0_5"].to_numpy(),
            cell_id,
            "macro_f1_at_0_5",
        )
        calibrated_macro = paired_bootstrap(
            cell["adaptive_minus_base_macro_f1_at_oof_threshold"].to_numpy(),
            cell_id,
            "macro_f1_at_oof_threshold",
        )
        auroc = paired_bootstrap(
            cell["adaptive_minus_base_auroc"].to_numpy(),
            cell_id,
            "auroc",
        )
        intercept = paired_bootstrap(
            cell["adaptive_minus_base_intercept"].to_numpy(),
            cell_id,
            "intercept",
        )
        positive_rate = paired_bootstrap(
            cell["adaptive_minus_base_dev_positive_rate_at_0_5"].to_numpy(),
            cell_id,
            "positive_rate",
        )
        threshold_guard = calibrated_macro["ci_lower"] > 0.0
        prior_guards = (
            bool_value(selected.guard_positive_macro_f1_ci)
            and bool_value(selected.guard_local_stability)
            and bool_value(selected.guard_auroc_noninferiority)
        )
        augmentation_on = bool(prior_guards and threshold_guard)
        decision_reason = (
            "all_development_safety_guards_passed"
            if augmentation_on
            else "training_only_threshold_diagnostic_failed_fallback_to_base"
        )
        final_rows.append(
            {
                "cell_id": cell_id,
                "augmentation": "ON" if augmentation_on else "OFF",
                "selected_ratio": float(selected.selected_ratio) if augmentation_on else 0.0,
                "selected_synthetic_weight": float(selected.selected_synthetic_weight)
                if augmentation_on
                else 0.0,
                "final_training_arm": "adaptive_hybrid" if augmentation_on else "base",
                "decision_reason": decision_reason,
                "diagnostic_macro_f1_ci_lower": calibrated_macro["ci_lower"],
                "diagnostic_macro_f1_ci_upper": calibrated_macro["ci_upper"],
                "diagnostic_auroc_ci_lower": auroc["ci_lower"],
                "test_evaluation_allowed": True,
            }
        )
        summary_rows.append(
            {
                "cell_id": cell_id,
                "ratio": float(selected.selected_ratio),
                "synthetic_weight": float(selected.selected_synthetic_weight),
                "default_threshold_delta_mean": default_macro["mean"],
                "default_threshold_delta_ci_lower": default_macro["ci_lower"],
                "default_threshold_delta_ci_upper": default_macro["ci_upper"],
                "training_oof_threshold_delta_mean": calibrated_macro["mean"],
                "training_oof_threshold_delta_ci_lower": calibrated_macro["ci_lower"],
                "training_oof_threshold_delta_ci_upper": calibrated_macro["ci_upper"],
                "auroc_delta_mean": auroc["mean"],
                "auroc_delta_ci_lower": auroc["ci_lower"],
                "auroc_delta_ci_upper": auroc["ci_upper"],
                "intercept_delta_mean": intercept["mean"],
                "dev_positive_rate_delta_at_0_5_mean": positive_rate["mean"],
                "threshold_guard_passed": threshold_guard,
                "final_augmentation_on": augmentation_on,
            }
        )

    summary_path = output_root / "decision_boundary_summary.csv"
    selection_path = output_root / "adaptive_selection.csv"
    if summary_rows:
        write_csv(summary_path, summary_rows, list(summary_rows[0].keys()))
    else:
        write_csv(
            summary_path,
            [],
            [
                "cell_id",
                "ratio",
                "synthetic_weight",
                "training_oof_threshold_delta_mean",
                "training_oof_threshold_delta_ci_lower",
                "training_oof_threshold_delta_ci_upper",
                "final_augmentation_on",
            ],
        )
    write_csv(selection_path, final_rows, list(final_rows[0].keys()))

    final_protocol = {
        "protocol_version": LOCK_VERSION,
        "created_at": utc_now(),
        "experiment_name": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "development_only": True,
        "test_data_accessed": False,
        "threshold_selection": {
            "source": "training-only five-fold out-of-fold probabilities",
            "objective": "Macro-F1",
            "tie_break": "threshold nearest 0.5, then lower threshold",
            "development_labels_used_for_threshold_selection": False,
        },
        "classifier": CLASSIFIER_CONFIG,
        "final_adaptive_policy": final_rows,
        "prohibited_after_lock": [
            "changing ON/OFF decisions after viewing test results",
            "changing ratio or synthetic weight after viewing test results",
            "searching a second positive candidate for a cell that fell back to Base",
            "tuning a test threshold",
        ],
        "input_artifact_sha256": artifact_hashes,
    }
    final_protocol_path = output_root / "final_adaptive_protocol.json"
    write_json(final_protocol_path, final_protocol)

    lock_path = output_root / "ADAPTIVE_SELECTION_LOCK.json"
    files = {
        repeat_results_path.name: sha256_file(repeat_results_path),
        summary_path.name: sha256_file(summary_path),
        selection_path.name: sha256_file(selection_path),
        final_protocol_path.name: sha256_file(final_protocol_path),
    }
    lock = {
        "lock_version": LOCK_VERSION,
        "created_at": utc_now(),
        "experiment_name": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "status": "adaptive_policy_frozen_ready_for_one_shot_test",
        "test_data_accessed": False,
        "files": files,
        "input_lock_sha256": {
            "adaptive_protocol": sha256_file(
                adaptive_root / "BINARY_ADAPTIVE_PROTOCOL_LOCK.json"
            ),
            "synthetic_weight_grid": sha256_file(
                weight_grid_root / "SYNTHETIC_WEIGHT_GRID_LOCK.json"
            ),
        },
    }
    write_json(lock_path, lock)
    print(f"Final adaptive selection lock written: {lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())