"""Run the Binary Matched-Size development-set downstream grid."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
import sqlite3
import sys
import time
import warnings
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

_MODULE_ROOT = Path(__file__).resolve().parent
if str(_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULE_ROOT))

from embed_binary_qwen import read_valid_outputs
from generate_binary_candidates import (
    CELL_LANGUAGE,
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    PATH_ID,
    PROTOCOL_VERSION,
    TARGET_CELLS,
    canonical_json,
    load_and_validate_inputs,
    parse_int,
    progress_log,
    read_json,
    sha256_bytes,
    sha256_file,
    utc_now,
    verify_lock,
    write_csv,
    write_json,
)


BASE_N = 280
REPEAT_SEEDS = tuple(range(1000, 1050))
RATIOS = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0)
LOWER_BOUNDS = (0.10, 0.20, 0.30, 0.40)
UPPER_BOUNDS = (0.60, 0.70, 0.80, 0.90)
SELECTOR_VERSION = "parent-round-robin-sha256-v1"
DATABASE_SCHEMA_VERSION = "binary-development-grid-v1"
LOCK_NAME = "DEVELOPMENT_GRID_LOCK.json"
NORM_ATOL = 1e-5
PLANNED_BASE_FITS = 5 * 50
PLANNED_MATCHED_FITS = 5 * 50 * len(RATIOS)
PLANNED_HYBRID_CONDITIONS = 5 * 50 * len(RATIOS) * 17
PLANNED_TOTAL_FITS = (
    PLANNED_BASE_FITS + PLANNED_MATCHED_FITS + PLANNED_HYBRID_CONDITIONS
)

CLASSIFIER_CONFIG: dict[str, Any] = {
    "classifier": "sklearn.linear_model.LogisticRegression",
    "penalty": "l2",
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 2000,
    "tol": 1e-4,
    "fit_intercept": True,
    "class_weight": None,
    "random_state": "repeat_seed",
}

FIT_FIELDS = [
    "experiment_name", "experiment_id", "protocol_version", "cell_id", "language",
    "repeat_seed", "training_condition", "ratio", "similarity_condition_id",
    "similarity_enabled", "similarity_lower", "similarity_upper", "n_add",
    "total_train_n", "label_0_n", "label_1_n", "macro_f1", "auroc",
    "auroc_reason", "status", "fit_seconds", "n_iter", "converged",
    "convergence_warning", "exception_type", "exception_message",
]

PAIRED_FIELDS = [
    "experiment_name", "experiment_id", "protocol_version", "cell_id", "language",
    "repeat_seed", "ratio", "n_add", "total_train_n", "similarity_condition_id",
    "similarity_enabled", "similarity_lower", "similarity_upper", "feasible",
    "infeasible_reason", "base_macro_f1", "matched_all_real_macro_f1",
    "hybrid_macro_f1", "delta_synth_macro_f1", "delta_real_macro_f1",
    "g_real_macro_f1", "base_auroc", "matched_all_real_auroc", "hybrid_auroc",
    "delta_synth_auroc", "delta_real_auroc", "g_real_auroc",
    "base_label_0_n", "base_label_1_n", "additional_label_0_n",
    "additional_label_1_n", "synthetic_label_0_n", "synthetic_label_1_n",
    "eligible_synthetic_n", "selected_synthetic_n", "selected_parent_count",
    "selection_sha256", "base_status", "matched_status", "hybrid_status",
]

AUDIT_FIELDS = [
    "cell_id", "repeat_seed", "ratio", "similarity_condition_id", "lower", "upper",
    "target_n", "target_label_vector", "eligible_n", "eligible_label_vector",
    "selected_n", "selected_label_vector", "selected_global_generation_indices",
    "selected_parent_count", "selection_sha256", "feasible", "infeasible_reason",
]

_WORKER_CONTEXT: dict[str, dict[str, Any]] = {}


def similarity_conditions() -> tuple[dict[str, Any], ...]:
    conditions: list[dict[str, Any]] = [
        {
            "similarity_condition_id": "OFF",
            "similarity_enabled": False,
            "lower": None,
            "upper": None,
        }
    ]
    for lower in LOWER_BOUNDS:
        for upper in UPPER_BOUNDS:
            conditions.append(
                {
                    "similarity_condition_id": f"L{lower:.2f}_U{upper:.2f}",
                    "similarity_enabled": True,
                    "lower": lower,
                    "upper": upper,
                }
            )
    return tuple(conditions)


SIMILARITY_CONDITIONS = similarity_conditions()


def ratio_key(value: float) -> str:
    return f"{value:.2f}"


def condition_key(
    cell_id: str,
    repeat_seed: int,
    training_condition: str,
    ratio: float | None,
    similarity_condition_id: str | None,
) -> str:
    ratio_value = "NA" if ratio is None else ratio_key(ratio)
    similarity_value = similarity_condition_id or "NA"
    return (
        f"{cell_id}|{repeat_seed}|{training_condition}|"
        f"{ratio_value}|{similarity_value}"
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def vectors_are_normalized(array: Any) -> bool:
    import numpy as np

    for start in range(0, array.shape[0], 2_048):
        values = np.asarray(array[start : start + 2_048])
        if not np.all(np.isfinite(values)):
            return False
        norms = np.linalg.norm(values, axis=1)
        if not np.allclose(norms, 1.0, rtol=0.0, atol=NORM_ATOL):
            return False
    return True


def input_lock_hashes(experiment_root: Path) -> dict[str, str]:
    lock_paths = {
        "experiment": experiment_root / "BINARY_MATCHED_SIZE_EXPERIMENT_LOCK.json",
        "generation": experiment_root / "Generation" / "GENERATION_LOCK.json",
        "qwen": experiment_root / "Embeddings" / "Qwen3" / "QWEN_EMBEDDING_LOCK.json",
        "bge": experiment_root / "Embeddings" / "BGE_M3" / "BGE_EMBEDDING_LOCK.json",
        "qwen_eval": experiment_root / "Embeddings" / "Qwen3Eval" / "QWEN_EVAL_EMBEDDING_LOCK.json",
    }
    verify_lock(experiment_root, "BINARY_MATCHED_SIZE_EXPERIMENT_LOCK.json")
    verify_lock(experiment_root / "Generation", "GENERATION_LOCK.json")
    verify_lock(experiment_root / "Embeddings" / "Qwen3", "QWEN_EMBEDDING_LOCK.json")
    verify_lock(experiment_root / "Embeddings" / "BGE_M3", "BGE_EMBEDDING_LOCK.json")
    verify_lock(
        experiment_root / "Embeddings" / "Qwen3Eval",
        "QWEN_EVAL_EMBEDDING_LOCK.json",
    )
    return {name: sha256_file(path) for name, path in lock_paths.items()}


def validate_repetitions(
    manifest_root: Path,
    banks: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    expected_ratio_keys = [ratio_key(ratio) for ratio in RATIOS]
    for cell_id in TARGET_CELLS:
        bank_labels = {
            str(row["real_row_id"]): int(row["label_int"])
            for row in banks[cell_id]
        }
        if len(bank_labels) != 1_400:
            raise RuntimeError(f"{cell_id}: experiment bank does not contain 1,400 rows")
        path = manifest_root / cell_id / "repetitions.jsonl"
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RuntimeError(f"{cell_id}: repetition line {line_number} is not an object")
                rows.append(value)
        if len(rows) != 50:
            raise RuntimeError(f"{cell_id}: repetition count {len(rows)} != 50")
        if {parse_int(row.get("repeat_seed"), "repeat_seed") for row in rows} != set(REPEAT_SEEDS):
            raise RuntimeError(f"{cell_id}: repeat seeds are not exactly 1000..1049")
        normalized: list[dict[str, Any]] = []
        for row in sorted(rows, key=lambda item: int(item["repeat_seed"])):
            repeat_seed = parse_int(row.get("repeat_seed"), "repeat_seed")
            if row.get("cell_id") != cell_id or row.get("protocol_version") != PROTOCOL_VERSION:
                raise RuntimeError(f"{cell_id}/{repeat_seed}: repetition identity mismatch")
            if parse_int(row.get("base_n"), "base_n") != BASE_N:
                raise RuntimeError(f"{cell_id}/{repeat_seed}: base_n is not {BASE_N}")
            base = [str(value) for value in row.get("base_row_ids", [])]
            additional = [str(value) for value in row.get("additional_real_master_order", [])]
            if len(base) != BASE_N or len(set(base)) != BASE_N:
                raise RuntimeError(f"{cell_id}/{repeat_seed}: invalid base_row_ids")
            if len(additional) != 1_120 or len(set(additional)) != 1_120:
                raise RuntimeError(f"{cell_id}/{repeat_seed}: invalid additional master order")
            if set(base) & set(additional):
                raise RuntimeError(f"{cell_id}/{repeat_seed}: base/additional overlap")
            if set(base) | set(additional) != set(bank_labels):
                raise RuntimeError(f"{cell_id}/{repeat_seed}: repetition rows do not partition bank")
            observed_base_vector = Counter(str(bank_labels[row_id]) for row_id in base)
            expected_base_vector = {
                str(key): parse_int(value, "base_label_vector")
                for key, value in (row.get("base_label_vector") or {}).items()
            }
            if dict(observed_base_vector) != expected_base_vector:
                raise RuntimeError(f"{cell_id}/{repeat_seed}: base label vector mismatch")
            ratio_rows = row.get("ratios")
            if not isinstance(ratio_rows, list) or len(ratio_rows) != len(RATIOS):
                raise RuntimeError(f"{cell_id}/{repeat_seed}: invalid ratio rows")
            ratio_map: dict[str, dict[str, Any]] = {}
            for ratio_row in ratio_rows:
                ratio = float(ratio_row["ratio"])
                key = ratio_key(ratio)
                if key in ratio_map:
                    raise RuntimeError(f"{cell_id}/{repeat_seed}: duplicate ratio {ratio}")
                n_add = parse_int(ratio_row.get("n_add"), "n_add")
                expected_n_add = int(ratio * BASE_N)
                if n_add != expected_n_add:
                    raise RuntimeError(f"{cell_id}/{repeat_seed}/{ratio}: n_add mismatch")
                if parse_int(ratio_row.get("additional_real_prefix_n"), "additional_real_prefix_n") != n_add:
                    raise RuntimeError(f"{cell_id}/{repeat_seed}/{ratio}: prefix n mismatch")
                if parse_int(ratio_row.get("synthetic_target_n"), "synthetic_target_n") != n_add:
                    raise RuntimeError(f"{cell_id}/{repeat_seed}/{ratio}: synthetic n mismatch")
                if parse_int(ratio_row.get("total_train_n"), "total_train_n") != BASE_N + n_add:
                    raise RuntimeError(f"{cell_id}/{repeat_seed}/{ratio}: total n mismatch")
                additional_vector = {
                    str(label): parse_int(count, "additional_label_vector")
                    for label, count in ratio_row["additional_label_vector"].items()
                }
                synthetic_vector = {
                    str(label): parse_int(count, "synthetic_target_label_vector")
                    for label, count in ratio_row["synthetic_target_label_vector"].items()
                }
                observed_additional = Counter(
                    str(bank_labels[row_id]) for row_id in additional[:n_add]
                )
                if additional_vector != synthetic_vector or dict(observed_additional) != additional_vector:
                    raise RuntimeError(f"{cell_id}/{repeat_seed}/{ratio}: label vector mismatch")
                ratio_map[key] = {
                    "ratio": ratio,
                    "n_add": n_add,
                    "total_train_n": BASE_N + n_add,
                    "additional_label_vector": additional_vector,
                    "synthetic_target_label_vector": synthetic_vector,
                }
            if list(ratio_map) != expected_ratio_keys:
                raise RuntimeError(f"{cell_id}/{repeat_seed}: ratio grid/order mismatch")
            normalized.append(
                {
                    "cell_id": cell_id,
                    "repeat_seed": repeat_seed,
                    "base_row_ids": base,
                    "additional_real_master_order": additional,
                    "base_label_vector": expected_base_vector,
                    "ratios": ratio_map,
                }
            )
        result[cell_id] = normalized
    return result


def validate_and_load_embeddings(
    experiment_root: Path,
    banks: Mapping[str, Sequence[Mapping[str, Any]]],
    valid_by_cell: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, dict[str, Any]]:
    import numpy as np

    qwen_root = experiment_root / "Embeddings" / "Qwen3"
    eval_root = experiment_root / "Embeddings" / "Qwen3Eval"
    cosine_path = experiment_root / "Embeddings" / "BGE_M3" / "cosine_similarity.csv"
    cosine_rows = read_csv(cosine_path)
    cosine_by_cell: dict[str, dict[int, dict[str, Any]]] = {
        cell_id: {} for cell_id in TARGET_CELLS
    }
    for raw in cosine_rows:
        cell_id = str(raw.get("cell_id") or "")
        if cell_id not in cosine_by_cell:
            raise RuntimeError(f"cosine file contains unexpected cell: {cell_id}")
        global_index = parse_int(raw.get("global_generation_index"), "cosine.global_generation_index")
        if global_index in cosine_by_cell[cell_id]:
            raise RuntimeError(f"duplicate cosine global_generation_index: {global_index}")
        cosine = float(raw["cosine_similarity"])
        if not math.isfinite(cosine):
            raise RuntimeError(f"non-finite cosine: {global_index}")
        cosine_by_cell[cell_id][global_index] = {
            "real_row_id": str(raw.get("real_row_id") or ""),
            "cosine_similarity": cosine,
        }

    result: dict[str, dict[str, Any]] = {}
    seen_global: set[int] = set()
    for cell_id in TARGET_CELLS:
        bank_labels = {
            str(row["real_row_id"]): int(row["label_int"])
            for row in banks[cell_id]
        }
        valid_rows = list(valid_by_cell[cell_id])
        valid_by_global = {
            int(row["global_generation_index"]): row for row in valid_rows
        }
        if len(valid_by_global) != len(valid_rows):
            raise RuntimeError(f"{cell_id}: duplicate valid synthetic key")
        if seen_global & set(valid_by_global):
            raise RuntimeError(f"{cell_id}: global_generation_index repeats across cells")
        seen_global.update(valid_by_global)

        qwen_path = qwen_root / f"{cell_id}.npy"
        qwen_array = np.load(qwen_path, mmap_mode="r")
        qwen_manifest = read_csv(qwen_root / f"{cell_id}_manifest.csv")
        if qwen_array.dtype != np.float32 or qwen_array.ndim != 2 or qwen_array.shape[1] != 4096:
            raise RuntimeError(f"{cell_id}: invalid Qwen training array")
        if len(qwen_manifest) != qwen_array.shape[0] or not vectors_are_normalized(qwen_array):
            raise RuntimeError(f"{cell_id}: Qwen training manifest/array validation failed")
        real_indices: dict[str, int] = {}
        synthetic_indices: dict[int, int] = {}
        for expected_index, row in enumerate(qwen_manifest):
            row_index = parse_int(row.get("embedding_row_index"), "embedding_row_index")
            if row_index != expected_index:
                raise RuntimeError(f"{cell_id}: Qwen row index mismatch")
            record_type = str(row.get("record_type") or "")
            if record_type == "real":
                row_id = str(row.get("real_row_id") or "")
                if row_id not in bank_labels or row_id in real_indices:
                    raise RuntimeError(f"{cell_id}: invalid/duplicate Qwen real mapping")
                real_indices[row_id] = row_index
            elif record_type == "synthetic":
                global_index = parse_int(row.get("global_generation_index"), "global_generation_index")
                if global_index not in valid_by_global or global_index in synthetic_indices:
                    raise RuntimeError(f"{cell_id}: invalid/duplicate Qwen synthetic mapping")
                synthetic_indices[global_index] = row_index
            else:
                raise RuntimeError(f"{cell_id}: invalid Qwen record_type {record_type!r}")
        if len(real_indices) != 1_400 or set(real_indices) != set(bank_labels):
            raise RuntimeError(f"{cell_id}: Qwen real embedding count/mapping mismatch")
        if set(synthetic_indices) != set(valid_by_global):
            raise RuntimeError(f"{cell_id}: Qwen synthetic count/mapping mismatch")

        dev_path = eval_root / f"{cell_id}_dev.npy"
        dev_array = np.load(dev_path, mmap_mode="r")
        dev_manifest = read_csv(eval_root / f"{cell_id}_dev_manifest.csv")
        if dev_array.dtype != np.float32 or dev_array.ndim != 2 or dev_array.shape[1] != 4096:
            raise RuntimeError(f"{cell_id}: invalid Qwen dev array")
        if len(dev_manifest) != dev_array.shape[0] or not vectors_are_normalized(dev_array):
            raise RuntimeError(f"{cell_id}: Qwen dev manifest/array validation failed")
        dev_labels: list[int] = []
        for expected_index, row in enumerate(dev_manifest):
            if parse_int(row.get("embedding_row_index"), "dev.embedding_row_index") != expected_index:
                raise RuntimeError(f"{cell_id}: dev embedding index mismatch")
            if str(row.get("split") or "") != "dev":
                raise RuntimeError(f"{cell_id}: non-dev row in dev manifest")
            label = parse_int(row.get("label_int"), "dev.label_int")
            if label not in (0, 1):
                raise RuntimeError(f"{cell_id}: invalid dev label")
            dev_labels.append(label)

        cosine_map = cosine_by_cell[cell_id]
        if set(cosine_map) != set(valid_by_global):
            raise RuntimeError(f"{cell_id}: BGE cosine/valid synthetic mapping mismatch")
        synthetic_records: dict[int, dict[str, Any]] = {}
        for global_index, valid_row in valid_by_global.items():
            real_row_id = str(valid_row["real_row_id"])
            if real_row_id not in bank_labels:
                raise RuntimeError(f"{cell_id}: synthetic parent absent from bank")
            label_int = parse_int(valid_row.get("label_int"), "valid.label_int")
            expected_label = "negative" if label_int == 0 else "positive"
            if (
                label_int not in (0, 1)
                or str(valid_row.get("inherited_label")) != expected_label
                or str(valid_row.get("generated_label_raw")) != expected_label
            ):
                raise RuntimeError(f"{cell_id}/{global_index}: synthetic label mismatch")
            cosine_row = cosine_map[global_index]
            if cosine_row["real_row_id"] != real_row_id:
                raise RuntimeError(f"{cell_id}/{global_index}: cosine parent mismatch")
            synthetic_records[global_index] = {
                "global_generation_index": global_index,
                "real_row_id": real_row_id,
                "label_int": label_int,
                "qwen_embedding_row_index": synthetic_indices[global_index],
                "cosine_similarity": cosine_row["cosine_similarity"],
            }
        result[cell_id] = {
            "qwen_path": str(qwen_path),
            "dev_path": str(dev_path),
            "dev_labels": dev_labels,
            "bank_labels": bank_labels,
            "real_indices": real_indices,
            "synthetic_records": synthetic_records,
            "valid_synthetic_count": len(synthetic_records),
        }
        del qwen_array, dev_array
    if len(cosine_rows) != sum(len(valid_by_cell[cell_id]) for cell_id in TARGET_CELLS):
        raise RuntimeError("global cosine row count does not match valid generation count")
    return result


def candidate_hash(
    cell_id: str,
    repeat_seed: int,
    similarity_condition_id: str,
    label: int,
    global_generation_index: int,
) -> str:
    value = "|".join(
        (
            EXPERIMENT_ID,
            cell_id,
            str(repeat_seed),
            similarity_condition_id,
            str(label),
            str(global_generation_index),
        )
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def synthetic_master_orders(
    cell_id: str,
    repeat: Mapping[str, Any],
    condition: Mapping[str, Any],
    synthetic_records: Mapping[int, Mapping[str, Any]],
) -> tuple[dict[int, list[int]], dict[int, int]]:
    base_order = list(repeat["base_row_ids"])
    base_set = set(base_order)
    grouped: dict[int, dict[str, list[Mapping[str, Any]]]] = {
        0: defaultdict(list),
        1: defaultdict(list),
    }
    for record in synthetic_records.values():
        parent = str(record["real_row_id"])
        if parent not in base_set:
            continue
        cosine = float(record["cosine_similarity"])
        if condition["similarity_enabled"] and not (
            float(condition["lower"]) <= cosine <= float(condition["upper"])
        ):
            continue
        label = int(record["label_int"])
        grouped[label][parent].append(record)
    orders: dict[int, list[int]] = {0: [], 1: []}
    eligible_vector: dict[int, int] = {0: 0, 1: 0}
    for label in (0, 1):
        queues: dict[str, list[int]] = {}
        for parent in base_order:
            candidates = grouped[label].get(parent, [])
            candidates = sorted(
                candidates,
                key=lambda row: (
                    candidate_hash(
                        cell_id,
                        int(repeat["repeat_seed"]),
                        str(condition["similarity_condition_id"]),
                        label,
                        int(row["global_generation_index"]),
                    ),
                    int(row["global_generation_index"]),
                ),
            )
            if candidates:
                queues[parent] = [
                    int(row["global_generation_index"]) for row in candidates
                ]
        eligible_vector[label] = sum(len(queue) for queue in queues.values())
        offsets = {parent: 0 for parent in queues}
        while True:
            selected_this_round = 0
            for parent in base_order:
                queue = queues.get(parent)
                if queue is None:
                    continue
                offset = offsets[parent]
                if offset < len(queue):
                    orders[label].append(queue[offset])
                    offsets[parent] = offset + 1
                    selected_this_round += 1
            if selected_this_round == 0:
                break
        if len(orders[label]) != eligible_vector[label] or len(set(orders[label])) != len(orders[label]):
            raise AssertionError(f"{cell_id}: invalid synthetic master order for label {label}")
    return orders, eligible_vector


def select_hybrid(
    cell_id: str,
    repeat: Mapping[str, Any],
    ratio_row: Mapping[str, Any],
    condition: Mapping[str, Any],
    orders: Mapping[int, Sequence[int]],
    eligible_vector: Mapping[int, int],
    synthetic_records: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    target_vector = {
        int(label): int(count)
        for label, count in ratio_row["synthetic_target_label_vector"].items()
    }
    shortages = {
        label: target_vector[label] - len(orders[label])
        for label in (0, 1)
        if len(orders[label]) < target_vector[label]
    }
    feasible = not shortages
    selected_by_label = {
        label: list(orders[label][: target_vector[label]]) if feasible else []
        for label in (0, 1)
    }
    selected = selected_by_label[0] + selected_by_label[1]
    selected_vector = {label: len(selected_by_label[label]) for label in (0, 1)}
    selected_parents = {
        str(synthetic_records[index]["real_row_id"]) for index in selected
    }
    selection_sha256 = sha256_bytes(canonical_json(selected).encode("utf-8"))
    reason = None
    if shortages:
        reason = "insufficient_eligible_synthetic:" + canonical_json(shortages)
    if feasible:
        n_add = int(ratio_row["n_add"])
        if len(selected) != n_add:
            raise AssertionError("Hybrid synthetic count does not equal n_add")
        if BASE_N + len(selected) != int(ratio_row["total_train_n"]):
            raise AssertionError("Hybrid total size mismatch")
        if selected_vector != target_vector:
            raise AssertionError("Hybrid label vector mismatch")
    return {
        "cell_id": cell_id,
        "repeat_seed": int(repeat["repeat_seed"]),
        "ratio": float(ratio_row["ratio"]),
        "similarity_condition_id": condition["similarity_condition_id"],
        "lower": condition["lower"],
        "upper": condition["upper"],
        "target_n": int(ratio_row["n_add"]),
        "target_label_vector": {str(key): value for key, value in target_vector.items()},
        "eligible_n": sum(eligible_vector.values()),
        "eligible_label_vector": {str(key): int(value) for key, value in eligible_vector.items()},
        "selected_n": len(selected),
        "selected_label_vector": {str(key): value for key, value in selected_vector.items()},
        "selected_global_generation_indices": selected,
        "selected_parent_count": len(selected_parents),
        "selection_sha256": selection_sha256,
        "feasible": feasible,
        "infeasible_reason": reason,
    }


def metadata_contract(
    lock_hashes: Mapping[str, str],
    manifest_lock_sha256: str,
) -> dict[str, str]:
    values: dict[str, Any] = {
        "experiment_name": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "database_schema_version": DATABASE_SCHEMA_VERSION,
        "input_lock_sha256": dict(lock_hashes),
        "manifest_lock_sha256": manifest_lock_sha256,
        "classifier_config": CLASSIFIER_CONFIG,
        "grid_config": {
            "ratios": RATIOS,
            "lower_bounds": LOWER_BOUNDS,
            "upper_bounds": UPPER_BOUNDS,
            "similarity_conditions": SIMILARITY_CONDITIONS,
            "base_n": BASE_N,
            "repeat_seeds": REPEAT_SEEDS,
        },
        "selector_version": SELECTOR_VERSION,
    }
    return {key: canonical_json(value) for key, value in values.items()}


def open_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=60.0)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA busy_timeout=60000")
    connection.execute("PRAGMA synchronous=FULL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS fits (
            condition_key TEXT PRIMARY KEY,
            cell_id TEXT NOT NULL,
            language TEXT NOT NULL,
            repeat_seed INTEGER NOT NULL,
            training_condition TEXT NOT NULL,
            ratio REAL,
            similarity_condition_id TEXT,
            similarity_enabled INTEGER NOT NULL,
            similarity_lower REAL,
            similarity_upper REAL,
            n_add INTEGER NOT NULL,
            total_train_n INTEGER NOT NULL,
            label_0_n INTEGER NOT NULL,
            label_1_n INTEGER NOT NULL,
            macro_f1 REAL,
            auroc REAL,
            auroc_reason TEXT,
            status TEXT NOT NULL,
            fit_seconds REAL,
            n_iter INTEGER,
            converged INTEGER,
            convergence_warning TEXT,
            exception_type TEXT,
            exception_message TEXT,
            target_label_vector_json TEXT,
            eligible_n INTEGER,
            eligible_label_vector_json TEXT,
            selected_n INTEGER,
            selected_label_vector_json TEXT,
            selected_parent_count INTEGER,
            selection_sha256 TEXT,
            selected_indices_json TEXT,
            infeasible_reason TEXT,
            completed_at TEXT NOT NULL,
            UNIQUE(cell_id, repeat_seed, training_condition, ratio, similarity_condition_id)
        );
        """
    )
    return connection


def initialize_database(
    connection: sqlite3.Connection,
    contract: Mapping[str, str],
    resume: bool,
) -> None:
    existing = dict(connection.execute("SELECT key, value FROM metadata"))
    fit_count = int(connection.execute("SELECT COUNT(*) FROM fits").fetchone()[0])
    if resume:
        if not existing:
            raise RuntimeError("--resume requires an initialized downstream database")
        mismatches = [key for key, value in contract.items() if existing.get(key) != value]
        if mismatches:
            raise RuntimeError(f"resume metadata mismatch: {', '.join(sorted(mismatches))}")
        return
    if existing or fit_count:
        raise RuntimeError("downstream database already exists; use --resume")
    with connection:
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            [*contract.items(), ("created_at", canonical_json(utc_now()))],
        )


def terminal_keys(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            "SELECT condition_key FROM fits WHERE status IN "
            "('completed', 'completed_with_warning', 'infeasible')"
        )
    }


def fit_row_from_task(task: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "condition_key": task["condition_key"],
        "cell_id": task["cell_id"],
        "language": task["language"],
        "repeat_seed": task["repeat_seed"],
        "training_condition": task["training_condition"],
        "ratio": task.get("ratio"),
        "similarity_condition_id": task.get("similarity_condition_id"),
        "similarity_enabled": bool(task.get("similarity_enabled", False)),
        "similarity_lower": task.get("similarity_lower"),
        "similarity_upper": task.get("similarity_upper"),
        "n_add": task["n_add"],
        "total_train_n": task["total_train_n"],
        "label_0_n": task["label_0_n"],
        "label_1_n": task["label_1_n"],
        "macro_f1": result.get("macro_f1"),
        "auroc": result.get("auroc"),
        "auroc_reason": result.get("auroc_reason"),
        "status": result["status"],
        "fit_seconds": result.get("fit_seconds"),
        "n_iter": result.get("n_iter"),
        "converged": result.get("converged"),
        "convergence_warning": result.get("convergence_warning"),
        "exception_type": result.get("exception_type"),
        "exception_message": result.get("exception_message"),
        "target_label_vector": task.get("target_label_vector"),
        "eligible_n": task.get("eligible_n"),
        "eligible_label_vector": task.get("eligible_label_vector"),
        "selected_n": task.get("selected_n"),
        "selected_label_vector": task.get("selected_label_vector"),
        "selected_parent_count": task.get("selected_parent_count"),
        "selection_sha256": task.get("selection_sha256"),
        "selected_indices": task.get("selected_global_generation_indices"),
        "infeasible_reason": task.get("infeasible_reason"),
        "completed_at": utc_now(),
    }


def commit_fit(connection: sqlite3.Connection, row: Mapping[str, Any]) -> None:
    with connection:
        connection.execute(
            """INSERT OR REPLACE INTO fits(
                condition_key, cell_id, language, repeat_seed, training_condition,
                ratio, similarity_condition_id, similarity_enabled, similarity_lower,
                similarity_upper, n_add, total_train_n, label_0_n, label_1_n,
                macro_f1, auroc, auroc_reason, status, fit_seconds, n_iter, converged,
                convergence_warning, exception_type, exception_message,
                target_label_vector_json, eligible_n, eligible_label_vector_json,
                selected_n, selected_label_vector_json, selected_parent_count,
                selection_sha256, selected_indices_json, infeasible_reason, completed_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                row["condition_key"], row["cell_id"], row["language"], row["repeat_seed"],
                row["training_condition"], row["ratio"], row["similarity_condition_id"],
                int(row["similarity_enabled"]), row["similarity_lower"], row["similarity_upper"],
                row["n_add"], row["total_train_n"], row["label_0_n"], row["label_1_n"],
                row["macro_f1"], row["auroc"], row["auroc_reason"], row["status"],
                row["fit_seconds"], row["n_iter"],
                None if row["converged"] is None else int(row["converged"]),
                row["convergence_warning"], row["exception_type"], row["exception_message"],
                canonical_json(row["target_label_vector"]) if row["target_label_vector"] is not None else None,
                row["eligible_n"],
                canonical_json(row["eligible_label_vector"]) if row["eligible_label_vector"] is not None else None,
                row["selected_n"],
                canonical_json(row["selected_label_vector"]) if row["selected_label_vector"] is not None else None,
                row["selected_parent_count"], row["selection_sha256"],
                canonical_json(row["selected_indices"]) if row["selected_indices"] is not None else None,
                row["infeasible_reason"], row["completed_at"],
            ),
        )


def initialize_worker(context: Mapping[str, Mapping[str, Any]]) -> None:
    global _WORKER_CONTEXT
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    import numpy as np

    _WORKER_CONTEXT = {}
    for cell_id, cell in context.items():
        _WORKER_CONTEXT[cell_id] = {
            "qwen": np.load(cell["qwen_path"], mmap_mode="r"),
            "dev": np.load(cell["dev_path"], mmap_mode="r"),
            "dev_labels": np.asarray(cell["dev_labels"], dtype=np.int64),
        }


def execute_fit(task: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, roc_auc_score

    started = time.perf_counter()
    result: dict[str, Any] = {
        "status": "failed",
        "fit_seconds": None,
        "n_iter": None,
        "converged": None,
        "convergence_warning": None,
        "exception_type": None,
        "exception_message": None,
        "macro_f1": None,
        "auroc": None,
        "auroc_reason": None,
    }
    try:
        cell = _WORKER_CONTEXT[str(task["cell_id"])]
        train_indices = np.asarray(task["train_indices"], dtype=np.int64)
        y_train = np.asarray(task["train_labels"], dtype=np.int64)
        x_train = np.asarray(cell["qwen"][train_indices], dtype=np.float32)
        x_dev = cell["dev"]
        y_dev = cell["dev_labels"]
        classifier = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=2000,
            tol=1e-4,
            fit_intercept=True,
            class_weight=None,
            random_state=int(task["repeat_seed"]),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            classifier.fit(x_train, y_train)
        convergence_messages = [
            str(item.message)
            for item in caught
            if issubclass(item.category, ConvergenceWarning)
        ]
        y_pred = classifier.predict(x_dev)
        macro_f1 = float(
            f1_score(
                y_dev,
                y_pred,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        )
        auroc = None
        auroc_reason = None
        if len(np.unique(y_dev)) != 2:
            auroc_reason = "dev_does_not_contain_both_classes"
        elif 1 not in classifier.classes_:
            auroc_reason = "classifier_classes_missing_label_1"
        else:
            positive_column = int(np.flatnonzero(classifier.classes_ == 1)[0])
            probabilities = classifier.predict_proba(x_dev)[:, positive_column]
            auroc = float(roc_auc_score(y_dev, probabilities))
        result.update(
            status="completed_with_warning" if convergence_messages else "completed",
            n_iter=int(np.max(classifier.n_iter_)),
            converged=not convergence_messages,
            convergence_warning=" | ".join(convergence_messages) or None,
            macro_f1=macro_f1,
            auroc=auroc,
            auroc_reason=auroc_reason,
        )
    except Exception as exc:
        result.update(
            status="failed",
            exception_type=type(exc).__name__,
            exception_message=str(exc),
        )
    result["fit_seconds"] = time.perf_counter() - started
    return result


def task_stream(
    repetitions: Mapping[str, Sequence[Mapping[str, Any]]],
    data: Mapping[str, Mapping[str, Any]],
    skip_keys: set[str],
) -> Iterator[dict[str, Any]]:
    for cell_id in TARGET_CELLS:
        language = CELL_LANGUAGE[cell_id]
        real_indices = data[cell_id]["real_indices"]
        bank_labels = data[cell_id]["bank_labels"]
        synthetic_records = data[cell_id]["synthetic_records"]
        for repeat in repetitions[cell_id]:
            repeat_seed = int(repeat["repeat_seed"])
            base_ids = list(repeat["base_row_ids"])
            base_indices = [int(real_indices[row_id]) for row_id in base_ids]
            base_labels = [int(bank_labels[row_id]) for row_id in base_ids]
            base_vector = Counter(base_labels)
            base_key = condition_key(cell_id, repeat_seed, "base", None, None)
            if base_key not in skip_keys:
                yield {
                    "run_fit": True,
                    "condition_key": base_key,
                    "cell_id": cell_id,
                    "language": language,
                    "repeat_seed": repeat_seed,
                    "training_condition": "base",
                    "ratio": None,
                    "similarity_condition_id": None,
                    "similarity_enabled": False,
                    "similarity_lower": None,
                    "similarity_upper": None,
                    "n_add": 0,
                    "total_train_n": BASE_N,
                    "label_0_n": int(base_vector[0]),
                    "label_1_n": int(base_vector[1]),
                    "train_indices": base_indices,
                    "train_labels": base_labels,
                }

            for ratio in RATIOS:
                ratio_row = repeat["ratios"][ratio_key(ratio)]
                n_add = int(ratio_row["n_add"])
                additional_ids = list(repeat["additional_real_master_order"][:n_add])
                additional_labels = [int(bank_labels[row_id]) for row_id in additional_ids]
                additional_vector = Counter(additional_labels)
                expected_additional = {
                    int(label): int(count)
                    for label, count in ratio_row["additional_label_vector"].items()
                }
                if dict(additional_vector) != expected_additional:
                    raise AssertionError("Matched additional label vector mismatch")
                matched_indices = base_indices + [int(real_indices[row_id]) for row_id in additional_ids]
                matched_labels = base_labels + additional_labels
                matched_vector = Counter(matched_labels)
                if len(matched_indices) != BASE_N + n_add:
                    raise AssertionError("Matched All-real total size mismatch")
                matched_key = condition_key(cell_id, repeat_seed, "matched_all_real", ratio, None)
                if matched_key not in skip_keys:
                    yield {
                        "run_fit": True,
                        "condition_key": matched_key,
                        "cell_id": cell_id,
                        "language": language,
                        "repeat_seed": repeat_seed,
                        "training_condition": "matched_all_real",
                        "ratio": ratio,
                        "similarity_condition_id": None,
                        "similarity_enabled": False,
                        "similarity_lower": None,
                        "similarity_upper": None,
                        "n_add": n_add,
                        "total_train_n": BASE_N + n_add,
                        "label_0_n": int(matched_vector[0]),
                        "label_1_n": int(matched_vector[1]),
                        "train_indices": matched_indices,
                        "train_labels": matched_labels,
                    }

            for similarity in SIMILARITY_CONDITIONS:
                orders, eligible_vector = synthetic_master_orders(
                    cell_id,
                    repeat,
                    similarity,
                    synthetic_records,
                )
                for ratio in RATIOS:
                    ratio_row = repeat["ratios"][ratio_key(ratio)]
                    hybrid_key = condition_key(
                        cell_id,
                        repeat_seed,
                        "hybrid",
                        ratio,
                        str(similarity["similarity_condition_id"]),
                    )
                    if hybrid_key in skip_keys:
                        continue
                    selection = select_hybrid(
                        cell_id,
                        repeat,
                        ratio_row,
                        similarity,
                        orders,
                        eligible_vector,
                        synthetic_records,
                    )
                    target_vector = {
                        int(label): int(count)
                        for label, count in selection["target_label_vector"].items()
                    }
                    hybrid_vector = {
                        label: int(base_vector[label]) + target_vector[label]
                        for label in (0, 1)
                    }
                    common = {
                        "condition_key": hybrid_key,
                        "cell_id": cell_id,
                        "language": language,
                        "repeat_seed": repeat_seed,
                        "training_condition": "hybrid",
                        "ratio": ratio,
                        "similarity_condition_id": similarity["similarity_condition_id"],
                        "similarity_enabled": similarity["similarity_enabled"],
                        "similarity_lower": similarity["lower"],
                        "similarity_upper": similarity["upper"],
                        "n_add": int(ratio_row["n_add"]),
                        "total_train_n": int(ratio_row["total_train_n"]),
                        "label_0_n": hybrid_vector[0],
                        "label_1_n": hybrid_vector[1],
                        **selection,
                    }
                    if not selection["feasible"]:
                        yield {"run_fit": False, **common}
                        continue
                    selected = selection["selected_global_generation_indices"]
                    synthetic_indices = [
                        int(synthetic_records[index]["qwen_embedding_row_index"])
                        for index in selected
                    ]
                    synthetic_labels = [
                        int(synthetic_records[index]["label_int"])
                        for index in selected
                    ]
                    if Counter(synthetic_labels) != Counter(target_vector):
                        raise AssertionError("Hybrid selected label vector mismatch")
                    train_indices = base_indices + synthetic_indices
                    train_labels = base_labels + synthetic_labels
                    if len(train_indices) != int(ratio_row["total_train_n"]):
                        raise AssertionError("Hybrid training total size mismatch")
                    if Counter(train_labels) != Counter(
                        base_labels
                        + [
                            int(bank_labels[row_id])
                            for row_id in repeat["additional_real_master_order"][
                                : int(ratio_row["n_add"])
                            ]
                        ]
                    ):
                        raise AssertionError("Hybrid/Matched label vector mismatch")
                    yield {
                        "run_fit": True,
                        **common,
                        "train_indices": train_indices,
                        "train_labels": train_labels,
                    }


def infeasible_result() -> dict[str, Any]:
    return {
        "status": "infeasible",
        "fit_seconds": 0.0,
        "n_iter": None,
        "converged": None,
        "convergence_warning": None,
        "exception_type": None,
        "exception_message": None,
        "macro_f1": None,
        "auroc": None,
        "auroc_reason": None,
    }


def current_status(connection: sqlite3.Connection) -> dict[str, int]:
    values = dict(connection.execute("SELECT status, COUNT(*) FROM fits GROUP BY status"))
    return {str(key): int(value) for key, value in values.items()}


def run_grid(
    connection: sqlite3.Connection,
    repetitions: Mapping[str, Sequence[Mapping[str, Any]]],
    data: Mapping[str, Mapping[str, Any]],
    workers: int,
    progress_every: int,
) -> None:
    skip_keys = terminal_keys(connection)
    stream = task_stream(repetitions, data, skip_keys)
    worker_context = {
        cell_id: {
            "qwen_path": data[cell_id]["qwen_path"],
            "dev_path": data[cell_id]["dev_path"],
            "dev_labels": data[cell_id]["dev_labels"],
        }
        for cell_id in TARGET_CELLS
    }
    processed = len(skip_keys)
    started = time.perf_counter()

    def record(task: Mapping[str, Any], result: Mapping[str, Any]) -> None:
        nonlocal processed
        commit_fit(connection, fit_row_from_task(task, result))
        processed += 1
        if processed % progress_every == 0 or processed == PLANNED_TOTAL_FITS:
            elapsed = time.perf_counter() - started
            progress_log(
                f"development grid | terminal={processed}/{PLANNED_TOTAL_FITS} | "
                f"status={current_status(connection)} | elapsed_seconds={elapsed:.1f}"
            )

    if workers == 1:
        initialize_worker(worker_context)
        for task in stream:
            result = execute_fit(task) if task["run_fit"] else infeasible_result()
            record(task, result)
        return

    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=initialize_worker,
        initargs=(worker_context,),
    ) as executor:
        futures: dict[Any, dict[str, Any]] = {}
        exhausted = False
        while not exhausted or futures:
            while not exhausted and len(futures) < workers * 2:
                try:
                    task = next(stream)
                except StopIteration:
                    exhausted = True
                    break
                if not task["run_fit"]:
                    record(task, infeasible_result())
                    continue
                future = executor.submit(execute_fit, task)
                futures[future] = task
            if futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    task = futures.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {
                            "status": "failed",
                            "fit_seconds": None,
                            "n_iter": None,
                            "converged": None,
                            "convergence_warning": None,
                            "exception_type": type(exc).__name__,
                            "exception_message": str(exc),
                            "macro_f1": None,
                            "auroc": None,
                            "auroc_reason": None,
                        }
                    record(task, result)


def database_rows(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    connection.row_factory = sqlite3.Row
    rows = [dict(row) for row in connection.execute(
        "SELECT * FROM fits ORDER BY cell_id, repeat_seed, "
        "COALESCE(ratio, -1), COALESCE(similarity_condition_id, '')"
    )]
    connection.row_factory = None
    for row in rows:
        for source, target in (
            ("target_label_vector_json", "target_label_vector"),
            ("eligible_label_vector_json", "eligible_label_vector"),
            ("selected_label_vector_json", "selected_label_vector"),
            ("selected_indices_json", "selected_global_generation_indices"),
        ):
            value = row.pop(source)
            row[target] = json.loads(value) if value is not None else None
    return rows


def fit_export_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "experiment_name": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        **{field: row.get(field) for field in FIT_FIELDS if field not in {
            "experiment_name", "experiment_id", "protocol_version"
        }},
    }


def metric_delta(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def build_paired_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    base = {
        (row["cell_id"], int(row["repeat_seed"])): row
        for row in rows
        if row["training_condition"] == "base"
    }
    matched = {
        (row["cell_id"], int(row["repeat_seed"]), ratio_key(float(row["ratio"]))): row
        for row in rows
        if row["training_condition"] == "matched_all_real"
    }
    paired: list[dict[str, Any]] = []
    for hybrid in sorted(
        (row for row in rows if row["training_condition"] == "hybrid"),
        key=lambda row: (
            row["cell_id"],
            int(row["repeat_seed"]),
            float(row["ratio"]),
            str(row["similarity_condition_id"]),
        ),
    ):
        base_row = base.get((hybrid["cell_id"], int(hybrid["repeat_seed"])))
        matched_row = matched.get(
            (
                hybrid["cell_id"],
                int(hybrid["repeat_seed"]),
                ratio_key(float(hybrid["ratio"])),
            )
        )
        if base_row is None or matched_row is None:
            raise RuntimeError("paired result is missing base or matched result")
        feasible = hybrid["status"] != "infeasible"
        target = hybrid["target_label_vector"] or {"0": 0, "1": 0}
        selected = hybrid["selected_label_vector"] or {"0": 0, "1": 0}
        paired.append(
            {
                "experiment_name": EXPERIMENT_NAME,
                "experiment_id": EXPERIMENT_ID,
                "protocol_version": PROTOCOL_VERSION,
                "cell_id": hybrid["cell_id"],
                "language": hybrid["language"],
                "repeat_seed": hybrid["repeat_seed"],
                "ratio": hybrid["ratio"],
                "n_add": hybrid["n_add"],
                "total_train_n": hybrid["total_train_n"],
                "similarity_condition_id": hybrid["similarity_condition_id"],
                "similarity_enabled": bool(hybrid["similarity_enabled"]),
                "similarity_lower": hybrid["similarity_lower"],
                "similarity_upper": hybrid["similarity_upper"],
                "feasible": feasible,
                "infeasible_reason": hybrid["infeasible_reason"],
                "base_macro_f1": base_row["macro_f1"],
                "matched_all_real_macro_f1": matched_row["macro_f1"],
                "hybrid_macro_f1": hybrid["macro_f1"],
                "delta_synth_macro_f1": metric_delta(hybrid["macro_f1"], base_row["macro_f1"]),
                "delta_real_macro_f1": metric_delta(matched_row["macro_f1"], base_row["macro_f1"]),
                "g_real_macro_f1": metric_delta(hybrid["macro_f1"], matched_row["macro_f1"]),
                "base_auroc": base_row["auroc"],
                "matched_all_real_auroc": matched_row["auroc"],
                "hybrid_auroc": hybrid["auroc"],
                "delta_synth_auroc": metric_delta(hybrid["auroc"], base_row["auroc"]),
                "delta_real_auroc": metric_delta(matched_row["auroc"], base_row["auroc"]),
                "g_real_auroc": metric_delta(hybrid["auroc"], matched_row["auroc"]),
                "base_label_0_n": base_row["label_0_n"],
                "base_label_1_n": base_row["label_1_n"],
                "additional_label_0_n": int(target.get("0", 0)),
                "additional_label_1_n": int(target.get("1", 0)),
                "synthetic_label_0_n": int(selected.get("0", 0)),
                "synthetic_label_1_n": int(selected.get("1", 0)),
                "eligible_synthetic_n": hybrid["eligible_n"],
                "selected_synthetic_n": hybrid["selected_n"],
                "selected_parent_count": hybrid["selected_parent_count"],
                "selection_sha256": hybrid["selection_sha256"],
                "base_status": base_row["status"],
                "matched_status": matched_row["status"],
                "hybrid_status": hybrid["status"],
            }
        )
    return paired


def audit_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "cell_id": row["cell_id"],
        "repeat_seed": row["repeat_seed"],
        "ratio": row["ratio"],
        "similarity_condition_id": row["similarity_condition_id"],
        "lower": row["similarity_lower"],
        "upper": row["similarity_upper"],
        "target_n": row["n_add"],
        "target_label_vector": row["target_label_vector"],
        "eligible_n": row["eligible_n"],
        "eligible_label_vector": row["eligible_label_vector"],
        "selected_n": row["selected_n"],
        "selected_label_vector": row["selected_label_vector"],
        "selected_global_generation_indices": row["selected_global_generation_indices"],
        "selected_parent_count": row["selected_parent_count"],
        "selection_sha256": row["selection_sha256"],
        "feasible": row["status"] != "infeasible",
        "infeasible_reason": row["infeasible_reason"],
    }


def write_deterministic_gzip_jsonl(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            for row in rows:
                compressed.write(
                    (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
                )
    os.replace(temporary, path)


def package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in ("numpy", "scikit-learn", "scipy"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def scope_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    base = [row for row in rows if row["training_condition"] == "base"]
    matched = [row for row in rows if row["training_condition"] == "matched_all_real"]
    hybrid = [row for row in rows if row["training_condition"] == "hybrid"]
    terminal_success = {"completed", "completed_with_warning"}
    infeasible_by_ratio = Counter(
        ratio_key(float(row["ratio"])) for row in hybrid if row["status"] == "infeasible"
    )
    infeasible_by_band = Counter(
        str(row["similarity_condition_id"])
        for row in hybrid
        if row["status"] == "infeasible"
    )
    off_eligible = {
        (row["cell_id"], row["repeat_seed"], ratio_key(float(row["ratio"]))): int(row["eligible_n"])
        for row in hybrid
        if row["similarity_condition_id"] == "OFF"
    }
    retention_numerator: Counter[str] = Counter()
    retention_denominator: Counter[str] = Counter()
    for row in hybrid:
        condition_id = str(row["similarity_condition_id"])
        key = (row["cell_id"], row["repeat_seed"], ratio_key(float(row["ratio"])))
        retention_numerator[condition_id] += int(row["eligible_n"])
        retention_denominator[condition_id] += off_eligible.get(key, 0)
    retention = {
        condition_id: (
            retention_numerator[condition_id] / retention_denominator[condition_id]
            if retention_denominator[condition_id]
            else None
        )
        for condition_id in sorted(retention_numerator)
    }
    parent_counts = [
        int(row["selected_parent_count"])
        for row in hybrid
        if row["status"] in terminal_success
    ]
    return {
        "planned_base_fits": len(base),
        "completed_base_fits": sum(row["status"] in terminal_success for row in base),
        "planned_matched_fits": len(matched),
        "completed_matched_fits": sum(row["status"] in terminal_success for row in matched),
        "planned_hybrid_conditions": len(hybrid),
        "completed_hybrid_fits": sum(row["status"] in terminal_success for row in hybrid),
        "completed_with_warning": sum(row["status"] == "completed_with_warning" for row in rows),
        "infeasible_count": sum(row["status"] == "infeasible" for row in hybrid),
        "hard_failure_count": sum(row["status"] == "failed" for row in rows),
        "ratio_infeasible_count": dict(sorted(infeasible_by_ratio.items())),
        "band_infeasible_count": dict(sorted(infeasible_by_band.items())),
        "band_retention": retention,
        "mean_selected_parent_count": (
            sum(parent_counts) / len(parent_counts) if parent_counts else None
        ),
    }


def output_hashes(output_root: Path) -> dict[str, str]:
    excluded = {
        LOCK_NAME,
        "downstream_status.sqlite3",
        "downstream_status.sqlite3-wal",
        "downstream_status.sqlite3-shm",
    }
    return {
        path.relative_to(output_root).as_posix(): sha256_file(path)
        for path in sorted(output_root.rglob("*"))
        if path.is_file() and path.name not in excluded
    }


def export_results(
    connection: sqlite3.Connection,
    output_root: Path,
    lock_hashes: Mapping[str, str],
    manifest_lock_sha256: str,
) -> dict[str, Any]:
    rows = database_rows(connection)
    base_rows = [row for row in rows if row["training_condition"] == "base"]
    matched_rows = [row for row in rows if row["training_condition"] == "matched_all_real"]
    hybrid_rows = [row for row in rows if row["training_condition"] == "hybrid"]
    if len(base_rows) != PLANNED_BASE_FITS:
        raise RuntimeError(f"base result count {len(base_rows)} != {PLANNED_BASE_FITS}")
    if len(matched_rows) != PLANNED_MATCHED_FITS:
        raise RuntimeError(f"matched result count {len(matched_rows)} != {PLANNED_MATCHED_FITS}")
    if len(hybrid_rows) != PLANNED_HYBRID_CONDITIONS:
        raise RuntimeError(
            f"hybrid result count {len(hybrid_rows)} != {PLANNED_HYBRID_CONDITIONS}"
        )

    base_rows.sort(key=lambda row: (row["cell_id"], int(row["repeat_seed"])))
    matched_rows.sort(
        key=lambda row: (row["cell_id"], int(row["repeat_seed"]), float(row["ratio"]))
    )
    hybrid_rows.sort(
        key=lambda row: (
            row["cell_id"],
            int(row["repeat_seed"]),
            float(row["ratio"]),
            str(row["similarity_condition_id"]),
        )
    )
    write_csv(
        output_root / "base_results.csv",
        [fit_export_row(row) for row in base_rows],
        FIT_FIELDS,
    )
    write_csv(
        output_root / "matched_all_real_results.csv",
        [fit_export_row(row) for row in matched_rows],
        FIT_FIELDS,
    )
    write_csv(
        output_root / "hybrid_results.csv",
        [fit_export_row(row) for row in hybrid_rows],
        FIT_FIELDS,
    )
    paired = build_paired_rows(rows)
    if len(paired) != PLANNED_HYBRID_CONDITIONS:
        raise RuntimeError(f"paired result count {len(paired)} != {PLANNED_HYBRID_CONDITIONS}")
    write_csv(output_root / "paired_results.csv", paired, PAIRED_FIELDS)

    audits = [audit_row(row) for row in hybrid_rows]
    write_deterministic_gzip_jsonl(
        output_root / "selection_audit.jsonl.gz",
        audits,
    )
    infeasible = [row for row in audits if not row["feasible"]]
    write_csv(
        output_root / "infeasible_conditions.csv",
        infeasible,
        AUDIT_FIELDS,
    )

    cells = {
        cell_id: scope_report([row for row in rows if row["cell_id"] == cell_id])
        for cell_id in TARGET_CELLS
    }
    overall = scope_report(rows)
    report = {
        "report_version": "binary-development-grid-v1",
        "created_at": utc_now(),
        "experiment_name": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "classifier_config": CLASSIFIER_CONFIG,
        "grid_config": {
            "base_n": BASE_N,
            "repeat_seeds": list(REPEAT_SEEDS),
            "ratios": list(RATIOS),
            "lower_bounds": list(LOWER_BOUNDS),
            "upper_bounds": list(UPPER_BOUNDS),
            "similarity_conditions": list(SIMILARITY_CONDITIONS),
        },
        "selector_version": SELECTOR_VERSION,
        "input_lock_hashes": dict(lock_hashes),
        "manifest_lock_sha256": manifest_lock_sha256,
        "package_versions": package_versions(),
        "cells": cells,
        "overall": overall,
    }
    write_json(output_root / "development_report.json", report)
    report_fields = ["scope", "cell_id", *overall.keys()]
    report_rows: list[dict[str, Any]] = []
    for cell_id in TARGET_CELLS:
        row = {"scope": "cell", "cell_id": cell_id, **cells[cell_id]}
        report_rows.append(
            {
                key: canonical_json(value) if isinstance(value, dict) else value
                for key, value in row.items()
            }
        )
    overall_row = {"scope": "overall", "cell_id": "__all__", **overall}
    report_rows.append(
        {
            key: canonical_json(value) if isinstance(value, dict) else value
            for key, value in overall_row.items()
        }
    )
    write_csv(
        output_root / "development_report.csv",
        report_rows,
        report_fields,
    )
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    terminal_success = {"completed", "completed_with_warning"}
    hard_failures = sum(row["status"] == "failed" for row in rows)
    base_success = sum(row["status"] in terminal_success for row in base_rows)
    matched_success = sum(row["status"] in terminal_success for row in matched_rows)
    hybrid_terminal = sum(
        row["status"] in terminal_success or row["status"] == "infeasible"
        for row in hybrid_rows
    )
    for row in hybrid_rows:
        if row["status"] == "infeasible":
            continue
        if row["selected_n"] != row["n_add"]:
            raise RuntimeError("feasible Hybrid selected_n does not equal n_add")
        if row["selected_label_vector"] != row["target_label_vector"]:
            raise RuntimeError("feasible Hybrid label vector mismatch")
    success = (
        base_success == PLANNED_BASE_FITS
        and matched_success == PLANNED_MATCHED_FITS
        and hybrid_terminal == PLANNED_HYBRID_CONDITIONS
        and hard_failures == 0
        and len(paired) == PLANNED_HYBRID_CONDITIONS
    )
    if success:
        files = output_hashes(output_root)
        for relative, expected in files.items():
            if sha256_file(output_root / relative) != expected:
                raise RuntimeError(f"output hash verification failed: {relative}")
        write_json(
            output_root / LOCK_NAME,
            {
                "lock_version": "binary-development-grid-v1",
                "created_at": utc_now(),
                "experiment_name": EXPERIMENT_NAME,
                "experiment_id": EXPERIMENT_ID,
                "status": "completed",
                "input_lock_sha256": {
                    **dict(lock_hashes),
                    "manifest": manifest_lock_sha256,
                },
                "files": files,
            },
        )
    else:
        lock_path = output_root / LOCK_NAME
        if lock_path.exists():
            lock_path.unlink()
    return report


def dry_run_summary(
    repetitions: Mapping[str, Sequence[Mapping[str, Any]]],
    data: Mapping[str, Mapping[str, Any]],
    output_root: Path,
) -> dict[str, Any]:
    base_count = 0
    matched_count = 0
    hybrid_count = 0
    infeasible_count = 0
    infeasible_by_ratio: Counter[str] = Counter()
    infeasible_by_band: Counter[str] = Counter()
    for task in task_stream(repetitions, data, set()):
        condition = task["training_condition"]
        if condition == "base":
            base_count += 1
        elif condition == "matched_all_real":
            matched_count += 1
        else:
            hybrid_count += 1
            if not task["run_fit"]:
                infeasible_count += 1
                infeasible_by_ratio[ratio_key(float(task["ratio"]))] += 1
                infeasible_by_band[str(task["similarity_condition_id"])] += 1
    if (
        base_count != PLANNED_BASE_FITS
        or matched_count != PLANNED_MATCHED_FITS
        or hybrid_count != PLANNED_HYBRID_CONDITIONS
    ):
        raise RuntimeError("dry-run planned condition counts are incorrect")
    return {
        "dry_run": True,
        "planned_base_fits": base_count,
        "planned_matched_fits": matched_count,
        "planned_hybrid_conditions": hybrid_count,
        "maximum_actual_fit_count": PLANNED_TOTAL_FITS,
        "precomputed_infeasible_count": infeasible_count,
        "ratio_infeasible_count": dict(sorted(infeasible_by_ratio.items())),
        "band_infeasible_count": dict(sorted(infeasible_by_band.items())),
        "expected_output_root": str(output_root),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Run the Binary Matched-Size development downstream grid"
    )
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=project_root / "Data" / "ExperimentManifests" / "v1",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "Results" / PATH_ID,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.workers <= 0 or args.progress_every <= 0:
        raise SystemExit("workers and progress-every must be positive")
    if args.dry_run and args.resume:
        raise SystemExit("--dry-run and --resume cannot be combined")
    project_root = Path(__file__).resolve().parents[2]
    manifest_root = args.manifest_root.resolve()
    experiment_root = args.output_root.resolve()
    downstream_root = experiment_root / "Downstream" / "DevelopmentGrid"
    try:
        lock_hashes = input_lock_hashes(experiment_root)
        prepared_root = project_root / "Data" / "Prepared"
        manifest_info, banks, _plan = load_and_validate_inputs(
            prepared_root,
            manifest_root,
        )
        repetitions = validate_repetitions(manifest_root, banks)
        valid_by_cell = read_valid_outputs(experiment_root / "Generation")
        data = validate_and_load_embeddings(
            experiment_root,
            banks,
            valid_by_cell,
        )
        if args.dry_run:
            print(
                json.dumps(
                    dry_run_summary(
                        repetitions,
                        data,
                        downstream_root,
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

        downstream_root.mkdir(parents=True, exist_ok=True)
        lock_path = downstream_root / LOCK_NAME
        if lock_path.exists():
            lock_path.unlink()
        connection = open_database(
            downstream_root / "downstream_status.sqlite3"
        )
        try:
            contract = metadata_contract(
                lock_hashes,
                str(manifest_info["lock_sha256"]),
            )
            initialize_database(connection, contract, args.resume)
            progress_log(
                f"development grid {'resume' if args.resume else 'new run'} | "
                f"existing_status={current_status(connection)} | workers={args.workers}"
            )
            run_grid(
                connection,
                repetitions,
                data,
                args.workers,
                args.progress_every,
            )
            report = export_results(
                connection,
                downstream_root,
                lock_hashes,
                str(manifest_info["lock_sha256"]),
            )
        finally:
            connection.close()
        print(json.dumps(report["overall"], ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(
            f"binary downstream experiment failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
