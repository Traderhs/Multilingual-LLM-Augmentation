"""Summarize the development grid and freeze one representative condition."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

_MODULE_ROOT = Path(__file__).resolve().parent
if str(_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULE_ROOT))

from generate_binary_candidates import (
    CELL_LANGUAGE,
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    PATH_ID,
    TARGET_CELLS,
    canonical_json,
    read_json,
    sha256_file,
    utc_now,
    verify_lock,
    write_csv,
    write_json,
)
from run_binary_downstream_experiment import (
    BASE_N,
    PAIRED_FIELDS,
    RATIOS,
    REPEAT_SEEDS,
    SIMILARITY_CONDITIONS,
    ratio_key,
)


BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_BASE_SEED = 20260713
CI_LEVEL = 0.95
NONINFERIORITY_MARGIN = -0.01
GAIN_RECOVERY_MIN_DELTA_REAL = 0.005
EXPECTED_INPUT_ROWS = 46_750
LOCK_NAME = "CONDITION_SELECTION_LOCK.json"
NUMERIC_ATOL = 1e-9

EFFECTS = ("delta_synth", "delta_real", "g_real")
METRICS = ("macro_f1", "auroc")
EFFECT_COLUMNS = {
    ("delta_synth", "macro_f1"): "delta_synth_macro_f1",
    ("delta_real", "macro_f1"): "delta_real_macro_f1",
    ("g_real", "macro_f1"): "g_real_macro_f1",
    ("delta_synth", "auroc"): "delta_synth_auroc",
    ("delta_real", "auroc"): "delta_real_auroc",
    ("g_real", "auroc"): "g_real_auroc",
}

DATASET_FIELDS = [
    "experiment_name", "experiment_id", "cell_id", "language", "ratio",
    "similarity_condition_id", "similarity_enabled", "similarity_lower",
    "similarity_upper", "metric", "effect", "n", "mean", "std",
    "standard_error", "median", "minimum", "maximum", "ci_lower", "ci_upper",
    "positive_count", "positive_proportion", "zero_count", "negative_count",
]

POOLED_FIELDS = [
    "experiment_name", "experiment_id", "ratio", "similarity_condition_id",
    "similarity_enabled", "similarity_lower", "similarity_upper", "metric",
    "effect", "analysis_complete", "dataset_count", "repeat_count_per_dataset",
    "mean", "bootstrap_standard_error", "ci_lower", "ci_upper",
    "dataset_mean_minimum", "dataset_mean_maximum", "positive_dataset_count",
    "negative_dataset_count", "positive_repeat_proportion", "gain_recovery",
    "gain_recovery_reportable", "gain_recovery_reason",
]

CRITERIA_FIELDS = [
    "ratio", "similarity_condition_id", "similarity_enabled", "similarity_lower",
    "similarity_upper", "analysis_complete", "selection_eligible",
    "pooled_delta_synth_mean", "pooled_delta_synth_se",
    "pooled_delta_synth_ci_lower", "pooled_delta_synth_ci_upper",
    "pooled_delta_real_mean", "pooled_delta_real_ci_lower",
    "pooled_delta_real_ci_upper", "pooled_g_real_mean", "pooled_g_real_ci_lower",
    "pooled_g_real_ci_upper", "positive_dataset_count", "damaged_dataset_count",
    "c1_pass", "c2_pass", "c3_pass", "c4_pass", "c5_pass", "c1_to_c5_pass",
    "c6_pass", "c7_pass", "stable_candidate", "stable_region_id",
    "ratio_run_id", "ratio_run_start", "ratio_run_end", "ratio_run_length",
    "adjacent_band_pass_count", "adjacent_supported_ratio_count",
]

REGION_MEMBER_FIELDS = [
    "stable_region_id", "filter_mode", "ratio", "similarity_condition_id",
    "similarity_enabled", "similarity_lower", "similarity_upper",
    "pooled_delta_synth_mean", "condition_key",
]

ONE_SE_FIELDS = [
    "ratio", "similarity_condition_id", "similarity_enabled", "similarity_lower",
    "similarity_upper", "stable_region_id", "pooled_delta_synth_mean",
    "pooled_delta_synth_se", "one_se_threshold", "equivalent_candidate",
    "selected",
]


def condition_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    ratio = float(row["ratio"])
    ratio_index = RATIOS.index(ratio)
    enabled = bool(row["similarity_enabled"])
    lower = row.get("similarity_lower")
    upper = row.get("similarity_upper")
    return (
        ratio_index,
        1 if enabled else 0,
        -1.0 if lower is None else float(lower),
        -1.0 if upper is None else float(upper),
    )


def full_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        *condition_sort_key(row),
        str(row.get("cell_id", "")),
        METRICS.index(str(row["metric"])) if row.get("metric") in METRICS else -1,
        EFFECTS.index(str(row["effect"])) if row.get("effect") in EFFECTS else -1,
    )


def similarity_map() -> dict[str, dict[str, Any]]:
    return {
        str(row["similarity_condition_id"]): dict(row)
        for row in SIMILARITY_CONDITIONS
    }


SIMILARITY_MAP = similarity_map()


def parse_optional_float(value: Any, field: str) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError as exc:
        raise RuntimeError(f"{field} is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise RuntimeError(f"{field} is not finite: {value!r}")
    return number


def parse_bool(value: Any, field: str) -> bool:
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    raise RuntimeError(f"{field} is not boolean: {value!r}")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def close_enough(observed: float, expected: float) -> bool:
    return math.isclose(observed, expected, rel_tol=0.0, abs_tol=NUMERIC_ATOL)


def condition_key(row: Mapping[str, Any]) -> tuple[str, int, str, str]:
    return (
        str(row["cell_id"]),
        int(row["repeat_seed"]),
        ratio_key(float(row["ratio"])),
        str(row["similarity_condition_id"]),
    )


def analysis_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return (
        ratio_key(float(row["ratio"])),
        str(row["similarity_condition_id"]),
    )


def complete_condition_count(rows: Sequence[Mapping[str, Any]]) -> int:
    feasible_seeds: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    for row in rows:
        if row["feasible"]:
            feasible_seeds[
                (str(row["cell_id"]), *analysis_key(row))
            ].add(int(row["repeat_seed"]))
    return sum(
        all(
            feasible_seeds[(cell_id, ratio_key(ratio), str(condition["similarity_condition_id"]))]
            == set(REPEAT_SEEDS)
            for cell_id in TARGET_CELLS
        )
        for ratio in RATIOS
        for condition in SIMILARITY_CONDITIONS
    )


def validate_input(
    development_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    lock = verify_lock(development_root, "DEVELOPMENT_GRID_LOCK.json")
    lock_sha256 = sha256_file(development_root / "DEVELOPMENT_GRID_LOCK.json")
    report = read_json(development_root / "development_report.json")
    if not isinstance(report, dict):
        raise RuntimeError("development_report.json is not an object")
    overall = report.get("overall")
    if not isinstance(overall, dict) or int(overall.get("hard_failure_count", -1)) != 0:
        raise RuntimeError("development grid reports hard failures")
    report_input_hashes = dict(report.get("input_lock_hashes") or {})
    lock_input_hashes = dict(lock.get("input_lock_sha256") or {})
    for name, value in report_input_hashes.items():
        if name == "manifest":
            continue
        if lock_input_hashes.get(name) != value:
            raise RuntimeError(f"input lock hash mismatch for {name}")
    manifest_hash = report.get("manifest_lock_sha256")
    if manifest_hash is not None and lock_input_hashes.get("manifest") != manifest_hash:
        raise RuntimeError("manifest lock hash mismatch")

    raw_rows = read_csv(development_root / "paired_results.csv")
    if len(raw_rows) != EXPECTED_INPUT_ROWS:
        raise RuntimeError(
            f"paired_results.csv has {len(raw_rows)} rows, expected {EXPECTED_INPUT_ROWS}"
        )
    missing_fields = set(PAIRED_FIELDS) - set(raw_rows[0] if raw_rows else {})
    if missing_fields:
        raise RuntimeError(f"paired_results columns missing: {sorted(missing_fields)}")
    rows: list[dict[str, Any]] = []
    keys: set[tuple[str, int, str, str]] = set()
    expected_statuses = {"completed", "completed_with_warning"}
    for index, raw in enumerate(raw_rows):
        cell_id = str(raw.get("cell_id") or "")
        repeat_seed = int(raw["repeat_seed"])
        ratio = float(raw["ratio"])
        similarity_id = str(raw.get("similarity_condition_id") or "")
        if cell_id not in TARGET_CELLS:
            raise RuntimeError(f"row {index}: unexpected cell {cell_id}")
        if repeat_seed not in REPEAT_SEEDS:
            raise RuntimeError(f"row {index}: unexpected repeat_seed {repeat_seed}")
        if ratio not in RATIOS:
            raise RuntimeError(f"row {index}: unexpected ratio {ratio}")
        if similarity_id not in SIMILARITY_MAP:
            raise RuntimeError(f"row {index}: unexpected similarity condition {similarity_id}")
        similarity = SIMILARITY_MAP[similarity_id]
        feasible = parse_bool(raw.get("feasible"), "feasible")
        if parse_bool(raw.get("similarity_enabled"), "similarity_enabled") != bool(similarity["similarity_enabled"]):
            raise RuntimeError(f"row {index}: similarity_enabled mismatch")
        raw_lower = parse_optional_float(raw.get("similarity_lower"), "similarity_lower")
        raw_upper = parse_optional_float(raw.get("similarity_upper"), "similarity_upper")
        if raw_lower != similarity["lower"] or raw_upper != similarity["upper"]:
            raise RuntimeError(f"row {index}: similarity bounds mismatch")
        row: dict[str, Any] = {
            **raw,
            "cell_id": cell_id,
            "language": str(raw.get("language") or CELL_LANGUAGE[cell_id]),
            "repeat_seed": repeat_seed,
            "ratio": ratio,
            "similarity_condition_id": similarity_id,
            "similarity_enabled": bool(similarity["similarity_enabled"]),
            "similarity_lower": similarity["lower"],
            "similarity_upper": similarity["upper"],
            "feasible": feasible,
            "n_add": int(raw["n_add"]),
            "total_train_n": int(raw["total_train_n"]),
        }
        key = condition_key(row)
        if key in keys:
            raise RuntimeError(f"duplicate paired condition key: {key}")
        keys.add(key)
        for field in (
            "base_macro_f1", "matched_all_real_macro_f1", "hybrid_macro_f1",
            "delta_synth_macro_f1", "delta_real_macro_f1", "g_real_macro_f1",
            "base_auroc", "matched_all_real_auroc", "hybrid_auroc",
            "delta_synth_auroc", "delta_real_auroc", "g_real_auroc",
        ):
            row[field] = parse_optional_float(raw.get(field), field)
        for field in (
            "base_label_0_n", "base_label_1_n", "additional_label_0_n",
            "additional_label_1_n", "synthetic_label_0_n", "synthetic_label_1_n",
            "eligible_synthetic_n", "selected_synthetic_n", "selected_parent_count",
        ):
            row[field] = int(raw[field])
        if feasible:
            for status_field in ("base_status", "matched_status", "hybrid_status"):
                if str(raw.get(status_field)) not in expected_statuses:
                    raise RuntimeError(f"row {index}: feasible {status_field} is not completed")
            macro_fields = (
                "base_macro_f1", "matched_all_real_macro_f1", "hybrid_macro_f1",
            )
            if any(row[field] is None for field in macro_fields):
                raise RuntimeError(f"row {index}: feasible Macro-F1 is missing")
            expected_deltas = {
                "delta_synth_macro_f1": row["hybrid_macro_f1"] - row["base_macro_f1"],
                "delta_real_macro_f1": row["matched_all_real_macro_f1"] - row["base_macro_f1"],
                "g_real_macro_f1": row["hybrid_macro_f1"] - row["matched_all_real_macro_f1"],
            }
            for field, expected in expected_deltas.items():
                if row[field] is None or not close_enough(row[field], expected):
                    raise RuntimeError(f"row {index}: {field} mismatch")
            auroc_scores = (row["base_auroc"], row["matched_all_real_auroc"], row["hybrid_auroc"])
            if all(value is not None for value in auroc_scores):
                expected_auroc = {
                    "delta_synth_auroc": row["hybrid_auroc"] - row["base_auroc"],
                    "delta_real_auroc": row["matched_all_real_auroc"] - row["base_auroc"],
                    "g_real_auroc": row["hybrid_auroc"] - row["matched_all_real_auroc"],
                }
                for field, expected in expected_auroc.items():
                    if row[field] is None or not close_enough(row[field], expected):
                        raise RuntimeError(f"row {index}: {field} mismatch")
            elif any(row[field] is not None for field in (
                "delta_synth_auroc", "delta_real_auroc", "g_real_auroc"
            )):
                raise RuntimeError(f"row {index}: partial AUROC delta values")
            if row["selected_synthetic_n"] != row["n_add"]:
                raise RuntimeError(f"row {index}: feasible equal-N mismatch")
            if (
                row["additional_label_0_n"] != row["synthetic_label_0_n"]
                or row["additional_label_1_n"] != row["synthetic_label_1_n"]
            ):
                raise RuntimeError(f"row {index}: feasible label-vector mismatch")
            if row["total_train_n"] != BASE_N + row["n_add"]:
                raise RuntimeError(f"row {index}: feasible total train size mismatch")
        rows.append(row)

    expected_analysis_keys = {
        (ratio_key(ratio), str(condition["similarity_condition_id"]))
        for ratio in RATIOS
        for condition in SIMILARITY_CONDITIONS
    }
    observed_analysis_keys = {analysis_key(row) for row in rows}
    if observed_analysis_keys != expected_analysis_keys:
        raise RuntimeError("paired results do not contain the fixed 187-condition grid")
    for cell_id in TARGET_CELLS:
        cell_rows = [row for row in rows if row["cell_id"] == cell_id]
        if {row["repeat_seed"] for row in cell_rows} != set(REPEAT_SEEDS):
            raise RuntimeError(f"{cell_id}: repeat seed coverage mismatch")
        if len(cell_rows) != 50 * len(RATIOS) * len(SIMILARITY_CONDITIONS):
            raise RuntimeError(f"{cell_id}: paired grid row count mismatch")
    return rows, report, lock_sha256


def bootstrap_seed(*parts: Any) -> int:
    source = "|".join(str(part) for part in (BOOTSTRAP_BASE_SEED, *parts))
    return int.from_bytes(hashlib.sha256(source.encode("utf-8")).digest()[:4], "big")


def percentile_interval(values: Any) -> tuple[float, float]:
    import numpy as np

    alpha = (1.0 - CI_LEVEL) / 2.0
    lower, upper = np.quantile(values, [alpha, 1.0 - alpha])
    return float(lower), float(upper)


def descriptive_summary(
    values: Sequence[float],
    seed: int,
) -> dict[str, Any]:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    n = int(array.size)
    if n == 0:
        return {
            "n": 0, "mean": None, "std": None, "standard_error": None,
            "median": None, "minimum": None, "maximum": None, "ci_lower": None,
            "ci_upper": None, "positive_count": 0, "positive_proportion": None,
            "zero_count": 0, "negative_count": 0,
        }
    rng = np.random.default_rng(seed)
    sampled_indices = rng.integers(0, n, size=(BOOTSTRAP_REPLICATES, n))
    replicates = array[sampled_indices].mean(axis=1)
    ci_lower, ci_upper = percentile_interval(replicates)
    std = float(np.std(array, ddof=1)) if n > 1 else None
    return {
        "n": n,
        "mean": float(np.mean(array)),
        "std": std,
        "standard_error": std / math.sqrt(n) if std is not None else None,
        "median": float(np.median(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "positive_count": int(np.sum(array > 0)),
        "positive_proportion": float(np.mean(array > 0)),
        "zero_count": int(np.sum(array == 0)),
        "negative_count": int(np.sum(array < 0)),
    }


def build_dataset_summaries(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["cell_id"], ratio_key(float(row["ratio"])), row["similarity_condition_id"])].append(row)
    summaries: list[dict[str, Any]] = []
    for cell_id in TARGET_CELLS:
        for ratio in RATIOS:
            for condition in SIMILARITY_CONDITIONS:
                similarity_id = str(condition["similarity_condition_id"])
                condition_rows = sorted(
                    grouped.get((cell_id, ratio_key(ratio), similarity_id), []),
                    key=lambda row: int(row["repeat_seed"]),
                )
                feasible_rows = [row for row in condition_rows if row["feasible"]]
                for metric in METRICS:
                    for effect in EFFECTS:
                        field = EFFECT_COLUMNS[(effect, metric)]
                        values = [
                            float(row[field])
                            for row in feasible_rows
                            if row[field] is not None
                        ]
                        stats = descriptive_summary(
                            values,
                            bootstrap_seed(
                                cell_id,
                                ratio_key(ratio),
                                similarity_id,
                                effect,
                                metric,
                            ),
                        )
                        summaries.append(
                            {
                                "experiment_name": EXPERIMENT_NAME,
                                "experiment_id": EXPERIMENT_ID,
                                "cell_id": cell_id,
                                "language": CELL_LANGUAGE[cell_id],
                                "ratio": ratio,
                                "similarity_condition_id": similarity_id,
                                "similarity_enabled": condition["similarity_enabled"],
                                "similarity_lower": condition["lower"],
                                "similarity_upper": condition["upper"],
                                "metric": metric,
                                "effect": effect,
                                **stats,
                            }
                        )
    summaries.sort(key=full_sort_key)
    return summaries


def build_pooled_summaries(
    rows: Sequence[Mapping[str, Any]],
    dataset_summaries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    import numpy as np

    rows_by_condition: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_condition[analysis_key(row)].append(row)
    dataset_index = {
        (
            row["cell_id"], ratio_key(float(row["ratio"])),
            row["similarity_condition_id"], row["metric"], row["effect"],
        ): row
        for row in dataset_summaries
    }
    pooled: list[dict[str, Any]] = []
    for ratio in RATIOS:
        for condition in SIMILARITY_CONDITIONS:
            similarity_id = str(condition["similarity_condition_id"])
            condition_rows = rows_by_condition[(ratio_key(ratio), similarity_id)]
            by_cell: dict[str, list[Mapping[str, Any]]] = {
                cell_id: sorted(
                    [row for row in condition_rows if row["cell_id"] == cell_id and row["feasible"]],
                    key=lambda row: int(row["repeat_seed"]),
                )
                for cell_id in TARGET_CELLS
            }
            macro_complete = all(
                len(by_cell[cell_id]) == 50
                and {row["repeat_seed"] for row in by_cell[cell_id]} == set(REPEAT_SEEDS)
                for cell_id in TARGET_CELLS
            )
            for metric in METRICS:
                metric_complete = macro_complete and all(
                    all(row[EFFECT_COLUMNS[(effect, metric)]] is not None for effect in EFFECTS)
                    for cell_id in TARGET_CELLS
                    for row in by_cell[cell_id]
                )
                replicate_by_effect: dict[str, Any] = {}
                values_by_effect: dict[str, list[float]] = {effect: [] for effect in EFFECTS}
                if metric_complete:
                    replicate_by_effect = {
                        effect: np.zeros(BOOTSTRAP_REPLICATES, dtype=np.float64)
                        for effect in EFFECTS
                    }
                    rng = np.random.default_rng(
                        bootstrap_seed("pooled", ratio_key(ratio), similarity_id, metric)
                    )
                    for cell_id in TARGET_CELLS:
                        sampled_indices = rng.integers(
                            0,
                            50,
                            size=(BOOTSTRAP_REPLICATES, 50),
                        )
                        for effect in EFFECTS:
                            field = EFFECT_COLUMNS[(effect, metric)]
                            values = np.asarray(
                                [float(row[field]) for row in by_cell[cell_id]],
                                dtype=np.float64,
                            )
                            values_by_effect[effect].extend(values.tolist())
                            replicate_by_effect[effect] += values[sampled_indices].mean(axis=1) / 5.0
                for effect in EFFECTS:
                    dataset_rows = [
                        dataset_index[(cell_id, ratio_key(ratio), similarity_id, metric, effect)]
                        for cell_id in TARGET_CELLS
                    ]
                    dataset_means = [
                        float(row["mean"]) for row in dataset_rows if row["mean"] is not None
                    ]
                    if metric_complete:
                        replicates = replicate_by_effect[effect]
                        ci_lower, ci_upper = percentile_interval(replicates)
                        values = values_by_effect[effect]
                        mean = float(sum(float(row["mean"]) for row in dataset_rows) / 5.0)
                        standard_error = float(np.std(replicates, ddof=1))
                        positive_repeat_proportion = float(np.mean(np.asarray(values) > 0))
                    else:
                        mean = standard_error = ci_lower = ci_upper = None
                        positive_repeat_proportion = None
                    pooled.append(
                        {
                            "experiment_name": EXPERIMENT_NAME,
                            "experiment_id": EXPERIMENT_ID,
                            "ratio": ratio,
                            "similarity_condition_id": similarity_id,
                            "similarity_enabled": condition["similarity_enabled"],
                            "similarity_lower": condition["lower"],
                            "similarity_upper": condition["upper"],
                            "metric": metric,
                            "effect": effect,
                            "analysis_complete": metric_complete,
                            "dataset_count": 5,
                            "repeat_count_per_dataset": 50 if metric_complete else None,
                            "mean": mean,
                            "bootstrap_standard_error": standard_error,
                            "ci_lower": ci_lower,
                            "ci_upper": ci_upper,
                            "dataset_mean_minimum": min(dataset_means) if dataset_means else None,
                            "dataset_mean_maximum": max(dataset_means) if dataset_means else None,
                            "positive_dataset_count": sum(value > 0 for value in dataset_means),
                            "negative_dataset_count": sum(value < 0 for value in dataset_means),
                            "positive_repeat_proportion": positive_repeat_proportion,
                            "gain_recovery": None,
                            "gain_recovery_reportable": False,
                            "gain_recovery_reason": None,
                        }
                    )
    pooled_index = {
        (ratio_key(float(row["ratio"])), row["similarity_condition_id"], row["metric"], row["effect"]): row
        for row in pooled
    }
    for ratio in RATIOS:
        for condition in SIMILARITY_CONDITIONS:
            similarity_id = str(condition["similarity_condition_id"])
            delta_synth = pooled_index[(ratio_key(ratio), similarity_id, "macro_f1", "delta_synth")]
            delta_real = pooled_index[(ratio_key(ratio), similarity_id, "macro_f1", "delta_real")]
            if not delta_synth["analysis_complete"] or not delta_real["analysis_complete"]:
                recovery = None
                reportable = False
                reason = "analysis_incomplete"
            elif float(delta_real["ci_lower"]) <= 0:
                recovery = None
                reportable = False
                reason = "pooled_delta_real_ci_lower_not_positive"
            elif float(delta_real["mean"]) < GAIN_RECOVERY_MIN_DELTA_REAL:
                recovery = None
                reportable = False
                reason = "pooled_delta_real_mean_below_0.005"
            else:
                recovery = float(delta_synth["mean"]) / float(delta_real["mean"])
                reportable = True
                reason = None
            for effect in EFFECTS:
                row = pooled_index[(ratio_key(ratio), similarity_id, "macro_f1", effect)]
                row["gain_recovery"] = recovery
                row["gain_recovery_reportable"] = reportable
                row["gain_recovery_reason"] = reason
    pooled.sort(key=full_sort_key)
    return pooled


def selection_priority(row: Mapping[str, Any]) -> tuple[Any, ...]:
    enabled = bool(row["similarity_enabled"])
    lower = row.get("similarity_lower")
    upper = row.get("similarity_upper")
    width = 0.0 if not enabled else float(upper) - float(lower)
    return (
        RATIOS.index(float(row["ratio"])),
        1 if enabled else 0,
        -width,
        -1.0 if lower is None else float(lower),
        0.0 if upper is None else -float(upper),
        str(row["similarity_condition_id"]),
    )


def adjacent_band(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if not left["similarity_enabled"] or not right["similarity_enabled"]:
        return False
    same_lower = close_enough(float(left["similarity_lower"]), float(right["similarity_lower"]))
    same_upper = close_enough(float(left["similarity_upper"]), float(right["similarity_upper"]))
    lower_step = close_enough(abs(float(left["similarity_lower"]) - float(right["similarity_lower"])), 0.1)
    upper_step = close_enough(abs(float(left["similarity_upper"]) - float(right["similarity_upper"])), 0.1)
    return (same_lower and upper_step) or (same_upper and lower_step)


def build_criteria_matrix(
    dataset_rows: Sequence[Mapping[str, Any]],
    pooled_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    dataset_index = {
        (str(row["cell_id"]), ratio_key(float(row["ratio"])), str(row["similarity_condition_id"]), str(row["metric"]), str(row["effect"])): row
        for row in dataset_rows
    }
    pooled_index = {
        (ratio_key(float(row["ratio"])), str(row["similarity_condition_id"]), str(row["metric"]), str(row["effect"])): row
        for row in pooled_rows
    }
    criteria: list[dict[str, Any]] = []
    for ratio in RATIOS:
        for condition in SIMILARITY_CONDITIONS:
            similarity_id = str(condition["similarity_condition_id"])
            effects = {
                effect: pooled_index[(ratio_key(ratio), similarity_id, "macro_f1", effect)]
                for effect in EFFECTS
            }
            complete = all(bool(row["analysis_complete"]) for row in effects.values())
            dataset_delta = [
                dataset_index[(cell_id, ratio_key(ratio), similarity_id, "macro_f1", "delta_synth")]
                for cell_id in TARGET_CELLS
            ]
            positive_count = sum(
                row["mean"] is not None and float(row["mean"]) > 0
                for row in dataset_delta
            )
            damaged_count = sum(
                row["ci_upper"] is not None and float(row["ci_upper"]) < 0
                for row in dataset_delta
            )
            delta_synth = effects["delta_synth"]
            delta_real = effects["delta_real"]
            g_real = effects["g_real"]
            c1 = complete and float(delta_synth["ci_lower"]) > 0
            c2 = complete and positive_count >= 4
            c3 = complete and damaged_count == 0
            c4 = complete and float(delta_real["ci_lower"]) > 0
            c5 = complete and float(g_real["ci_lower"]) > NONINFERIORITY_MARGIN
            criteria.append(
                {
                    "ratio": ratio,
                    "similarity_condition_id": similarity_id,
                    "similarity_enabled": bool(condition["similarity_enabled"]),
                    "similarity_lower": condition["lower"],
                    "similarity_upper": condition["upper"],
                    "analysis_complete": complete,
                    "selection_eligible": complete,
                    "pooled_delta_synth_mean": delta_synth["mean"],
                    "pooled_delta_synth_se": delta_synth["bootstrap_standard_error"],
                    "pooled_delta_synth_ci_lower": delta_synth["ci_lower"],
                    "pooled_delta_synth_ci_upper": delta_synth["ci_upper"],
                    "pooled_delta_real_mean": delta_real["mean"],
                    "pooled_delta_real_ci_lower": delta_real["ci_lower"],
                    "pooled_delta_real_ci_upper": delta_real["ci_upper"],
                    "pooled_g_real_mean": g_real["mean"],
                    "pooled_g_real_ci_lower": g_real["ci_lower"],
                    "pooled_g_real_ci_upper": g_real["ci_upper"],
                    "positive_dataset_count": positive_count,
                    "damaged_dataset_count": damaged_count,
                    "c1_pass": c1,
                    "c2_pass": c2,
                    "c3_pass": c3,
                    "c4_pass": c4,
                    "c5_pass": c5,
                    "c1_to_c5_pass": c1 and c2 and c3 and c4 and c5,
                    "c6_pass": False,
                    "c7_pass": None,
                    "stable_candidate": False,
                    "stable_region_id": None,
                    "ratio_run_id": None,
                    "ratio_run_start": None,
                    "ratio_run_end": None,
                    "ratio_run_length": None,
                    "adjacent_band_pass_count": None,
                    "adjacent_supported_ratio_count": None,
                }
            )

    by_similarity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in criteria:
        by_similarity[str(row["similarity_condition_id"])].append(row)
    for similarity_id, rows in by_similarity.items():
        rows.sort(key=lambda row: RATIOS.index(float(row["ratio"])))
        run_number = 0
        index = 0
        while index < len(rows):
            if not rows[index]["c1_to_c5_pass"]:
                index += 1
                continue
            end = index
            while end + 1 < len(rows) and rows[end + 1]["c1_to_c5_pass"]:
                end += 1
            run_number += 1
            run = rows[index:end + 1]
            run_id = f"{similarity_id}-R{run_number:02d}"
            for member in run:
                member["ratio_run_id"] = run_id
                member["ratio_run_start"] = run[0]["ratio"]
                member["ratio_run_end"] = run[-1]["ratio"]
                member["ratio_run_length"] = len(run)
                member["c6_pass"] = len(run) >= 3
            index = end + 1

    criteria_index = {
        (ratio_key(float(row["ratio"])), str(row["similarity_condition_id"])): row
        for row in criteria
    }
    for rows in by_similarity.values():
        for row in rows:
            if not row["similarity_enabled"]:
                row["stable_candidate"] = bool(row["c1_to_c5_pass"] and row["c6_pass"])
                continue
            if not row["c6_pass"]:
                row["c7_pass"] = False
                row["adjacent_band_pass_count"] = 0
                row["adjacent_supported_ratio_count"] = 0
                continue
            run = [candidate for candidate in rows if candidate["ratio_run_id"] == row["ratio_run_id"]]
            pass_count = 0
            supported_ratios = 0
            for run_member in run:
                adjacent_passes = 0
                for condition in SIMILARITY_CONDITIONS:
                    candidate = criteria_index[(ratio_key(float(run_member["ratio"])), str(condition["similarity_condition_id"]))]
                    if adjacent_band(run_member, candidate) and candidate["c1_to_c5_pass"]:
                        adjacent_passes += 1
                pass_count += adjacent_passes
                supported_ratios += adjacent_passes > 0
            c7 = supported_ratios >= 2
            for run_member in run:
                run_member["adjacent_band_pass_count"] = pass_count
                run_member["adjacent_supported_ratio_count"] = supported_ratios
                run_member["c7_pass"] = c7
                run_member["stable_candidate"] = bool(run_member["c1_to_c5_pass"] and run_member["c6_pass"] and c7)
    criteria.sort(key=condition_sort_key)
    return criteria


def build_stable_regions(
    criteria: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stable = [row for row in criteria if row["stable_candidate"]]
    remaining = {analysis_key(row): row for row in stable}
    components: list[list[dict[str, Any]]] = []
    while remaining:
        start_key = min(remaining, key=lambda key: condition_sort_key(remaining[key]))
        queue = deque([remaining.pop(start_key)])
        component: list[dict[str, Any]] = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for key, candidate in list(remaining.items()):
                same_mode = bool(current["similarity_enabled"]) == bool(candidate["similarity_enabled"])
                ratio_distance = abs(RATIOS.index(float(current["ratio"])) - RATIOS.index(float(candidate["ratio"])))
                if not same_mode:
                    connected = False
                elif not current["similarity_enabled"]:
                    connected = ratio_distance == 1
                else:
                    same_band = current["similarity_condition_id"] == candidate["similarity_condition_id"]
                    same_ratio = close_enough(float(current["ratio"]), float(candidate["ratio"]))
                    connected = (same_band and ratio_distance == 1) or (same_ratio and adjacent_band(current, candidate))
                if connected:
                    queue.append(remaining.pop(key))
        components.append(sorted(component, key=condition_sort_key))

    regions: list[dict[str, Any]] = []
    members: list[dict[str, Any]] = []
    for number, component in enumerate(components, 1):
        region_id = f"SR{number:03d}"
        for row in component:
            row["stable_region_id"] = region_id
            members.append(
                {
                    "stable_region_id": region_id,
                    "filter_mode": "ON" if row["similarity_enabled"] else "OFF",
                    "ratio": row["ratio"],
                    "similarity_condition_id": row["similarity_condition_id"],
                    "similarity_enabled": row["similarity_enabled"],
                    "similarity_lower": row["similarity_lower"],
                    "similarity_upper": row["similarity_upper"],
                    "pooled_delta_synth_mean": row["pooled_delta_synth_mean"],
                    "condition_key": f"{ratio_key(float(row['ratio']))}|{row['similarity_condition_id']}",
                }
            )
        regions.append(
            {
                "stable_region_id": region_id,
                "filter_mode": "ON" if component[0]["similarity_enabled"] else "OFF",
                "cell_count": len(component),
                "ratio_count": len({float(row["ratio"]) for row in component}),
                "band_count": len({str(row["similarity_condition_id"]) for row in component}),
                "minimum_ratio": min(float(row["ratio"]) for row in component),
                "maximum_ratio": max(float(row["ratio"]) for row in component),
                "member_condition_keys": [
                    f"{ratio_key(float(row['ratio']))}|{row['similarity_condition_id']}"
                    for row in component
                ],
                "maximum_pooled_delta_synth": max(float(row["pooled_delta_synth_mean"]) for row in component),
                "mean_pooled_delta_synth": sum(float(row["pooled_delta_synth_mean"]) for row in component) / len(component),
            }
        )
    return regions, members


def select_condition(criteria: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    stable = [row for row in criteria if row["stable_candidate"]]
    candidates: list[dict[str, Any]] = []
    if stable:
        best = min(stable, key=lambda row: (-float(row["pooled_delta_synth_mean"]), selection_priority(row)))
        threshold = float(best["pooled_delta_synth_mean"]) - float(best["pooled_delta_synth_se"])
        equivalent = [row for row in stable if row["c5_pass"] and float(row["pooled_delta_synth_mean"]) >= threshold]
        selected = min(equivalent, key=selection_priority)
        method = "stable_one_standard_error"
        reason = None
        for row in stable:
            candidates.append(
                {
                    "ratio": row["ratio"], "similarity_condition_id": row["similarity_condition_id"],
                    "similarity_enabled": row["similarity_enabled"], "similarity_lower": row["similarity_lower"],
                    "similarity_upper": row["similarity_upper"], "stable_region_id": row["stable_region_id"],
                    "pooled_delta_synth_mean": row["pooled_delta_synth_mean"],
                    "pooled_delta_synth_se": row["pooled_delta_synth_se"], "one_se_threshold": threshold,
                    "equivalent_candidate": row in equivalent, "selected": row is selected,
                }
            )
    else:
        matches = [row for row in criteria if close_enough(float(row["ratio"]), 1.0) and not row["similarity_enabled"]]
        if len(matches) != 1 or not matches[0]["analysis_complete"]:
            raise RuntimeError("fixed fallback ratio=1.0, similarity=OFF is incomplete")
        selected = matches[0]
        best = None
        threshold = None
        method = "fixed_fallback"
        reason = "no_stable_candidate"
    payload = {
        "selection_method": method,
        "fallback_reason": reason,
        "ratio": selected["ratio"],
        "similarity_condition_id": selected["similarity_condition_id"],
        "similarity_enabled": selected["similarity_enabled"],
        "similarity_lower": selected["similarity_lower"],
        "similarity_upper": selected["similarity_upper"],
        "stable_region_id": selected["stable_region_id"],
        "pooled_delta_synth_mean": selected["pooled_delta_synth_mean"],
        "pooled_delta_synth_standard_error": selected["pooled_delta_synth_se"],
        "best_stable_condition": None if best is None else {
            "ratio": best["ratio"], "similarity_condition_id": best["similarity_condition_id"],
            "pooled_delta_synth_mean": best["pooled_delta_synth_mean"],
            "pooled_delta_synth_standard_error": best["pooled_delta_synth_se"],
        },
        "one_standard_error_threshold": threshold,
    }
    candidates.sort(key=condition_sort_key)
    return payload, candidates


def package_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for package in ("numpy",):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def output_hashes(root: Path, names: Iterable[str]) -> dict[str, str]:
    return {name: sha256_file(root / name) for name in sorted(names)}


def build_selected_condition(
    selection: Mapping[str, Any],
    criteria: Sequence[Mapping[str, Any]],
    pooled_rows: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    development_lock_sha256: str,
) -> dict[str, Any]:
    selected = next(
        row for row in criteria
        if close_enough(float(row["ratio"]), float(selection["ratio"]))
        and row["similarity_condition_id"] == selection["similarity_condition_id"]
    )
    pooled_index = {
        (str(row["metric"]), str(row["effect"])): row
        for row in pooled_rows
        if close_enough(float(row["ratio"]), float(selected["ratio"]))
        and row["similarity_condition_id"] == selected["similarity_condition_id"]
    }
    delta_synth = pooled_index[("macro_f1", "delta_synth")]
    delta_real = pooled_index[("macro_f1", "delta_real")]
    g_real = pooled_index[("macro_f1", "g_real")]
    best = selection["best_stable_condition"]
    clear = selection["selection_method"] == "stable_one_standard_error"
    lower = selected["similarity_lower"]
    upper = selected["similarity_upper"]
    return {
        "experiment_name": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "selected_at": utc_now(),
        "selection_status": "stable_region_selected" if clear else "clear_effect_region_absent_fallback",
        "clear_effect_region_found": clear,
        "selected_ratio": selected["ratio"],
        "selected_similarity_condition_id": selected["similarity_condition_id"],
        "selected_similarity_enabled": selected["similarity_enabled"],
        "selected_similarity_lower": lower,
        "selected_similarity_upper": upper,
        "selected_band_width": None if lower is None else float(upper) - float(lower),
        "selected_stable_region_id": selected["stable_region_id"],
        "selected_pooled_delta_synth_mean": delta_synth["mean"],
        "selected_pooled_delta_synth_se": delta_synth["bootstrap_standard_error"],
        "selected_pooled_delta_synth_ci": [delta_synth["ci_lower"], delta_synth["ci_upper"]],
        "selected_pooled_delta_real_mean": delta_real["mean"],
        "selected_pooled_delta_real_ci": [delta_real["ci_lower"], delta_real["ci_upper"]],
        "selected_pooled_g_real_mean": g_real["mean"],
        "selected_pooled_g_real_ci": [g_real["ci_lower"], g_real["ci_upper"]],
        "selected_gain_recovery": delta_synth["gain_recovery"],
        "best_condition": best,
        "best_mean": None if best is None else best["pooled_delta_synth_mean"],
        "best_se": None if best is None else best["pooled_delta_synth_standard_error"],
        "one_se_threshold": selection["one_standard_error_threshold"],
        "equivalent_candidate_count": sum(bool(row["equivalent_candidate"]) for row in candidates),
        "criteria": {
            "c1": "pooled delta_synth Macro-F1 CI lower > 0",
            "c2": "at least 4 dataset delta_synth means > 0",
            "c3": "no dataset delta_synth CI upper < 0",
            "c4": "pooled delta_real Macro-F1 CI lower > 0",
            "c5": f"pooled g_real Macro-F1 CI lower > {NONINFERIORITY_MARGIN}",
            "c6": "at least 3 consecutive RATIOS within one similarity condition",
            "c7": "ON run has adjacent C1-C5 band support at at least 2 ratios",
        },
        "bootstrap_configuration": {
            "replicates": BOOTSTRAP_REPLICATES,
            "base_seed": BOOTSTRAP_BASE_SEED,
            "ci_level": CI_LEVEL,
            "pooled_dataset_weight": 0.2,
        },
        "input_lock_sha256": {"development_grid": development_lock_sha256},
    }


def report_rows(report: Mapping[str, Any]) -> list[dict[str, str]]:
    rows = []
    for key, value in report.items():
        rows.append({"field": key, "value": canonical_json(value) if isinstance(value, (dict, list)) else value})
    return rows


def run_analysis(
    experiment_root: Path,
    selection_root: Path,
    paired_rows: Sequence[Mapping[str, Any]],
    development_report: Mapping[str, Any],
    development_lock_sha256: str,
) -> dict[str, Any]:
    dataset_rows = build_dataset_summaries(paired_rows)
    pooled_rows = build_pooled_summaries(paired_rows, dataset_rows)
    criteria = build_criteria_matrix(dataset_rows, pooled_rows)
    regions, region_members = build_stable_regions(criteria)
    selection, candidates = select_condition(criteria)
    selected = build_selected_condition(
        selection, criteria, pooled_rows, candidates, development_lock_sha256
    )
    selection_root.mkdir(parents=True, exist_ok=True)
    write_csv(selection_root / "dataset_condition_summary.csv", dataset_rows, DATASET_FIELDS)
    write_csv(selection_root / "pooled_condition_summary.csv", pooled_rows, POOLED_FIELDS)
    write_csv(selection_root / "criteria_matrix.csv", criteria, CRITERIA_FIELDS)
    write_json(
        selection_root / "stable_regions.json",
        {"clear_effect_region_found": bool(regions), "stable_region_count": len(regions), "regions": regions},
    )
    write_csv(selection_root / "stable_region_members.csv", region_members, REGION_MEMBER_FIELDS)
    write_csv(selection_root / "one_se_candidates.csv", candidates, ONE_SE_FIELDS)
    write_json(selection_root / "selected_condition.json", selected)

    preliminary_names = [
        "dataset_condition_summary.csv", "pooled_condition_summary.csv",
        "criteria_matrix.csv", "stable_regions.json", "stable_region_members.csv",
        "one_se_candidates.csv", "selected_condition.json",
    ]
    report = {
        "experiment_name": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "created_at": utc_now(),
        "input_row_count": len(paired_rows),
        "complete_condition_count": sum(bool(row["analysis_complete"]) for row in criteria),
        "incomplete_condition_count": sum(not bool(row["analysis_complete"]) for row in criteria),
        "c1_pass_count": sum(bool(row["c1_pass"]) for row in criteria),
        "c2_pass_count": sum(bool(row["c2_pass"]) for row in criteria),
        "c3_pass_count": sum(bool(row["c3_pass"]) for row in criteria),
        "c4_pass_count": sum(bool(row["c4_pass"]) for row in criteria),
        "c5_pass_count": sum(bool(row["c5_pass"]) for row in criteria),
        "c1_to_c5_pass_count": sum(bool(row["c1_to_c5_pass"]) for row in criteria),
        "c6_pass_count": sum(bool(row["c6_pass"]) for row in criteria),
        "c7_pass_count": sum(row["c7_pass"] is True for row in criteria),
        "stable_candidate_count": sum(bool(row["stable_candidate"]) for row in criteria),
        "stable_region_count": len(regions),
        "one_se_candidate_count": sum(bool(row["equivalent_candidate"]) for row in candidates),
        "selected_condition": selected,
        "fallback_used": not selected["clear_effect_region_found"],
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_base_seed": BOOTSTRAP_BASE_SEED,
        "noninferiority_margin": NONINFERIORITY_MARGIN,
        "package_versions": package_versions(),
        "input_lock_hashes": {
            "development_grid": development_lock_sha256,
            **dict(development_report.get("input_lock_hashes") or {}),
        },
        "output_file_hashes": output_hashes(selection_root, preliminary_names),
    }
    write_json(selection_root / "development_selection_report.json", report)
    write_csv(
        selection_root / "development_selection_report.csv",
        report_rows(report),
        ("field", "value"),
    )
    all_names = preliminary_names + [
        "development_selection_report.json", "development_selection_report.csv"
    ]
    files = output_hashes(selection_root, all_names)
    write_json(
        selection_root / LOCK_NAME,
        {
            "lock_version": "binary-condition-selection-v1",
            "created_at": utc_now(),
            "experiment_name": EXPERIMENT_NAME,
            "experiment_id": EXPERIMENT_ID,
            "status": "completed",
            "development_grid_lock_sha256": development_lock_sha256,
            "analysis_configuration": {
                "ratios": list(RATIOS),
                "similarity_condition_count": len(SIMILARITY_CONDITIONS),
                "repeat_seeds": list(REPEAT_SEEDS),
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_base_seed": BOOTSTRAP_BASE_SEED,
                "ci_level": CI_LEVEL,
                "noninferiority_margin": NONINFERIORITY_MARGIN,
            },
            "selected_condition_sha256": sha256_file(selection_root / "selected_condition.json"),
            "files": files,
        },
    )
    verify_lock(selection_root, LOCK_NAME)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Summarize the Binary Matched-Size development grid")
    parser.add_argument("--output-root", type=Path, default=project_root / "Results" / PATH_ID)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    experiment_root = args.output_root.resolve()
    development_root = experiment_root / "Downstream" / "DevelopmentGrid"
    selection_root = experiment_root / "Downstream" / "DevelopmentSelection"
    try:
        paired_rows, development_report, development_lock_sha256 = validate_input(development_root)
        complete_count = complete_condition_count(paired_rows)
        if args.dry_run:
            print(json.dumps({
                "dry_run": True,
                "input_row_count": len(paired_rows),
                "complete_condition_count": complete_count,
                "incomplete_condition_count": len(RATIOS) * len(SIMILARITY_CONDITIONS) - complete_count,
                "planned_dataset_bootstraps": len(TARGET_CELLS) * len(RATIOS) * len(SIMILARITY_CONDITIONS) * len(EFFECTS) * len(METRICS),
                "planned_pooled_bootstraps": len(RATIOS) * len(SIMILARITY_CONDITIONS) * len(METRICS),
                "bootstrap_replicates_each": BOOTSTRAP_REPLICATES,
                "expected_output_root": str(selection_root),
            }, ensure_ascii=False, indent=2))
            return 0
        lock_path = selection_root / LOCK_NAME
        if lock_path.is_file() and not args.force:
            verify_lock(selection_root, LOCK_NAME)
            print(f"valid {LOCK_NAME} already exists; use --force to recompute")
            return 0
        if lock_path.exists():
            lock_path.unlink()
        report = run_analysis(
            experiment_root, selection_root, paired_rows,
            development_report, development_lock_sha256,
        )
        print(json.dumps({
            "selection_status": report["selected_condition"]["selection_status"],
            "stable_region_count": report["stable_region_count"],
            "selected_ratio": report["selected_condition"]["selected_ratio"],
            "selected_similarity_condition_id": report["selected_condition"]["selected_similarity_condition_id"],
        }, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"binary downstream summary failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
