"""Friedman assessment for the manuscript's final three-condition comparison.

The analysis follows the exact condition terminology used in the current
manuscript's ``Frozen One-Shot Test`` section and final comparison figure. Only the
following test conditions are compared:

* Base
* Selected Hybrid
* Matched All-real

The script reads the locked one-shot test results. Source-only arms that are not
part of this manuscript comparison are ignored. Augmentation-OFF datasets are
excluded because their selected condition reproduces Base exactly.

For each augmentation-ON dataset and each manuscript metric, ``repeat_seed`` is
the Friedman blocking factor. The four omnibus p-values (two datasets by two
metrics in the current experiment) are Holm-adjusted together. Significant
omnibus tests are followed by the three prespecified paired Wilcoxon tests,
with Holm correction within each dataset--metric family:

* delta_synth: Selected Hybrid minus Base
* delta_real: Matched All-real minus Base
* g_real: Selected Hybrid minus Matched All-real

Kendall's W and rank-biserial correlations are also reported.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import chi2, rankdata, wilcoxon


ANALYSIS_VERSION = "binary-final-three-condition-friedman-v1"
LOCK_NAME = "FRIEDMAN_ANALYSIS_LOCK.json"
SOURCE_LOCK_NAME = "ADAPTIVE_ONE_SHOT_TEST_LOCK.json"
EXPECTED_SOURCE_STATUS = "completed_one_shot_test_evaluation"
EXPECTED_REPEATS = 50

# The source one-shot CSV uses implementation-facing arm identifiers. They are
# mapped immediately to the exact condition names used in the manuscript.
SOURCE_ARM_TO_CONDITION = {
    "base": "Base",
    "frozen_adaptive": "Selected Hybrid",
    "matched_all_real": "Matched All-real",
}
CONDITION_ORDER = (
    "Base",
    "Selected Hybrid",
    "Matched All-real",
)
METRICS = (
    "macro_f1_at_oof_threshold",
    "auroc",
)
METRIC_LABELS = {
    "macro_f1_at_oof_threshold": "Macro-F1 at OOF threshold",
    "auroc": "AUROC",
}

# The orientation is fixed so that the reported mean difference directly equals
# the effect definition used in the experiment framework.
POSTHOC_COMPARISONS = (
    ("delta_synth", "Selected Hybrid", "Base"),
    ("delta_real", "Matched All-real", "Base"),
    ("g_real", "Selected Hybrid", "Matched All-real"),
)

OMNIBUS_FIELDS = [
    "analysis",
    "cell_id",
    "language",
    "metric",
    "metric_label",
    "n_blocks",
    "n_conditions",
    "condition_order",
    "friedman_chi_square",
    "degrees_of_freedom",
    "p_value",
    "p_value_holm",
    "holm_family",
    "kendalls_w",
    "significant_raw",
    "significant_holm",
]

MEAN_RANK_FIELDS = [
    "analysis",
    "cell_id",
    "language",
    "metric",
    "metric_label",
    "condition",
    "condition_index",
    "mean_rank",
    "rank_1_is_best",
]

POSTHOC_FIELDS = [
    "analysis",
    "cell_id",
    "language",
    "metric",
    "metric_label",
    "effect_name",
    "condition_a",
    "condition_b",
    "difference_definition",
    "n_pairs",
    "n_nonzero_pairs",
    "mean_a",
    "mean_b",
    "mean_difference_a_minus_b",
    "median_difference_a_minus_b",
    "wilcoxon_statistic",
    "p_value",
    "p_value_holm",
    "holm_family",
    "rank_biserial_correlation",
    "better_condition",
    "all_differences_zero",
    "omnibus_p_value",
    "omnibus_p_value_holm",
    "posthoc_trigger",
    "significant_holm",
]


@dataclass
class TestFamily:
    cell_id: str
    language: str
    metric: str
    conditions: tuple[str, ...]
    values: dict[str, np.ndarray]
    statistic: float = math.nan
    p_value: float = math.nan
    p_value_holm: float = math.nan
    kendalls_w: float = math.nan

    @property
    def n_blocks(self) -> int:
        return int(next(iter(self.values.values())).size)


# ---------------------------------------------------------------------------
# Reproducible I/O
# ---------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise RuntimeError(f"missing input CSV: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"missing JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def verify_source_lock(input_root: Path) -> dict[str, Any]:
    """Verify every file recorded in the frozen one-shot test lock."""
    lock_path = input_root / SOURCE_LOCK_NAME
    lock = read_json(lock_path)
    if str(lock.get("status")) != EXPECTED_SOURCE_STATUS:
        raise RuntimeError(
            f"unexpected one-shot lock status: {lock.get('status')!r}; "
            f"expected {EXPECTED_SOURCE_STATUS!r}"
        )
    if lock.get("policy_changed_after_test") is not False:
        raise RuntimeError("one-shot lock does not confirm policy_changed_after_test=False")
    files = lock.get("files")
    if not isinstance(files, dict):
        raise RuntimeError(f"invalid files map in {lock_path}")
    for relative, expected_hash in files.items():
        path = input_root / str(relative)
        if not path.is_file():
            raise RuntimeError(f"locked file is missing: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != str(expected_hash):
            raise RuntimeError(
                f"hash mismatch for {path}: expected={expected_hash}, actual={actual_hash}"
            )
    return lock


# ---------------------------------------------------------------------------
# Parsing and validation
# ---------------------------------------------------------------------------


def parse_finite_float(value: Any, field: str) -> float:
    text = str(value).strip()
    if not text:
        raise RuntimeError(f"{field} is blank")
    try:
        number = float(text)
    except ValueError as exc:
        raise RuntimeError(f"{field} is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise RuntimeError(f"{field} is not finite: {value!r}")
    return number


def infer_language(cell_id: str, row_language: str | None = None) -> str:
    candidate = str(row_language or "").strip()
    if candidate:
        return candidate
    prefix = cell_id.split("_", 1)[0].lower()
    return {
        "en": "English",
        "ko": "Korean",
        "bn": "Bengali",
        "ha": "Hausa",
        "ml": "Malayalam",
    }.get(prefix, prefix)


def validate_one_shot_input(
    input_root: Path,
) -> tuple[list[TestFamily], list[str], dict[str, int]]:
    raw_rows = read_csv(input_root / "test_repeat_results.csv")
    if not raw_rows:
        raise RuntimeError("test_repeat_results.csv is empty")

    required_columns = {
        "cell_id",
        "repeat_seed",
        "arm",
        "augmentation",
        *METRICS,
    }
    missing = required_columns - set(raw_rows[0])
    if missing:
        raise RuntimeError(f"test_repeat_results.csv columns missing: {sorted(missing)}")

    rows_by_cell: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row_number, row in enumerate(raw_rows, 2):
        cell_id = str(row.get("cell_id") or "").strip()
        if not cell_id:
            raise RuntimeError(f"row {row_number}: blank cell_id")
        rows_by_cell[cell_id].append(row)

    on_cells: list[str] = []
    excluded_off_cells: list[str] = []
    families: list[TestFamily] = []
    source_counts: dict[str, int] = {}

    included_source_arms = tuple(SOURCE_ARM_TO_CONDITION)

    for cell_id in sorted(rows_by_cell):
        rows = rows_by_cell[cell_id]
        augmentation_values = {
            str(row.get("augmentation") or "").strip().upper() for row in rows
        }
        if len(augmentation_values) != 1:
            raise RuntimeError(
                f"{cell_id}: inconsistent augmentation values {sorted(augmentation_values)}"
            )
        augmentation = next(iter(augmentation_values))
        source_counts[cell_id] = len(rows)

        if augmentation == "OFF":
            excluded_off_cells.append(cell_id)
            continue
        if augmentation != "ON":
            raise RuntimeError(f"{cell_id}: unexpected augmentation value {augmentation!r}")

        on_cells.append(cell_id)
        index: dict[tuple[int, str], dict[str, str]] = {}
        for row_number, row in enumerate(rows, 2):
            try:
                seed = int(str(row["repeat_seed"]).strip())
            except ValueError as exc:
                raise RuntimeError(
                    f"{cell_id}: non-integer repeat_seed {row['repeat_seed']!r}"
                ) from exc
            source_arm = str(row.get("arm") or "").strip()
            if source_arm not in SOURCE_ARM_TO_CONDITION:
                # The locked source may contain additional diagnostic comparators,
                # but they are outside the manuscript's final three-condition test.
                continue
            key = (seed, source_arm)
            if key in index:
                raise RuntimeError(
                    f"duplicate selected-comparison result key: {(cell_id, seed, source_arm)}"
                )
            index[key] = row

        seeds = sorted({seed for seed, _ in index})
        if len(seeds) != EXPECTED_REPEATS:
            raise RuntimeError(
                f"{cell_id}: expected {EXPECTED_REPEATS} repeated seeds, got {len(seeds)}"
            )
        expected_keys = {
            (seed, source_arm) for seed in seeds for source_arm in included_source_arms
        }
        if set(index) != expected_keys:
            missing_keys = sorted(expected_keys - set(index))[:10]
            extra_keys = sorted(set(index) - expected_keys)[:10]
            raise RuntimeError(
                f"{cell_id}: incomplete three-condition repeated block; "
                f"missing={missing_keys}, extra={extra_keys}"
            )

        language = infer_language(
            cell_id,
            next(
                (
                    row.get("language") or row.get("dataset_language")
                    for row in rows
                    if row.get("language") or row.get("dataset_language")
                ),
                None,
            ),
        )

        for metric in METRICS:
            values: dict[str, np.ndarray] = {}
            for source_arm, manuscript_condition in SOURCE_ARM_TO_CONDITION.items():
                values[manuscript_condition] = np.asarray(
                    [
                        parse_finite_float(index[(seed, source_arm)][metric], metric)
                        for seed in seeds
                    ],
                    dtype=np.float64,
                )
            families.append(
                TestFamily(
                    cell_id=cell_id,
                    language=language,
                    metric=metric,
                    conditions=CONDITION_ORDER,
                    values=values,
                )
            )

    if not on_cells:
        raise RuntimeError("no augmentation-ON cells were found")
    expected_family_count = len(on_cells) * len(METRICS)
    if len(families) != expected_family_count:
        raise RuntimeError(
            f"constructed {len(families)} test families; expected {expected_family_count}"
        )
    return families, excluded_off_cells, source_counts


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values in the original order."""
    if not p_values:
        return []
    values = np.asarray(p_values, dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
        raise RuntimeError("Holm adjustment received invalid p-values")
    order = np.argsort(values, kind="mergesort")
    adjusted_sorted = np.empty(values.size, dtype=np.float64)
    running = 0.0
    m = int(values.size)
    for position, original_index in enumerate(order):
        candidate = min(1.0, (m - position) * float(values[original_index]))
        running = max(running, candidate)
        adjusted_sorted[position] = running
    adjusted = np.empty(values.size, dtype=np.float64)
    for position, original_index in enumerate(order):
        adjusted[original_index] = adjusted_sorted[position]
    return adjusted.tolist()


def friedman_test(matrix: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    """Tie-corrected Friedman chi-square, p-value, Kendall's W, and mean ranks."""
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2:
        raise RuntimeError("Friedman matrix must be two-dimensional")
    n, k = values.shape
    if n < 2 or k < 3:
        raise RuntimeError(f"Friedman requires n>=2 and k>=3; got n={n}, k={k}")
    if not np.all(np.isfinite(values)):
        raise RuntimeError("Friedman matrix contains non-finite values")

    ascending_ranks = np.vstack([rankdata(row, method="average") for row in values])
    rank_sums = ascending_ranks.sum(axis=0)
    statistic = (
        12.0 / (n * k * (k + 1.0)) * float(np.sum(rank_sums**2))
        - 3.0 * n * (k + 1.0)
    )

    tie_sum = 0.0
    for row in values:
        _, counts = np.unique(row, return_counts=True)
        tie_sum += float(np.sum(counts**3 - counts))
    tie_correction = 1.0 - tie_sum / (n * (k**3 - k))
    if tie_correction <= 0.0:
        statistic = 0.0
        p_value = 1.0
        kendalls_w = 0.0
    else:
        statistic = max(0.0, statistic / tie_correction)
        p_value = float(chi2.sf(statistic, k - 1))
        kendalls_w = min(1.0, max(0.0, statistic / (n * (k - 1.0))))

    best_ranks = np.vstack([rankdata(-row, method="average") for row in values])
    mean_best_ranks = best_ranks.mean(axis=0)
    return float(statistic), p_value, float(kendalls_w), mean_best_ranks


def rank_biserial(differences: np.ndarray) -> float:
    nonzero = np.asarray(differences, dtype=np.float64)
    nonzero = nonzero[nonzero != 0.0]
    if nonzero.size == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero), method="average")
    positive = float(ranks[nonzero > 0.0].sum())
    negative = float(ranks[nonzero < 0.0].sum())
    denominator = positive + negative
    return 0.0 if denominator == 0.0 else (positive - negative) / denominator


def paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1:
        raise RuntimeError("paired Wilcoxon inputs must be equal-length vectors")
    differences = left - right
    nonzero_count = int(np.count_nonzero(differences))
    all_zero = nonzero_count == 0
    if all_zero:
        statistic = 0.0
        p_value = 1.0
    else:
        result = wilcoxon(
            left,
            right,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
            method="auto",
        )
        statistic = float(result.statistic)
        p_value = float(result.pvalue)

    mean_difference = float(np.mean(differences))
    median_difference = float(np.median(differences))
    if median_difference > 0.0 or (median_difference == 0.0 and mean_difference > 0.0):
        better = "a"
    elif median_difference < 0.0 or (median_difference == 0.0 and mean_difference < 0.0):
        better = "b"
    else:
        better = "tie"
    return {
        "n_pairs": int(left.size),
        "n_nonzero_pairs": nonzero_count,
        "mean_a": float(np.mean(left)),
        "mean_b": float(np.mean(right)),
        "mean_difference_a_minus_b": mean_difference,
        "median_difference_a_minus_b": median_difference,
        "wilcoxon_statistic": statistic,
        "p_value": p_value,
        "rank_biserial_correlation": float(rank_biserial(differences)),
        "better": better,
        "all_differences_zero": all_zero,
    }


def run_omnibus(
    families: Sequence[TestFamily], alpha: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rank_rows: list[dict[str, Any]] = []
    for family in families:
        matrix = np.column_stack([family.values[c] for c in family.conditions])
        statistic, p_value, kendalls_w, mean_ranks = friedman_test(matrix)
        family.statistic = statistic
        family.p_value = p_value
        family.kendalls_w = kendalls_w
        for index, (condition, mean_rank) in enumerate(
            zip(family.conditions, mean_ranks), 1
        ):
            rank_rows.append(
                {
                    "analysis": "final_three_condition",
                    "cell_id": family.cell_id,
                    "language": family.language,
                    "metric": family.metric,
                    "metric_label": METRIC_LABELS[family.metric],
                    "condition": condition,
                    "condition_index": index,
                    "mean_rank": float(mean_rank),
                    "rank_1_is_best": True,
                }
            )

    adjusted = holm_adjust([family.p_value for family in families])
    for family, p_adjusted in zip(families, adjusted):
        family.p_value_holm = p_adjusted

    omnibus_rows: list[dict[str, Any]] = []
    for family in families:
        omnibus_rows.append(
            {
                "analysis": "final_three_condition",
                "cell_id": family.cell_id,
                "language": family.language,
                "metric": family.metric,
                "metric_label": METRIC_LABELS[family.metric],
                "n_blocks": family.n_blocks,
                "n_conditions": len(family.conditions),
                "condition_order": canonical_json(list(family.conditions)),
                "friedman_chi_square": family.statistic,
                "degrees_of_freedom": len(family.conditions) - 1,
                "p_value": family.p_value,
                "p_value_holm": family.p_value_holm,
                "holm_family": "final_three_condition_omnibus",
                "kendalls_w": family.kendalls_w,
                "significant_raw": family.p_value < alpha,
                "significant_holm": family.p_value_holm < alpha,
            }
        )
    return omnibus_rows, rank_rows


def should_run_posthoc(family: TestFamily, trigger: str, alpha: float) -> bool:
    if trigger == "none":
        return False
    if trigger == "always":
        return True
    if trigger == "raw":
        return family.p_value < alpha
    if trigger == "holm":
        return family.p_value_holm < alpha
    raise RuntimeError(f"unexpected post-hoc trigger: {trigger}")


def build_posthoc_rows(
    families: Sequence[TestFamily], trigger: str, alpha: float
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for family in families:
        if not should_run_posthoc(family, trigger, alpha):
            continue
        pending: list[dict[str, Any]] = []
        for effect_name, condition_a, condition_b in POSTHOC_COMPARISONS:
            stats = paired_wilcoxon(
                family.values[condition_a], family.values[condition_b]
            )
            better_code = stats.pop("better")
            better_condition = (
                condition_a
                if better_code == "a"
                else condition_b
                if better_code == "b"
                else "tie"
            )
            pending.append(
                {
                    "analysis": "final_three_condition",
                    "cell_id": family.cell_id,
                    "language": family.language,
                    "metric": family.metric,
                    "metric_label": METRIC_LABELS[family.metric],
                    "effect_name": effect_name,
                    "condition_a": condition_a,
                    "condition_b": condition_b,
                    "difference_definition": f"{condition_a} - {condition_b}",
                    **stats,
                    "better_condition": better_condition,
                    "holm_family": (
                        f"final_three_condition|{family.cell_id}|{family.metric}"
                    ),
                    "omnibus_p_value": family.p_value,
                    "omnibus_p_value_holm": family.p_value_holm,
                    "posthoc_trigger": trigger,
                }
            )
        adjusted = holm_adjust([float(row["p_value"]) for row in pending])
        for row, p_adjusted in zip(pending, adjusted):
            row["p_value_holm"] = p_adjusted
            row["significant_holm"] = p_adjusted < alpha
        output.extend(pending)
    return output


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("numpy", "scipy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    return versions


def output_hashes(root: Path, names: Iterable[str]) -> dict[str, str]:
    return {name: sha256_file(root / name) for name in names}


def sort_outputs(
    omnibus: list[dict[str, Any]],
    ranks: list[dict[str, Any]],
    posthoc: list[dict[str, Any]],
) -> None:
    metric_order = {name: index for index, name in enumerate(METRICS)}
    condition_order = {name: index for index, name in enumerate(CONDITION_ORDER)}
    effect_order = {name: index for index, (name, _, _) in enumerate(POSTHOC_COMPARISONS)}
    omnibus.sort(key=lambda row: (str(row["cell_id"]), metric_order[str(row["metric"])]))
    ranks.sort(
        key=lambda row: (
            str(row["cell_id"]),
            metric_order[str(row["metric"])],
            condition_order[str(row["condition"])],
        )
    )
    posthoc.sort(
        key=lambda row: (
            str(row["cell_id"]),
            metric_order[str(row["metric"])],
            effect_order[str(row["effect_name"])],
        )
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    default_input = (
        project_root
        / "Results"
        / "BinaryMatchedSizeExperiment"
        / "Downstream"
        / "AdaptivePolicy"
        / "OneShotTest"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Run the manuscript-aligned three-condition Friedman/Wilcoxon-Holm "
            "assessment on the locked one-shot test results"
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=default_input,
        help="Directory containing test_repeat_results.csv and the one-shot lock",
    )
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=None,
        help="Output directory (default: <input-root>/FinalThreeConditionFriedmanV1)",
    )
    parser.add_argument(
        "--posthoc-trigger",
        choices=("holm", "raw", "always", "none"),
        default="holm",
        help="When to run the three prespecified paired Wilcoxon tests",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--version", action="version", version=ANALYSIS_VERSION)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--skip-source-lock-validation",
        action="store_true",
        help="Allow exploratory use on an unlocked copy (not recommended for manuscript results)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not 0.0 < args.alpha < 1.0:
        raise SystemExit("--alpha must be between 0 and 1")

    input_root = args.input_root.resolve()
    analysis_root = (
        args.analysis_root.resolve()
        if args.analysis_root is not None
        else input_root / "FinalThreeConditionFriedmanV1"
    )

    print(f"SCRIPT_VERSION={ANALYSIS_VERSION}")
    print(f"SCRIPT_PATH={Path(__file__).resolve()}")
    print(f"INPUT_ROOT={input_root}")
    print(f"ANALYSIS_ROOT={analysis_root}")
    print(f"CONDITIONS={canonical_json(list(CONDITION_ORDER))}")

    try:
        source_lock = (
            None if args.skip_source_lock_validation else verify_source_lock(input_root)
        )
        families, excluded_off_cells, source_counts = validate_one_shot_input(input_root)
        on_cells = sorted({family.cell_id for family in families})

        if args.dry_run:
            print(
                json.dumps(
                    {
                        "dry_run": True,
                        "input_root": str(input_root),
                        "analysis_root": str(analysis_root),
                        "included_on_cells": on_cells,
                        "excluded_off_cells": excluded_off_cells,
                        "conditions": list(CONDITION_ORDER),
                        "metrics": list(METRICS),
                        "omnibus_test_count": len(families),
                        "maximum_posthoc_pair_count": (
                            len(families) * len(POSTHOC_COMPARISONS)
                        ),
                        "posthoc_trigger": args.posthoc_trigger,
                        "alpha": args.alpha,
                        "source_lock_validated": not args.skip_source_lock_validation,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

        lock_path = analysis_root / LOCK_NAME
        if lock_path.is_file() and not args.force:
            lock = read_json(lock_path)
            files = lock.get("files")
            if not isinstance(files, dict):
                raise RuntimeError(f"invalid files map in {lock_path}")
            for name, expected_hash in files.items():
                path = analysis_root / str(name)
                if not path.is_file() or sha256_file(path) != str(expected_hash):
                    raise RuntimeError(f"existing analysis lock is invalid: {path}")
            print(f"valid {LOCK_NAME} already exists; use --force to recompute")
            return 0

        analysis_root.mkdir(parents=True, exist_ok=True)
        if lock_path.exists():
            lock_path.unlink()

        omnibus_rows, rank_rows = run_omnibus(families, args.alpha)
        posthoc_rows = build_posthoc_rows(families, args.posthoc_trigger, args.alpha)
        sort_outputs(omnibus_rows, rank_rows, posthoc_rows)

        output_names = [
            "friedman_omnibus.csv",
            "friedman_mean_ranks.csv",
            "friedman_posthoc_wilcoxon_holm.csv",
            "friedman_report.json",
        ]
        write_csv(analysis_root / output_names[0], omnibus_rows, OMNIBUS_FIELDS)
        write_csv(analysis_root / output_names[1], rank_rows, MEAN_RANK_FIELDS)
        write_csv(analysis_root / output_names[2], posthoc_rows, POSTHOC_FIELDS)

        report = {
            "report_version": ANALYSIS_VERSION,
            "created_at": utc_now(),
            "source_experiment": "Frozen One-Shot Test",
            "analysis": "final_three_condition",
            "scope": (
                "Only Base, Selected Hybrid, and Matched All-real from augmentation-ON "
                "datasets in the locked one-shot test are compared. Development-grid "
                "searches and source-only diagnostic comparators are excluded."
            ),
            "included_on_cells": on_cells,
            "excluded_off_cells": excluded_off_cells,
            "source_row_counts_by_cell": source_counts,
            "conditions": list(CONDITION_ORDER),
            "blocking_factor": "repeat_seed",
            "expected_blocks_per_cell": EXPECTED_REPEATS,
            "metrics": list(METRICS),
            "metric_labels": METRIC_LABELS,
            "alpha": args.alpha,
            "posthoc_trigger": args.posthoc_trigger,
            "multiple_testing": {
                "omnibus": "Holm correction across all ON-cell x metric Friedman tests",
                "posthoc": (
                    "Holm correction across delta_synth, delta_real, and g_real "
                    "within each significant omnibus test"
                ),
            },
            "posthoc_comparisons": [
                {
                    "effect_name": effect_name,
                    "condition_a": condition_a,
                    "condition_b": condition_b,
                    "difference": f"{condition_a} - {condition_b}",
                }
                for effect_name, condition_a, condition_b in POSTHOC_COMPARISONS
            ],
            "rank_direction": "rank 1 is best; larger metric values are better",
            "source_lock_validated": not args.skip_source_lock_validation,
            "source_lock_sha256": (
                sha256_file(input_root / SOURCE_LOCK_NAME)
                if source_lock is not None
                else None
            ),
            "input_hashes": {
                "test_repeat_results.csv": sha256_file(
                    input_root / "test_repeat_results.csv"
                ),
                **(
                    {SOURCE_LOCK_NAME: sha256_file(input_root / SOURCE_LOCK_NAME)}
                    if source_lock is not None
                    else {}
                ),
            },
            "package_versions": package_versions(),
            "counts": {
                "omnibus_tests": len(omnibus_rows),
                "raw_significant_omnibus": sum(
                    bool(row["significant_raw"]) for row in omnibus_rows
                ),
                "holm_significant_omnibus": sum(
                    bool(row["significant_holm"]) for row in omnibus_rows
                ),
                "mean_rank_rows": len(rank_rows),
                "posthoc_tests": len(posthoc_rows),
                "holm_significant_posthoc": sum(
                    bool(row["significant_holm"]) for row in posthoc_rows
                ),
            },
        }
        write_json(analysis_root / output_names[3], report)

        files = output_hashes(analysis_root, output_names)
        write_json(
            analysis_root / LOCK_NAME,
            {
                "lock_version": ANALYSIS_VERSION,
                "created_at": utc_now(),
                "status": "completed",
                "analysis_configuration": {
                    "analysis": "final_three_condition",
                    "included_on_cells": on_cells,
                    "excluded_off_cells": excluded_off_cells,
                    "conditions": list(CONDITION_ORDER),
                    "metrics": list(METRICS),
                    "blocking_factor": "repeat_seed",
                    "expected_blocks_per_cell": EXPECTED_REPEATS,
                    "alpha": args.alpha,
                    "posthoc_trigger": args.posthoc_trigger,
                },
                "input_hashes": report["input_hashes"],
                "files": files,
            },
        )

        print(
            json.dumps(
                {
                    "analysis_status": "completed",
                    "analysis_root": str(analysis_root),
                    "included_on_cells": on_cells,
                    "excluded_off_cells": excluded_off_cells,
                    **report["counts"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except Exception as exc:
        print(
            f"Final three-condition Friedman analysis failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())