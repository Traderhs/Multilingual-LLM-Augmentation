"""Freeze a conservative validation-gated policy for binary synthetic augmentation.

This script reads only completed development-grid artifacts. It never reads test
labels, test embeddings, or test results. The output lock is intended to be
created before any one-shot test evaluation.

Policy overview
---------------
1. Candidate set: one real-only Base arm plus every positive-weight
   (ratio, synthetic_weight) condition in paired_weight_results.csv.
2. Per dataset, choose a preliminary candidate by a one-standard-error rule:
   keep candidates within one SE of the best development Macro-F1 and choose
   the least synthetic exposure. Base wins whenever it is eligible.
3. A positive candidate is provisionally allowed only when all safety guards
   pass: paired bootstrap CI vs Base is positive, at least two adjacent grid
   neighbors also have positive bootstrap CIs, and AUROC is non-inferior within
   the frozen margin.
4. The decision-boundary diagnostic is still required. This script marks
   positive candidates as ``pending_diagnostic`` rather than final ON.

The script deliberately does not search for a fallback candidate after a guard
fails. Falling back to another positive condition after seeing a failure would
turn the procedure into post-hoc result hunting; the safe fallback is Base.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


EXPERIMENT_NAME = "Binary Validation-Gated Adaptive Augmentation"
EXPERIMENT_ID = "binary_validation_gated_adaptive_augmentation"
LOCK_VERSION = "binary-adaptive-protocol-v1"
PROTOCOL_SEED = 20260713
PRIMARY_METRIC = "macro_f1"
BOOTSTRAP_REPS = 10_000
MIN_STABLE_NEIGHBORS = 2
AUROC_NONINFERIORITY_MARGIN = 0.010
EXPECTED_REPEATS = 50

REQUIRED_COLUMNS = {
    "cell_id",
    "repeat_seed",
    "ratio",
    "synthetic_weight",
    "base_macro_f1",
    "weighted_hybrid_macro_f1",
    "delta_vs_base_macro_f1",
    "base_auroc",
    "weighted_hybrid_auroc",
    "delta_vs_base_auroc",
    "weighted_status",
    "selection_sha256",
}


@dataclass(frozen=True)
class BootstrapSummary:
    mean: float
    standard_error: float
    ci_lower: float
    ci_upper: float


@dataclass(frozen=True)
class CandidateChoice:
    cell_id: str
    selected_arm: str
    selected_ratio: float
    selected_synthetic_weight: float
    selected_effective_synthetic_ratio: float
    selected_mean_macro_f1: float
    selected_standard_error_macro_f1: float
    best_mean_macro_f1: float
    best_standard_error_macro_f1: float
    one_se_floor: float
    delta_vs_base_macro_f1_mean: float
    delta_vs_base_macro_f1_ci_lower: float
    delta_vs_base_macro_f1_ci_upper: float
    delta_vs_base_auroc_mean: float
    delta_vs_base_auroc_ci_lower: float
    delta_vs_base_auroc_ci_upper: float
    positive_ci_neighbor_count: int
    provisional_policy_state: str
    guard_positive_macro_f1_ci: bool
    guard_local_stability: bool
    guard_auroc_noninferiority: bool
    diagnostic_required: bool
    final_test_allowed: bool
    fallback_reason: str


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


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def verify_weight_grid_lock(weight_grid_root: Path) -> dict[str, Any]:
    lock_path = weight_grid_root / "SYNTHETIC_WEIGHT_GRID_LOCK.json"
    if not lock_path.is_file():
        raise FileNotFoundError(f"Missing weight-grid lock: {lock_path}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "completed":
        raise RuntimeError(f"Weight-grid lock is not completed: {lock.get('status')!r}")
    files = lock.get("files")
    if not isinstance(files, dict):
        raise RuntimeError("Weight-grid lock has no files map")
    for relative, expected in files.items():
        path = weight_grid_root / relative
        if not path.is_file():
            # The SQLite database is not needed for policy freezing. It still has
            # to exist in the original result directory; copied analysis bundles
            # may legitimately omit it.
            if relative == "synthetic_weight_status.sqlite3":
                continue
            raise FileNotFoundError(f"Locked file is missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"Hash mismatch for {relative}: expected={expected}, actual={actual}"
            )
    return lock


def load_results(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"paired_weight_results.csv is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("paired_weight_results.csv is empty")
    if not (frame["weighted_status"] == "completed").all():
        bad = frame.loc[frame["weighted_status"] != "completed", "weighted_status"].value_counts()
        raise RuntimeError(f"Non-completed weighted rows exist: {bad.to_dict()}")

    duplicate_key = ["cell_id", "repeat_seed", "ratio", "synthetic_weight"]
    duplicates = frame.duplicated(duplicate_key, keep=False)
    if duplicates.any():
        raise RuntimeError(
            "Duplicate condition rows exist: "
            + frame.loc[duplicates, duplicate_key].head(20).to_json(orient="records")
        )

    for cell_id, cell in frame.groupby("cell_id", sort=True):
        repeat_count = cell["repeat_seed"].nunique()
        if repeat_count != EXPECTED_REPEATS:
            raise RuntimeError(
                f"{cell_id}: expected {EXPECTED_REPEATS} repeats, observed {repeat_count}"
            )
        counts = cell.groupby(["ratio", "synthetic_weight"])["repeat_seed"].nunique()
        if not (counts == EXPECTED_REPEATS).all():
            raise RuntimeError(f"{cell_id}: incomplete ratio-weight condition detected")
        # Weight zero is a reused Base anchor and therefore must be exactly equal.
        zero = cell[np.isclose(cell["synthetic_weight"], 0.0)]
        if not np.allclose(
            zero["weighted_hybrid_macro_f1"], zero["base_macro_f1"], rtol=0.0, atol=1e-12
        ):
            raise RuntimeError(f"{cell_id}: weight-zero Macro-F1 is not the Base anchor")
    return frame


def paired_bootstrap(values: np.ndarray, *, key: Sequence[object]) -> BootstrapSummary:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size != EXPECTED_REPEATS:
        raise ValueError(f"Expected {EXPECTED_REPEATS} paired values, got {values.shape}")
    rng = np.random.default_rng(stable_seed(*key))
    indices = rng.integers(0, values.size, size=(BOOTSTRAP_REPS, values.size))
    means = values[indices].mean(axis=1)
    return BootstrapSummary(
        mean=float(values.mean()),
        standard_error=float(values.std(ddof=1) / math.sqrt(values.size)),
        ci_lower=float(np.quantile(means, 0.025)),
        ci_upper=float(np.quantile(means, 0.975)),
    )


def summarize_grid(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[tuple[str, float, float], dict[str, BootstrapSummary]]]:
    rows: list[dict[str, Any]] = []
    boot: dict[tuple[str, float, float], dict[str, BootstrapSummary]] = {}
    for (cell_id, ratio, weight), condition in frame.groupby(
        ["cell_id", "ratio", "synthetic_weight"], sort=True
    ):
        condition = condition.sort_values("repeat_seed")
        macro = paired_bootstrap(
            condition["delta_vs_base_macro_f1"].to_numpy(),
            key=(cell_id, ratio, weight, "macro_f1"),
        )
        auroc = paired_bootstrap(
            condition["delta_vs_base_auroc"].to_numpy(),
            key=(cell_id, ratio, weight, "auroc"),
        )
        boot[(str(cell_id), float(ratio), float(weight))] = {
            "macro_f1": macro,
            "auroc": auroc,
        }
        rows.append(
            {
                "cell_id": str(cell_id),
                "ratio": float(ratio),
                "synthetic_weight": float(weight),
                "effective_synthetic_ratio": float(ratio) * float(weight),
                "mean_macro_f1": float(condition["weighted_hybrid_macro_f1"].mean()),
                "se_macro_f1": float(
                    condition["weighted_hybrid_macro_f1"].std(ddof=1)
                    / math.sqrt(EXPECTED_REPEATS)
                ),
                "mean_auroc": float(condition["weighted_hybrid_auroc"].mean()),
                "delta_vs_base_macro_f1_mean": macro.mean,
                "delta_vs_base_macro_f1_se": macro.standard_error,
                "delta_vs_base_macro_f1_ci_lower": macro.ci_lower,
                "delta_vs_base_macro_f1_ci_upper": macro.ci_upper,
                "delta_vs_base_auroc_mean": auroc.mean,
                "delta_vs_base_auroc_se": auroc.standard_error,
                "delta_vs_base_auroc_ci_lower": auroc.ci_lower,
                "delta_vs_base_auroc_ci_upper": auroc.ci_upper,
            }
        )
    return pd.DataFrame(rows), boot


def unique_candidates(cell_summary: pd.DataFrame) -> pd.DataFrame:
    positive = cell_summary[cell_summary["synthetic_weight"] > 0].copy()
    zero = cell_summary[np.isclose(cell_summary["synthetic_weight"], 0.0)]
    if zero.empty:
        raise RuntimeError("No weight-zero Base anchor exists")
    first = zero.sort_values(["ratio", "synthetic_weight"], kind="stable").iloc[0].copy()
    first["ratio"] = 0.0
    first["synthetic_weight"] = 0.0
    first["effective_synthetic_ratio"] = 0.0
    return pd.concat([pd.DataFrame([first]), positive], ignore_index=True)


def adjacent_conditions(
    summary: pd.DataFrame,
    ratio: float,
    weight: float,
) -> pd.DataFrame:
    ratios = sorted(float(value) for value in summary["ratio"].unique())
    weights = sorted(float(value) for value in summary["synthetic_weight"].unique() if value > 0)
    ratio_index = ratios.index(float(ratio))
    weight_index = weights.index(float(weight))
    keys: set[tuple[float, float]] = set()
    if ratio_index > 0:
        keys.add((ratios[ratio_index - 1], float(weight)))
    if ratio_index + 1 < len(ratios):
        keys.add((ratios[ratio_index + 1], float(weight)))
    if weight_index > 0:
        keys.add((float(ratio), weights[weight_index - 1]))
    if weight_index + 1 < len(weights):
        keys.add((float(ratio), weights[weight_index + 1]))
    mask = pd.Series(False, index=summary.index)
    for neighbor_ratio, neighbor_weight in keys:
        mask |= np.isclose(summary["ratio"], neighbor_ratio) & np.isclose(
            summary["synthetic_weight"], neighbor_weight
        )
    return summary[mask].copy()


def choose_cell(
    cell_id: str,
    cell_summary: pd.DataFrame,
) -> CandidateChoice:
    candidates = unique_candidates(cell_summary)
    best = candidates.sort_values(
        ["mean_macro_f1", "effective_synthetic_ratio", "ratio", "synthetic_weight"],
        ascending=[False, True, True, True],
        kind="stable",
    ).iloc[0]
    one_se_floor = float(best["mean_macro_f1"] - best["se_macro_f1"])
    eligible = candidates[candidates["mean_macro_f1"] >= one_se_floor - 1e-15].copy()
    selected = eligible.sort_values(
        ["effective_synthetic_ratio", "ratio", "synthetic_weight", "mean_macro_f1"],
        ascending=[True, True, True, False],
        kind="stable",
    ).iloc[0]

    selected_ratio = float(selected["ratio"])
    selected_weight = float(selected["synthetic_weight"])
    is_base = math.isclose(selected_weight, 0.0, abs_tol=1e-15)

    if is_base:
        positive_ci_neighbor_count = 0
        guard_macro = False
        guard_neighbors = False
        guard_auroc = True
        state = "augmentation_off_base_selected"
        fallback_reason = "base_within_one_standard_error_of_best"
        diagnostic_required = False
        final_test_allowed = True
    else:
        neighbors = adjacent_conditions(cell_summary, selected_ratio, selected_weight)
        positive_ci_neighbor_count = int(
            (neighbors["delta_vs_base_macro_f1_ci_lower"] > 0.0).sum()
        )
        guard_macro = bool(selected["delta_vs_base_macro_f1_ci_lower"] > 0.0)
        guard_neighbors = positive_ci_neighbor_count >= MIN_STABLE_NEIGHBORS
        guard_auroc = bool(
            selected["delta_vs_base_auroc_ci_lower"] >= -AUROC_NONINFERIORITY_MARGIN
        )
        if guard_macro and guard_neighbors and guard_auroc:
            state = "pending_decision_boundary_diagnostic"
            fallback_reason = ""
            diagnostic_required = True
            final_test_allowed = False
        else:
            state = "augmentation_off_safety_guard_failed"
            failures = []
            if not guard_macro:
                failures.append("macro_f1_ci_not_positive")
            if not guard_neighbors:
                failures.append("insufficient_positive_ci_neighbors")
            if not guard_auroc:
                failures.append("auroc_noninferiority_failed")
            fallback_reason = ";".join(failures)
            diagnostic_required = False
            final_test_allowed = True

    return CandidateChoice(
        cell_id=cell_id,
        selected_arm="base" if is_base else "weighted_hybrid",
        selected_ratio=selected_ratio,
        selected_synthetic_weight=selected_weight,
        selected_effective_synthetic_ratio=float(selected["effective_synthetic_ratio"]),
        selected_mean_macro_f1=float(selected["mean_macro_f1"]),
        selected_standard_error_macro_f1=float(selected["se_macro_f1"]),
        best_mean_macro_f1=float(best["mean_macro_f1"]),
        best_standard_error_macro_f1=float(best["se_macro_f1"]),
        one_se_floor=one_se_floor,
        delta_vs_base_macro_f1_mean=float(selected["delta_vs_base_macro_f1_mean"]),
        delta_vs_base_macro_f1_ci_lower=float(selected["delta_vs_base_macro_f1_ci_lower"]),
        delta_vs_base_macro_f1_ci_upper=float(selected["delta_vs_base_macro_f1_ci_upper"]),
        delta_vs_base_auroc_mean=float(selected["delta_vs_base_auroc_mean"]),
        delta_vs_base_auroc_ci_lower=float(selected["delta_vs_base_auroc_ci_lower"]),
        delta_vs_base_auroc_ci_upper=float(selected["delta_vs_base_auroc_ci_upper"]),
        positive_ci_neighbor_count=positive_ci_neighbor_count,
        provisional_policy_state=state,
        guard_positive_macro_f1_ci=guard_macro,
        guard_local_stability=guard_neighbors,
        guard_auroc_noninferiority=guard_auroc,
        diagnostic_required=diagnostic_required,
        final_test_allowed=final_test_allowed,
        fallback_reason=fallback_reason,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weight-grid-root",
        type=Path,
        required=True,
        help="Directory containing paired_weight_results.csv and SYNTHETIC_WEIGHT_GRID_LOCK.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="New, dedicated adaptive-policy output directory",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory only when intentionally rerunning before test access",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    weight_grid_root = args.weight_grid_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_root}. Use --overwrite only before test access."
        )
    output_root.mkdir(parents=True, exist_ok=True)

    weight_lock = verify_weight_grid_lock(weight_grid_root)
    paired_path = weight_grid_root / "paired_weight_results.csv"
    frame = load_results(paired_path)
    grid_summary, _boot = summarize_grid(frame)

    choices = [
        choose_cell(cell_id, grid_summary[grid_summary["cell_id"] == cell_id].copy())
        for cell_id in sorted(grid_summary["cell_id"].unique())
    ]

    candidate_summary_path = output_root / "adaptive_candidate_summary.csv"
    preselection_path = output_root / "adaptive_preselection.csv"
    protocol_path = output_root / "adaptive_protocol.json"
    lock_path = output_root / "BINARY_ADAPTIVE_PROTOCOL_LOCK.json"

    grid_summary.to_csv(candidate_summary_path, index=False, encoding="utf-8-sig")
    write_csv(
        preselection_path,
        [asdict(choice) for choice in choices],
        fieldnames=list(asdict(choices[0]).keys()),
    )

    protocol = {
        "protocol_version": LOCK_VERSION,
        "created_at": utc_now(),
        "experiment_name": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "development_only": True,
        "test_data_accessed": False,
        "primary_metric": PRIMARY_METRIC,
        "candidate_space": {
            "base": {"ratio": 0.0, "synthetic_weight": 0.0},
            "positive_candidates": "all completed similarity-OFF ratio × positive-weight conditions in paired_weight_results.csv",
        },
        "selection_rule": {
            "name": "dataset-specific one-standard-error minimum-exposure rule",
            "best_candidate": "highest mean development Macro-F1 across 50 paired repeats",
            "eligibility": "mean Macro-F1 >= best mean - SE(best mean)",
            "tie_break": [
                "smallest effective_synthetic_ratio",
                "smallest ratio",
                "smallest synthetic_weight",
                "highest mean Macro-F1",
            ],
            "base_priority": "Base is selected whenever it is one-SE eligible",
            "no_second_chance_rule": "A failed positive candidate falls back to Base; another positive candidate is not searched.",
        },
        "safety_guards": {
            "paired_bootstrap_repetitions": BOOTSTRAP_REPS,
            "bootstrap_seed": PROTOCOL_SEED,
            "macro_f1": "selected candidate 95% paired-bootstrap CI lower bound vs Base > 0",
            "local_stability": {
                "minimum_adjacent_conditions_with_positive_ci": MIN_STABLE_NEIGHBORS,
                "adjacency": "one immediately neighboring ratio at fixed weight or one immediately neighboring weight at fixed ratio",
            },
            "auroc_noninferiority": {
                "requirement": "selected candidate 95% paired-bootstrap CI lower bound vs Base >= -margin",
                "margin": AUROC_NONINFERIORITY_MARGIN,
            },
            "decision_boundary_diagnostic": {
                "required_for_positive_candidate": True,
                "calibration_source": "training-only 5-fold out-of-fold predictions",
                "calibration_target": "Macro-F1",
                "pass_rule": "calibrated-threshold Macro-F1 paired-bootstrap CI lower bound vs calibrated Base > 0",
                "prohibited": [
                    "choosing threshold on development labels and evaluating on the same development labels",
                    "searching another positive candidate after diagnostic failure",
                    "reading test data before final policy lock",
                ],
            },
        },
        "preselection": [asdict(choice) for choice in choices],
        "input_artifacts": {
            "weight_grid_lock_sha256": sha256_file(weight_grid_root / "SYNTHETIC_WEIGHT_GRID_LOCK.json"),
            "paired_weight_results_sha256": sha256_file(paired_path),
            "weight_grid_lock_version": weight_lock.get("lock_version"),
        },
    }
    write_json(protocol_path, protocol)

    lock = {
        "lock_version": LOCK_VERSION,
        "created_at": utc_now(),
        "experiment_name": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "status": "protocol_frozen_pending_diagnostic",
        "test_data_accessed": False,
        "files": {
            candidate_summary_path.name: sha256_file(candidate_summary_path),
            preselection_path.name: sha256_file(preselection_path),
            protocol_path.name: sha256_file(protocol_path),
        },
        "input_lock_sha256": {
            "synthetic_weight_grid": sha256_file(
                weight_grid_root / "SYNTHETIC_WEIGHT_GRID_LOCK.json"
            ),
        },
        "protocol_sha256": hashlib.sha256(canonical_json(protocol).encode("utf-8")).hexdigest(),
    }
    write_json(lock_path, lock)

    print(f"Frozen adaptive protocol: {lock_path}")
    for choice in choices:
        print(
            f"{choice.cell_id}: {choice.provisional_policy_state} | "
            f"ratio={choice.selected_ratio:g} | weight={choice.selected_synthetic_weight:g} | "
            f"delta={choice.delta_vs_base_macro_f1_mean:+.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
