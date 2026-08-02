"""Orchestrate and verify the Binary Matched-Size Augmentation Experiment."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

_MODULE_ROOT = Path(__file__).resolve().parent
if str(_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULE_ROOT))

from generate_binary_candidates import (
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    PATH_ID,
    TARGET_CELLS,
    TOTAL_REQUESTS,
    read_json,
    sha256_file,
    utc_now,
    verify_lock,
    write_json,
)


FINAL_LOCK_NAME = "BINARY_MATCHED_SIZE_EXPERIMENT_LOCK.json"
NORM_ATOL = 1e-5


def append_endpoints(command: list[str], option: str, values: list[list[str]] | None) -> None:
    for value in values or []:
        command.extend([option, *value])


def run_command(command: Sequence[str]) -> None:
    subprocess.run(list(command), check=True)


def stage_lock_valid(root: Path, name: str) -> bool:
    try:
        verify_lock(root, name)
    except Exception:
        return False
    return True


def generation_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(_MODULE_ROOT / "generate_binary_candidates.py"),
        "--prepared-root", str(args.prepared_root),
        "--manifest-root", str(args.manifest_root),
        "--output-root", str(args.output_root),
        "--timeout-seconds", str(args.timeout_seconds),
        "--export-shard-size", str(args.export_shard_size),
    ]
    if args.api_key:
        command.extend(["--api-key", args.api_key])
    if args.resume:
        command.append("--resume")
    if args.dry_run:
        command.append("--dry-run")
    append_endpoints(command, "--endpoint", args.endpoint)
    return command


def embedding_command(args: argparse.Namespace, script: str, option: str, values: list[list[str]] | None) -> list[str]:
    command = [
        sys.executable,
        str(_MODULE_ROOT / script),
        "--prepared-root", str(args.prepared_root),
        "--manifest-root", str(args.manifest_root),
        "--output-root", str(args.output_root),
        "--timeout-seconds", str(args.timeout_seconds),
        "--embedding-batch-size", str(args.embedding_batch_size),
    ]
    if args.api_key:
        command.extend(["--api-key", args.api_key])
    append_endpoints(command, option, values)
    return command


def count_csv_rows(path: Path) -> tuple[int, list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return len(rows), rows


def vectors_are_normalized(array: Any, absolute_tolerance: float) -> bool:
    import numpy as np

    for start in range(0, array.shape[0], 2_048):
        norms = np.linalg.norm(np.asarray(array[start : start + 2_048]), axis=1)
        if not np.allclose(norms, 1.0, rtol=0.0, atol=absolute_tolerance):
            return False
    return True


def validate_final_integrity(experiment_root: Path) -> dict[str, Any]:
    import numpy as np

    generation_root = experiment_root / "Generation"
    qwen_root = experiment_root / "Embeddings" / "Qwen3"
    bge_root = experiment_root / "Embeddings" / "BGE_M3"
    generation_lock = verify_lock(generation_root, "GENERATION_LOCK.json")
    qwen_lock = verify_lock(qwen_root, "QWEN_EMBEDDING_LOCK.json")
    bge_lock = verify_lock(bge_root, "BGE_EMBEDDING_LOCK.json")

    database = generation_root / "generation_status.sqlite3"
    connection = sqlite3.connect(database)
    try:
        counts = dict(connection.execute("SELECT status, COUNT(*) FROM requests GROUP BY status"))
        total = sum(counts.values())
        duplicates = connection.execute(
            """SELECT COUNT(*) FROM (
                SELECT cell_id, real_row_id, candidate_index, COUNT(*) AS n
                FROM requests GROUP BY cell_id, real_row_id, candidate_index HAVING n > 1
            )"""
        ).fetchone()[0]
        global_unique = connection.execute("SELECT COUNT(DISTINCT global_generation_index) FROM requests").fetchone()[0]
        valid_key_unique = connection.execute(
            "SELECT COUNT(DISTINCT cell_id || ':' || real_row_id || ':' || candidate_index) FROM requests WHERE status='valid'"
        ).fetchone()[0]
    finally:
        connection.close()
    valid_total = int(counts.get("valid", 0))
    failed_total = int(counts.get("failed", 0))
    pending_total = int(counts.get("pending", 0))
    if total != TOTAL_REQUESTS or valid_total + failed_total != TOTAL_REQUESTS or pending_total != 0:
        raise RuntimeError(f"generation terminal counts invalid: {counts}")
    if duplicates or global_unique != TOTAL_REQUESTS or valid_key_unique != valid_total:
        raise RuntimeError("generation key uniqueness validation failed")

    stage_counts: dict[str, Any] = {}
    for cell_id in TARGET_CELLS:
        qwen = np.load(qwen_root / f"{cell_id}.npy", mmap_mode="r")
        bge = np.load(bge_root / f"{cell_id}.npy", mmap_mode="r")
        qwen_count, qwen_manifest = count_csv_rows(qwen_root / f"{cell_id}_manifest.csv")
        bge_count, bge_manifest = count_csv_rows(bge_root / f"{cell_id}_manifest.csv")
        valid_cell = int(counts_for_cell(database, cell_id).get("valid", 0))
        expected_rows = 1_400 + valid_cell
        if qwen.dtype != np.float32 or qwen.ndim != 2 or qwen.shape != (expected_rows, 4096):
            raise RuntimeError(f"{cell_id}: invalid Qwen array {qwen.shape}/{qwen.dtype}")
        if bge.dtype != np.float32 or bge.ndim != 2 or bge.shape != (expected_rows, 1024):
            raise RuntimeError(f"{cell_id}: invalid BGE array {bge.shape}/{bge.dtype}")
        if qwen_count != expected_rows or bge_count != expected_rows:
            raise RuntimeError(f"{cell_id}: embedding manifest/array row mismatch")
        if sum(row["record_type"] == "real" for row in qwen_manifest) != 1_400:
            raise RuntimeError(f"{cell_id}: Qwen real count mismatch")
        if sum(row["record_type"] == "synthetic" for row in qwen_manifest) != valid_cell:
            raise RuntimeError(f"{cell_id}: Qwen synthetic count mismatch")
        if sum(row["record_type"] == "real" for row in bge_manifest) != 1_400:
            raise RuntimeError(f"{cell_id}: BGE real count mismatch")
        if sum(row["record_type"] == "synthetic" for row in bge_manifest) != valid_cell:
            raise RuntimeError(f"{cell_id}: BGE synthetic count mismatch")
        if not vectors_are_normalized(qwen, NORM_ATOL):
            raise RuntimeError(f"{cell_id}: Qwen vectors are not L2 normalized")
        if not vectors_are_normalized(bge, NORM_ATOL):
            raise RuntimeError(f"{cell_id}: BGE vectors are not L2 normalized")
        stage_counts[cell_id] = {"valid_generation": valid_cell, "qwen_rows": qwen_count, "bge_rows": bge_count}
        del qwen, bge

    cosine_count, cosine_rows = count_csv_rows(bge_root / "cosine_similarity.csv")
    cosine_keys = [int(row["global_generation_index"]) for row in cosine_rows]
    if cosine_count != valid_total or len(set(cosine_keys)) != valid_total:
        raise RuntimeError(f"cosine row count/uniqueness invalid: {cosine_count} != {valid_total}")
    if any(not np.isfinite(float(row["cosine_similarity"])) for row in cosine_rows):
        raise RuntimeError("cosine file contains NaN or infinite values")

    return {
        "integrity_version": "binary-matched-size-experiment-v1",
        "validated_at": utc_now(),
        "experiment_name": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "l2_norm_absolute_tolerance": NORM_ATOL,
        "generation": {"valid": valid_total, "failed": failed_total, "pending": pending_total},
        "cells": stage_counts,
        "cosine_rows": cosine_count,
        "stage_lock_sha256": {
            "generation": sha256_file(generation_root / "GENERATION_LOCK.json"),
            "qwen": sha256_file(qwen_root / "QWEN_EMBEDDING_LOCK.json"),
            "bge": sha256_file(bge_root / "BGE_EMBEDDING_LOCK.json"),
        },
        "stage_locks": {
            "generation": generation_lock["lock_version"],
            "qwen": qwen_lock["lock_version"],
            "bge": bge_lock["lock_version"],
        },
    }


def counts_for_cell(database: Path, cell_id: str) -> dict[str, int]:
    connection = sqlite3.connect(database)
    try:
        return {str(status): int(count) for status, count in connection.execute(
            "SELECT status, COUNT(*) FROM requests WHERE cell_id=? GROUP BY status", (cell_id,)
        )}
    finally:
        connection.close()


def write_final_lock(experiment_root: Path, integrity: dict[str, Any]) -> None:
    write_json(experiment_root / FINAL_LOCK_NAME, {
        "lock_version": "binary-matched-size-experiment-v1",
        "created_at": utc_now(),
        "experiment_name": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "status": "completed",
        "files": {
            "Generation/GENERATION_LOCK.json": sha256_file(experiment_root / "Generation" / "GENERATION_LOCK.json"),
            "Embeddings/Qwen3/QWEN_EMBEDDING_LOCK.json": sha256_file(experiment_root / "Embeddings" / "Qwen3" / "QWEN_EMBEDDING_LOCK.json"),
            "Embeddings/BGE_M3/BGE_EMBEDDING_LOCK.json": sha256_file(experiment_root / "Embeddings" / "BGE_M3" / "BGE_EMBEDDING_LOCK.json"),
        },
        "integrity": integrity,
    })


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Run " + EXPERIMENT_NAME)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--generation-only", action="store_true")
    modes.add_argument("--qwen-only", action="store_true")
    modes.add_argument("--bge-only", action="store_true")
    parser.add_argument("--prepared-root", type=Path, default=project_root / "Data" / "Prepared")
    parser.add_argument("--manifest-root", type=Path, default=project_root / "Data" / "ExperimentManifests" / "v1")
    parser.add_argument("--output-root", type=Path, default=project_root / "Results" / PATH_ID)
    parser.add_argument("--endpoint", action="append", nargs=3, metavar=("API_URL", "MODEL", "CONCURRENCY"))
    parser.add_argument("--qwen-endpoint", action="append", nargs=3, metavar=("API_URL", "MODEL", "CONCURRENCY"))
    parser.add_argument("--bge-endpoint", action="append", nargs=3, metavar=("API_URL", "MODEL", "CONCURRENCY"))
    parser.add_argument("--api-key", default=os.environ.get("LMSTUDIO_API_KEY"))
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--export-shard-size", type=int, default=1_000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.prepared_root = args.prepared_root.resolve()
    args.manifest_root = args.manifest_root.resolve()
    args.output_root = args.output_root.resolve()
    if args.timeout_seconds <= 0 or args.embedding_batch_size <= 0 or args.export_shard_size <= 0:
        raise SystemExit("timeout, batch size, and shard size must be positive")
    if args.dry_run and (args.qwen_only or args.bge_only):
        raise SystemExit("--dry-run validates the generation contract and cannot be combined with embedding-only modes")
    try:
        generation_root = args.output_root / "Generation"
        qwen_root = args.output_root / "Embeddings" / "Qwen3"
        bge_root = args.output_root / "Embeddings" / "BGE_M3"
        run_generation_stage = not args.qwen_only and not args.bge_only
        run_qwen_stage = not args.generation_only and not args.bge_only and not args.dry_run
        run_bge_stage = not args.generation_only and not args.qwen_only and not args.dry_run

        if run_generation_stage:
            if args.dry_run or not stage_lock_valid(generation_root, "GENERATION_LOCK.json"):
                run_command(generation_command(args))
            else:
                print("generation lock valid; skipping generation")
        if args.dry_run or args.generation_only:
            return 0

        verify_lock(generation_root, "GENERATION_LOCK.json")
        if run_qwen_stage:
            if not stage_lock_valid(qwen_root, "QWEN_EMBEDDING_LOCK.json"):
                run_command(embedding_command(args, "embed_binary_qwen.py", "--qwen-endpoint", args.qwen_endpoint))
            else:
                print("Qwen lock valid; skipping Qwen embedding")
        if args.qwen_only:
            if not stage_lock_valid(qwen_root, "QWEN_EMBEDDING_LOCK.json"):
                run_command(embedding_command(args, "embed_binary_qwen.py", "--qwen-endpoint", args.qwen_endpoint))
            return 0

        verify_lock(qwen_root, "QWEN_EMBEDDING_LOCK.json")
        if run_bge_stage:
            if not stage_lock_valid(bge_root, "BGE_EMBEDDING_LOCK.json"):
                run_command(embedding_command(args, "embed_binary_bge.py", "--bge-endpoint", args.bge_endpoint))
            else:
                print("BGE lock valid; skipping BGE embedding")
        if args.bge_only:
            if not stage_lock_valid(bge_root, "BGE_EMBEDDING_LOCK.json"):
                run_command(embedding_command(args, "embed_binary_bge.py", "--bge-endpoint", args.bge_endpoint))
            return 0

        verify_lock(bge_root, "BGE_EMBEDDING_LOCK.json")
        integrity = validate_final_integrity(args.output_root)
        write_final_lock(args.output_root, integrity)
        print(json.dumps(integrity, ensure_ascii=False, indent=2))
        return 0
    except subprocess.CalledProcessError as exc:
        print(f"experiment stage failed with exit code {exc.returncode}", file=sys.stderr)
        return exc.returncode or 1
    except Exception as exc:
        print(f"experiment failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
