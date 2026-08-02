"""Embed experiment records with BGE-M3 and compute seed-synthetic cosine."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Sequence

_MODULE_ROOT = Path(__file__).resolve().parent
_SOURCES_ROOT = _MODULE_ROOT.parent
for candidate in (_MODULE_ROOT, _SOURCES_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from Common.lmstudio import DEFAULT_LMSTUDIO_EMBEDDINGS_API_URL, EndpointConfig, loaded_lmstudio_models, parse_lmstudio_endpoints
from embed_binary_qwen import (
    read_valid_outputs,
    request_batches,
    source_entries,
    tokenizer_revision,
    write_errors,
)
from generate_binary_candidates import (
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    PATH_ID,
    TARGET_CELLS,
    load_and_validate_inputs,
    progress_log,
    sha256_bytes,
    sha256_file,
    utc_now,
    verify_lock,
    write_csv,
    write_json,
)


ENDPOINT_MODEL_ID = "text-embedding-bge-m3"
TOKENIZER_ID = "BAAI/bge-m3"
EXPECTED_DIMENSION = 1024
MAX_LENGTH = 512
LOCK_NAME = "BGE_EMBEDDING_LOCK.json"
MANIFEST_FIELDS = [
    "experiment_name", "experiment_id", "embedding_row_index", "cell_id",
    "record_type", "real_row_id", "candidate_index", "global_generation_index",
    "text_sha256", "original_token_count", "token_count", "truncated",
    "embedding_dimension", "status",
]
COSINE_FIELDS = [
    "experiment_name", "experiment_id", "cell_id", "language", "real_row_id",
    "candidate_index", "global_generation_index", "seed_embedding_row_index",
    "synthetic_embedding_row_index", "cosine_similarity",
]


def lock_files(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != LOCK_NAME
    }


def encode_for_bge(tokenizer: Any, text: str) -> tuple[str, int, int, bool]:
    original = tokenizer(text, add_special_tokens=True, truncation=False, padding=False).get("input_ids")
    if not isinstance(original, list):
        raise RuntimeError("tokenizer did not return input_ids")
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    ).get("input_ids")
    if not isinstance(encoded, list):
        raise RuntimeError("tokenizer did not return truncated input_ids")
    truncated = len(original) > MAX_LENGTH
    embedding_text = tokenizer.decode(encoded, skip_special_tokens=True, clean_up_tokenization_spaces=False) if truncated else text
    separator = tokenizer.sep_token or tokenizer.eos_token
    if not isinstance(separator, str) or not separator:
        raise RuntimeError("BGE tokenizer has no SEP or EOS token")
    if not embedding_text.endswith(separator):
        embedding_text += separator
    return embedding_text, len(original), len(encoded), truncated


def run_bge_embeddings(
    prepared_root: Path,
    manifest_root: Path,
    experiment_root: Path,
    endpoints: Sequence[EndpointConfig],
    api_key: str | None,
    timeout_seconds: int,
    batch_size: int,
) -> dict[str, Any]:
    import numpy as np
    from transformers import AutoTokenizer

    verify_lock(experiment_root / "Generation", "GENERATION_LOCK.json")
    verify_lock(experiment_root / "Embeddings" / "Qwen3", "QWEN_EMBEDDING_LOCK.json")
    _lock_info, banks, _plan = load_and_validate_inputs(prepared_root, manifest_root)
    valid = read_valid_outputs(experiment_root / "Generation")
    entries = source_entries(banks, valid)
    output_root = experiment_root / "Embeddings" / "BGE_M3"
    output_root.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, local_files_only=True)
    tokenizer.truncation_side = "right"
    metadata: dict[str, Any] = {
        "metadata_version": "binary-matched-size-bge-v1",
        "created_at": utc_now(),
        "experiment_name": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "endpoint_model_id": ENDPOINT_MODEL_ID,
        "tokenizer_id": TOKENIZER_ID,
        "tokenizer_revision": tokenizer_revision(tokenizer),
        "backend": "lm_studio_openai_compatible_embeddings",
        "expected_dimension": EXPECTED_DIMENSION,
        "max_length": MAX_LENGTH,
        "truncation": "right",
        "dtype": "float32",
        "l2_normalized": True,
        "endpoints": [{"api_url": url, "model": model, "max_concurrency": concurrency} for url, model, concurrency in endpoints],
        "cells": {},
        "status": "running",
    }
    all_cosines: list[dict[str, Any]] = []
    total_errors = 0
    for cell_id in TARGET_CELLS:
        prepared: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        for entry in entries[cell_id]:
            try:
                value, original_count, count, truncated = encode_for_bge(tokenizer, str(entry["text"]))
                prepared.append({
                    **entry,
                    "embedding_input": value,
                    "original_token_count": original_count,
                    "token_count": count,
                    "truncated": truncated,
                })
            except Exception as exc:
                errors.append({**entry, "error_type": "tokenizer_error", "error": f"{type(exc).__name__}: {exc}"})
        entry_batches = [prepared[start:start + batch_size] for start in range(0, len(prepared), batch_size)]
        batches = [[str(row["embedding_input"]) for row in batch] for batch in entry_batches]
        results = request_batches(batches, endpoints, api_key, timeout_seconds)
        vectors: list[Any] = []
        manifest_rows: list[dict[str, Any]] = []
        for batch_index, batch_entries in enumerate(entry_batches):
            raw_vectors, error = results[batch_index]
            if error is not None or raw_vectors is None or len(raw_vectors) != len(batch_entries):
                for entry in batch_entries:
                    errors.append({**entry, "error_type": "endpoint_embedding_error", "error": error or "embedding batch size mismatch"})
                continue
            for entry, raw_vector in zip(batch_entries, raw_vectors, strict=True):
                try:
                    vector = np.asarray(raw_vector, dtype=np.float32)
                    if vector.ndim != 1 or vector.shape[0] != EXPECTED_DIMENSION:
                        raise ValueError(f"dimension {vector.shape} != ({EXPECTED_DIMENSION},)")
                    if not np.all(np.isfinite(vector)):
                        raise ValueError("vector contains NaN or infinite values")
                    norm = float(np.linalg.norm(vector))
                    if not math.isfinite(norm) or norm <= 0:
                        raise ValueError("vector has zero or invalid norm")
                    vector = (vector / norm).astype(np.float32)
                except Exception as exc:
                    errors.append({**entry, "error_type": "embedding_validation_error", "error": f"{type(exc).__name__}: {exc}"})
                    continue
                row_index = len(vectors)
                vectors.append(vector)
                manifest_rows.append({
                    "experiment_name": EXPERIMENT_NAME,
                    "experiment_id": EXPERIMENT_ID,
                    "embedding_row_index": row_index,
                    "cell_id": cell_id,
                    "record_type": entry["record_type"],
                    "real_row_id": entry["real_row_id"],
                    "candidate_index": entry["candidate_index"],
                    "global_generation_index": entry["global_generation_index"],
                    "text_sha256": sha256_bytes(str(entry["text"]).encode("utf-8")),
                    "original_token_count": entry["original_token_count"],
                    "token_count": entry["token_count"],
                    "truncated": entry["truncated"],
                    "embedding_dimension": EXPECTED_DIMENSION,
                    "status": "ok",
                })
        array = np.asarray(vectors, dtype=np.float32) if vectors else np.empty((0, EXPECTED_DIMENSION), dtype=np.float32)
        np.save(output_root / f"{cell_id}.npy", array)
        write_csv(output_root / f"{cell_id}_manifest.csv", manifest_rows, MANIFEST_FIELDS)
        write_errors(output_root / f"{cell_id}_errors.jsonl", errors)

        seed_index: dict[str, int] = {}
        synthetic_index: dict[int, tuple[int, dict[str, Any]]] = {}
        for row in manifest_rows:
            index = int(row["embedding_row_index"])
            if row["record_type"] == "real":
                key = str(row["real_row_id"])
                if key in seed_index:
                    raise RuntimeError(f"duplicate BGE real embedding: {cell_id}/{key}")
                seed_index[key] = index
            else:
                key = int(row["global_generation_index"])
                if key in synthetic_index:
                    raise RuntimeError(f"duplicate BGE synthetic embedding: {key}")
                synthetic_index[key] = (index, row)
        cell_cosines: list[dict[str, Any]] = []
        for generation_row in valid[cell_id]:
            global_index = int(generation_row["global_generation_index"])
            parent = str(generation_row["real_row_id"])
            if parent not in seed_index:
                errors.append({"cell_id": cell_id, "global_generation_index": global_index, "error_type": "missing_parent_real_embedding", "error": parent})
                continue
            if global_index not in synthetic_index:
                errors.append({"cell_id": cell_id, "global_generation_index": global_index, "error_type": "missing_synthetic_mapping", "error": str(global_index)})
                continue
            real_index = seed_index[parent]
            generated_index, _manifest_row = synthetic_index[global_index]
            cosine = float(np.dot(array[real_index], array[generated_index]))
            if not math.isfinite(cosine):
                errors.append({"cell_id": cell_id, "global_generation_index": global_index, "error_type": "invalid_cosine", "error": str(cosine)})
                continue
            cell_cosines.append({
                "experiment_name": EXPERIMENT_NAME,
                "experiment_id": EXPERIMENT_ID,
                "cell_id": cell_id,
                "language": generation_row["language"],
                "real_row_id": parent,
                "candidate_index": int(generation_row["candidate_index"]),
                "global_generation_index": global_index,
                "seed_embedding_row_index": real_index,
                "synthetic_embedding_row_index": generated_index,
                "cosine_similarity": cosine,
            })
        write_errors(output_root / f"{cell_id}_errors.jsonl", errors)
        all_cosines.extend(cell_cosines)
        real_count = len(seed_index)
        synthetic_count = len(synthetic_index)
        metadata["cells"][cell_id] = {
            "array_shape": list(array.shape),
            "real_count": real_count,
            "synthetic_count": synthetic_count,
            "expected_synthetic_count": len(valid[cell_id]),
            "cosine_count": len(cell_cosines),
            "errors": len(errors),
        }
        total_errors += len(errors)
    all_cosines.sort(key=lambda row: int(row["global_generation_index"]))
    if len({int(row["global_generation_index"]) for row in all_cosines}) != len(all_cosines):
        raise RuntimeError("duplicate cosine global_generation_index")
    write_csv(output_root / "cosine_similarity.csv", all_cosines, COSINE_FIELDS)
    expected_total = sum(len(valid[cell_id]) for cell_id in TARGET_CELLS)
    complete = total_errors == 0 and len(all_cosines) == expected_total and all(
        metadata["cells"][cell_id]["real_count"] == 1_400
        and metadata["cells"][cell_id]["synthetic_count"] == metadata["cells"][cell_id]["expected_synthetic_count"]
        and metadata["cells"][cell_id]["cosine_count"] == metadata["cells"][cell_id]["expected_synthetic_count"]
        for cell_id in TARGET_CELLS
    )
    metadata["total_cosine_count"] = len(all_cosines)
    metadata["expected_total_cosine_count"] = expected_total
    metadata["status"] = "completed" if complete else "failed"
    write_json(output_root / "metadata.json", metadata)
    if not complete:
        raise RuntimeError("BGE embedding/cosine incomplete; see metadata.json and errors JSONL")
    write_json(output_root / LOCK_NAME, {
        "lock_version": "binary-matched-size-bge-v1",
        "created_at": utc_now(),
        "experiment_name": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "files": lock_files(output_root),
    })
    return metadata


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="BGE-M3 embedding for " + EXPERIMENT_NAME)
    parser.add_argument("--prepared-root", type=Path, default=project_root / "Data" / "Prepared")
    parser.add_argument("--manifest-root", type=Path, default=project_root / "Data" / "ExperimentManifests" / "v1")
    parser.add_argument("--output-root", type=Path, default=project_root / "Results" / PATH_ID)
    parser.add_argument("--bge-endpoint", action="append", nargs=3, metavar=("API_URL", "MODEL", "CONCURRENCY"))
    parser.add_argument("--api-key", default=os.environ.get("LMSTUDIO_API_KEY"))
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.timeout_seconds <= 0 or args.embedding_batch_size <= 0:
        raise SystemExit("timeout and embedding batch size must be positive")
    try:
        endpoints = parse_lmstudio_endpoints(args.bge_endpoint, ENDPOINT_MODEL_ID, DEFAULT_LMSTUDIO_EMBEDDINGS_API_URL)
        if any(model != ENDPOINT_MODEL_ID for _, model, _ in endpoints):
            raise ValueError(f"all BGE endpoints must use {ENDPOINT_MODEL_ID}")
        with loaded_lmstudio_models(endpoints, api_key=args.api_key, timeout_seconds=args.timeout_seconds, log=progress_log):
            metadata = run_bge_embeddings(
                args.prepared_root.resolve(), args.manifest_root.resolve(), args.output_root.resolve(),
                endpoints, args.api_key, args.timeout_seconds, args.embedding_batch_size,
            )
        print(json.dumps(metadata["cells"], ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"BGE embedding failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
