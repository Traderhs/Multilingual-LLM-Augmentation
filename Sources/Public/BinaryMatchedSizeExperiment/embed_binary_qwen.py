"""Embed locked real and valid synthetic records with Qwen3-Embedding-8B."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

_MODULE_ROOT = Path(__file__).resolve().parent
_SOURCES_ROOT = _MODULE_ROOT.parent
for candidate in (_MODULE_ROOT, _SOURCES_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from Common.lmstudio import (
    DEFAULT_LMSTUDIO_EMBEDDINGS_API_URL,
    EndpointConfig,
    loaded_lmstudio_models,
    parse_lmstudio_endpoints,
    request_lmstudio_embeddings,
)
from generate_binary_candidates import (
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    PATH_ID,
    TARGET_CELLS,
    load_and_validate_inputs,
    progress_log,
    read_json,
    sha256_bytes,
    sha256_file,
    utc_now,
    verify_lock,
    write_csv,
    write_json,
)


ENDPOINT_MODEL_ID = "text-embedding-qwen3-embedding-8b"
TOKENIZER_ID = "Qwen/Qwen3-Embedding-8B"
EXPECTED_DIMENSION = 4096
INSTRUCTION = "Represent this text for supervised text classification."
LOCK_NAME = "QWEN_EMBEDDING_LOCK.json"
MANIFEST_FIELDS = [
    "experiment_name", "experiment_id", "embedding_row_index", "cell_id",
    "record_type", "real_row_id", "candidate_index", "global_generation_index",
    "text_sha256", "token_count", "embedding_dimension", "status",
]


def read_valid_outputs(generation_root: Path) -> dict[str, list[dict[str, Any]]]:
    verify_lock(generation_root, "GENERATION_LOCK.json")
    result: dict[str, list[dict[str, Any]]] = {}
    seen_global: set[int] = set()
    for cell_id in TARGET_CELLS:
        rows: list[dict[str, Any]] = []
        for path in sorted((generation_root / "valid_outputs" / cell_id).glob("part_*.jsonl")):
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        index = int(row["global_generation_index"])
                        if index in seen_global:
                            raise RuntimeError(f"duplicate valid global_generation_index: {index}")
                        seen_global.add(index)
                        rows.append(row)
        rows.sort(key=lambda row: int(row["global_generation_index"]))
        result[cell_id] = rows
    return result


def source_entries(
    banks: Mapping[str, Sequence[Mapping[str, Any]]],
    valid: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for cell_id in TARGET_CELLS:
        rows = [
            {
                "cell_id": cell_id,
                "record_type": "real",
                "real_row_id": row["real_row_id"],
                "candidate_index": None,
                "global_generation_index": None,
                "text": row["seed_text"],
            }
            for row in banks[cell_id]
        ]
        rows.extend(
            {
                "cell_id": cell_id,
                "record_type": "synthetic",
                "real_row_id": row["real_row_id"],
                "candidate_index": int(row["candidate_index"]),
                "global_generation_index": int(row["global_generation_index"]),
                "text": row["generated_text"],
            }
            for row in valid[cell_id]
        )
        result[cell_id] = rows
    return result


def tokenizer_revision(tokenizer: Any) -> str | None:
    kwargs = getattr(tokenizer, "init_kwargs", None)
    value = kwargs.get("_commit_hash") if isinstance(kwargs, dict) else None
    return value if isinstance(value, str) and value else None


def context_limit(tokenizer: Any) -> int | None:
    value = getattr(tokenizer, "model_max_length", None)
    return value if isinstance(value, int) and 0 < value < 10**8 else None


def token_count(tokenizer: Any, text: str) -> int:
    encoded = tokenizer(text, add_special_tokens=True, truncation=False, padding=False)
    ids = encoded.get("input_ids")
    if not isinstance(ids, list):
        raise RuntimeError("tokenizer did not return input_ids")
    return len(ids)


def embedding_input(tokenizer: Any, text: str) -> str:
    separator = tokenizer.sep_token or tokenizer.eos_token
    if not isinstance(separator, str) or not separator:
        raise RuntimeError("Qwen tokenizer has no SEP or EOS token")
    value = f"Instruct: {INSTRUCTION}\nQuery:{text}"
    return value if value.endswith(separator) else value + separator


def request_batches(
    batches: Sequence[Sequence[str]],
    endpoints: Sequence[EndpointConfig],
    api_key: str | None,
    timeout_seconds: int,
) -> dict[int, tuple[list[list[float]] | None, str | None]]:
    executors = [ThreadPoolExecutor(max_workers=concurrency) for _, _, concurrency in endpoints]
    slots = [index for index, (_, _, concurrency) in enumerate(endpoints) for _ in range(concurrency)]
    futures = {}
    results: dict[int, tuple[list[list[float]] | None, str | None]] = {}
    try:
        for batch_index, batch in enumerate(batches):
            endpoint_index = slots[batch_index % len(slots)]
            url, model, _ = endpoints[endpoint_index]
            future = executors[endpoint_index].submit(
                request_lmstudio_embeddings,
                api_url=url,
                model=model,
                inputs=batch,
                api_key=api_key,
                timeout_seconds=timeout_seconds,
            )
            futures[future] = (batch_index, url)
        for completed, future in enumerate(as_completed(futures), 1):
            batch_index, url = futures[future]
            try:
                results[batch_index] = (future.result(), None)
                status = "ok"
            except Exception as exc:
                results[batch_index] = (None, f"{type(exc).__name__}: {exc}")
                status = "failed"
            progress_log(f"Qwen embedding | batches={completed}/{len(batches)} | endpoint={url} | status={status}")
    finally:
        for executor in executors:
            executor.shutdown(wait=True, cancel_futures=True)
    return results


def write_errors(path: Path, errors: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in errors:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def lock_files(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != LOCK_NAME
    }


def run_qwen_embeddings(
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

    _lock_info, banks, _plan = load_and_validate_inputs(prepared_root, manifest_root)
    generation_root = experiment_root / "Generation"
    valid = read_valid_outputs(generation_root)
    entries = source_entries(banks, valid)
    output_root = experiment_root / "Embeddings" / "Qwen3"
    output_root.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, local_files_only=True, padding_side="left")
    limit = context_limit(tokenizer)
    metadata: dict[str, Any] = {
        "metadata_version": "binary-matched-size-qwen-v1",
        "created_at": utc_now(),
        "experiment_name": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "endpoint_model_id": ENDPOINT_MODEL_ID,
        "tokenizer_id": TOKENIZER_ID,
        "tokenizer_revision": tokenizer_revision(tokenizer),
        "backend": "lm_studio_openai_compatible_embeddings",
        "expected_dimension": EXPECTED_DIMENSION,
        "instruction": INSTRUCTION,
        "dtype": "float32",
        "l2_normalized": True,
        "truncation": False,
        "context_limit": limit,
        "endpoints": [{"api_url": url, "model": model, "max_concurrency": concurrency} for url, model, concurrency in endpoints],
        "cells": {},
        "status": "running",
    }
    total_errors = 0
    for cell_id in TARGET_CELLS:
        prepared: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        for entry in entries[cell_id]:
            try:
                value = embedding_input(tokenizer, str(entry["text"]))
                count = token_count(tokenizer, value)
                if limit is not None and count > limit:
                    raise OverflowError(f"token_count={count} exceeds context_limit={limit}")
                prepared.append({**entry, "embedding_input": value, "token_count": count})
            except Exception as exc:
                errors.append({**entry, "error_type": "context_overflow" if isinstance(exc, OverflowError) else "tokenizer_error", "error": f"{type(exc).__name__}: {exc}"})
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
                    "token_count": entry["token_count"],
                    "embedding_dimension": EXPECTED_DIMENSION,
                    "status": "ok",
                })
        array = np.asarray(vectors, dtype=np.float32) if vectors else np.empty((0, EXPECTED_DIMENSION), dtype=np.float32)
        np.save(output_root / f"{cell_id}.npy", array)
        write_csv(output_root / f"{cell_id}_manifest.csv", manifest_rows, MANIFEST_FIELDS)
        write_errors(output_root / f"{cell_id}_errors.jsonl", errors)
        real_count = sum(row["record_type"] == "real" for row in manifest_rows)
        synthetic_count = sum(row["record_type"] == "synthetic" for row in manifest_rows)
        metadata["cells"][cell_id] = {
            "array_shape": list(array.shape),
            "real_count": real_count,
            "synthetic_count": synthetic_count,
            "expected_synthetic_count": len(valid[cell_id]),
            "errors": len(errors),
        }
        total_errors += len(errors)
    complete = total_errors == 0 and all(
        metadata["cells"][cell_id]["real_count"] == 1_400
        and metadata["cells"][cell_id]["synthetic_count"] == metadata["cells"][cell_id]["expected_synthetic_count"]
        for cell_id in TARGET_CELLS
    )
    metadata["status"] = "completed" if complete else "failed"
    write_json(output_root / "metadata.json", metadata)
    if not complete:
        raise RuntimeError("Qwen embedding incomplete; see metadata.json and errors JSONL")
    write_json(output_root / LOCK_NAME, {
        "lock_version": "binary-matched-size-qwen-v1",
        "created_at": utc_now(),
        "experiment_name": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "files": lock_files(output_root),
    })
    return metadata


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Qwen3 embedding for " + EXPERIMENT_NAME)
    parser.add_argument("--prepared-root", type=Path, default=project_root / "Data" / "Prepared")
    parser.add_argument("--manifest-root", type=Path, default=project_root / "Data" / "ExperimentManifests" / "v1")
    parser.add_argument("--output-root", type=Path, default=project_root / "Results" / PATH_ID)
    parser.add_argument("--qwen-endpoint", action="append", nargs=3, metavar=("API_URL", "MODEL", "CONCURRENCY"))
    parser.add_argument("--api-key", default=os.environ.get("LMSTUDIO_API_KEY"))
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.timeout_seconds <= 0 or args.embedding_batch_size <= 0:
        raise SystemExit("timeout and embedding batch size must be positive")
    try:
        endpoints = parse_lmstudio_endpoints(args.qwen_endpoint, ENDPOINT_MODEL_ID, DEFAULT_LMSTUDIO_EMBEDDINGS_API_URL)
        if any(model != ENDPOINT_MODEL_ID for _, model, _ in endpoints):
            raise ValueError(f"all Qwen endpoints must use {ENDPOINT_MODEL_ID}")
        with loaded_lmstudio_models(endpoints, api_key=args.api_key, timeout_seconds=args.timeout_seconds, log=progress_log):
            metadata = run_qwen_embeddings(
                args.prepared_root.resolve(), args.manifest_root.resolve(), args.output_root.resolve(),
                endpoints, args.api_key, args.timeout_seconds, args.embedding_batch_size,
            )
        print(json.dumps(metadata["cells"], ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"Qwen embedding failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
