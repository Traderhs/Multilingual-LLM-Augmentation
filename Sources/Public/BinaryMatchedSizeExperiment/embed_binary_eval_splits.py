"""Embed locked binary dev/test evaluation splits with Qwen3-Embedding-8B."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
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
)
from embed_binary_qwen import (
    context_limit,
    embedding_input,
    request_batches,
    token_count,
    tokenizer_revision,
    write_errors,
)
from generate_binary_candidates import (
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    PATH_ID,
    TARGET_CELLS,
    load_and_validate_inputs,
    normalize_text,
    parse_int,
    progress_log,
    sha256_bytes,
    sha256_file,
    utc_now,
    write_csv,
    write_json,
)


ENDPOINT_MODEL_ID = "text-embedding-qwen3-embedding-8b"
TOKENIZER_ID = "Qwen/Qwen3-Embedding-8B"
EXPECTED_DIMENSION = 4096
INSTRUCTION = "Represent this text for supervised text classification."
LOCK_NAME = "QWEN_EVAL_EMBEDDING_LOCK.json"
NORM_ATOL = 1e-5
REQUEST_BATCH_WINDOW = 64

EXPECTED_ROWS = {
    "en_binary_sst2": {"dev": 10_046, "test": 872},
    "ko_binary_nsmc": {"dev": 21_798, "test": 49_007},
    "bn_binary_cinexdrama": {"dev": 1_174, "test": 1_175},
    "ha_binary_hausa_movie_review": {"dev": 304, "test": 305},
    "ml_binary_dravidian_codemix": {"dev": 1_028, "test": 1_033},
}
MANIFEST_FIELDS = [
    "experiment_name",
    "experiment_id",
    "embedding_row_index",
    "cell_id",
    "split",
    "eval_row_index",
    "label_int",
    "text_sha256",
    "token_count",
    "embedding_dimension",
    "status",
]


def load_eval_splits(
    prepared_root: Path,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    result: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for cell_id in TARGET_CELLS:
        split_rows: dict[str, list[dict[str, Any]]] = {}
        split_hashes: dict[str, set[str]] = {}
        for split in ("dev", "test"):
            path = prepared_root / cell_id / f"{split}.csv"
            rows: list[dict[str, Any]] = []
            hashes: set[str] = set()
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if not {"text", "label"}.issubset(reader.fieldnames or []):
                    raise RuntimeError(
                        f"{cell_id}/{split}: prepared CSV lacks text/label columns"
                    )
                for eval_row_index, raw in enumerate(reader):
                    text = normalize_text(raw.get("text"))
                    if not text:
                        raise RuntimeError(
                            f"{cell_id}/{split}: empty text at row {eval_row_index}"
                        )
                    try:
                        label_int = parse_int(
                            raw.get("label"),
                            f"{cell_id}/{split}.label[{eval_row_index}]",
                        )
                    except Exception as exc:
                        raise RuntimeError(str(exc)) from exc
                    if label_int not in (0, 1):
                        raise RuntimeError(
                            f"{cell_id}/{split}: label {label_int} is not binary "
                            f"at row {eval_row_index}"
                        )
                    text_sha256 = sha256_bytes(text.encode("utf-8"))
                    if text_sha256 in hashes:
                        raise RuntimeError(
                            f"{cell_id}/{split}: duplicate text SHA-256 "
                            f"at row {eval_row_index}: {text_sha256}"
                        )
                    hashes.add(text_sha256)
                    rows.append(
                        {
                            "cell_id": cell_id,
                            "split": split,
                            "eval_row_index": eval_row_index,
                            "label_int": label_int,
                            "text": text,
                            "text_sha256": text_sha256,
                        }
                    )
            expected = EXPECTED_ROWS[cell_id][split]
            if len(rows) != expected:
                raise RuntimeError(
                    f"{cell_id}/{split}: row count {len(rows)} != expected {expected}"
                )
            split_rows[split] = rows
            split_hashes[split] = hashes
        overlap = split_hashes["dev"] & split_hashes["test"]
        if overlap:
            raise RuntimeError(
                f"{cell_id}: dev/test text SHA-256 overlap: {sorted(overlap)[:10]}"
            )
        result[cell_id] = split_rows
    return result


def error_record(
    entry: Mapping[str, Any], error_type: str, error: str
) -> dict[str, Any]:
    return {
        "cell_id": entry["cell_id"],
        "split": entry["split"],
        "eval_row_index": entry["eval_row_index"],
        "label_int": entry["label_int"],
        "text_sha256": entry["text_sha256"],
        "error_type": error_type,
        "error": error,
    }


def embed_split(
    output_root: Path,
    tokenizer: Any,
    limit: int | None,
    entries: Sequence[Mapping[str, Any]],
    cell_id: str,
    split: str,
    batch_size: int,
    endpoints: Sequence[EndpointConfig],
    api_key: str | None,
    timeout_seconds: int,
) -> dict[str, Any]:
    import numpy as np

    prepared: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for entry in entries:
        try:
            value = embedding_input(tokenizer, str(entry["text"]))
            count = token_count(tokenizer, value)
            if limit is not None and count > limit:
                raise OverflowError(
                    f"token_count={count} exceeds context_limit={limit}"
                )
            prepared.append(
                {**entry, "embedding_input": value, "token_count": count}
            )
        except Exception as exc:
            error_type = (
                "context_overflow" if isinstance(exc, OverflowError)
                else "tokenizer_error"
            )
            errors.append(error_record(entry, error_type, f"{type(exc).__name__}: {exc}"))

    entry_batches = [
        prepared[start : start + batch_size]
        for start in range(0, len(prepared), batch_size)
    ]
    array = np.empty((len(prepared), EXPECTED_DIMENSION), dtype=np.float32)
    manifest_rows: list[dict[str, Any]] = []
    embedded_rows = 0
    for window_start in range(0, len(entry_batches), REQUEST_BATCH_WINDOW):
        window_entries = entry_batches[
            window_start : window_start + REQUEST_BATCH_WINDOW
        ]
        text_batches = [
            [str(entry["embedding_input"]) for entry in batch]
            for batch in window_entries
        ]
        results = request_batches(
            text_batches,
            endpoints,
            api_key,
            timeout_seconds,
        )
        for local_batch_index, batch_entries in enumerate(window_entries):
            raw_vectors, batch_error = results[local_batch_index]
            if (
                batch_error is not None
                or raw_vectors is None
                or len(raw_vectors) != len(batch_entries)
            ):
                error = batch_error or "embedding batch size mismatch"
                for entry in batch_entries:
                    errors.append(
                        error_record(entry, "endpoint_embedding_error", error)
                    )
                continue
            for entry, raw_vector in zip(batch_entries, raw_vectors, strict=True):
                try:
                    vector = np.asarray(raw_vector, dtype=np.float32)
                    if vector.ndim != 1 or vector.shape[0] != EXPECTED_DIMENSION:
                        raise ValueError(
                            f"dimension {vector.shape} != ({EXPECTED_DIMENSION},)"
                        )
                    if not np.all(np.isfinite(vector)):
                        raise ValueError("vector contains NaN or infinite values")
                    norm = float(np.linalg.norm(vector))
                    if not math.isfinite(norm) or norm <= 0:
                        raise ValueError("vector has zero or invalid norm")
                    vector = (vector / norm).astype(np.float32)
                except Exception as exc:
                    errors.append(
                        error_record(
                            entry,
                            "embedding_validation_error",
                            f"{type(exc).__name__}: {exc}",
                        )
                    )
                    continue
                array[embedded_rows] = vector
                manifest_rows.append(
                    {
                        "experiment_name": EXPERIMENT_NAME,
                        "experiment_id": EXPERIMENT_ID,
                        "embedding_row_index": embedded_rows,
                        "cell_id": cell_id,
                        "split": split,
                        "eval_row_index": entry["eval_row_index"],
                        "label_int": entry["label_int"],
                        "text_sha256": entry["text_sha256"],
                        "token_count": entry["token_count"],
                        "embedding_dimension": EXPECTED_DIMENSION,
                        "status": "ok",
                    }
                )
                embedded_rows += 1
        completed_batches = min(
            window_start + len(window_entries), len(entry_batches)
        )
        progress_log(
            f"Qwen eval embedding | cell={cell_id} | split={split} | "
            f"batches={completed_batches}/{len(entry_batches)} | "
            f"embedded_rows={embedded_rows}/{len(entries)} | errors={len(errors)}"
        )

    array = array[:embedded_rows]
    prefix = f"{cell_id}_{split}"
    np.save(output_root / f"{prefix}.npy", array)
    write_csv(
        output_root / f"{prefix}_manifest.csv",
        manifest_rows,
        MANIFEST_FIELDS,
    )
    write_errors(output_root / f"{prefix}_errors.jsonl", errors)
    return {
        "expected_rows": len(entries),
        "embedded_rows": embedded_rows,
        "array_shape": list(array.shape),
        "label_counts": dict(
            sorted(Counter(str(entry["label_int"]) for entry in entries).items())
        ),
        "errors": len(errors),
    }


def embed_eval_splits(
    prepared: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    experiment_root: Path,
    endpoints: Sequence[EndpointConfig],
    api_key: str | None,
    timeout_seconds: int,
    batch_size: int,
) -> tuple[Path, dict[str, Any]]:
    from transformers import AutoTokenizer

    output_root = experiment_root / "Embeddings" / "Qwen3Eval"
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / LOCK_NAME
    if lock_path.exists():
        lock_path.unlink()
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_ID,
        local_files_only=True,
        padding_side="left",
    )
    limit = context_limit(tokenizer)
    metadata: dict[str, Any] = {
        "metadata_version": "binary-matched-size-qwen-eval-v1",
        "created_at": utc_now(),
        "experiment_name": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "endpoint_model_id": ENDPOINT_MODEL_ID,
        "tokenizer_id": TOKENIZER_ID,
        "tokenizer_revision": tokenizer_revision(tokenizer),
        "backend": "lm_studio_openai_compatible_embeddings",
        "instruction": INSTRUCTION,
        "expected_dimension": EXPECTED_DIMENSION,
        "dtype": "float32",
        "l2_normalized": True,
        "truncation": False,
        "context_limit": limit,
        "endpoints": [
            {
                "api_url": url,
                "model": model,
                "max_concurrency": concurrency,
            }
            for url, model, concurrency in endpoints
        ],
        "cells": {},
        "status": "running",
    }
    for cell_id in TARGET_CELLS:
        metadata["cells"][cell_id] = {}
        for split in ("dev", "test"):
            metadata["cells"][cell_id][split] = embed_split(
                output_root,
                tokenizer,
                limit,
                prepared[cell_id][split],
                cell_id,
                split,
                batch_size,
                endpoints,
                api_key,
                timeout_seconds,
            )
    return output_root, metadata


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_outputs(
    output_root: Path,
    prepared: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    metadata: Mapping[str, Any],
) -> None:
    import numpy as np

    for cell_id in TARGET_CELLS:
        for split in ("dev", "test"):
            prefix = f"{cell_id}_{split}"
            expected_rows = len(prepared[cell_id][split])
            array = np.load(output_root / f"{prefix}.npy", mmap_mode="r")
            manifest = read_manifest(output_root / f"{prefix}_manifest.csv")
            error_path = output_root / f"{prefix}_errors.jsonl"
            with error_path.open("r", encoding="utf-8") as handle:
                errors = sum(1 for line in handle if line.strip())
            if errors:
                raise RuntimeError(
                    f"{cell_id}/{split}: {errors} embedding errors"
                )
            if array.dtype != np.float32:
                raise RuntimeError(
                    f"{cell_id}/{split}: dtype {array.dtype} is not float32"
                )
            if array.shape != (expected_rows, EXPECTED_DIMENSION):
                raise RuntimeError(
                    f"{cell_id}/{split}: array shape {array.shape} != "
                    f"({expected_rows}, {EXPECTED_DIMENSION})"
                )
            if len(manifest) != expected_rows or len(manifest) != array.shape[0]:
                raise RuntimeError(
                    f"{cell_id}/{split}: manifest/array row count mismatch"
                )
            for index, row in enumerate(manifest):
                if int(row["embedding_row_index"]) != index:
                    raise RuntimeError(
                        f"{cell_id}/{split}: embedding_row_index mismatch at {index}"
                    )
                if int(row["eval_row_index"]) != int(
                    prepared[cell_id][split][index]["eval_row_index"]
                ):
                    raise RuntimeError(
                        f"{cell_id}/{split}: eval_row_index mismatch at {index}"
                    )
                if int(row["label_int"]) not in (0, 1):
                    raise RuntimeError(
                        f"{cell_id}/{split}: invalid manifest label at {index}"
                    )
                if int(row["embedding_dimension"]) != EXPECTED_DIMENSION:
                    raise RuntimeError(
                        f"{cell_id}/{split}: invalid manifest dimension at {index}"
                    )
            for start in range(0, expected_rows, 2_048):
                values = np.asarray(array[start : start + 2_048])
                if not np.all(np.isfinite(values)):
                    raise RuntimeError(
                        f"{cell_id}/{split}: saved vectors contain non-finite values"
                    )
                norms = np.linalg.norm(values, axis=1)
                if not np.allclose(norms, 1.0, rtol=0.0, atol=NORM_ATOL):
                    raise RuntimeError(
                        f"{cell_id}/{split}: saved vectors are not L2 normalized"
                    )
            cell_split = metadata["cells"][cell_id][split]
            if (
                cell_split["expected_rows"] != expected_rows
                or cell_split["embedded_rows"] != expected_rows
                or cell_split["array_shape"]
                != [expected_rows, EXPECTED_DIMENSION]
                or cell_split["errors"] != 0
            ):
                raise RuntimeError(
                    f"{cell_id}/{split}: metadata integrity mismatch"
                )


def lock_files(output_root: Path) -> dict[str, str]:
    return {
        path.relative_to(output_root).as_posix(): sha256_file(path)
        for path in sorted(output_root.rglob("*"))
        if path.is_file() and path.name != LOCK_NAME
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Qwen3 dev/test embedding for " + EXPERIMENT_NAME
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
        "--output-root",
        type=Path,
        default=project_root / "Results" / PATH_ID,
    )
    parser.add_argument(
        "--qwen-endpoint",
        action="append",
        nargs=3,
        metavar=("API_URL", "MODEL", "CONCURRENCY"),
    )
    parser.add_argument("--api-key", default=os.environ.get("LMSTUDIO_API_KEY"))
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.timeout_seconds <= 0 or args.embedding_batch_size <= 0:
        raise SystemExit("timeout and embedding batch size must be positive")
    try:
        endpoints = parse_lmstudio_endpoints(
            args.qwen_endpoint,
            ENDPOINT_MODEL_ID,
            DEFAULT_LMSTUDIO_EMBEDDINGS_API_URL,
        )
        if any(model != ENDPOINT_MODEL_ID for _, model, _ in endpoints):
            raise ValueError(
                f"all Qwen endpoints must use {ENDPOINT_MODEL_ID}"
            )
        prepared_root = args.prepared_root.resolve()
        manifest_root = args.manifest_root.resolve()
        experiment_root = args.output_root.resolve()

        load_and_validate_inputs(prepared_root, manifest_root)
        prepared = load_eval_splits(prepared_root)

        with loaded_lmstudio_models(
            endpoints,
            api_key=args.api_key,
            timeout_seconds=args.timeout_seconds,
            log=progress_log,
        ):
            output_root, metadata = embed_eval_splits(
                prepared,
                experiment_root,
                endpoints,
                args.api_key,
                args.timeout_seconds,
                args.embedding_batch_size,
            )

        try:
            validate_outputs(output_root, prepared, metadata)
        except Exception:
            metadata["status"] = "failed"
            write_json(output_root / "metadata.json", metadata)
            raise
        metadata["status"] = "completed"
        write_json(output_root / "metadata.json", metadata)
        write_json(
            output_root / LOCK_NAME,
            {
                "lock_version": "binary-matched-size-qwen-eval-v1",
                "created_at": utc_now(),
                "experiment_name": EXPERIMENT_NAME,
                "experiment_id": EXPERIMENT_ID,
                "files": lock_files(output_root),
            },
        )
        print(json.dumps(metadata["cells"], ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(
            f"Qwen eval embedding failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
