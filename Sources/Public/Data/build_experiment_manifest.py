from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PROTOCOL_VERSION = "journal-matched-size-v1"
B = 280
MAX_RATIO = 4.0
MAX_REAL_MULTIPLIER = 5  # Base B plus up to 4B additional real samples.
RATIOS = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0)
REPEAT_SEEDS = tuple(range(1000, 1050))
EXPERIMENT_BANK_SEED = 42
GENERATION_SEED_BASE = 42
CANDIDATES_PER_SEED = 20

# The five binary cells and their admissible labels are fixed by the protocol.
EXPECTED_LABELS: dict[str, tuple[int, ...]] = {
    "en_binary_sst2": (0, 1),
    "ko_binary_nsmc": (0, 1),
    "bn_binary_cinexdrama": (0, 1),
    "ha_binary_hausa_movie_review": (0, 1),
    "ml_binary_dravidian_codemix": (0, 1),
}

BINARY_CELLS = tuple(EXPECTED_LABELS)


@dataclass(frozen=True)
class DatasetAudit:
    cell_id: str
    split_rows: dict[str, int]
    split_sha256: dict[str, str]
    train_label_counts: dict[int, int]
    base_label_vector: dict[int, int]
    experiment_bank_label_counts: dict[int, int]


def parse_args() -> argparse.Namespace:
    script = Path(__file__).resolve()
    inferred_root = script.parents[2] if len(script.parents) >= 3 else Path.cwd()
    parser = argparse.ArgumentParser(
        description=(
            "Build the B=280 matched-size experiment bank, R=50 paired "
            "repetition manifests, and Stage A generation plan for the five "
            "prepared binary-classification cells."
        )
    )
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=inferred_root / "Data" / "Prepared",
        help="Root directory containing the prepared binary datasets.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=inferred_root / "Data" / "ExperimentManifests" / "v1",
        help="Directory in which the locked experiment manifests are written.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output directory using the same protocol.",
    )
    return parser.parse_args()


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFC", str(value))
    return re.sub(r"\s+", " ", text).strip()


def canonical_value(value: object) -> str:
    if pd.isna(value):
        return "<NA>"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return unicodedata.normalize("NFC", str(value)).strip()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def as_int_labels(series: pd.Series, cell_id: str, split: str) -> pd.Series:
    converted = pd.to_numeric(series, errors="raise")
    if not np.all(np.equal(converted, np.floor(converted))):
        raise ValueError(f"{cell_id}/{split}: label contains a non-integer value")
    return converted.astype(int)


def load_split(cell_id: str, split: str, prepared_root: Path) -> pd.DataFrame:
    path = prepared_root / cell_id / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"required file not found: {path}")

    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {"text", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{cell_id}/{split}: missing required columns {sorted(missing)}")
    frame = frame.copy()
    frame["text"] = frame["text"].map(normalize_text)
    frame["label"] = as_int_labels(frame["label"], cell_id, split)
    if frame["text"].eq("").any() or frame["label"].isna().any():
        raise ValueError(f"{cell_id}/{split}: empty text or label found")

    expected = set(EXPECTED_LABELS[cell_id])
    observed = set(frame["label"].unique())
    if observed != expected:
        raise ValueError(
            f"{cell_id}/{split}: label set mismatch. expected={sorted(expected)}, "
            f"observed={sorted(observed)}"
        )

    frame.insert(0, "_source_row", np.arange(len(frame), dtype=np.int64))
    frame["_text_hash"] = frame["text"].map(
        lambda text: sha256_bytes(text.encode("utf-8"))
    )

    source_columns = [column for column in frame.columns if not column.startswith("_")]

    def row_id(row: pd.Series) -> str:
        # Raw values and source-row order are locked so the same file yields stable IDs.
        values = [canonical_value(row[column]) for column in sorted(source_columns)]
        values.append(str(int(row["_source_row"])))
        payload = f"{cell_id}\x1f{split}\x1f" + "\x1f".join(values)
        return sha256_bytes(payload.encode("utf-8"))[:24]

    frame["row_id"] = frame.apply(row_id, axis=1)
    if frame["row_id"].duplicated().any():
        raise AssertionError(f"{cell_id}/{split}: duplicate row_id")
    return frame


def validate_split_disjointness(
    cell_id: str, frames: dict[str, pd.DataFrame]
) -> None:
    pairs = (("train", "dev"), ("train", "test"), ("dev", "test"))
    for left, right in pairs:
        overlap = set(frames[left]["_text_hash"]) & set(frames[right]["_text_hash"])
        if overlap:
            raise ValueError(
                f"{cell_id}: {left}/{right} text overlap ({len(overlap)} rows)"
            )


def allocate_base_label_vector(
    counts: dict[int, int], total_b: int = B
) -> dict[int, int]:
    """Allocate q_d(B) near the train distribution subject to 5q_c <= n_c."""
    labels = sorted(counts)
    capacities = {label: counts[label] // MAX_REAL_MULTIPLIER for label in labels}
    if any(capacities[label] < 1 for label in labels):
        raise ValueError(f"per-class 5B feasibility failed: capacities={capacities}")
    if sum(capacities.values()) < total_b:
        raise ValueError(
            "cannot construct B from the per-class capacities: "
            f"sum(floor(n_c/5))={sum(capacities.values())} < B={total_b}"
        )

    total = sum(counts.values())
    ideal = {label: total_b * counts[label] / total for label in labels}
    allocation = {
        label: max(1, min(math.floor(ideal[label]), capacities[label]))
        for label in labels
    }

    while sum(allocation.values()) < total_b:
        eligible = [label for label in labels if allocation[label] < capacities[label]]
        if not eligible:
            raise AssertionError("no eligible class while increasing the B allocation")
        chosen = max(
            eligible,
            key=lambda label: (
                ideal[label] - allocation[label],
                counts[label],
                -labels.index(label),
            ),
        )
        allocation[chosen] += 1

    while sum(allocation.values()) > total_b:
        eligible = [label for label in labels if allocation[label] > 1]
        if not eligible:
            raise AssertionError("no eligible class while decreasing the B allocation")
        chosen = max(
            eligible,
            key=lambda label: (
                allocation[label] - ideal[label],
                -counts[label],
                labels.index(label),
            ),
        )
        allocation[chosen] -= 1

    if sum(allocation.values()) != total_b:
        raise AssertionError("base label vector total mismatch")
    for label in labels:
        if MAX_REAL_MULTIPLIER * allocation[label] > counts[label]:
            raise AssertionError(
                f"label={label}: 5*q={MAX_REAL_MULTIPLIER * allocation[label]} "
                f"> train count={counts[label]}"
            )
    return allocation


def choose_experiment_bank(
    cell_id: str, train: pd.DataFrame, q: dict[int, int]
) -> pd.DataFrame:
    selected_parts: list[pd.DataFrame] = []
    for label in sorted(q):
        group = train[train["label"] == label]
        required = MAX_REAL_MULTIPLIER * q[label]
        rng = np.random.default_rng(
            stable_seed(PROTOCOL_VERSION, EXPERIMENT_BANK_SEED, cell_id, label)
        )
        positions = rng.permutation(len(group))[:required]
        part = group.iloc[positions].copy()
        part["bank_class_position"] = np.arange(required, dtype=np.int64)
        selected_parts.append(part)

    bank = pd.concat(selected_parts, ignore_index=True)
    if len(bank) != MAX_REAL_MULTIPLIER * B:
        raise AssertionError(
            f"{cell_id}: experiment bank={len(bank)}, expected={MAX_REAL_MULTIPLIER * B}"
        )
    if bank["row_id"].duplicated().any():
        raise AssertionError(f"{cell_id}: duplicate experiment-bank row_id")
    return bank


def nested_stratified_order(
    class_queues: dict[int, list[str]], q: dict[int, int]
) -> list[str]:
    """Build the 4B additional-real master order so every ratio is a prefix."""
    labels = sorted(q)
    used = {label: 0 for label in labels}
    pointers = {label: 0 for label in labels}
    total_length = int(MAX_RATIO * B)
    order: list[str] = []

    for step in range(1, total_length + 1):
        eligible = [
            label
            for label in labels
            if pointers[label] < len(class_queues[label])
        ]
        if not eligible:
            raise AssertionError("additional-real queue was exhausted early")

        chosen = max(
            eligible,
            key=lambda label: (
                step * q[label] / B - used[label],
                -labels.index(label),
            ),
        )
        order.append(class_queues[chosen][pointers[chosen]])
        pointers[chosen] += 1
        used[chosen] += 1

    expected = {label: int(MAX_RATIO * q[label]) for label in labels}
    if used != expected:
        raise AssertionError(f"final additional-real label vector mismatch: {used} != {expected}")
    return order


def build_repetition_records(
    cell_id: str,
    bank: pd.DataFrame,
    q: dict[int, int],
) -> list[dict[str, Any]]:
    id_to_label = dict(zip(bank["row_id"], bank["label"].astype(int), strict=True))
    by_label = {
        label: bank.loc[bank["label"] == label, "row_id"].tolist()
        for label in sorted(q)
    }
    all_bank_ids = set(bank["row_id"])
    records: list[dict[str, Any]] = []

    for repeat_seed in REPEAT_SEEDS:
        base_ids: list[str] = []
        remaining_by_label: dict[int, list[str]] = {}

        for label in sorted(q):
            ids = by_label[label].copy()
            rng = np.random.default_rng(
                stable_seed(PROTOCOL_VERSION, cell_id, repeat_seed, label)
            )
            rng.shuffle(ids)
            base_ids.extend(ids[: q[label]])
            remaining_by_label[label] = ids[q[label] :]

        rng_base = np.random.default_rng(
            stable_seed(PROTOCOL_VERSION, cell_id, repeat_seed, "base-order")
        )
        rng_base.shuffle(base_ids)
        additional_order = nested_stratified_order(remaining_by_label, q)

        if len(base_ids) != B or len(additional_order) != int(MAX_RATIO * B):
            raise AssertionError(f"{cell_id}/{repeat_seed}: sample-count mismatch")
        if set(base_ids) & set(additional_order):
            raise AssertionError(f"{cell_id}/{repeat_seed}: Base/additional overlap")
        if set(base_ids) | set(additional_order) != all_bank_ids:
            raise AssertionError(f"{cell_id}/{repeat_seed}: incomplete experiment-bank partition")

        ratio_specs: list[dict[str, Any]] = []
        previous_ids: set[str] = set()
        for ratio in RATIOS:
            n_add_float = ratio * B
            if not float(n_add_float).is_integer():
                raise AssertionError(f"rB is not an integer: r={ratio}, B={B}")
            n_add = int(n_add_float)
            selected_ids = additional_order[:n_add]
            selected_set = set(selected_ids)
            if not previous_ids.issubset(selected_set):
                raise AssertionError(f"{cell_id}/{repeat_seed}: ratio prefixes are not nested")
            previous_ids = selected_set

            label_counts = Counter(id_to_label[row_id] for row_id in selected_ids)
            ratio_specs.append(
                {
                    "ratio": ratio,
                    "n_add": n_add,
                    "total_train_n": B + n_add,
                    "additional_real_prefix_n": n_add,
                    "additional_label_vector": {
                        str(label): int(label_counts.get(label, 0))
                        for label in sorted(q)
                    },
                    # Candidate IDs are filled after generation; counts and labels are fixed here.
                    "synthetic_target_n": n_add,
                    "synthetic_target_label_vector": {
                        str(label): int(label_counts.get(label, 0))
                        for label in sorted(q)
                    },
                }
            )

        base_counts = Counter(id_to_label[row_id] for row_id in base_ids)
        if dict(base_counts) != q:
            raise AssertionError(
                f"{cell_id}/{repeat_seed}: base label vector {dict(base_counts)} != {q}"
            )

        records.append(
            {
                "protocol_version": PROTOCOL_VERSION,
                "cell_id": cell_id,
                "repeat_seed": repeat_seed,
                "base_n": B,
                "base_label_vector": {str(k): int(v) for k, v in q.items()},
                "base_row_ids": base_ids,
                "additional_real_master_order": additional_order,
                "ratios": ratio_specs,
            }
        )
    return records


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def build_stage_a_generation_plan(
    banks: dict[str, pd.DataFrame], output_root: Path
) -> int:
    rows: list[dict[str, Any]] = []
    global_index = 0
    for cell_id in BINARY_CELLS:
        bank = banks[cell_id].sort_values("row_id", kind="stable")
        for record in bank.itertuples(index=False):
            for candidate_index in range(CANDIDATES_PER_SEED):
                rows.append(
                    {
                        "protocol_version": PROTOCOL_VERSION,
                        "cell_id": cell_id,
                        "real_row_id": record.row_id,
                        "label": int(record.label),
                        "candidate_index": candidate_index,
                        "global_generation_index": global_index,
                        "generation_seed": GENERATION_SEED_BASE + global_index,
                    }
                )
                global_index += 1

    plan = pd.DataFrame(rows)
    expected = len(BINARY_CELLS) * MAX_REAL_MULTIPLIER * B * CANDIDATES_PER_SEED
    if len(plan) != expected:
        raise AssertionError(f"Stage A generation plan={len(plan)}, expected={expected}")
    plan.to_csv(
        output_root / "stage_a_generation_plan.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return len(plan)


def create_lock(output_root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(output_root.rglob("*")):
        if not path.is_file() or path.name == "MANIFEST_LOCK.json":
            continue
        relative = path.relative_to(output_root).as_posix()
        hashes[relative] = sha256_file(path)
    write_json(output_root / "MANIFEST_LOCK.json", hashes)
    return hashes


def main() -> None:
    args = parse_args()
    prepared_root = args.prepared_root.resolve()
    output_root = args.output_root.resolve()

    if not prepared_root.exists():
        raise FileNotFoundError(f"prepared root not found: {prepared_root}")
    if output_root.exists():
        if not args.force:
            raise FileExistsError(
                f"output path already exists: {output_root}\n"
                "Stop to preserve the locked copy, or use --force to regenerate it intentionally."
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    audits: list[DatasetAudit] = []
    banks: dict[str, pd.DataFrame] = {}

    for cell_id in EXPECTED_LABELS:
        frames = {
            split: load_split(cell_id, split, prepared_root)
            for split in ("train", "dev", "test")
        }
        validate_split_disjointness(cell_id, frames)

        train = frames["train"]
        counts = {
            int(label): int(count)
            for label, count in train["label"].value_counts().sort_index().items()
        }
        q = allocate_base_label_vector(counts)
        bank = choose_experiment_bank(cell_id, train, q)
        banks[cell_id] = bank
        records = build_repetition_records(cell_id, bank, q)

        cell_dir = output_root / cell_id
        cell_dir.mkdir()
        output_columns = [
            "row_id",
            "_source_row",
            "_text_hash",
            "text",
            "label",
            "bank_class_position",
        ]
        bank.loc[:, output_columns].to_csv(
            cell_dir / "experiment_bank.csv", index=False, encoding="utf-8-sig"
        )
        write_jsonl(cell_dir / "repetitions.jsonl", records)

        split_paths = {
            split: prepared_root / cell_id / f"{split}.csv"
            for split in ("train", "dev", "test")
        }
        audit = DatasetAudit(
            cell_id=cell_id,
            split_rows={split: len(frames[split]) for split in frames},
            split_sha256={split: sha256_file(path) for split, path in split_paths.items()},
            train_label_counts=counts,
            base_label_vector=q,
            experiment_bank_label_counts={
                int(label): int(count)
                for label, count in bank["label"].value_counts().sort_index().items()
            },
        )
        audits.append(audit)
        write_json(
            cell_dir / "cell_manifest.json",
            {
                "protocol_version": PROTOCOL_VERSION,
                "cell_id": audit.cell_id,
                "prepared_split_rows": audit.split_rows,
                "prepared_split_sha256": audit.split_sha256,
                "train_label_counts": {str(k): v for k, v in counts.items()},
                "base_n": B,
                "base_label_vector": {str(k): v for k, v in q.items()},
                "experiment_bank_n": MAX_REAL_MULTIPLIER * B,
                "experiment_bank_label_counts": {
                    str(k): v for k, v in audit.experiment_bank_label_counts.items()
                },
                "repeat_seeds": list(REPEAT_SEEDS),
                "ratios": list(RATIOS),
            },
        )

    generation_rows = build_stage_a_generation_plan(banks, output_root)
    protocol_manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "B": B,
        "max_ratio": MAX_RATIO,
        "max_total_real_n": MAX_REAL_MULTIPLIER * B,
        "ratios": list(RATIOS),
        "repeat_seeds": list(REPEAT_SEEDS),
        "experiment_bank_seed": EXPERIMENT_BANK_SEED,
        "candidates_per_seed": CANDIDATES_PER_SEED,
        "stage_a_generation_rows": generation_rows,
        "stage_a_generation_rows_per_binary_cell": (
            MAX_REAL_MULTIPLIER * B * CANDIDATES_PER_SEED
        ),
        "cells": [
            {
                "cell_id": audit.cell_id,
                "split_rows": audit.split_rows,
                "train_label_counts": {
                    str(k): v for k, v in audit.train_label_counts.items()
                },
                "base_label_vector": {
                    str(k): v for k, v in audit.base_label_vector.items()
                },
                "experiment_bank_label_counts": {
                    str(k): v
                    for k, v in audit.experiment_bank_label_counts.items()
                },
            }
            for audit in audits
        ],
    }
    write_json(output_root / "protocol_manifest.json", protocol_manifest)
    lock = create_lock(output_root)

    print("=" * 68)
    print("EXPERIMENT MANIFEST LOCKED")
    print("=" * 68)
    print(f"protocol              : {PROTOCOL_VERSION}")
    print(f"prepared root         : {prepared_root}")
    print(f"output root           : {output_root}")
    print(f"cells                 : {len(audits)} / {len(EXPECTED_LABELS)}")
    print(f"common B              : {B}")
    print(f"experiment bank/cell  : {MAX_REAL_MULTIPLIER * B}")
    print(f"paired repetitions    : {len(REPEAT_SEEDS)}")
    print(f"Stage A candidates    : {generation_rows:,}")
    print(f"locked files          : {len(lock)}")
    print("Subsequent stages must stop if MANIFEST_LOCK.json changes.")
    print("=" * 68)


if __name__ == "__main__":
    main()
