from __future__ import annotations

import argparse
import re
import shutil
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


SEED = 20260713
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW = PROJECT_ROOT / "Data" / "Raw"
OUTPUT = PROJECT_ROOT / "Data" / "Prepared"
SPLITS = ("train", "dev", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the five binary-classification datasets used in the study."
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=RAW,
        help="Root directory containing the raw dataset folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT,
        help="Directory in which prepared train/dev/test CSV files are written.",
    )
    return parser.parse_args()


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(value))).strip()


def normalize_label(value: object) -> object:
    if pd.isna(value) or str(value).strip() == "":
        return np.nan
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value).strip()
    return value


def clean_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    columns = ["text", "label"]
    df = df.loc[:, columns].copy()
    before = len(df)
    df["text"] = df["text"].map(normalize_text)
    df["label"] = df["label"].map(normalize_label)
    df = df[(df["text"] != "") & df["label"].notna()]
    conflicts = df.groupby("text", sort=False)["label"].nunique()
    conflict_texts = set(conflicts[conflicts > 1].index)
    if conflict_texts:
        df = df[~df["text"].isin(conflict_texts)]
    df = df.drop_duplicates(["text", "label"], keep="first").reset_index(drop=True)
    return df, before - len(df)


def clean_official(
    frames: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], int]:
    prepared: dict[str, pd.DataFrame] = {}
    before = sum(len(df) for df in frames.values())
    for split, frame in frames.items():
        columns = ["text", "label"]
        current = frame.loc[:, columns].copy()
        current["text"] = current["text"].map(normalize_text)
        current["label"] = current["label"].map(normalize_label)
        current = current[(current["text"] != "") & current["label"].notna()]
        prepared[split] = current

    combined = pd.concat(prepared.values(), ignore_index=True)
    conflicts = combined.groupby("text", sort=False)["label"].nunique()
    conflict_texts = set(conflicts[conflicts > 1].index)
    seen: set[str] = set()
    for split in ("test", "dev", "train"):
        if split not in prepared:
            continue
        current = prepared[split]
        if conflict_texts:
            current = current[~current["text"].isin(conflict_texts)]
        current = current.drop_duplicates(["text", "label"], keep="first")
        current = current[~current["text"].isin(seen)].reset_index(drop=True)
        seen.update(current["text"])
        prepared[split] = current
    return prepared, before - sum(len(df) for df in prepared.values())


def stratified_split(
    df: pd.DataFrame, train_size: float = 0.70, dev_size: float = 0.15
) -> dict[str, pd.DataFrame]:
    holdout_size = 1.0 - train_size
    train, holdout = train_test_split(
        df, test_size=holdout_size, random_state=SEED, stratify=df["label"]
    )
    dev_fraction = dev_size / holdout_size
    dev, test = train_test_split(
        holdout,
        train_size=dev_fraction,
        random_state=SEED,
        stratify=holdout["label"],
    )
    return {
        "train": train.reset_index(drop=True),
        "dev": dev.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def split_official_train(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[dict[str, pd.DataFrame], int]:
    cleaned, removed = clean_official({"train": train, "test": test})
    train_part, dev = train_test_split(
        cleaned["train"],
        test_size=0.15,
        random_state=SEED,
        stratify=cleaned["train"]["label"],
    )
    return {
        "train": train_part.reset_index(drop=True),
        "dev": dev.reset_index(drop=True),
        "test": cleaned["test"].reset_index(drop=True),
    }, removed


def validate_splits(splits: dict[str, pd.DataFrame]) -> None:
    expected = set(pd.concat(splits.values(), ignore_index=True)["label"].unique())
    texts: dict[str, set[str]] = {}
    for split in SPLITS:
        if split not in splits or splits[split].empty:
            raise ValueError(f"{split}.csv is missing or empty")
        frame = splits[split]
        if frame["text"].isna().any() or frame["label"].isna().any():
            raise ValueError(f"{split}.csv contains null text or label")
        if set(frame["label"].unique()) != expected:
            raise ValueError(f"{split}.csv does not contain every class")
        texts[split] = set(frame["text"])
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        if texts[left] & texts[right]:
            raise ValueError(f"text overlap between {left} and {right}")


def replace_output(cell_id: str, source: Path) -> None:
    target = OUTPUT / cell_id
    if target.exists():
        shutil.rmtree(target)
    source.replace(target)


def save_splits(cell_id: str, splits: dict[str, pd.DataFrame]) -> None:
    validate_splits(splits)
    temp = OUTPUT / f".{cell_id}.tmp"
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)
    for split in SPLITS:
        splits[split].loc[:, ["text", "label"]].to_csv(
            temp / f"{split}.csv", index=False, encoding="utf-8-sig"
        )
    replace_output(cell_id, temp)


def frame_counts(splits: dict[str, pd.DataFrame]) -> tuple[dict[str, int], dict[int, int]]:
    sizes = {split: len(splits[split]) for split in SPLITS}
    labels = Counter(pd.concat(splits.values(), ignore_index=True)["label"].astype(int))
    return sizes, dict(sorted(labels.items()))


def prepare_sst2() -> tuple[dict[str, pd.DataFrame], int]:
    base = RAW / "SST-2"
    train = pd.read_csv(base / "train.tsv", sep="\t").rename(columns={"sentence": "text"})
    test = pd.read_csv(base / "dev.tsv", sep="\t").rename(columns={"sentence": "text"})
    return split_official_train(train, test)


def prepare_nsmc() -> tuple[dict[str, pd.DataFrame], int]:
    base = RAW / "nsmc-master"
    train = pd.read_csv(base / "ratings_train.txt", sep="\t").rename(
        columns={"document": "text"}
    )
    test = pd.read_csv(base / "ratings_test.txt", sep="\t").rename(
        columns={"document": "text"}
    )
    return split_official_train(train, test)


def load_hausa() -> pd.DataFrame:
    path = RAW / "HausaMovieReview-main" / "Data" / "HausaMovieReview.csv"
    df = pd.read_csv(path)
    language = df["Language"].astype(str).str.strip().str.lower()
    df = df[language == "hausa"].copy()
    labels = df["label"].astype(str).str.strip().str.lower().replace(
        {"negetive": "negative"}
    )
    return pd.DataFrame({"text": df["text"], "sentiment": labels})


def prepare_cinexdrama() -> tuple[dict[str, pd.DataFrame], int]:
    path = RAW / "CineXDrama" / "CineXDrama.csv"
    df = pd.read_csv(path)
    df = df[df["Relevance"] == 1]
    frame = pd.DataFrame(
        {"text": df["Comments"], "label": df["Sentiment"].map({0.0: 0, 1.0: 1})}
    )
    frame, removed = clean_frame(frame)
    return stratified_split(frame), removed


def prepare_hausa_binary() -> tuple[dict[str, pd.DataFrame], int]:
    df = load_hausa()
    frame = pd.DataFrame(
        {"text": df["text"], "label": df["sentiment"].map({"negative": 0, "positive": 1})}
    )
    frame, removed = clean_frame(frame)
    return stratified_split(frame), removed


def extract_dravidian() -> Path:
    base = RAW / "DravidianCodeMix-Dataset-main"
    archive = base / "DravidianCodeMix-2020.zip"
    target = base / "DravidianCodeMix"
    target.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        for split in ("train", "dev", "test"):
            member = f"DravidianCodeMix/mal_full_sentiment_{split}.csv"
            output = target / f"mal_full_sentiment_{split}.csv"
            if not output.exists():
                with zipped.open(member) as source, output.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
    return target


def read_dravidian(path: Path) -> pd.DataFrame:
    rows: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            parts = line.rstrip("\r\n").rsplit(";", 2)
            if len(parts) == 3:
                rows.append((parts[0], parts[1].strip().lower()))
    return pd.DataFrame(rows, columns=["text", "sentiment"])


def prepare_dravidian() -> tuple[dict[str, pd.DataFrame], int]:
    base = extract_dravidian()
    frames: dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        raw = read_dravidian(base / f"mal_full_sentiment_{split}.csv")
        frames[split] = pd.DataFrame(
            {"text": raw["text"], "label": raw["sentiment"].map({"negative": 0, "positive": 1})}
        )
    return clean_official(frames)


PREPARERS = {
    "en_binary_sst2": prepare_sst2,
    "ko_binary_nsmc": prepare_nsmc,
    "bn_binary_cinexdrama": prepare_cinexdrama,
    "ha_binary_hausa_movie_review": prepare_hausa_binary,
    "ml_binary_dravidian_codemix": prepare_dravidian,
}


SOURCE_FILES = {
    "en_binary_sst2": "SST-2/train.tsv, dev.tsv",
    "ko_binary_nsmc": "nsmc-master/ratings_train.txt, ratings_test.txt",
    "bn_binary_cinexdrama": "CineXDrama/CineXDrama.csv",
    "ha_binary_hausa_movie_review": "HausaMovieReview-main/Data/HausaMovieReview.csv",
    "ml_binary_dravidian_codemix": "DravidianCodeMix-Dataset-main/DravidianCodeMix-2020.zip",
}


def safe_reason(error: Exception) -> str:
    return re.sub(r"\s+", " ", str(error).splitlines()[0]).strip()[:240] or type(error).__name__


def main() -> None:
    global RAW, OUTPUT

    args = parse_args()
    RAW = args.raw_root.expanduser().resolve()
    OUTPUT = args.output_root.expanduser().resolve()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    completed: dict[str, tuple[dict[str, int], dict[int, int], int]] = {}
    skipped: dict[str, str] = {}
    order = list(SOURCE_FILES)

    for cell_id in order:
        try:
            splits, removed = PREPARERS[cell_id]()
            save_splits(cell_id, splits)
            sizes, labels = frame_counts(splits)
            completed[cell_id] = (sizes, labels, removed)
        except Exception as error:
            skipped[cell_id] = safe_reason(error)

    print("=" * 60)
    print("BINARY DATASET PREPARATION COMPLETE")
    print("=" * 60)
    print(f"{'cell_id':38} {'train':>8} {'dev':>8} {'test':>8}")
    for cell_id in order:
        if cell_id not in completed:
            continue
        sizes, _, _ = completed[cell_id]
        print(
            f"{cell_id:38} {sizes['train']:8d} {sizes['dev']:8d} {sizes['test']:8d}"
        )
    print()
    for cell_id in order:
        if cell_id not in completed:
            continue
        _, labels, removed = completed[cell_id]
        class_text = ", ".join(f"{label}={count}" for label, count in labels.items())
        print(f"{cell_id}: classes {class_text}; removed {removed}")
    print()
    print(f"Completed cells: {len(completed)} / {len(order)}")
    print(f"Skipped cells: {len(skipped)} / {len(order)}")
    if skipped:
        print()
        print("Skipped:")
        for cell_id in order:
            if cell_id in skipped:
                print(f"- {cell_id}: {skipped[cell_id]}")
    print()
    print(f"Output root: {OUTPUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
