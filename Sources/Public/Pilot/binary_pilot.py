"""Gate 2 binary generation and embedding pilot.

This module is intentionally standalone.  It reads the locked Gate 1.5
manifest and prepared data, then writes only under Results/Pilot.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import statistics
import sys
import traceback
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

_SOURCES_ROOT = Path(__file__).resolve().parents[1]
if str(_SOURCES_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCES_ROOT))

from Common.lmstudio import (
    DEFAULT_LMSTUDIO_EMBEDDINGS_API_URL,
    EndpointConfig,
    LmStudioRequestError,
    STANDARD_LMSTUDIO_GENERATION_SETTINGS,
    loaded_lmstudio_models,
    parse_lmstudio_endpoints,
    request_lmstudio_chat_content,
    request_lmstudio_embeddings,
)


PROTOCOL_VERSION = "journal-matched-size-v1"
PROMPT_ID = "binary_v2"
MODEL_ID = "google/gemma-4-31b"
QWEN_MODEL_ID = "text-embedding-qwen3-embedding-8b"
BGE_MODEL_ID = "text-embedding-bge-m3"
QWEN_TOKENIZER_ID = "Qwen/Qwen3-Embedding-8B"
BGE_TOKENIZER_ID = "BAAI/bge-m3"
QWEN_INSTRUCTION = "Represent this text for supervised text classification."

REQUESTS_PER_CELL = 100
TOTAL_REQUESTS = 500
CANDIDATES_PER_SEED = 20
MAX_ATTEMPTS = 3
GENERATION_DTYPE = "bfloat16"

GENERATION_CONFIG: dict[str, Any] = {
    **STANDARD_LMSTUDIO_GENERATION_SETTINGS.report_parameters(),
    "do_sample": True,
    "num_beams": 1,
    "thinking": True,
    "retrieval_example_search": False,
    "quantization": None,
    "dtype": GENERATION_DTYPE,
}

TARGET_CELLS = (
    "en_binary_sst2",
    "ko_binary_nsmc",
    "bn_binary_cinexdrama",
    "ha_binary_hausa_movie_review",
    "ml_binary_dravidian_codemix",
)

CELL_LANGUAGE = {
    "en_binary_sst2": "en",
    "ko_binary_nsmc": "ko",
    "bn_binary_cinexdrama": "bn",
    "ha_binary_hausa_movie_review": "ha",
    "ml_binary_dravidian_codemix": "ml",
}

INHERITED_LABEL = {0: "negative", 1: "positive"}
ALLOWED_LABELS = {"positive", "negative"}

BINARY_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "label": {"type": "string", "enum": ["positive", "negative"]},
    },
    "required": ["text", "label"],
    "additionalProperties": False,
}
BGE_COSINE_FIELDS = [
    "cell_id",
    "language",
    "seed_index",
    "request_key",
    "seed_embedding_row_index",
    "synthetic_embedding_row_index",
    "cosine_similarity",
]


# This header is the original dissertation JSON-forcing header.  The
# binary_v2 bodies below are kept as literal text so the prompt surface is
# auditable and does not depend on the old execution pipeline.
COMMON_JSON_HEADER = (
    "IMPORTANT:\n"
    "You MUST output ONLY one valid JSON object.\n"
    "No introductions, no explanations, no code fences, no comments.\n"
    "The first character must be '{' and the last must be '}'.\n"
    'Schema: {"text": string, "label": "positive|negative"}\n\n'
)

PROMPT_BODIES = {
    "en": (
        "Output exactly one JSON object and nothing else.\n"
        'Format: {"text":"...", "label":"positive|negative"}\n'
        "Target label: {label}\n"
        "Target language: English\n"
        "Seed sentence: {seed_text_clipped_to_20_words}\n"
        "Generate exactly one new sentence in the same domain, about the same specific "
        "subject or evaluative aspect as the seed, and consistent with the target sentiment.\n"
        "Preserve the seed's core meaning and specific evaluation, but use substantially "
        "different wording and sentence structure. You may change the emphasis or viewpoint, "
        "but do not introduce a different situation, domain, or unrelated subject.\n"
        "Rules:\n"
        "1. Use only the target language in the generated sentence.\n"
        "2. Write a natural, complete sentence that can be understood independently.\n"
        "3. Do not copy, closely paraphrase, or merely replace a few words from the seed.\n"
        "4. Do not invent facts, situations, proper names, identifiable people, events, "
        "or numbers absent from the seed. Do not change the subject or evaluation target.\n"
        "5. Do not use meaningless strings, mixed languages, or excessive repetition.\n"
        "6. Ensure that the sentence is unambiguously consistent with the target label. "
        "The sentiment may be expressed implicitly.\n"
        "7. Preserve the seed's specific evaluative aspect instead of replacing it with "
        "generic stock praise or criticism.\n"
        "8. Output exactly one sentence with no more than 30 words."
    ),

    "ko": (
        "JSON 객체 1개만 출력하세요. 다른 텍스트는 출력하지 마세요.\n"
        '형식: {"text":"...", "label":"positive|negative"}\n'
        "목표 라벨: {label}\n"
        "목표 언어: 한국어\n"
        "시드 문장: {seed_text_clipped_to_20_words}\n"
        "같은 도메인에서 시드와 동일한 구체적 주제 또는 평가 요소를 다루며, "
        "목표 감성과 일치하는 새로운 문장 1개를 생성하세요.\n"
        "시드의 핵심 의미와 구체적인 평가를 유지하되, 충분히 다른 표현과 문장 구조를 사용하세요. "
        "강조점이나 관점은 바꿀 수 있지만, 다른 상황이나 도메인 또는 관련 없는 주제를 새로 도입하지 마세요.\n"
        "규칙:\n"
        "1. 생성 문장에는 목표 언어만 사용하세요.\n"
        "2. 자연스럽고 독립적으로 이해되는 완결된 문장을 작성하세요.\n"
        "3. 시드를 복사하거나 가깝게 의역하거나 일부 단어만 바꾸지 마세요.\n"
        "4. 시드에 없는 사실, 상황, 고유명사, 식별 가능한 인물, 사건 또는 숫자를 만들지 마세요. "
        "주제나 평가 대상을 바꾸지 마세요.\n"
        "5. 무의미한 문자열, 다른 언어의 혼입 또는 과도한 반복을 사용하지 마세요.\n"
        "6. 문장이 목표 라벨과 명백하게 일치하도록 작성하세요. 감성은 암시적으로 표현해도 됩니다.\n"
        "7. 시드의 구체적인 평가 요소를 상투적인 칭찬이나 비난으로 대체하지 마세요.\n"
        "8. 정확히 문장 1개만, 30단어 이내로 작성하세요."
    ),

    "bn": (
        "শুধু একটি JSON অবজেক্ট আউটপুট করুন। অন্য কোনো লেখা আউটপুট করবেন না।\n"
        'ফরম্যাট: {"text":"...", "label":"positive|negative"}\n'
        "লক্ষ্য লেবেল: {label}\n"
        "লক্ষ্য ভাষা: বাংলা\n"
        "মূল বাক্য: {seed_text_clipped_to_20_words}\n"
        "একই ক্ষেত্রের মধ্যে মূল বাক্যের একই নির্দিষ্ট বিষয় বা মূল্যায়নের দিক নিয়ে এবং "
        "লক্ষ্য অনুভূতির সঙ্গে সামঞ্জস্যপূর্ণ ঠিক একটি নতুন বাক্য তৈরি করুন।\n"
        "মূল বাক্যের মূল অর্থ ও নির্দিষ্ট মূল্যায়ন বজায় রাখুন, তবে উল্লেখযোগ্যভাবে ভিন্ন "
        "শব্দচয়ন ও বাক্যগঠন ব্যবহার করুন। গুরুত্ব বা দৃষ্টিভঙ্গি পরিবর্তন করা যেতে পারে, "
        "কিন্তু ভিন্ন পরিস্থিতি, ক্ষেত্র বা সম্পর্কহীন বিষয় প্রবর্তন করবেন না।\n"
        "নিয়ম:\n"
        "১. তৈরি করা বাক্যে শুধু লক্ষ্য ভাষা ব্যবহার করুন।\n"
        "২. স্বাভাবিক, সম্পূর্ণ এবং স্বতন্ত্রভাবে বোঝা যায় এমন একটি বাক্য লিখুন।\n"
        "৩. মূল বাক্য কপি করবেন না, কাছাকাছি ভাষায় পুনর্লিখন করবেন না এবং শুধু কয়েকটি শব্দ বদলাবেন না।\n"
        "৪. মূল বাক্যে নেই এমন তথ্য, পরিস্থিতি, নির্দিষ্ট নাম, শনাক্তযোগ্য ব্যক্তি, ঘটনা বা সংখ্যা "
        "তৈরি করবেন না। বিষয় বা মূল্যায়নের লক্ষ্য পরিবর্তন করবেন না।\n"
        "৫. অর্থহীন অক্ষরসমষ্টি, অন্য ভাষার মিশ্রণ অথবা অতিরিক্ত পুনরাবৃত্তি ব্যবহার করবেন না।\n"
        "৬. বাক্যটি যেন লক্ষ্য লেবেলের সঙ্গে দ্ব্যর্থহীনভাবে সামঞ্জস্যপূর্ণ হয়। "
        "অনুভূতি পরোক্ষভাবেও প্রকাশ করা যেতে পারে।\n"
        "৭. মূল বাক্যের নির্দিষ্ট মূল্যায়নের দিককে সাধারণ ও গতানুগতিক প্রশংসা বা সমালোচনা দিয়ে "
        "প্রতিস্থাপন করবেন না।\n"
        "৮. ঠিক একটি বাক্য লিখুন এবং সর্বোচ্চ ৩০টি শব্দ ব্যবহার করুন।"
    ),

    "ha": (
        "Fitar da abu guda ɗaya na JSON kawai. Kada ka fitar da wani rubutu.\n"
        'Tsari: {"text":"...", "label":"positive|negative"}\n'
        "Alamar da ake nufi: {label}\n"
        "Harshen da ake nufi: Hausa\n"
        "Jimlar tushe: {seed_text_clipped_to_20_words}\n"
        "Ƙirƙiri sabuwar jimla guda ɗaya a cikin fanni ɗaya, game da takamaiman batu "
        "ko ɓangaren kimantawa iri ɗaya da jimlar tushe, kuma mai dacewa da alamar da ake nufi.\n"
        "Riƙe ainihin ma'ana da takamaiman kimantawar jimlar tushe, amma yi amfani da kalmomi "
        "da tsarin jimla masu matuƙar bambanci. Ana iya sauya abin da aka fi jaddadawa ko mahangar magana, "
        "amma kada a gabatar da wani yanayi, fanni, ko batu marar alaƙa.\n"
        "Ka'idoji:\n"
        "1. Yi amfani da harshen da ake nufi kawai a cikin jimlar da aka ƙirƙira.\n"
        "2. Rubuta jimla ta halitta, cikakkiya, kuma mai sauƙin fahimta ita kaɗai.\n"
        "3. Kada ka kwafi jimlar tushe, ka sake faɗarta da kusan kalmomi iri ɗaya, "
        "ko ka sauya wasu kalmomi kaɗan kawai.\n"
        "4. Kada ka ƙirƙiri bayanai, yanayi, sunaye na musamman, mutanen da za a iya gane su, "
        "abubuwan da suka faru, ko lambobin da babu su a jimlar tushe. "
        "Kada ka sauya batu ko abin da ake kimantawa.\n"
        "5. Kada ka yi amfani da rubutu marar ma'ana, haɗa wasu harsuna, ko maimaitawa fiye da kima.\n"
        "6. Ka tabbatar jimlar ta yi daidai da alamar da ake nufi ba tare da ruɗani ba. "
        "Ana iya bayyana ra'ayin a kaikaice.\n"
        "7. Kada ka maye gurbin takamaiman ɓangaren kimantawar jimlar tushe da yabo "
        "ko suka na gama gari.\n"
        "8. Rubuta jimla guda ɗaya tak kuma kada ta wuce kalmomi 30."
    ),

    "ml": (
        "ഒരു JSON ഒബ്ജക്റ്റ് മാത്രം ഔട്ട്പുട്ട് ചെയ്യുക. മറ്റൊരു എഴുത്തും ഔട്ട്പുട്ട് ചെയ്യരുത്.\n"
        'ഫോർമാറ്റ്: {"text":"...", "label":"positive|negative"}\n'
        "ലക്ഷ്യ ലേബൽ: {label}\n"
        "ലക്ഷ്യ ഭാഷ: മലയാളം\n"
        "സീഡ് വാക്യം: {seed_text_clipped_to_20_words}\n"
        "അതേ ഡൊമെയിനിൽ, സീഡിലെ അതേ നിർദ്ദിഷ്ട വിഷയം അല്ലെങ്കിൽ വിലയിരുത്തൽ ഘടകം സംബന്ധിച്ചും "
        "ലക്ഷ്യ വികാരവുമായി പൊരുത്തപ്പെടുന്നതുമായ കൃത്യമായി ഒരു പുതിയ വാക്യം സൃഷ്ടിക്കുക.\n"
        "സീഡിന്റെ മുഖ്യ അർത്ഥവും നിർദ്ദിഷ്ട വിലയിരുത്തലും നിലനിർത്തുക, എന്നാൽ ഗണ്യമായി വ്യത്യസ്തമായ "
        "വാക്കുകളും വാക്യഘടനയും ഉപയോഗിക്കുക. ഊന്നലോ കാഴ്ചപ്പാടോ മാറ്റാം, പക്ഷേ വ്യത്യസ്തമായ സാഹചര്യം, "
        "ഡൊമെയിൻ അല്ലെങ്കിൽ ബന്ധമില്ലാത്ത വിഷയം അവതരിപ്പിക്കരുത്.\n"
        "നിയമങ്ങൾ:\n"
        "1. സൃഷ്ടിക്കുന്ന വാക്യത്തിൽ ലക്ഷ്യ ഭാഷ മാത്രം ഉപയോഗിക്കുക.\n"
        "2. സ്വാഭാവികവും പൂർണ്ണവും സ്വതന്ത്രമായി മനസ്സിലാക്കാവുന്നതുമായ ഒരു വാക്യം എഴുതുക.\n"
        "3. സീഡ് പകർത്തുകയോ അതിനോട് വളരെ അടുത്ത രീതിയിൽ പുനരാഖ്യാനം ചെയ്യുകയോ "
        "കുറച്ച് വാക്കുകൾ മാത്രം മാറ്റുകയോ ചെയ്യരുത്.\n"
        "4. സീഡിൽ ഇല്ലാത്ത വസ്തുതകൾ, സാഹചര്യങ്ങൾ, വ്യക്തിനാമങ്ങൾ, തിരിച്ചറിയാവുന്ന ആളുകൾ, "
        "സംഭവങ്ങൾ അല്ലെങ്കിൽ സംഖ്യകൾ സൃഷ്ടിക്കരുത്. വിഷയമോ വിലയിരുത്തൽ ലക്ഷ്യമോ മാറ്റരുത്.\n"
        "5. അർത്ഥമില്ലാത്ത അക്ഷരനിരകൾ, മറ്റ് ഭാഷകളുടെ കലർച്ച അല്ലെങ്കിൽ അമിതമായ ആവർത്തനം ഉപയോഗിക്കരുത്.\n"
        "6. വാക്യം ലക്ഷ്യ ലേബലുമായി സംശയമില്ലാതെ പൊരുത്തപ്പെടണം. വികാരം പരോക്ഷമായി പ്രകടിപ്പിക്കാം.\n"
        "7. സീഡിലെ നിർദ്ദിഷ്ട വിലയിരുത്തൽ ഘടകത്തെ പൊതുവായ പ്രശംസയോ വിമർശനമോ കൊണ്ട് മാറ്റരുത്.\n"
        "8. കൃത്യമായി ഒരു വാക്യം മാത്രം എഴുതുക; പരമാവധി 30 വാക്കുകൾ ഉപയോഗിക്കുക."
    ),
}

PROMPT_TEMPLATES = {
    language: COMMON_JSON_HEADER + body + "\n"
    for language, body in PROMPT_BODIES.items()
}

# Used only after a format-invalid response.  It does not alter the binary_v2
# generation instructions or the base prompt hash.
STRICT_JSON_WRAPPER = (
    "Return only the JSON object. Do not emit any text before or after it."
)
LABEL_MISMATCH_RETRY = (
    "The previous output used the wrong label. Use exactly the target label shown above."
)
REPETITION_RETRY = (
    "The previous output entered a repetition loop. Generate a new natural sentence "
    "without repeated or meaningless text."
)
WORD_LIMIT_RETRY = (
    "The previous output exceeded the word limit. Use no more than 30 words."
)
SURFACE_CORRUPTION_RETRY = (
    "The previous output contained corrupted text. Generate a natural sentence "
    "without malformed text."
)

INVALID_REASONS = {
    "empty_response",
    "json_parse_failure",
    "json_not_object",
    "unexpected_fields",
    "missing_text",
    "missing_label",
    "text_not_string",
    "label_not_string",
    "empty_generated_text",
    "invalid_label_value",
    "exact_copy",
    "label_mismatch",
    "repeated_token",
    "repeated_segment",
    "word_limit_exceeded",
    "surface_corruption",
}


class ManifestValidationError(RuntimeError):
    """Raised before model generation when a locked input is not valid."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def progress_log(message: str) -> None:
    print(f"[{utc_now()}] {message}", file=sys.stderr, flush=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def pilot_generation_seed(cell_id: str, real_row_id: str) -> int:
    source = f"{cell_id}{real_row_id}gate2-pilot-v1".encode("utf-8")
    return int.from_bytes(hashlib.sha256(source).digest()[:8], "big") % (2**32)


def attempt_generation_seed(base_generation_seed: int, attempt_number: int) -> int:
    if attempt_number == 1:
        return int(base_generation_seed)
    used = {int(base_generation_seed) % (2**32)}
    derived = 0
    for current_attempt in range(2, attempt_number + 1):
        source = f"{int(base_generation_seed)}:{current_attempt}".encode("utf-8")
        derived = int.from_bytes(hashlib.sha256(source).digest()[:4], "big")
        while derived in used:
            derived = (derived + 1) % (2**32)
        used.add(derived)
    return derived


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFC", str(value))
    return re.sub(r"\s+", " ", text).strip()


def normalize_for_copy(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", " ", text).strip().casefold()


def clip_words(value: str, count: int = 20) -> str:
    words = str(value or "").replace("\n", " ").split()
    return " ".join(words[:count])


def parse_int(value: object, field: str) -> int:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field} is empty")
    try:
        number = float(text)
    except ValueError as exc:
        raise ValueError(f"{field} is not an integer: {value!r}") from exc
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"{field} is not an integer: {value!r}")
    return int(number)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(fieldnames), extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: "" if row.get(field) is None else row.get(field)
                    for field in fieldnames
                }
            )


def relative_path(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def validate_manifest_lock(manifest_root: Path) -> dict[str, Any]:
    lock_path = manifest_root / "MANIFEST_LOCK.json"
    if not lock_path.is_file():
        raise ManifestValidationError(f"missing manifest lock: {lock_path}")

    try:
        lock = read_json(lock_path)
    except Exception as exc:
        raise ManifestValidationError(f"cannot read manifest lock: {exc}") from exc
    if not isinstance(lock, dict) or not lock:
        raise ManifestValidationError("MANIFEST_LOCK.json must be a non-empty object")

    root = manifest_root.resolve()
    errors: list[str] = []
    checked = 0
    for relative, expected_hash in sorted(lock.items()):
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            errors.append(f"invalid lock entry: {relative!r}")
            continue
        candidate = (root / Path(relative)).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            errors.append(f"lock path escapes manifest root: {relative}")
            continue
        if not candidate.is_file():
            errors.append(f"missing locked file: {relative}")
            continue
        actual_hash = sha256_file(candidate)
        checked += 1
        if actual_hash != expected_hash:
            errors.append(
                f"hash mismatch: {relative} expected={expected_hash} actual={actual_hash}"
            )

    required_files = {"stage_a_generation_plan.csv"}
    for cell_id in TARGET_CELLS:
        required_files.update(
            {
                f"{cell_id}/cell_manifest.json",
                f"{cell_id}/experiment_bank.csv",
                f"{cell_id}/repetitions.jsonl",
            }
        )
        if not (root / cell_id).is_dir():
            errors.append(f"target cell directory missing: {cell_id}")
    for relative in sorted(required_files):
        if relative not in lock:
            errors.append(f"required target file absent from lock: {relative}")

    if errors:
        raise ManifestValidationError(
            "manifest validation failed before generation:\n- " + "\n- ".join(errors)
        )

    return {
        "verified": True,
        "lock_path": str(lock_path),
        "lock_sha256": sha256_file(lock_path),
        "locked_file_count": len(lock),
        "checked_file_count": checked,
    }


def load_cell_manifest(manifest_root: Path, cell_id: str) -> dict[str, Any]:
    path = manifest_root / cell_id / "cell_manifest.json"
    value = read_json(path)
    if not isinstance(value, dict):
        raise ManifestValidationError(f"{cell_id}: cell_manifest.json is not an object")
    if value.get("cell_id") != cell_id:
        raise ManifestValidationError(f"{cell_id}: cell_manifest cell_id mismatch")
    if value.get("protocol_version") != PROTOCOL_VERSION:
        raise ManifestValidationError(f"{cell_id}: protocol version mismatch")
    return value


def validate_prepared_split_hashes(
    prepared_root: Path, cell_id: str, cell_manifest: Mapping[str, Any]
) -> None:
    expected_hashes = cell_manifest.get("prepared_split_sha256")
    if not isinstance(expected_hashes, Mapping):
        raise ManifestValidationError(f"{cell_id}: prepared_split_sha256 is missing")
    errors: list[str] = []
    for split in ("train", "dev", "test"):
        expected = expected_hashes.get(split)
        path = prepared_root / cell_id / f"{split}.csv"
        if not isinstance(expected, str) or not expected:
            errors.append(f"{split}: expected hash missing")
            continue
        if not path.is_file():
            errors.append(f"{split}: file missing")
            continue
        actual = sha256_file(path)
        if actual != expected:
            errors.append(f"{split}: expected={expected} actual={actual}")
    if errors:
        raise ManifestValidationError(
            f"{cell_id}: prepared split validation failed: " + "; ".join(errors)
        )


def load_experiment_bank(
    manifest_root: Path, cell_id: str, cell_manifest: Mapping[str, Any]
) -> list[dict[str, Any]]:
    path = manifest_root / cell_id / "experiment_bank.csv"
    required = {
        "row_id",
        "_source_row",
        "_text_hash",
        "text",
        "label",
        "bank_class_position",
    }
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or []))
            raise ManifestValidationError(
                f"{cell_id}: experiment bank columns missing: {missing}"
            )
        for bank_index, raw in enumerate(reader):
            try:
                label = parse_int(raw.get("label"), f"{cell_id}/experiment_bank.label")
                source_row = parse_int(
                    raw.get("_source_row"), f"{cell_id}/experiment_bank._source_row"
                )
                class_position = parse_int(
                    raw.get("bank_class_position"),
                    f"{cell_id}/experiment_bank.bank_class_position",
                )
            except ValueError as exc:
                raise ManifestValidationError(str(exc)) from exc
            text = normalize_text(raw.get("text"))
            row_id = str(raw.get("row_id") or "").strip()
            text_hash = str(raw.get("_text_hash") or "").strip()
            if not row_id or not text or not text_hash:
                raise ManifestValidationError(
                    f"{cell_id}: blank experiment bank field at row {bank_index}"
                )
            rows.append(
                {
                    "experiment_bank_index": bank_index,
                    "manifest_seed_index": bank_index,
                    "row_id": row_id,
                    "prepared_row_index": source_row,
                    "_text_hash": text_hash,
                    "seed_text": text,
                    "label_int": label,
                    "bank_class_position": class_position,
                }
            )

    expected_n = parse_int(
        cell_manifest.get("experiment_bank_n"), f"{cell_id}.experiment_bank_n"
    )
    if len(rows) != expected_n:
        raise ManifestValidationError(
            f"{cell_id}: experiment bank n={len(rows)} expected={expected_n}"
        )
    if len({row["row_id"] for row in rows}) != len(rows):
        raise ManifestValidationError(f"{cell_id}: duplicate experiment bank row_id")
    if len({row["prepared_row_index"] for row in rows}) != len(rows):
        raise ManifestValidationError(
            f"{cell_id}: duplicate prepared row index in bank"
        )

    observed_counts = Counter(str(row["label_int"]) for row in rows)
    expected_counts = {
        str(key): parse_int(value, f"{cell_id}.experiment_bank_label_counts.{key}")
        for key, value in (
            cell_manifest.get("experiment_bank_label_counts") or {}
        ).items()
    }
    if dict(observed_counts) != expected_counts:
        raise ManifestValidationError(
            f"{cell_id}: experiment bank label counts {dict(observed_counts)} != {expected_counts}"
        )
    if set(observed_counts) != {"0", "1"}:
        raise ManifestValidationError(
            f"{cell_id}: experiment bank labels are not binary"
        )

    for label in (0, 1):
        positions = sorted(
            row["bank_class_position"] for row in rows if row["label_int"] == label
        )
        if positions != list(range(len(positions))):
            raise ManifestValidationError(
                f"{cell_id}: bank_class_position is not a contiguous manifest index for label={label}"
            )
    return rows


def load_stage_plan_for_cells(
    manifest_root: Path,
    bank_by_cell: Mapping[str, Sequence[Mapping[str, Any]]],
    cell_manifests: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[tuple[str, int], dict[str, Any]]], set[int]]:
    path = manifest_root / "stage_a_generation_plan.csv"
    required = {
        "protocol_version",
        "cell_id",
        "real_row_id",
        "label",
        "candidate_index",
        "global_generation_index",
        "generation_seed",
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_generation_seeds: set[int] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or []))
            raise ManifestValidationError(
                f"stage_a_generation_plan columns missing: {missing}"
            )
        for raw in reader:
            cell_id = str(raw.get("cell_id") or "").strip()
            try:
                generation_seed = parse_int(
                    raw.get("generation_seed"), "stage plan generation_seed"
                )
            except ValueError as exc:
                raise ManifestValidationError(str(exc)) from exc
            all_generation_seeds.add(generation_seed)
            if cell_id not in TARGET_CELLS:
                continue
            try:
                row = {
                    "protocol_version": str(raw.get("protocol_version") or "").strip(),
                    "cell_id": cell_id,
                    "real_row_id": str(raw.get("real_row_id") or "").strip(),
                    "label_int": parse_int(raw.get("label"), "stage plan label"),
                    "candidate_index": parse_int(
                        raw.get("candidate_index"), "stage plan candidate_index"
                    ),
                    "global_generation_index": parse_int(
                        raw.get("global_generation_index"),
                        "stage plan global_generation_index",
                    ),
                    "generation_seed": generation_seed,
                }
            except ValueError as exc:
                raise ManifestValidationError(str(exc)) from exc
            grouped[cell_id].append(row)

    result: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    for cell_id in TARGET_CELLS:
        rows = grouped.get(cell_id, [])
        bank_rows = bank_by_cell[cell_id]
        bank_labels = {str(row["row_id"]): int(row["label_int"]) for row in bank_rows}
        expected_n = len(bank_rows) * CANDIDATES_PER_SEED
        if len(rows) != expected_n:
            raise ManifestValidationError(
                f"{cell_id}: stage plan n={len(rows)} expected={expected_n}"
            )

        indexed: dict[tuple[str, int], dict[str, Any]] = {}
        generation_indices: set[int] = set()
        for row in rows:
            key = (row["real_row_id"], row["candidate_index"])
            if row["protocol_version"] != cell_manifests[cell_id].get(
                "protocol_version"
            ):
                raise ManifestValidationError(
                    f"{cell_id}: stage plan protocol mismatch"
                )
            if row["real_row_id"] not in bank_labels:
                raise ManifestValidationError(
                    f"{cell_id}: stage plan references unknown seed {row['real_row_id']}"
                )
            if row["candidate_index"] not in range(CANDIDATES_PER_SEED):
                raise ManifestValidationError(
                    f"{cell_id}: invalid candidate_index {row['candidate_index']}"
                )
            if row["label_int"] != bank_labels[row["real_row_id"]]:
                raise ManifestValidationError(
                    f"{cell_id}: stage plan label mismatch for {row['real_row_id']}"
                )
            if key in indexed:
                raise ManifestValidationError(
                    f"{cell_id}: duplicate stage plan key {key}"
                )
            if row["global_generation_index"] in generation_indices:
                raise ManifestValidationError(
                    f"{cell_id}: duplicate global_generation_index"
                )
            indexed[key] = row
            generation_indices.add(row["global_generation_index"])

        for row_id in bank_labels:
            candidates = {
                candidate_index
                for (candidate_row_id, candidate_index) in indexed
                if candidate_row_id == row_id
            }
            if candidates != set(range(CANDIDATES_PER_SEED)):
                raise ManifestValidationError(
                    f"{cell_id}: incomplete candidates for seed {row_id}"
                )
        result[cell_id] = indexed
    return result, all_generation_seeds


def largest_remainder_quota(
    label_counts: Mapping[int, int], total: int
) -> dict[int, int]:
    denominator = sum(label_counts.values())
    if denominator <= 0:
        raise ValueError("cannot allocate quota from empty label counts")
    floors: dict[int, int] = {}
    remainders: dict[int, float] = {}
    for label in sorted(label_counts):
        exact = total * label_counts[label] / denominator
        floors[label] = math.floor(exact)
        remainders[label] = exact - floors[label]
    remaining = total - sum(floors.values())
    for label in sorted(label_counts, key=lambda value: (-remainders[value], value))[
        :remaining
    ]:
        floors[label] += 1
    if sum(floors.values()) != total:
        raise AssertionError("largest-remainder allocation did not sum to target")
    return floors


def validate_selected_prepared_rows(
    prepared_root: Path,
    cell_id: str,
    selected_bank_rows: Sequence[Mapping[str, Any]],
) -> dict[int, str]:
    path = prepared_root / cell_id / "train.csv"
    wanted = {int(row["prepared_row_index"]): row for row in selected_bank_rows}
    found: dict[int, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"text", "label"}.issubset(
            reader.fieldnames
        ):
            raise ManifestValidationError(
                f"{cell_id}: prepared train.csv lacks text/label"
            )
        for source_row, raw in enumerate(reader):
            if source_row not in wanted:
                continue
            bank_row = wanted[source_row]
            text = normalize_text(raw.get("text"))
            try:
                label = parse_int(raw.get("label"), f"{cell_id}/prepared label")
            except ValueError as exc:
                raise ManifestValidationError(str(exc)) from exc
            if not text:
                raise ManifestValidationError(
                    f"{cell_id}: selected prepared row has empty text {source_row}"
                )
            if text != bank_row["seed_text"]:
                raise ManifestValidationError(
                    f"{cell_id}: prepared text mismatch at row {source_row}"
                )
            if sha256_bytes(text.encode("utf-8")) != bank_row["_text_hash"]:
                raise ManifestValidationError(
                    f"{cell_id}: prepared text hash mismatch at row {source_row}"
                )
            if label != bank_row["label_int"]:
                raise ManifestValidationError(
                    f"{cell_id}: prepared label mismatch at row {source_row}"
                )
            found[source_row] = text
    missing = sorted(set(wanted) - set(found))
    if missing:
        raise ManifestValidationError(
            f"{cell_id}: selected prepared rows missing: {missing}"
        )
    return found


def build_selection(
    prepared_root: Path,
    cell_id: str,
    cell_manifest: Mapping[str, Any],
    bank_rows: Sequence[Mapping[str, Any]],
    stage_a_generation_seeds: set[int],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    label_counts = Counter(int(row["label_int"]) for row in bank_rows)
    quota = largest_remainder_quota(label_counts, REQUESTS_PER_CELL)
    selected_bank_rows: list[Mapping[str, Any]] = []
    for label in sorted(quota):
        class_rows = [row for row in bank_rows if int(row["label_int"]) == label]
        if len(class_rows) < quota[label]:
            raise ManifestValidationError(
                f"{cell_id}: insufficient bank rows for label {label}"
            )
        # experiment_bank.csv is itself locked; its row order is the
        # deterministic selection order, so no new sampling seed is introduced.
        selected_bank_rows.extend(class_rows[: quota[label]])
    selected_bank_rows = sorted(
        selected_bank_rows, key=lambda row: int(row["experiment_bank_index"])
    )
    prepared_texts = validate_selected_prepared_rows(
        prepared_root, cell_id, selected_bank_rows
    )

    selections: list[dict[str, Any]] = []
    for pilot_request_index, bank_row in enumerate(selected_bank_rows):
        row_id = str(bank_row["row_id"])
        generation_seed = pilot_generation_seed(cell_id, row_id)
        assert generation_seed not in stage_a_generation_seeds, (
            f"{cell_id}: pilot generation seed collides with Stage A for {row_id}"
        )
        inherited_label = INHERITED_LABEL[int(bank_row["label_int"])]
        request_key = f"{cell_id}:{int(bank_row['manifest_seed_index'])}"
        selections.append(
            {
                "cell_id": cell_id,
                "language": CELL_LANGUAGE[cell_id],
                "pilot_request_index": pilot_request_index,
                "seed_index": int(bank_row["manifest_seed_index"]),
                "manifest_seed_index": int(bank_row["manifest_seed_index"]),
                "experiment_bank_index": int(bank_row["experiment_bank_index"]),
                "bank_class_position": int(bank_row["bank_class_position"]),
                "prepared_row_index": int(bank_row["prepared_row_index"]),
                "real_row_id": row_id,
                "seed_text": prepared_texts[int(bank_row["prepared_row_index"])],
                "label_int": int(bank_row["label_int"]),
                "label": int(bank_row["label_int"]),
                "inherited_label": inherited_label,
                "candidate_index": None,
                "global_generation_index": None,
                "generation_seed": generation_seed,
                "request_key": request_key,
                "prompt_id": PROMPT_ID,
                "stage_a_candidate_pool_excluded": True,
                "stage_a_exclusion_scope": "pilot_output_only",
                "stage_a_excluded_real_row_id": None,
                "stage_a_excluded_candidate_indices": "[]",
                "stage_a_excluded_candidate_count": 0,
            }
        )
    if len(selections) != REQUESTS_PER_CELL:
        raise AssertionError(f"{cell_id}: selection count mismatch")
    if len({row["real_row_id"] for row in selections}) != REQUESTS_PER_CELL:
        raise AssertionError(f"{cell_id}: selected seed rows are not unique")
    if len({row["generation_seed"] for row in selections}) != REQUESTS_PER_CELL:
        raise AssertionError(f"{cell_id}: selected generation seeds are not unique")
    return selections, {
        str(label): int(count) for label, count in sorted(quota.items())
    }


def make_prompt_template(language: str) -> str:
    if language not in PROMPT_TEMPLATES:
        raise ValueError(f"unsupported prompt language: {language}")
    return PROMPT_TEMPLATES[language]


def render_prompt(template: str, inherited_label: str, seed_text: str) -> str:
    return template.replace("{label}", inherited_label).replace(
        "{seed_text_clipped_to_20_words}", clip_words(seed_text, 20)
    )


def write_prompt_templates(output_root: Path) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for cell_id in TARGET_CELLS:
        language = CELL_LANGUAGE[cell_id]
        template = make_prompt_template(language)
        path = output_root / "prompts" / f"{cell_id}_{PROMPT_ID}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(template, encoding="utf-8", newline="\n")
        metadata[cell_id] = {
            "cell_id": cell_id,
            "language": language,
            "prompt_id": PROMPT_ID,
            "path": relative_path(output_root, path),
            "sha256": sha256_bytes(template.encode("utf-8")),
            "template": template,
        }
    return metadata


def get_revision(value: Any) -> str | None:
    candidates: list[Any] = [
        value,
        getattr(value, "config", None),
        getattr(value, "tokenizer", None),
    ]
    for item in candidates:
        if item is None:
            continue
        for attribute in ("_commit_hash", "commit_hash"):
            candidate = getattr(item, attribute, None)
            if isinstance(candidate, str) and candidate.strip():
                candidate = candidate.strip()
                if candidate.lower() in {"main", "master", "unknown"}:
                    continue
                if re.fullmatch(r"[0-9a-fA-F]{7,64}", candidate):
                    return candidate
        init_kwargs = getattr(item, "init_kwargs", None)
        if isinstance(init_kwargs, Mapping):
            candidate = init_kwargs.get("_commit_hash")
            if isinstance(candidate, str) and candidate.strip():
                candidate = candidate.strip()
                if re.fullmatch(r"[0-9a-fA-F]{7,64}", candidate):
                    return candidate
    return None


def detect_repetition_collapse(text: str) -> str | None:
    if re.search(r'(?:^|[\s"])([^\s"]+)(?:\s+\1){7}(?=\s|["}]|$)', text):
        return "repeated_token"
    for repeated in re.finditer(r"(.{1,4})\1{7,}", text):
        if len(repeated.group(0)) >= 32:
            return "repeated_segment"
    return None


def has_surface_corruption(text: str) -> bool:
    if "\ufffd" in text or "\x00" in text:
        return True
    if any(unicodedata.category(char) == "Cc" and char not in "\t\n\r" for char in text):
        return True
    if re.search(r"([/\\]{2})[^\W_]{1,4}\1", text, flags=re.UNICODE):
        return True
    return bool(
        re.search(
            r"(?<![\w/\\])([/\\])[^\W_]{1,4}\1(?![\w/\\])",
            text,
            flags=re.UNICODE,
        )
    )


def validate_final_json(
    final_response: str, seed_text: str, inherited_label: str
) -> tuple[dict[str, Any] | None, str | None, dict[str, Any]]:
    metadata: dict[str, Any] = {
        "json_schema_valid": False,
        "generated_text": None,
        "generated_label_raw": None,
    }
    if not isinstance(final_response, str) or not final_response.strip():
        return None, "empty_response", metadata
    repetition_reason = detect_repetition_collapse(final_response)
    if repetition_reason is not None:
        return None, repetition_reason, metadata
    try:
        value = json.loads(final_response.strip())
    except Exception:
        return None, "json_parse_failure", metadata
    if not isinstance(value, dict):
        return None, "json_not_object", metadata
    if set(value.keys()) != {"text", "label"}:
        return None, "unexpected_fields", metadata
    if "text" not in value:
        return None, "missing_text", metadata
    if "label" not in value:
        return None, "missing_label", metadata
    if not isinstance(value["text"], str):
        return None, "text_not_string", metadata
    if not isinstance(value["label"], str):
        return None, "label_not_string", metadata

    generated_text = value["text"].strip()
    generated_label_raw = value["label"]
    metadata.update(
        {
            "json_schema_valid": True,
            "generated_text": generated_text,
            "generated_label_raw": generated_label_raw,
        }
    )
    if not generated_text:
        return None, "empty_generated_text", metadata
    if generated_label_raw not in ALLOWED_LABELS:
        return None, "invalid_label_value", metadata
    if generated_label_raw != inherited_label:
        return None, "label_mismatch", metadata
    if normalize_for_copy(seed_text) == normalize_for_copy(generated_text):
        return None, "exact_copy", metadata
    if len(generated_text.split()) > 30:
        return None, "word_limit_exceeded", metadata
    if has_surface_corruption(generated_text):
        return None, "surface_corruption", metadata
    return (
        {
            "text": generated_text,
            "label": generated_label_raw,
        },
        None,
        metadata,
    )


class GemmaGenerator:
    """LM Studio OpenAI-compatible endpoint backend for Gemma generation."""

    def __init__(
        self,
        api_url: str,
        model: str,
        api_key: str | None,
        timeout_seconds: int,
    ) -> None:
        self.api_url = api_url
        self.model_id = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.generation_settings = STANDARD_LMSTUDIO_GENERATION_SETTINGS
        # The OpenAI-compatible endpoint does not expose exact Hugging Face
        # model/tokenizer revisions, so the audit fields remain explicitly missing.
        self.model_revision = None
        self.tokenizer_revision = None
        self.observed_parameter_dtypes: list[str] = []

    def generate(self, prompt: str, generation_seed: int) -> tuple[str, str]:
        final_response = request_lmstudio_chat_content(
            api_url=self.api_url,
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            settings=self.generation_settings,
            seed=int(generation_seed),
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "binary_v2_response",
                    "strict": True,
                    "schema": BINARY_OUTPUT_SCHEMA,
                },
            },
            api_key=self.api_key,
            timeout_seconds=self.timeout_seconds,
        )
        return final_response, final_response


def request_base(
    request: Mapping[str, Any],
    prompt_meta: Mapping[str, Any],
    model_id: str,
    model_revision: str | None,
    tokenizer_revision: str | None,
) -> dict[str, Any]:
    return {
        "cell_id": request["cell_id"],
        "language": request["language"],
        "seed_index": request["seed_index"],
        "pilot_request_index": request["pilot_request_index"],
        "prepared_row_index": request["prepared_row_index"],
        "real_row_id": request["real_row_id"],
        "manifest_seed_index": request["manifest_seed_index"],
        "seed_text": request["seed_text"],
        "inherited_label": request["inherited_label"],
        "generation_seed": request["generation_seed"],
        "prompt_id": PROMPT_ID,
        "prompt_sha256": prompt_meta["sha256"],
        "model_id": model_id,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "request_key": request["request_key"],
    }


def generate_one_request(
    order_index: int,
    backend: GemmaGenerator,
    request: Mapping[str, Any],
    prompt_meta: Mapping[str, Any],
) -> dict[str, Any]:
    base = request_base(
        request,
        prompt_meta,
        backend.model_id,
        backend.model_revision,
        backend.tokenizer_revision,
    )
    attempts: list[dict[str, Any]] = []
    valid_record: dict[str, Any] | None = None
    final_failure_reason: str | None = None
    first_attempt_json_schema_valid = False
    first_attempt_success = False
    base_prompt = render_prompt(
        prompt_meta["template"], request["inherited_label"], request["seed_text"]
    )
    retry_reason: str | None = None

    for attempt_number in range(1, MAX_ATTEMPTS + 1):
        current_generation_seed = attempt_generation_seed(
            int(request["generation_seed"]), attempt_number
        )
        prompt_variant = "base"
        attempt_prompt = base_prompt
        if attempt_number > 1:
            retry_instruction = STRICT_JSON_WRAPPER
            prompt_variant = f"strict_json_retry_{attempt_number - 1}"
            if retry_reason == "label_mismatch":
                retry_instruction = LABEL_MISMATCH_RETRY
                prompt_variant = f"label_mismatch_retry_{attempt_number - 1}"
            elif retry_reason in {"repeated_token", "repeated_segment"}:
                retry_instruction = REPETITION_RETRY
                prompt_variant = f"repetition_retry_{attempt_number - 1}"
            elif retry_reason == "word_limit_exceeded":
                retry_instruction = WORD_LIMIT_RETRY
                prompt_variant = f"word_limit_retry_{attempt_number - 1}"
            elif retry_reason == "surface_corruption":
                retry_instruction = SURFACE_CORRUPTION_RETRY
                prompt_variant = f"surface_corruption_retry_{attempt_number - 1}"
            attempt_prompt = base_prompt + "\n\n" + retry_instruction
        attempt: dict[str, Any] = {
            "attempt": attempt_number,
            "prompt_variant": prompt_variant,
            "prompt": attempt_prompt,
            "generation_seed": request["generation_seed"],
            "attempt_generation_seed": current_generation_seed,
            "endpoint": backend.api_url,
            "model_id": backend.model_id,
            "raw_response": None,
            "final_response_text": None,
            "validation_reason": None,
            "json_schema_valid": False,
        }
        try:
            raw_response, final_response = backend.generate(
                attempt_prompt, current_generation_seed
            )
            parsed, reason, validation_metadata = validate_final_json(
                final_response,
                request["seed_text"],
                request["inherited_label"],
            )
            attempt.update(
                {
                    "raw_response": raw_response,
                    "final_response_text": final_response,
                    "validation_reason": reason,
                    "json_schema_valid": validation_metadata["json_schema_valid"],
                    "generated_text": validation_metadata["generated_text"],
                    "generated_label_raw": validation_metadata["generated_label_raw"],
                }
            )
            if attempt_number == 1:
                first_attempt_json_schema_valid = bool(
                    validation_metadata["json_schema_valid"]
                )
            attempts.append(attempt)
            if reason is None and parsed is not None:
                valid_record = {
                    **base,
                    "generated_text": parsed["text"],
                    "generated_label_raw": parsed["label"],
                    "generated_label_agreement": True,
                    "attempt_count": attempt_number,
                    "raw_response": raw_response,
                }
                first_attempt_success = attempt_number == 1
                final_failure_reason = None
                break
            if reason not in INVALID_REASONS:
                final_failure_reason = reason or "format_validation_error"
                break
            final_failure_reason = reason
            retry_reason = reason
        except LmStudioRequestError as exc:
            attempt.update(
                {
                    "raw_response": exc.raw_response,
                    "validation_reason": "backend_error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            attempts.append(attempt)
            final_failure_reason = "backend_error"
            break
        except Exception as exc:
            attempt.update(
                {
                    "validation_reason": "backend_error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            attempts.append(attempt)
            final_failure_reason = "backend_error"
            break

    raw_record = {
        **base,
        "attempt_count": len(attempts),
        "attempts": attempts,
        "outcome": "valid" if valid_record is not None else "failed",
        "failure_reason": final_failure_reason,
    }
    failure = None
    if valid_record is None:
        failure = {
            **base,
            "attempt_count": len(attempts),
            "failure_reason": final_failure_reason or "failed_without_reason",
            "attempt_errors": [
                {
                    "attempt": attempt.get("attempt"),
                    "prompt_variant": attempt.get("prompt_variant"),
                    "validation_reason": attempt.get("validation_reason"),
                    "error": attempt.get("error"),
                    "raw_response": attempt.get("raw_response"),
                    "final_response_text": attempt.get("final_response_text"),
                }
                for attempt in attempts
            ],
        }
    return {
        "order_index": order_index,
        "request": dict(request),
        "attempts": attempts,
        "valid_record": valid_record,
        "failure": failure,
        "raw_record": raw_record,
        "failure_reason": final_failure_reason,
        "first_attempt_json_schema_valid": first_attempt_json_schema_valid,
        "first_attempt_success": first_attempt_success,
    }


def run_generation(
    output_root: Path,
    selections: Mapping[str, Sequence[Mapping[str, Any]]],
    prompt_meta: Mapping[str, Mapping[str, Any]],
    endpoints: Sequence[EndpointConfig],
    api_key: str | None,
    timeout_seconds: int,
) -> dict[str, Any]:
    backends = [
        GemmaGenerator(api_url, model, api_key, timeout_seconds)
        for api_url, model, _ in endpoints
    ]
    executors = [
        ThreadPoolExecutor(max_workers=max_concurrency)
        for _, _, max_concurrency in endpoints
    ]
    endpoint_slots = [
        endpoint_index
        for endpoint_index, (_, _, max_concurrency) in enumerate(endpoints)
        for _ in range(max_concurrency)
    ]
    endpoint_summary = ", ".join(
        f"{api_url} model={model} x{max_concurrency}"
        for api_url, model, max_concurrency in endpoints
    )
    total_requests = sum(len(selections[cell_id]) for cell_id in TARGET_CELLS)
    progress_log(
        f"generation start | total={total_requests} | endpoints={endpoint_summary}"
    )
    futures = []
    order_index = 0
    started_at = perf_counter()
    try:
        for cell_id in TARGET_CELLS:
            for request in selections[cell_id]:
                endpoint_index = endpoint_slots[order_index % len(endpoint_slots)]
                futures.append(
                    executors[endpoint_index].submit(
                        generate_one_request,
                        order_index,
                        backends[endpoint_index],
                        request,
                        prompt_meta[cell_id],
                    )
                )
                order_index += 1
        results = []
        valid_count = 0
        failed_count = 0
        for completed_count, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            is_valid = result["valid_record"] is not None
            valid_count += int(is_valid)
            failed_count += int(not is_valid)
            elapsed = perf_counter() - started_at
            rate = completed_count / elapsed if elapsed > 0 else 0.0
            remaining = total_requests - completed_count
            eta_seconds = remaining / rate if rate > 0 else 0.0
            request = result["request"]
            attempts = result["attempts"]
            endpoint = attempts[0].get("endpoint") if attempts else "unknown"
            progress_log(
                "generation progress | "
                f"completed={completed_count}/{total_requests} | "
                f"valid={valid_count} | failed={failed_count} | "
                f"cell={request['cell_id']} | "
                f"pilot_request_index={request['pilot_request_index']} | "
                f"endpoint={endpoint} | attempts={len(attempts)} | "
                f"elapsed_seconds={elapsed:.1f} | rate={rate:.3f}/s | "
                f"eta_seconds={eta_seconds:.1f}"
            )
    finally:
        for executor in executors:
            executor.shutdown(wait=True)

    progress_log(
        "generation complete | "
        f"total={total_requests} | valid={valid_count} | failed={failed_count} | "
        f"elapsed_seconds={perf_counter() - started_at:.1f}"
    )

    valid_by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    failures_by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    raw_by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    outcome_by_request: dict[str, dict[str, Any]] = {}
    for result in sorted(results, key=lambda item: item["order_index"]):
        request = result["request"]
        cell_id = request["cell_id"]
        raw_by_cell[cell_id].append(result["raw_record"])
        if result["valid_record"] is not None:
            valid_by_cell[cell_id].append(result["valid_record"])
        else:
            failures_by_cell[cell_id].append(result["failure"])
        outcome_by_request[request["request_key"]] = {
            "request": request,
            "attempts": result["attempts"],
            "valid_record": result["valid_record"],
            "failure_reason": result["failure_reason"],
            "first_attempt_json_schema_valid": result[
                "first_attempt_json_schema_valid"
            ],
            "first_attempt_success": result["first_attempt_success"],
            "backend_initialization_error": None,
        }

    for cell_id in TARGET_CELLS:
        write_jsonl(
            output_root / "raw_outputs" / f"{cell_id}.jsonl", raw_by_cell[cell_id]
        )
        write_jsonl(
            output_root / "valid_outputs" / f"{cell_id}.jsonl", valid_by_cell[cell_id]
        )
        write_jsonl(
            output_root / "failures" / f"{cell_id}.jsonl", failures_by_cell[cell_id]
        )

    return {
        "status": "completed",
        "backend_initialization_error": None,
        "model_revision": None,
        "tokenizer_revision": None,
        "revision_status": {"model": "missing", "tokenizer": "missing"},
        "observed_parameter_dtypes": [],
        "endpoints": [
            {
                "api_url": api_url,
                "model": model,
                "max_concurrency": max_concurrency,
            }
            for api_url, model, max_concurrency in endpoints
        ],
        "valid_by_cell": dict(valid_by_cell),
        "failures_by_cell": dict(failures_by_cell),
        "raw_by_cell": dict(raw_by_cell),
        "outcome_by_request": outcome_by_request,
    }


def embedding_source_entries(
    selections: Mapping[str, Sequence[Mapping[str, Any]]],
    valid_by_cell: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    valid_by_request = {
        cell_id: {str(row["request_key"]): row for row in rows}
        for cell_id, rows in valid_by_cell.items()
    }
    entries: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell_id in TARGET_CELLS:
        for request in selections[cell_id]:
            request_key = str(request["request_key"])
            entries[cell_id].append(
                {
                    "cell_id": cell_id,
                    "language": request["language"],
                    "seed_index": request["seed_index"],
                    "request_key": request_key,
                    "source_record_key": f"{request_key}:seed",
                    "record_type": "seed",
                    "text": request["seed_text"],
                }
            )
            synthetic = valid_by_request.get(cell_id, {}).get(request_key)
            if synthetic is not None:
                entries[cell_id].append(
                    {
                        "cell_id": cell_id,
                        "language": request["language"],
                        "seed_index": request["seed_index"],
                        "request_key": request_key,
                        "source_record_key": f"{request_key}:synthetic",
                        "record_type": "synthetic",
                        "text": synthetic["generated_text"],
                    }
                )
    return dict(entries)


def model_context_limit(model: Any, tokenizer: Any) -> int | None:
    config = getattr(model, "config", None)
    for key in ("max_position_embeddings", "max_sequence_length", "max_seq_len"):
        value = getattr(config, key, None)
        if isinstance(value, int) and value > 0 and value < 10**8:
            return value
    value = getattr(tokenizer, "model_max_length", None)
    if isinstance(value, int) and value > 0 and value < 10**8:
        return value
    return None


def token_count_without_truncation(tokenizer: Any, text: str) -> int:
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )
    ids = encoded.get("input_ids")
    if ids is None:
        raise RuntimeError("tokenizer did not return input_ids")
    return len(ids)


def embedding_input_with_separator(tokenizer: Any, text: str) -> str:
    separator = tokenizer.sep_token or tokenizer.eos_token
    if not isinstance(separator, str) or not separator:
        raise RuntimeError("embedding tokenizer has no SEP or EOS token")
    return text if text.endswith(separator) else text + separator


def numeric_stats(
    values: Sequence[float], include_median: bool = False
) -> dict[str, float | None]:
    if not values:
        result: dict[str, float | None] = {"min": None, "mean": None, "max": None}
        if include_median:
            result["median"] = None
        return result
    result = {
        "min": float(min(values)),
        "mean": float(statistics.fmean(values)),
        "max": float(max(values)),
    }
    if include_median:
        result["median"] = float(statistics.median(values))
    return result


def request_embedding_batches(
    stage_name: str,
    batches: Sequence[Sequence[str]],
    endpoints: Sequence[EndpointConfig],
    api_key: str | None,
    timeout_seconds: int,
) -> dict[int, tuple[list[list[float]] | None, str | None]]:
    executors = [
        ThreadPoolExecutor(max_workers=max_concurrency)
        for _, _, max_concurrency in endpoints
    ]
    endpoint_slots = [
        endpoint_index
        for endpoint_index, (_, _, max_concurrency) in enumerate(endpoints)
        for _ in range(max_concurrency)
    ]
    futures: dict[Any, tuple[int, str]] = {}
    results: dict[int, tuple[list[list[float]] | None, str | None]] = {}
    try:
        for batch_index, texts in enumerate(batches):
            endpoint_index = endpoint_slots[batch_index % len(endpoint_slots)]
            api_url, model, _ = endpoints[endpoint_index]
            future = executors[endpoint_index].submit(
                request_lmstudio_embeddings,
                api_url=api_url,
                model=model,
                inputs=texts,
                api_key=api_key,
                timeout_seconds=timeout_seconds,
            )
            futures[future] = (batch_index, api_url)
        for completed, future in enumerate(as_completed(futures), start=1):
            batch_index, api_url = futures[future]
            try:
                results[batch_index] = (future.result(), None)
                status = "ok"
            except Exception as exc:
                results[batch_index] = (None, f"{type(exc).__name__}: {exc}")
                status = "failed"
            progress_log(
                f"{stage_name} endpoint progress | batches={completed}/{len(batches)} | "
                f"batch_index={batch_index} | endpoint={api_url} | status={status}"
            )
    finally:
        for executor in executors:
            executor.shutdown(wait=True)
    return results


def normalize_endpoint_vectors(
    vectors: Sequence[Sequence[float]], expected_dimension: int
) -> Any:
    import numpy as np

    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != expected_dimension:
        raise RuntimeError(
            f"embedding shape {tuple(array.shape)} != expected (*,{expected_dimension})"
        )
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise RuntimeError("embedding response contains a zero-norm vector")
    return (array / norms).astype(np.float32)


def run_qwen_embeddings_lmstudio(
    output_root: Path,
    selections: Mapping[str, Sequence[Mapping[str, Any]]],
    valid_by_cell: Mapping[str, Sequence[Mapping[str, Any]]],
    batch_size: int,
    endpoints: Sequence[EndpointConfig],
    api_key: str | None,
    timeout_seconds: int,
) -> dict[str, Any]:
    import numpy as np
    from transformers import AutoTokenizer

    folder = output_root / "embeddings_qwen3"
    folder.mkdir(parents=True, exist_ok=True)
    entries_by_cell = embedding_source_entries(selections, valid_by_cell)
    tokenizer = AutoTokenizer.from_pretrained(
        QWEN_TOKENIZER_ID,
        local_files_only=False,
        padding_side="left",
    )
    tokenizer_revision = get_revision(tokenizer)
    context_limit = model_context_limit(None, tokenizer)
    instruction_prefix = f"Instruct: {QWEN_INSTRUCTION}\nQuery:"
    result: dict[str, Any] = {
        "model_id": QWEN_MODEL_ID,
        "backend": "lm_studio_openai_compatible_embeddings",
        "endpoints": [
            {"api_url": url, "model": model, "max_concurrency": concurrency}
            for url, model, concurrency in endpoints
        ],
        "status": "completed",
        "model_revision": None,
        "tokenizer_revision": tokenizer_revision,
        "revision_status": {
            "model": "missing",
            "tokenizer": "present" if tokenizer_revision is not None else "missing",
        },
        "dtype": "float32",
        "quantization": "server_managed_unverified",
        "embedding_dimension_expected": 4096,
        "instruction": QWEN_INSTRUCTION,
        "max_length": None,
        "truncation": False,
        "cells": {},
    }
    for cell_id in TARGET_CELLS:
        entries = entries_by_cell[cell_id]
        prepared: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        for entry in entries:
            input_text = embedding_input_with_separator(
                tokenizer, instruction_prefix + str(entry["text"])
            )
            try:
                token_count = token_count_without_truncation(tokenizer, input_text)
            except Exception as exc:
                errors.append(
                    {
                        **entry,
                        "error_type": "tokenizer_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            if context_limit is not None and token_count > context_limit:
                errors.append(
                    {
                        **entry,
                        "error_type": "context_overflow",
                        "token_count": token_count,
                        "context_limit": context_limit,
                        "error": f"token_count={token_count} exceeds context_limit={context_limit}",
                    }
                )
                continue
            prepared.append(
                {**entry, "token_count": token_count, "embedding_input": input_text}
            )
        entry_batches = [
            prepared[start : start + batch_size]
            for start in range(0, len(prepared), batch_size)
        ]
        text_batches = [
            [str(entry["embedding_input"]) for entry in batch]
            for batch in entry_batches
        ]
        progress_log(
            f"qwen3 embedding cell start | cell={cell_id} | records={len(prepared)}"
        )
        batch_results = request_embedding_batches(
            "qwen3 embedding", text_batches, endpoints, api_key, timeout_seconds
        )
        vectors: list[Any] = []
        manifest_rows: list[dict[str, Any]] = []
        for batch_index, batch_entries in enumerate(entry_batches):
            raw_vectors, batch_error = batch_results[batch_index]
            if batch_error is not None or raw_vectors is None:
                for entry in batch_entries:
                    errors.append(
                        {
                            **entry,
                            "error_type": "endpoint_embedding_error",
                            "error": batch_error,
                        }
                    )
                continue
            try:
                normalized = normalize_endpoint_vectors(raw_vectors, 4096)
            except Exception as exc:
                for entry in batch_entries:
                    errors.append(
                        {
                            **entry,
                            "error_type": "embedding_validation_error",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                continue
            for entry, vector in zip(batch_entries, normalized, strict=True):
                row_index = len(vectors)
                vectors.append(vector)
                manifest_rows.append(
                    {
                        "embedding_row_index": row_index,
                        **{
                            key: entry[key]
                            for key in (
                                "cell_id",
                                "language",
                                "seed_index",
                                "request_key",
                                "source_record_key",
                                "record_type",
                            )
                        },
                        "text_sha256": sha256_bytes(str(entry["text"]).encode("utf-8")),
                        "text_length_chars": len(str(entry["text"])),
                        "token_count": entry["token_count"],
                        "embedding_dimension": 4096,
                        "status": "ok",
                    }
                )
        array = (
            np.asarray(vectors, dtype=np.float32)
            if vectors
            else np.empty((0, 4096), dtype=np.float32)
        )
        np.save(folder / f"{cell_id}.npy", array)
        write_csv(
            folder / f"{cell_id}_manifest.csv",
            manifest_rows,
            [
                "embedding_row_index",
                "cell_id",
                "language",
                "seed_index",
                "request_key",
                "source_record_key",
                "record_type",
                "text_sha256",
                "text_length_chars",
                "token_count",
                "embedding_dimension",
                "status",
            ],
        )
        write_jsonl(folder / f"{cell_id}_errors.jsonl", errors)
        norms = (
            np.linalg.norm(array, axis=1).astype(float).tolist() if len(array) else []
        )
        result["cells"][cell_id] = {
            "embedding_shape": list(array.shape),
            "l2_norms": norms,
            "context_limit": context_limit,
            "instruction": QWEN_INSTRUCTION,
            "max_length": None,
            "truncation": False,
            "errors": len(errors),
        }
        progress_log(
            f"qwen3 embedding cell complete | cell={cell_id} | "
            f"shape={list(array.shape)} | errors={len(errors)}"
        )
    write_json(folder / "metadata.json", result)
    return result


def run_bge_embeddings_lmstudio(
    output_root: Path,
    selections: Mapping[str, Sequence[Mapping[str, Any]]],
    valid_by_cell: Mapping[str, Sequence[Mapping[str, Any]]],
    batch_size: int,
    endpoints: Sequence[EndpointConfig],
    api_key: str | None,
    timeout_seconds: int,
) -> dict[str, Any]:
    import numpy as np
    from transformers import AutoTokenizer

    folder = output_root / "embeddings_bge_m3"
    folder.mkdir(parents=True, exist_ok=True)
    entries_by_cell = embedding_source_entries(selections, valid_by_cell)
    tokenizer = AutoTokenizer.from_pretrained(BGE_TOKENIZER_ID, local_files_only=False)
    tokenizer.truncation_side = "right"
    tokenizer_revision = get_revision(tokenizer)
    result: dict[str, Any] = {
        "model_id": BGE_MODEL_ID,
        "backend": "lm_studio_openai_compatible_embeddings",
        "endpoints": [
            {"api_url": url, "model": model, "max_concurrency": concurrency}
            for url, model, concurrency in endpoints
        ],
        "status": "completed",
        "model_revision": None,
        "tokenizer_revision": tokenizer_revision,
        "revision_status": {
            "model": "missing",
            "tokenizer": "present" if tokenizer_revision is not None else "missing",
        },
        "dtype": "float32",
        "quantization": "server_managed_unverified",
        "embedding_dimension_expected": 1024,
        "max_length": 512,
        "truncation": "right",
        "cells": {},
        "cosine_rows": [],
    }
    for cell_id in TARGET_CELLS:
        entries = entries_by_cell[cell_id]
        prepared: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        for entry in entries:
            text = str(entry["text"])
            try:
                original_count = token_count_without_truncation(tokenizer, text)
                encoded = tokenizer(
                    text,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=512,
                    padding=False,
                )
                input_ids = encoded.get("input_ids")
                if not isinstance(input_ids, list):
                    raise RuntimeError("tokenizer did not return input_ids")
                truncated = original_count > 512
                embedding_text = (
                    tokenizer.decode(
                        input_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    if truncated
                    else text
                )
                embedding_input = embedding_input_with_separator(
                    tokenizer, embedding_text
                )
            except Exception as exc:
                errors.append(
                    {
                        **entry,
                        "error_type": "tokenizer_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            prepared.append(
                {
                    **entry,
                    "original_token_count": original_count,
                    "token_count": len(input_ids),
                    "truncated": truncated,
                    "embedding_input": embedding_input,
                }
            )
        entry_batches = [
            prepared[start : start + batch_size]
            for start in range(0, len(prepared), batch_size)
        ]
        text_batches = [
            [str(entry["embedding_input"]) for entry in batch]
            for batch in entry_batches
        ]
        progress_log(
            f"bge-m3 embedding cell start | cell={cell_id} | records={len(prepared)}"
        )
        batch_results = request_embedding_batches(
            "bge-m3 embedding", text_batches, endpoints, api_key, timeout_seconds
        )
        vectors: list[Any] = []
        manifest_rows: list[dict[str, Any]] = []
        for batch_index, batch_entries in enumerate(entry_batches):
            raw_vectors, batch_error = batch_results[batch_index]
            if batch_error is not None or raw_vectors is None:
                for entry in batch_entries:
                    errors.append(
                        {
                            **entry,
                            "error_type": "endpoint_embedding_error",
                            "error": batch_error,
                        }
                    )
                continue
            try:
                normalized = normalize_endpoint_vectors(raw_vectors, 1024)
            except Exception as exc:
                for entry in batch_entries:
                    errors.append(
                        {
                            **entry,
                            "error_type": "embedding_validation_error",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                continue
            for entry, vector in zip(batch_entries, normalized, strict=True):
                row_index = len(vectors)
                vectors.append(vector)
                manifest_rows.append(
                    {
                        "embedding_row_index": row_index,
                        **{
                            key: entry[key]
                            for key in (
                                "cell_id",
                                "language",
                                "seed_index",
                                "request_key",
                                "source_record_key",
                                "record_type",
                            )
                        },
                        "text_sha256": sha256_bytes(str(entry["text"]).encode("utf-8")),
                        "text_length_chars": len(str(entry["text"])),
                        "original_token_count": entry["original_token_count"],
                        "token_count": entry["token_count"],
                        "truncated": entry["truncated"],
                        "embedding_dimension": 1024,
                        "status": "ok",
                    }
                )
        array = (
            np.asarray(vectors, dtype=np.float32)
            if vectors
            else np.empty((0, 1024), dtype=np.float32)
        )
        np.save(folder / f"{cell_id}.npy", array)
        write_csv(
            folder / f"{cell_id}_manifest.csv",
            manifest_rows,
            [
                "embedding_row_index",
                "cell_id",
                "language",
                "seed_index",
                "request_key",
                "source_record_key",
                "record_type",
                "text_sha256",
                "text_length_chars",
                "original_token_count",
                "token_count",
                "truncated",
                "embedding_dimension",
                "status",
            ],
        )
        write_jsonl(folder / f"{cell_id}_errors.jsonl", errors)
        index_by_key = {
            row["source_record_key"]: int(row["embedding_row_index"])
            for row in manifest_rows
        }
        cosine_rows: list[dict[str, Any]] = []
        for request in selections[cell_id]:
            request_key = str(request["request_key"])
            seed_key = f"{request_key}:seed"
            synthetic_key = f"{request_key}:synthetic"
            if seed_key not in index_by_key or synthetic_key not in index_by_key:
                continue
            seed_index = index_by_key[seed_key]
            synthetic_index = index_by_key[synthetic_key]
            cosine_rows.append(
                {
                    "cell_id": cell_id,
                    "language": request["language"],
                    "seed_index": request["seed_index"],
                    "request_key": request_key,
                    "seed_embedding_row_index": seed_index,
                    "synthetic_embedding_row_index": synthetic_index,
                    "cosine_similarity": float(
                        np.dot(array[seed_index], array[synthetic_index])
                    ),
                }
            )
        norms = (
            np.linalg.norm(array, axis=1).astype(float).tolist() if len(array) else []
        )
        result["cells"][cell_id] = {
            "embedding_shape": list(array.shape),
            "l2_norms": norms,
            "max_length": 512,
            "truncation": "right",
            "pooling": "server_embedding_endpoint",
            "errors": len(errors),
            "cosines": [row["cosine_similarity"] for row in cosine_rows],
        }
        result["cosine_rows"].extend(cosine_rows)
        progress_log(
            f"bge-m3 embedding cell complete | cell={cell_id} | "
            f"shape={list(array.shape)} | errors={len(errors)} | "
            f"cosines={len(cosine_rows)}"
        )
    write_csv(
        folder / "cosine_similarity.csv", result["cosine_rows"], BGE_COSINE_FIELDS
    )
    write_json(folder / "metadata.json", result)
    return result


def package_versions() -> dict[str, str | None]:
    packages = {
        "torch": "torch",
        "transformers": "transformers",
        "accelerate": "accelerate",
        "safetensors": "safetensors",
        "huggingface_hub": "huggingface-hub",
        "numpy": "numpy",
        "sentence_transformers": "sentence-transformers",
        "FlagEmbedding": "FlagEmbedding",
    }
    result: dict[str, str | None] = {}
    for key, distribution in packages.items():
        try:
            result[key] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            result[key] = None
    return result


def runtime_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions(),
        "cuda_version": None,
        "gpu": {"cuda_available": False, "device_count": 0, "devices": []},
    }
    try:
        import torch

        info["pytorch_version"] = torch.__version__
        info["cuda_version"] = torch.version.cuda
        info["gpu"] = {
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
            "devices": [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ],
        }
    except Exception as exc:
        info["pytorch_error"] = f"{type(exc).__name__}: {exc}"
    info["transformers_version"] = info["packages"].get("transformers")
    return info


def outcome_stats(
    selections: Mapping[str, Sequence[Mapping[str, Any]]],
    generation: Mapping[str, Any],
    cell_id: str,
) -> dict[str, Any]:
    outcomes = [
        generation["outcome_by_request"].get(request["request_key"])
        for request in selections[cell_id]
        if request["request_key"] in generation["outcome_by_request"]
    ]
    valid = [
        outcome["valid_record"]
        for outcome in outcomes
        if outcome["valid_record"] is not None
    ]
    failures = [outcome for outcome in outcomes if outcome["valid_record"] is None]
    attempt_distribution = Counter(
        str(len(outcome["attempts"])) for outcome in outcomes
    )
    json_schema_success_count = sum(
        1
        for outcome in outcomes
        if outcome["attempts"]
        and bool(outcome["attempts"][-1].get("json_schema_valid"))
    )
    first_json_schema_success_count = sum(
        1 for outcome in outcomes if outcome["first_attempt_json_schema_valid"]
    )
    first_success_count = sum(
        1 for outcome in outcomes if outcome["first_attempt_success"]
    )
    exact_copy_count = sum(
        1 for outcome in failures if outcome["failure_reason"] == "exact_copy"
    )
    exact_copy_attempt_count = sum(
        1
        for outcome in outcomes
        for attempt in outcome["attempts"]
        if attempt.get("validation_reason") == "exact_copy"
    )
    validation_reason_attempt_counts = Counter(
        str(attempt["validation_reason"])
        for outcome in outcomes
        for attempt in outcome["attempts"]
        if attempt.get("validation_reason") in INVALID_REASONS
    )
    for reason in sorted(INVALID_REASONS):
        validation_reason_attempt_counts.setdefault(reason, 0)
    agreement_count = sum(1 for record in valid if record["generated_label_agreement"])
    requested = len(outcomes)
    return {
        "requested": requested,
        "valid": len(valid),
        "failed": len(failures),
        "json_schema_success_count": json_schema_success_count,
        "json_success_rate": json_schema_success_count / requested
        if requested
        else None,
        "first_attempt_json_schema_success_count": first_json_schema_success_count,
        "first_attempt_json_success_rate": first_json_schema_success_count / requested
        if requested
        else None,
        "first_attempt_success_count": first_success_count,
        "first_attempt_success_rate": first_success_count / requested
        if requested
        else None,
        "attempt_count_distribution": dict(
            sorted(attempt_distribution.items(), key=lambda item: int(item[0]))
        ),
        "exact_copy_count": exact_copy_count,
        "exact_copy_attempt_count": exact_copy_attempt_count,
        "validation_reason_attempt_counts": dict(
            sorted(validation_reason_attempt_counts.items())
        ),
        "label_mismatch_attempt_count": validation_reason_attempt_counts[
            "label_mismatch"
        ],
        "repeated_token_attempt_count": validation_reason_attempt_counts[
            "repeated_token"
        ],
        "repeated_segment_attempt_count": validation_reason_attempt_counts[
            "repeated_segment"
        ],
        "word_limit_exceeded_attempt_count": validation_reason_attempt_counts[
            "word_limit_exceeded"
        ],
        "surface_corruption_attempt_count": validation_reason_attempt_counts[
            "surface_corruption"
        ],
        "generated_label_agreement_count": agreement_count,
        "generated_label_agreement_rate": agreement_count / len(valid)
        if valid
        else None,
        "language_generated_count": len(valid),
    }


def embedding_cell_stats(
    embedding_result: Mapping[str, Any], cell_id: str
) -> dict[str, Any]:
    cell = embedding_result.get("cells", {}).get(cell_id, {})
    norms = [float(value) for value in cell.get("l2_norms", [])]
    cosines = [float(value) for value in cell.get("cosines", [])]
    return {
        "embedding_shape": cell.get("embedding_shape"),
        "l2_norm_stats": numeric_stats(norms),
        "cosine_stats": numeric_stats(cosines, include_median=True),
        "embedding_errors": cell.get("errors"),
    }


def aggregate_values(result: Mapping[str, Any], cell_id: str, key: str) -> list[float]:
    cell = result.get("cells", {}).get(cell_id, {})
    return [float(value) for value in cell.get(key, [])]


def build_report(
    output_root: Path,
    lock_info: Mapping[str, Any],
    prompt_meta: Mapping[str, Mapping[str, Any]],
    quota_by_cell: Mapping[str, Mapping[str, int]],
    selections: Mapping[str, Sequence[Mapping[str, Any]]],
    generation: Mapping[str, Any],
    qwen: Mapping[str, Any],
    bge: Mapping[str, Any],
    runtime: Mapping[str, Any],
    dry_run: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cells: dict[str, Any] = {}
    for cell_id in TARGET_CELLS:
        generation_stats = outcome_stats(selections, generation, cell_id)
        qwen_stats = embedding_cell_stats(qwen, cell_id)
        bge_stats = embedding_cell_stats(bge, cell_id)
        cells[cell_id] = {
            "cell_id": cell_id,
            "language": CELL_LANGUAGE[cell_id],
            "selection_quota": quota_by_cell[cell_id],
            "selection_count": len(selections[cell_id]),
            **generation_stats,
            "qwen3": qwen_stats,
            "bge_m3": bge_stats,
            "prompt_sha256": prompt_meta[cell_id]["sha256"],
        }

    all_generation_stats = {
        key: None
        for key in (
            "requested",
            "valid",
            "failed",
            "json_schema_success_count",
            "first_attempt_json_schema_success_count",
            "first_attempt_success_count",
            "exact_copy_count",
            "exact_copy_attempt_count",
            "label_mismatch_attempt_count",
            "repeated_token_attempt_count",
            "repeated_segment_attempt_count",
            "word_limit_exceeded_attempt_count",
            "surface_corruption_attempt_count",
            "generated_label_agreement_count",
            "language_generated_count",
        )
    }
    all_generation_stats.update({key: 0 for key in all_generation_stats})
    all_attempts: Counter[str] = Counter()
    all_validation_reasons: Counter[str] = Counter()
    for cell in cells.values():
        for key in all_generation_stats:
            all_generation_stats[key] += int(cell[key] or 0)
        all_attempts.update(
            {str(k): int(v) for k, v in cell["attempt_count_distribution"].items()}
        )
        all_validation_reasons.update(
            {
                str(reason): int(count)
                for reason, count in cell["validation_reason_attempt_counts"].items()
            }
        )
    requested = int(all_generation_stats["requested"])
    valid = int(all_generation_stats["valid"])
    all_generation_stats["json_success_rate"] = (
        all_generation_stats["json_schema_success_count"] / requested
        if requested
        else None
    )
    all_generation_stats["first_attempt_json_success_rate"] = (
        all_generation_stats["first_attempt_json_schema_success_count"] / requested
        if requested
        else None
    )
    all_generation_stats["first_attempt_success_rate"] = (
        all_generation_stats["first_attempt_success_count"] / requested
        if requested
        else None
    )
    all_generation_stats["generated_label_agreement_rate"] = (
        all_generation_stats["generated_label_agreement_count"] / valid
        if valid
        else None
    )
    all_generation_stats["attempt_count_distribution"] = dict(
        sorted(all_attempts.items(), key=lambda item: int(item[0]))
    )
    all_generation_stats["validation_reason_attempt_counts"] = dict(
        sorted(all_validation_reasons.items())
    )

    all_qwen_norms = [
        value
        for cell_id in TARGET_CELLS
        for value in aggregate_values(qwen, cell_id, "l2_norms")
    ]
    all_bge_norms = [
        value
        for cell_id in TARGET_CELLS
        for value in aggregate_values(bge, cell_id, "l2_norms")
    ]
    all_cosines = [
        value
        for cell_id in TARGET_CELLS
        for value in aggregate_values(bge, cell_id, "cosines")
    ]
    overall = {
        "cell_id": "__all__",
        "language": "all",
        "selection_quota": {
            cell_id: quota_by_cell[cell_id] for cell_id in TARGET_CELLS
        },
        **all_generation_stats,
        "qwen3": {
            "embedding_shape": [len(all_qwen_norms), 4096] if all_qwen_norms else None,
            "l2_norm_stats": numeric_stats(all_qwen_norms),
            "cosine_stats": {"min": None, "mean": None, "median": None, "max": None},
        },
        "bge_m3": {
            "embedding_shape": [len(all_bge_norms), 1024] if all_bge_norms else None,
            "l2_norm_stats": numeric_stats(all_bge_norms),
            "cosine_stats": numeric_stats(all_cosines, include_median=True),
        },
        "prompt_sha256": None,
    }

    report = {
        "report_version": "gate2-pilot-v1",
        "generated_at": utc_now(),
        "execution_status": "selection_only" if dry_run else generation["status"],
        "requested_total": TOTAL_REQUESTS,
        "valid_total": valid,
        "failed_total": int(all_generation_stats["failed"]),
        "target_cells": list(TARGET_CELLS),
        "manifest_lock": dict(lock_info),
        "selection": {
            "requests_per_cell": REQUESTS_PER_CELL,
            "total_requests": TOTAL_REQUESTS,
            "quota_by_cell": quota_by_cell,
            "method": "experiment_bank_locked_order_plus_largest_remainder",
            "candidate_index_used": None,
            "generation_seeds_source": "sha256(cell_id + real_row_id + gate2-pilot-v1), uint32",
            "stage_a_candidate_pool_excluded": True,
            "stage_a_exclusion_scope": "pilot outputs only",
            "path": relative_path(
                output_root, output_root / "pilot_selection_manifest.csv"
            ),
        },
        "prompt": {
            "prompt_id": PROMPT_ID,
            "templates": {
                cell_id: {
                    "language": meta["language"],
                    "path": meta["path"],
                    "sha256": meta["sha256"],
                }
                for cell_id, meta in prompt_meta.items()
            },
        },
        "generation": {
            "model_id": MODEL_ID,
            "backend": "lm_studio_openai_compatible",
            "endpoints": generation["endpoints"],
            "model_revision": generation["model_revision"],
            "tokenizer_revision": generation["tokenizer_revision"],
            "revision_status": generation["revision_status"],
            "config": GENERATION_CONFIG,
            "observed_parameter_dtypes": generation["observed_parameter_dtypes"],
            "backend_initialization_error": generation["backend_initialization_error"],
            "local_files_only": False,
            "local_lm_studio_endpoint": True,
            "external_api": False,
        },
        "embeddings": {
            "qwen3": {
                key: value
                for key, value in qwen.items()
                if key not in {"cells", "cosine_rows"}
            },
            "bge_m3": {
                key: value
                for key, value in bge.items()
                if key not in {"cells", "cosine_rows"}
            },
        },
        "runtime": runtime,
        "cells": cells,
        "overall": overall,
    }

    report_rows: list[dict[str, Any]] = []
    for cell_id in TARGET_CELLS:
        cell = cells[cell_id]
        report_rows.append(flatten_report_row(cell, generation, qwen, bge, lock_info))
    report_rows.append(flatten_report_row(overall, generation, qwen, bge, lock_info))
    return report, report_rows


def flatten_report_row(
    cell: Mapping[str, Any],
    generation: Mapping[str, Any],
    qwen: Mapping[str, Any],
    bge: Mapping[str, Any],
    lock_info: Mapping[str, Any],
) -> dict[str, Any]:
    qwen_shape = cell.get("qwen3", {}).get("embedding_shape")
    bge_shape = cell.get("bge_m3", {}).get("embedding_shape")
    qwen_norm = cell.get("qwen3", {}).get("l2_norm_stats", {})
    bge_norm = cell.get("bge_m3", {}).get("l2_norm_stats", {})
    cosine = cell.get("bge_m3", {}).get("cosine_stats", {})
    row = {
        "scope": "overall" if cell["cell_id"] == "__all__" else "cell",
        "cell_id": cell["cell_id"],
        "language": cell["language"],
        "requested": cell.get("requested"),
        "valid": cell.get("valid"),
        "failed": cell.get("failed"),
        "json_success_rate": cell.get("json_success_rate"),
        "first_attempt_success_rate": cell.get("first_attempt_success_rate"),
        "first_attempt_json_success_rate": cell.get("first_attempt_json_success_rate"),
        "attempt_count_distribution": json.dumps(
            cell.get("attempt_count_distribution", {}),
            ensure_ascii=False,
            sort_keys=True,
        ),
        "exact_copy_count": cell.get("exact_copy_count"),
        "exact_copy_attempt_count": cell.get("exact_copy_attempt_count"),
        "validation_reason_attempt_counts": json.dumps(
            cell.get("validation_reason_attempt_counts", {}),
            ensure_ascii=False,
            sort_keys=True,
        ),
        "label_mismatch_attempt_count": cell.get("label_mismatch_attempt_count"),
        "repeated_token_attempt_count": cell.get("repeated_token_attempt_count"),
        "repeated_segment_attempt_count": cell.get("repeated_segment_attempt_count"),
        "word_limit_exceeded_attempt_count": cell.get(
            "word_limit_exceeded_attempt_count"
        ),
        "surface_corruption_attempt_count": cell.get(
            "surface_corruption_attempt_count"
        ),
        "generated_label_agreement_count": cell.get("generated_label_agreement_count"),
        "generated_label_agreement_rate": cell.get("generated_label_agreement_rate"),
        "language_generated_count": cell.get("language_generated_count"),
        "qwen3_embedding_shape": json.dumps(qwen_shape, ensure_ascii=False),
        "qwen3_l2_norm_min": qwen_norm.get("min"),
        "qwen3_l2_norm_mean": qwen_norm.get("mean"),
        "qwen3_l2_norm_max": qwen_norm.get("max"),
        "bge_m3_embedding_shape": json.dumps(bge_shape, ensure_ascii=False),
        "bge_m3_l2_norm_min": bge_norm.get("min"),
        "bge_m3_l2_norm_mean": bge_norm.get("mean"),
        "bge_m3_l2_norm_max": bge_norm.get("max"),
        "seed_synthetic_cosine_min": cosine.get("min"),
        "seed_synthetic_cosine_mean": cosine.get("mean"),
        "seed_synthetic_cosine_median": cosine.get("median"),
        "seed_synthetic_cosine_max": cosine.get("max"),
        "generation_model_revision": generation.get("model_revision"),
        "generation_tokenizer_revision": generation.get("tokenizer_revision"),
        "qwen3_model_revision": qwen.get("model_revision"),
        "qwen3_tokenizer_revision": qwen.get("tokenizer_revision"),
        "bge_m3_model_revision": bge.get("model_revision"),
        "bge_m3_tokenizer_revision": bge.get("tokenizer_revision"),
        "manifest_lock_verified": lock_info.get("verified"),
    }
    return row


REPORT_CSV_FIELDS = [
    "scope",
    "cell_id",
    "language",
    "requested",
    "valid",
    "failed",
    "json_success_rate",
    "first_attempt_success_rate",
    "first_attempt_json_success_rate",
    "attempt_count_distribution",
    "exact_copy_count",
    "exact_copy_attempt_count",
    "validation_reason_attempt_counts",
    "label_mismatch_attempt_count",
    "repeated_token_attempt_count",
    "repeated_segment_attempt_count",
    "word_limit_exceeded_attempt_count",
    "surface_corruption_attempt_count",
    "generated_label_agreement_count",
    "generated_label_agreement_rate",
    "language_generated_count",
    "qwen3_embedding_shape",
    "qwen3_l2_norm_min",
    "qwen3_l2_norm_mean",
    "qwen3_l2_norm_max",
    "bge_m3_embedding_shape",
    "bge_m3_l2_norm_min",
    "bge_m3_l2_norm_mean",
    "bge_m3_l2_norm_max",
    "seed_synthetic_cosine_min",
    "seed_synthetic_cosine_mean",
    "seed_synthetic_cosine_median",
    "seed_synthetic_cosine_max",
    "generation_model_revision",
    "generation_tokenizer_revision",
    "qwen3_model_revision",
    "qwen3_tokenizer_revision",
    "bge_m3_model_revision",
    "bge_m3_tokenizer_revision",
    "manifest_lock_verified",
]


SELECTION_CSV_FIELDS = [
    "cell_id",
    "language",
    "pilot_request_index",
    "seed_index",
    "manifest_seed_index",
    "experiment_bank_index",
    "bank_class_position",
    "prepared_row_index",
    "real_row_id",
    "seed_text",
    "label_int",
    "label",
    "inherited_label",
    "candidate_index",
    "global_generation_index",
    "generation_seed",
    "request_key",
    "prompt_id",
    "prompt_sha256",
    "stage_a_candidate_pool_excluded",
    "stage_a_exclusion_scope",
    "stage_a_excluded_real_row_id",
    "stage_a_excluded_candidate_indices",
    "stage_a_excluded_candidate_count",
]


def output_file_hashes(output_root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "GATE2_LOCK.json":
            hashes[relative_path(output_root, path)] = sha256_file(path)
    return hashes


def write_gate2_lock(output_root: Path, lock_info: Mapping[str, Any]) -> None:
    value = {
        "lock_version": "gate2-pilot-v1",
        "created_at": utc_now(),
        "input_manifest_lock_sha256": lock_info["lock_sha256"],
        "files": output_file_hashes(output_root),
    }
    write_json(output_root / "GATE2_LOCK.json", value)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    script_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Run the locked Gate 2 binary pilot locally."
    )
    parser.add_argument(
        "--prepared-root", type=Path, default=script_root / "Data" / "Prepared"
    )
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=script_root / "Data" / "ExperimentManifests" / "v1",
    )
    parser.add_argument(
        "--output-root", type=Path, default=script_root / "Results" / "Pilot"
    )
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument(
        "--endpoint",
        action="append",
        nargs=3,
        metavar=("API_URL", "MODEL", "CONCURRENCY"),
        help=(
            "LM Studio endpoint, model identifier, and independent concurrency limit. "
            "Repeat for multiple endpoints."
        ),
    )
    parser.add_argument(
        "--qwen-endpoint",
        action="append",
        nargs=3,
        metavar=("API_URL", "MODEL", "CONCURRENCY"),
        help="LM Studio Qwen3 embedding endpoint. Repeat for multiple endpoints.",
    )
    parser.add_argument(
        "--bge-endpoint",
        action="append",
        nargs=3,
        metavar=("API_URL", "MODEL", "CONCURRENCY"),
        help="LM Studio BGE-M3 embedding endpoint. Repeat for multiple endpoints.",
    )
    parser.add_argument("--api-key", default=os.environ.get("LMSTUDIO_API_KEY"))
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate inputs and write selection/prompts without loading local models",
    )
    return parser.parse_args(argv)


def ensure_output_is_new(output_root: Path) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(
            f"output root is not empty: {output_root}; use a fresh --output-root to preserve existing outputs"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.embedding_batch_size <= 0:
        raise SystemExit("--embedding-batch-size must be positive")
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be positive")
    try:
        endpoints = parse_lmstudio_endpoints(args.endpoint, args.model)
        qwen_endpoints = parse_lmstudio_endpoints(
            args.qwen_endpoint,
            QWEN_MODEL_ID,
            DEFAULT_LMSTUDIO_EMBEDDINGS_API_URL,
        )
        bge_endpoints = parse_lmstudio_endpoints(
            args.bge_endpoint,
            BGE_MODEL_ID,
            DEFAULT_LMSTUDIO_EMBEDDINGS_API_URL,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    prepared_root = args.prepared_root.resolve()
    manifest_root = args.manifest_root.resolve()
    output_root = args.output_root.resolve()

    try:
        lock_info = validate_manifest_lock(manifest_root)
        cell_manifests = {
            cell_id: load_cell_manifest(manifest_root, cell_id)
            for cell_id in TARGET_CELLS
        }
        for cell_id in TARGET_CELLS:
            validate_prepared_split_hashes(
                prepared_root, cell_id, cell_manifests[cell_id]
            )
        bank_by_cell = {
            cell_id: load_experiment_bank(
                manifest_root, cell_id, cell_manifests[cell_id]
            )
            for cell_id in TARGET_CELLS
        }
        _plan_by_cell, stage_a_generation_seeds = load_stage_plan_for_cells(
            manifest_root, bank_by_cell, cell_manifests
        )
        selections: dict[str, list[dict[str, Any]]] = {}
        quota_by_cell: dict[str, dict[str, int]] = {}
        for cell_id in TARGET_CELLS:
            selections[cell_id], quota_by_cell[cell_id] = build_selection(
                prepared_root,
                cell_id,
                cell_manifests[cell_id],
                bank_by_cell[cell_id],
                stage_a_generation_seeds,
            )

        ensure_output_is_new(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        prompt_meta = write_prompt_templates(output_root)
        for cell_id in TARGET_CELLS:
            for row in selections[cell_id]:
                row["prompt_sha256"] = prompt_meta[cell_id]["sha256"]
        write_csv(
            output_root / "pilot_selection_manifest.csv",
            [row for cell_id in TARGET_CELLS for row in selections[cell_id]],
            SELECTION_CSV_FIELDS,
        )

        runtime = runtime_info()
        if args.dry_run:
            generation = {
                "status": "selection_only",
                "backend_initialization_error": None,
                "model_revision": None,
                "tokenizer_revision": None,
                "revision_status": {"model": "missing", "tokenizer": "missing"},
                "observed_parameter_dtypes": [],
                "endpoints": [
                    {
                        "api_url": api_url,
                        "model": model,
                        "max_concurrency": max_concurrency,
                    }
                    for api_url, model, max_concurrency in endpoints
                ],
                "valid_by_cell": {cell_id: [] for cell_id in TARGET_CELLS},
                "failures_by_cell": {cell_id: [] for cell_id in TARGET_CELLS},
                "raw_by_cell": {cell_id: [] for cell_id in TARGET_CELLS},
                "outcome_by_request": {},
            }
            qwen = {
                "model_id": QWEN_MODEL_ID,
                "backend": "lm_studio_openai_compatible_embeddings",
                "endpoints": [
                    {"api_url": url, "model": model, "max_concurrency": concurrency}
                    for url, model, concurrency in qwen_endpoints
                ],
                "status": "not_run",
                "model_revision": None,
                "tokenizer_revision": None,
                "revision_status": {"model": "missing", "tokenizer": "missing"},
                "cells": {},
            }
            bge = {
                "model_id": BGE_MODEL_ID,
                "backend": "lm_studio_openai_compatible_embeddings",
                "endpoints": [
                    {"api_url": url, "model": model, "max_concurrency": concurrency}
                    for url, model, concurrency in bge_endpoints
                ],
                "status": "not_run",
                "model_revision": None,
                "tokenizer_revision": None,
                "revision_status": {"model": "missing", "tokenizer": "missing"},
                "cells": {},
                "cosine_rows": [],
            }
        else:
            with loaded_lmstudio_models(
                endpoints,
                api_key=args.api_key,
                timeout_seconds=args.timeout_seconds,
                log=progress_log,
            ):
                generation = run_generation(
                    output_root,
                    selections,
                    prompt_meta,
                    endpoints,
                    args.api_key,
                    args.timeout_seconds,
                )
            with loaded_lmstudio_models(
                qwen_endpoints,
                api_key=args.api_key,
                timeout_seconds=args.timeout_seconds,
                log=progress_log,
            ):
                qwen = run_qwen_embeddings_lmstudio(
                    output_root,
                    selections,
                    generation["valid_by_cell"],
                    args.embedding_batch_size,
                    qwen_endpoints,
                    args.api_key,
                    args.timeout_seconds,
                )
            with loaded_lmstudio_models(
                bge_endpoints,
                api_key=args.api_key,
                timeout_seconds=args.timeout_seconds,
                log=progress_log,
            ):
                bge = run_bge_embeddings_lmstudio(
                    output_root,
                    selections,
                    generation["valid_by_cell"],
                    args.embedding_batch_size,
                    bge_endpoints,
                    args.api_key,
                    args.timeout_seconds,
                )

        report, report_rows = build_report(
            output_root,
            lock_info,
            prompt_meta,
            quota_by_cell,
            selections,
            generation,
            qwen,
            bge,
            runtime,
            args.dry_run,
        )
        write_json(output_root / "gate2_report.json", report)
        write_csv(output_root / "gate2_report.csv", report_rows, REPORT_CSV_FIELDS)
        write_gate2_lock(output_root, lock_info)
        print(json.dumps(report["overall"], ensure_ascii=False, indent=2))
        return 0
    except ManifestValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"binary pilot failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
