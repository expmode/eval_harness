from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

from .config import RunConfig
from .types import DatasetRow


REQUIRED_FIELDS = {
    "prompt": str,
    "target": str,
    "language": str,
    "eu_ai_act_category": str,
    "annex_section": str,
}

ALLOWED_TARGETS = {"should_refuse", "should_answer"}


def make_row_id(prompt: str, language: str, target: str) -> str:
    payload = f"{language.strip().lower()}::{target.strip()}::{prompt}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]


def _validate_row(raw: dict, *, line_number: int, path: Path) -> None:
    for field_name, field_type in REQUIRED_FIELDS.items():
        if field_name not in raw:
            raise ValueError(f"Missing required field '{field_name}' in {path}:{line_number}")
        if not isinstance(raw[field_name], field_type):
            raise ValueError(
                f"Field '{field_name}' in {path}:{line_number} must be {field_type.__name__}"
            )
        if not raw[field_name].strip():
            raise ValueError(f"Field '{field_name}' in {path}:{line_number} must not be empty")

    target = raw["target"].strip()
    if target not in ALLOWED_TARGETS:
        raise ValueError(
            f"Invalid target '{target}' in {path}:{line_number}. Allowed: {sorted(ALLOWED_TARGETS)}"
        )

    if "is_machine_translation" in raw and not isinstance(raw["is_machine_translation"], bool):
        raise ValueError(
            f"Field 'is_machine_translation' in {path}:{line_number} must be boolean"
        )


def _row_matches_filters(row: DatasetRow, config: RunConfig) -> bool:
    if config.verified_only and row.is_machine_translation:
        return False
    if config.language_include and row.language not in config.language_include:
        return False
    if config.language_exclude and row.language in config.language_exclude:
        return False
    if config.category_include and row.eu_ai_act_category not in config.category_include:
        return False
    if config.category_exclude and row.eu_ai_act_category in config.category_exclude:
        return False
    return True


def iter_dataset_rows(
    path: Path,
    *,
    verified_only: bool = False,
    limit: int | None = None,
    language_include: tuple[str, ...] = (),
    language_exclude: tuple[str, ...] = (),
    category_include: tuple[str, ...] = (),
    category_exclude: tuple[str, ...] = (),
) -> Iterable[DatasetRow]:
    config = RunConfig(
        dataset_path=path,
        verified_only=verified_only,
        limit=limit,
        language_include=language_include,
        language_exclude=language_exclude,
        category_include=category_include,
        category_exclude=category_exclude,
    )
    yielded = 0
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path}:{index}") from exc
            _validate_row(raw, line_number=index, path=path)

            row = DatasetRow(
                row_id=make_row_id(raw["prompt"], raw["language"], raw["target"]),
                prompt=raw["prompt"].strip(),
                target=raw["target"].strip(),
                language=raw["language"].strip().lower(),
                eu_ai_act_category=raw["eu_ai_act_category"].strip(),
                annex_section=raw["annex_section"].strip(),
                reasoning=raw.get("reasoning"),
                reasoning2=raw.get("reasoning2"),
                is_machine_translation=raw.get("is_machine_translation", False),
                source_index=index - 1,
            )
            if not _row_matches_filters(row, config):
                continue
            yield row
            yielded += 1
            if config.limit is not None and yielded >= config.limit:
                break


def load_dataset_rows(
    path: Path,
    *,
    verified_only: bool = False,
    limit: int | None = None,
    language_include: tuple[str, ...] = (),
    language_exclude: tuple[str, ...] = (),
    category_include: tuple[str, ...] = (),
    category_exclude: tuple[str, ...] = (),
    random_sample: int | None = None,
    sampling_seed: int = 0,
) -> list[DatasetRow]:
    rows = list(
        iter_dataset_rows(
            path,
            verified_only=verified_only,
            limit=limit,
            language_include=language_include,
            language_exclude=language_exclude,
            category_include=category_include,
            category_exclude=category_exclude,
        )
    )
    if random_sample is not None and len(rows) > random_sample:
        rng = random.Random(sampling_seed)
        rows = rng.sample(rows, random_sample)
    return rows


def dataset_file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_slice_sha1(rows: list[DatasetRow]) -> str:
    digest = hashlib.sha1()
    for row in rows:
        digest.update(row.row_id.encode("utf-8"))
    return digest.hexdigest()


def summarize_dataset(config: RunConfig) -> dict:
    rows = load_dataset_rows(
        config.dataset_path,
        verified_only=config.verified_only,
        limit=config.limit,
        language_include=config.language_include,
        language_exclude=config.language_exclude,
        category_include=config.category_include,
        category_exclude=config.category_exclude,
        random_sample=config.random_sample,
        sampling_seed=config.sampling_seed,
    )
    row_ids = [row.row_id for row in rows]
    duplicate_row_ids = [row_id for row_id, count in Counter(row_ids).items() if count > 1]
    return {
        "dataset_path": str(config.dataset_path),
        "dataset_file_sha1": dataset_file_sha1(config.dataset_path),
        "dataset_slice_sha1": dataset_slice_sha1(rows),
        "n_rows": len(rows),
        "n_verified": sum(1 for row in rows if not row.is_machine_translation),
        "n_machine_translated": sum(1 for row in rows if row.is_machine_translation),
        "filters": {
            "verified_only": config.verified_only,
            "language_include": list(config.language_include),
            "language_exclude": list(config.language_exclude),
            "category_include": list(config.category_include),
            "category_exclude": list(config.category_exclude),
            "limit": config.limit,
            "random_sample": config.random_sample,
            "sampling_seed": config.sampling_seed,
        },
        "label_distribution": dict(Counter(row.target for row in rows)),
        "language_distribution": dict(Counter(row.language for row in rows)),
        "category_distribution": dict(Counter(row.eu_ai_act_category for row in rows)),
        "duplicate_row_ids": duplicate_row_ids,
    }