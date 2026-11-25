"""Shared loader types and helpers."""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hgrag.config import DatasetSettings
from hgrag.data.schema import Document, Query


class MissingRawDataError(FileNotFoundError):
    """A raw file is missing; run ``hgrag data download`` first."""


@dataclass(frozen=True)
class BuildInputs:
    name: str
    raw_dir: Path  # data/raw/<raw_name>
    processed_root: Path  # data/processed (lets a loader read other built datasets)
    settings: DatasetSettings
    seed: int


@dataclass
class BuiltDataset:
    documents: list[Document]
    queries: list[Query]
    notes: list[str] = field(default_factory=list)


Builder = Callable[[BuildInputs], BuiltDataset]


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise MissingRawDataError(f"missing {path}; run: hgrag data download")
    return path


def read_json(path: Path) -> Any:
    return json.loads(require_file(path).read_text(encoding="utf-8"))


def read_json_lines(path: Path) -> list[Any]:
    with require_file(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def seeded_sample[T](items: Sequence[T], k: int | None, seed: int) -> list[T]:
    """``k`` items chosen with a fixed seed, kept in their original order (all if ``k`` is None)."""
    if k is None or k >= len(items):
        return list(items)
    chosen = sorted(random.Random(seed).sample(range(len(items)), k))
    return [items[i] for i in chosen]
