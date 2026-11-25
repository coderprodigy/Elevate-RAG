"""Reading and writing the processed dataset files."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from hgrag.config import AppConfig
from hgrag.data.schema import Document, Query, Split


def write_jsonl(path: Path, records: Iterable[BaseModel]) -> int:
    """Write one JSON object per line (atomically, via a temp file). Returns the count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")
            count += 1
    tmp.replace(path)
    return count


def read_jsonl[RecordT: BaseModel](path: Path, model: type[RecordT]) -> list[RecordT]:
    with path.open(encoding="utf-8") as handle:
        return [model.model_validate_json(line) for line in handle if line.strip()]


@dataclass(frozen=True)
class DatasetFiles:
    """Locations of one processed dataset: ``data/processed/<name>/``."""

    root: Path

    @property
    def corpus(self) -> Path:
        return self.root / "corpus.jsonl"

    @property
    def queries(self) -> Path:
        return self.root / "queries.jsonl"

    @property
    def splits(self) -> Path:
        return self.root / "splits.json"

    @property
    def datacard(self) -> Path:
        return self.root / "datacard.json"

    def exists(self) -> bool:
        return self.queries.is_file()


def raw_dir(config: AppConfig) -> Path:
    return config.paths.data_dir / "raw"


def processed_dir(config: AppConfig) -> Path:
    return config.paths.data_dir / "processed"


def dataset_files(config: AppConfig, name: str) -> DatasetFiles:
    return DatasetFiles(processed_dir(config) / name)


def load_corpus(files: DatasetFiles) -> list[Document]:
    return read_jsonl(files.corpus, Document) if files.corpus.is_file() else []


def load_queries(files: DatasetFiles, split: Split | None = None) -> list[Query]:
    queries = read_jsonl(files.queries, Query)
    return queries if split is None else [q for q in queries if q.split == split]
