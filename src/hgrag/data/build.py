"""Build one processed dataset: load → deduplicate → validate → split → write → data card."""

from __future__ import annotations

import logging

from hgrag.config import AppConfig, ConfigError
from hgrag.data.datacard import DataCard, compute_datacard
from hgrag.data.io import dataset_files, processed_dir, raw_dir, write_jsonl
from hgrag.data.loaders import DatasetSpec
from hgrag.data.loaders.base import BuildInputs
from hgrag.data.schema import Document, Query
from hgrag.data.splits import load_or_create_splits

logger = logging.getLogger(__name__)


class DataValidationError(ValueError):
    """The built data breaks an invariant (e.g. a gold document missing from the corpus)."""


def build_dataset(spec: DatasetSpec, config: AppConfig) -> DataCard:
    if spec.name not in config.datasets:
        raise ConfigError(f"no settings for '{spec.name}' in configs/datasets.yaml")
    settings = config.datasets[spec.name]
    files = dataset_files(config, spec.name)
    processed_dir(config).mkdir(parents=True, exist_ok=True)

    built = spec.builder(
        BuildInputs(
            name=spec.name,
            raw_dir=raw_dir(config) / spec.raw_name,
            processed_root=processed_dir(config),
            settings=settings,
            seed=config.seed,
        )
    )
    documents = _dedupe_documents(built.documents)
    queries, duplicates = _dedupe_queries(built.queries)
    notes = list(built.notes)
    if duplicates:
        notes.append(f"Dropped {duplicates} duplicate question(s).")
    _validate(spec.name, documents, queries)

    splits = load_or_create_splits(files.splits, queries, settings.dev_fraction, config.seed)
    queries = [q.model_copy(update={"split": splits[q.qid]}) for q in queries]

    write_jsonl(files.corpus, documents)
    write_jsonl(files.queries, queries)
    card = compute_datacard(spec.name, settings, documents, queries, notes)
    files.datacard.write_text(card.model_dump_json(indent=2), encoding="utf-8")
    logger.info(
        "%s: %d documents, %d queries (%d dev / %d test)",
        spec.name,
        card.n_documents,
        card.n_queries,
        card.n_dev,
        card.n_test,
    )
    return card


def _dedupe_documents(documents: list[Document]) -> list[Document]:
    """Keep the first occurrence of each doc_id (IDs are content hashes)."""
    unique: dict[str, Document] = {}
    for doc in documents:
        unique.setdefault(doc.doc_id, doc)
    return list(unique.values())


def _dedupe_queries(queries: list[Query]) -> tuple[list[Query], int]:
    unique: dict[str, Query] = {}
    for query in queries:
        unique.setdefault(query.qid, query)
    return list(unique.values()), len(queries) - len(unique)


def _validate(name: str, documents: list[Document], queries: list[Query]) -> None:
    doc_ids = {d.doc_id for d in documents}
    if documents:  # router-only datasets have no corpus
        missing = [q.qid for q in queries if not set(q.gold_doc_ids) <= doc_ids]
        if missing:
            raise DataValidationError(f"{name}: gold docs missing from corpus for {missing[:5]}")
    bad_unanswerable = [
        q.qid for q in queries if not q.answerable and (q.answers or q.gold_doc_ids)
    ]
    if bad_unanswerable:
        raise DataValidationError(
            f"{name}: unanswerable queries with answers {bad_unanswerable[:5]}"
        )
    empty = [q.qid for q in queries if not q.question]
    if empty:
        raise DataValidationError(f"{name}: empty questions {empty[:5]}")
