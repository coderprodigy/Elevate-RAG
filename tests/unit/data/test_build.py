import json
from pathlib import Path

import pytest

from hgrag.config import AppConfig
from hgrag.data.build import DataValidationError, _validate, build_dataset
from hgrag.data.datacard import DataCard, render_markdown
from hgrag.data.io import dataset_files, load_corpus, load_queries, raw_dir
from hgrag.data.loaders import REGISTRY, resolve
from hgrag.data.schema import Document, Query
from hgrag.data.splits import FrozenSplitError

from .raw_fixtures import write_multihop_rag


def _build_multihop(config: AppConfig) -> DataCard:
    write_multihop_rag(raw_dir(config) / "multihop_rag")
    return build_dataset(REGISTRY["multihop_rag"], config)


def test_build_writes_corpus_queries_splits_and_card(config: AppConfig) -> None:
    card = _build_multihop(config)
    files = dataset_files(config, "multihop_rag")

    assert len(load_corpus(files)) == card.n_documents == 2
    queries = load_queries(files)
    assert len(queries) == card.n_queries == 2
    assert all(q.split in ("dev", "test") for q in queries)
    assert card.n_dev + card.n_test == 2 and card.n_unanswerable == 1
    assert "multihop_rag" in render_markdown([card])


def test_rebuild_reuses_frozen_splits_and_detects_changes(config: AppConfig) -> None:
    _build_multihop(config)
    files = dataset_files(config, "multihop_rag")
    first = {q.qid: q.split for q in load_queries(files)}
    _build_multihop(config)
    assert {q.qid: q.split for q in load_queries(files)} == first

    raw_queries = raw_dir(config) / "multihop_rag" / "MultiHopRAG.json"
    raw_queries.write_text(json.dumps(json.loads(raw_queries.read_text())[:1]))
    with pytest.raises(FrozenSplitError):
        build_dataset(REGISTRY["multihop_rag"], config)


def test_missing_gold_document_fails_validation() -> None:
    doc = Document(doc_id="d1", title="t", text="x", source="s")
    query = Query(
        qid="q",
        question="q?",
        answers=["a"],
        gold_doc_ids=["d404"],
        query_type="single_hop",
        native_type="x",
    )
    with pytest.raises(DataValidationError, match="gold docs missing"):
        _validate("s", [doc], [query])


def test_resolve_keeps_build_order_with_ragrouter_last() -> None:
    names = [spec.name for spec in resolve(["ragrouter_bench", "nq"])]
    assert names == ["nq", "ragrouter_bench"]
    assert list(REGISTRY)[-1] == "ragrouter_bench"
    with pytest.raises(KeyError):
        resolve(["nope"])


def test_every_registered_dataset_has_settings(config: AppConfig) -> None:
    assert set(REGISTRY) == set(config.datasets)


def test_missing_raw_files_give_a_helpful_error(config: AppConfig, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="hgrag data download"):
        build_dataset(REGISTRY["hotpotqa"], config)
