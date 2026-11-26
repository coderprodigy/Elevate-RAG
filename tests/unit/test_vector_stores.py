"""Unit tests for the Vector_Store protocol and the embedded backends (Req 2.1, 2.3, 2.5, 2.6)."""

from __future__ import annotations

import numpy as np
import pytest

from elevate.config.schema import BackendSettings
from elevate.core.errors import BackendError, EmbeddingMismatchError
from elevate.stores.vector import (
    FaissVectorStore,
    HnswParams,
    NanoVectorStore,
    VectorItem,
    VectorStore,
    create_vector_store,
)

DIM = 8


def _item(i: int, source: str = "nq", vec: list[float] | None = None) -> VectorItem:
    rng = np.random.default_rng(i)
    vector = vec if vec is not None else rng.normal(size=DIM).tolist()
    return VectorItem(
        id=f"chunk-{i}",
        vector=vector,
        source=source,
        model_id="fake-embedder",
        model_version="1",
        metadata={"doc_id": f"doc-{i}"},
    )


def _stores() -> list[VectorStore]:
    return [
        FaissVectorStore(DIM, index_type="hnsw"),
        FaissVectorStore(DIM, index_type="flat"),
        NanoVectorStore(DIM),
    ]


@pytest.fixture(params=["faiss-hnsw", "faiss-flat", "nanovectordb"])
def store(request: pytest.FixtureRequest) -> VectorStore:
    return {
        "faiss-hnsw": lambda: FaissVectorStore(DIM, index_type="hnsw"),
        "faiss-flat": lambda: FaissVectorStore(DIM, index_type="flat"),
        "nanovectordb": lambda: NanoVectorStore(DIM),
    }[request.param]()


def test_backends_satisfy_protocol() -> None:
    for s in _stores():
        assert isinstance(s, VectorStore)


def test_insert_and_top_k_search(store: VectorStore) -> None:
    items = [_item(i) for i in range(20)]
    store.upsert(items)
    assert len(store) == 20
    assert store.ids() == {f"chunk-{i}" for i in range(20)}

    hits = store.search(items[3].vector, k=5)
    assert len(hits) == 5
    assert hits[0].id == "chunk-3"
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert hits[0].model_id == "fake-embedder"
    assert hits[0].model_version == "1"
    assert hits[0].metadata == {"doc_id": "doc-3"}


def test_k_larger_than_store_returns_all(store: VectorStore) -> None:
    store.upsert([_item(i) for i in range(3)])
    assert len(store.search(_item(0).vector, k=10)) == 3


def test_empty_store_returns_no_hits(store: VectorStore) -> None:
    assert store.search([1.0] * DIM, k=3) == []


def test_filtered_search_by_source(store: VectorStore) -> None:
    store.upsert([_item(i, source="nq" if i % 2 else "multihop") for i in range(20)])
    hits = store.search(_item(0).vector, k=4, source="nq")
    assert len(hits) == 4
    assert all(h.source == "nq" for h in hits)
    assert store.search(_item(0).vector, k=4, source="missing") == []


def test_delete_removes_items(store: VectorStore) -> None:
    store.upsert([_item(i) for i in range(10)])
    assert store.delete(["chunk-2", "chunk-2", "unknown"]) == 1
    assert "chunk-2" not in store.ids()
    assert all(h.id != "chunk-2" for h in store.search(_item(2).vector, k=9))
    assert store.delete(["unknown"]) == 0


def test_upsert_replaces_existing_id(store: VectorStore) -> None:
    store.upsert([_item(i) for i in range(5)])
    unit = [1.0] + [0.0] * (DIM - 1)
    store.upsert([_item(0, source="other", vec=unit)])
    assert len(store) == 5
    hit = store.search(unit, k=1)[0]
    assert hit.id == "chunk-0"
    assert hit.source == "other"
    assert hit.score == pytest.approx(1.0, abs=1e-5)


def test_rejects_mixed_embedding_models(store: VectorStore) -> None:
    store.upsert([_item(0)])
    other = VectorItem(id="x", vector=[1.0] * DIM, source="nq", model_id="other-embedder")
    with pytest.raises(EmbeddingMismatchError) as err:
        store.upsert([other])
    assert "other-embedder" in str(err.value) and "fake-embedder@1" in str(err.value)
    assert store.embedding_model == ("fake-embedder", "1")


@pytest.mark.parametrize(
    "vector", [[1.0] * (DIM - 1), [0.0] * DIM, [float("nan")] + [1.0] * (DIM - 1)]
)
def test_rejects_invalid_vectors(store: VectorStore, vector: list[float]) -> None:
    with pytest.raises(ValueError):
        store.upsert([_item(0, vec=vector)])
    assert len(store) == 0


def test_rejects_non_positive_k(store: VectorStore) -> None:
    with pytest.raises(ValueError):
        store.search([1.0] * DIM, k=0)


def test_hnsw_index_params_recorded() -> None:
    s = FaissVectorStore(DIM, hnsw=HnswParams(m=12, ef_construction=100, ef_search=40))
    s.upsert([_item(0)])
    params = s.index_params()
    assert params["backend"] == "faiss"
    assert params["index_type"] == "hnsw"
    assert (params["M"], params["ef_construction"], params["ef_search"]) == (12, 100, 40)
    assert params["metric"] == "cosine"
    assert params["dim"] == DIM
    assert params["embedding_model_id"] == "fake-embedder"
    assert params["embedding_model_version"] == "1"
    assert params["backend_version"]


def test_flat_index_params_have_no_hnsw_fields() -> None:
    params = FaissVectorStore(DIM, index_type="flat").index_params()
    assert params["index_type"] == "flat"
    assert "M" not in params


def test_hnsw_matches_exact_flat_on_small_index() -> None:
    hnsw, flat = FaissVectorStore(DIM), FaissVectorStore(DIM, index_type="flat")
    items = [_item(i) for i in range(200)]
    hnsw.upsert(items)
    flat.upsert(items)
    query = _item(999).vector
    assert [h.id for h in hnsw.search(query, k=10)] == [h.id for h in flat.search(query, k=10)]


def test_invalid_hnsw_params() -> None:
    with pytest.raises(ValueError):
        HnswParams(m=0)


@pytest.mark.parametrize("backend,version_pkg", [("faiss", "faiss"), ("nanovectordb", None)])
def test_health_check_reports_name_version_status(backend: str, version_pkg: str | None) -> None:
    s = create_vector_store(BackendSettings(vector=backend), dim=DIM)  # type: ignore[arg-type]
    status = s.health_check()
    assert status.backend == backend
    assert status.status == "ok"
    assert status.version and status.version != "unknown"
    assert backend in status.summary()


def test_default_backend_is_faiss() -> None:
    assert isinstance(create_vector_store(dim=DIM), FaissVectorStore)
    assert isinstance(
        create_vector_store(BackendSettings(vector="nanovectordb"), dim=DIM), NanoVectorStore
    )


def test_health_check_failure_raises_backend_error() -> None:
    s = FaissVectorStore(DIM)

    class Broken:
        __version__ = "0"

        def IndexFlatIP(self, dim: int) -> None:  # noqa: N802 - mirrors the faiss API
            raise RuntimeError("faiss is broken")

    s._faiss = Broken()  # type: ignore[assignment]
    with pytest.raises(BackendError) as err:
        s.health_check()
    assert err.value.backend == "faiss"
    assert "docs/SETUP.md#3-installation" in str(err.value)


def test_unavailable_backend_raises_backend_error() -> None:
    with pytest.raises(BackendError) as err:
        create_vector_store(BackendSettings(vector="qdrant"), dim=DIM)
    assert err.value.backend == "qdrant"
    assert err.value.doc_ref == "docs/SETUP.md#qdrant"
