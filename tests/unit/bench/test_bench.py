import os

import numpy as np
import pytest

from hgrag.bench.graph_bench import benchmark_graph_store
from hgrag.bench.graph_stores import NetworkXStore, make_graph_store
from hgrag.bench.memory import parse_docker_mem
from hgrag.bench.vector_bench import VectorBenchSettings, benchmark_store, exact_top_k, recall_at_k
from hgrag.bench.vector_stores import FaissStore, make_vector_store
from hgrag.config import HNSWSpec

HNSW = HNSWSpec(m=16, ef_construction=100, ef_search=64)
SMALL = VectorBenchSettings(ef_sweep=(8, 32, 128), concurrency=(1, 4), warmup_queries=10, repeats=1)


def _vectors(n: int, dim: int = 32, seed: int = 0) -> np.ndarray:
    data = np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32)
    return data / np.linalg.norm(data, axis=1, keepdims=True)


def test_recall_helpers() -> None:
    assert recall_at_k([[1, 2, 3]], [[3, 2, 9]], k=3) == pytest.approx(2 / 3)
    vectors = _vectors(100)
    assert exact_top_k(vectors, vectors[:5], k=1) == [[0], [1], [2], [3], [4]]


def test_exact_faiss_has_perfect_recall_and_hnsw_reaches_target() -> None:
    vectors, queries = _vectors(3000), _vectors(60, seed=1)
    flat = benchmark_store(FaissStore("flat", HNSW), vectors, queries, SMALL)
    assert flat.matched_recall == 1.0 and flat.filtered_recall == 1.0
    hnsw = benchmark_store(FaissStore("hnsw", HNSW), vectors, queries, SMALL)
    assert hnsw.reached_target and hnsw.matched_recall >= 0.95
    assert [p["ef"] for p in hnsw.curve] == [8, 32, 128]
    assert hnsw.curve[-1]["recall"] >= hnsw.curve[0]["recall"]  # more ef → more recall
    # Restrictive filters (10% of rows) cost HNSW recall at the matched ef (0.83 here); that is
    # why filtered search is measured separately. Exact search stays perfect.
    assert 0.5 <= hnsw.filtered_recall <= 1.0
    assert hnsw.write_to_searchable_ms < 1000 and set(hnsw.qps) == {1, 4}


def test_filtered_search_returns_only_the_group() -> None:
    store = FaissStore("hnsw", HNSW)
    vectors = _vectors(500)
    groups = [i % 10 for i in range(500)]
    store.build(vectors, groups)
    assert all(groups[r] == 3 for r in store.search(vectors[7], 10, 64, group=3))


def test_graph_bench_agrees_with_networkx() -> None:
    nodes = [f"n{i}" for i in range(200)]
    rng = np.random.default_rng(0)
    edges = [(nodes[a], nodes[b]) for a, b in rng.integers(0, 200, size=(400, 2)) if a != b]
    report = benchmark_graph_store(NetworkXStore(), nodes, edges, n_entities=50)
    assert report.agreement == 1.0 and report.nodes == 200
    store = NetworkXStore()
    store.load(["a", "b", "c"], [("a", "b"), ("b", "c")])
    assert store.neighbours("a", 1) == {"b"} and store.neighbours("a", 2) == {"b", "c"}


def test_docker_memory_parsing_and_unknown_stores() -> None:
    assert parse_docker_mem("1.5GiB / 4GiB") == 1536.0
    assert parse_docker_mem("512MiB / 4GiB") == 512.0
    assert parse_docker_mem("garbage") is None
    with pytest.raises(KeyError):
        make_vector_store("nope", HNSW)
    with pytest.raises(KeyError):
        make_graph_store("nope")


@pytest.mark.integration
@pytest.mark.parametrize("name", ["qdrant", "pgvector", "milvus"])
def test_server_vector_stores_match_faiss(name: str) -> None:
    """Needs the docker stacks (docker/README.md); run with `make test-int`."""
    if os.environ.get("HGRAG_SKIP_DB_TESTS"):
        pytest.skip("database tests disabled")
    try:
        store = make_vector_store(name, HNSW)
    except Exception as error:  # client missing or service down
        pytest.skip(f"{name} unavailable: {error}")
    report = benchmark_store(store, _vectors(2000), _vectors(30, seed=1), SMALL)
    store.drop()
    assert report.reached_target
