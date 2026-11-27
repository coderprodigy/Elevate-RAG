"""Database measurements with aligned inputs and explicit run settings."""

import resource
import sys
import time
import uuid
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from .config import Settings
from .data import Chunk, stable_hash, write_json
from .graph import Graph
from .providers import Vector
from .reporting import write_csv
from .retrieval import FaissIndex, Hit
from .storage import open_vector


def latency(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return dict.fromkeys(("p50_ms", "p95_ms", "p99_ms"))
    return dict(
        zip(
            ("p50_ms", "p95_ms", "p99_ms"),
            map(float, np.asarray(np.quantile(values, [0.5, 0.95, 0.99])) * 1000),
            strict=True,
        )
    )


def peak_process_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss if sys.platform == "darwin" else rss * 1024)


def exact_recall(
    expected: Sequence[Hit],
    actual: Sequence[Hit],
    exact_scores: dict[str, float],
    *,
    tolerance: float = 1e-6,
) -> float:
    """Credit any exact tie at the cutoff without giving it a higher rank."""
    if not expected:
        return 1.0 if not actual else 0.0
    cutoff = expected[-1].score
    strict = {hit.chunk_id for hit in expected if hit.score > cutoff + tolerance}
    retrieved = {hit.chunk_id for hit in actual if hit.chunk_id in exact_scores}
    boundary = sum(
        abs(exact_scores[key] - cutoff) <= tolerance
        for key in retrieved - strict
        if key in exact_scores
    )
    return (len(retrieved & strict) + min(boundary, len(expected) - len(strict))) / len(expected)


def vector_microbenchmark(
    settings: Settings,
    chunks: Sequence[Chunk],
    matrix: Vector,
    queries: Vector,
    *,
    backend: str,
    out: Path,
    ef_values: Sequence[int],
    warmup: int = 1000,
    repeats: int = 3,
) -> dict[str, Any]:
    if len(chunks) != len(matrix) or len(matrix) < 11 or len(queries) < 1:
        raise ValueError("vector microbenchmark needs at least 11 passages and one query vector")
    if not ef_values or min(ef_values) < 10 or warmup < 0 or repeats < 1:
        raise ValueError("invalid microbenchmark ef values, warmup or repeats")
    baseline = FaissIndex(matrix.shape[1])
    baseline.upsert(chunks[:-1], matrix[:-1])
    exact = [baseline.search(query, 10) for query in queries]
    index_by_id = {chunk.chunk_id: (i, chunk.source) for i, chunk in enumerate(chunks[:-1])}
    sources = [chunks[i % (len(chunks) - 1)].source for i in range(len(queries))]
    filtered_exact = [
        baseline.search(query, 10, source=source)
        for query, source in zip(queries, sources, strict=True)
    ]
    namespace = f"{settings.namespace[:24]}_a_{uuid.uuid4().hex[:12]}"
    start = time.perf_counter()
    index = open_vector(backend, namespace, matrix.shape[1], settings)
    try:
        index.upsert(chunks[:-1], matrix[:-1])
        build_seconds = time.perf_counter() - start
        records: list[dict[str, Any]] = []
        for filtered in (False, True):
            targets = filtered_exact if filtered else exact
            for ef in sorted(set(ef_values)):

                def query_at(i: int) -> list[Hit]:
                    return index.search(
                        queries[i], 10, source=sources[i] if filtered else None, ef=ef
                    )

                for workers in (1, 8, 32):
                    repetitions: list[dict[str, Any]] = []
                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        for _ in range(repeats):
                            for _result in pool.map(
                                query_at, (i % len(queries) for i in range(warmup))
                            ):
                                pass

                            def measure(i: int) -> tuple[list[Hit], float]:
                                begun = time.perf_counter()
                                found = query_at(i)
                                return found, time.perf_counter() - begun

                            begun = time.perf_counter()
                            raw = list(pool.map(measure, range(len(queries))))
                            spent = time.perf_counter() - begun
                            recalls = []
                            for i, (found, _) in enumerate(raw):
                                actual_scores = {}
                                for hit in found:
                                    j, src = index_by_id[hit.chunk_id]
                                    if not filtered or src == sources[i]:
                                        actual_scores[hit.chunk_id] = float(matrix[j] @ queries[i])
                                recalls.append(exact_recall(targets[i], found, actual_scores))
                            repetitions.append(
                                {
                                    "qps": len(raw) / spent,
                                    "recall_at_10": sum(recalls) / len(raw),
                                    **latency([pair[1] for pair in raw]),
                                }
                            )
                    summary = {
                        key: median(row[key] for row in repetitions) for key in repetitions[0]
                    }
                    records.append(
                        {
                            "backend": backend,
                            "filtered": filtered,
                            "concurrency": workers,
                            "ef": ef,
                            **summary,
                        }
                    )
        matching = [
            row
            for row in records
            if row["recall_at_10"] >= 0.95 and not row["filtered"] and row["concurrency"] == 1
        ]
        start = time.perf_counter()
        index.upsert(chunks[-1:], matrix[-1:])
        deadline = start + settings.timeout
        write_visibility = None
        while time.perf_counter() < deadline:
            if any(
                hit.chunk_id == chunks[-1].chunk_id
                for hit in index.search(matrix[-1], 10, source=chunks[-1].source)
            ):
                write_visibility = time.perf_counter() - start
                break
            time.sleep(0.05)
        result = {
            "backend": backend,
            "namespace": namespace,
            "queries": len(queries),
            "build_seconds": build_seconds,
            "client_peak_rss_bytes": peak_process_bytes(),
            "server_peak_ram_bytes": None,
            "server_disk_bytes": None,
            "write_to_searchable_seconds": write_visibility,
            "warmup": warmup,
            "repeats": repeats,
            "hnsw": {"m": settings.hnsw_m, "ef_construction": settings.ef_construction},
            "matched_target": min(matching, key=lambda row: row["ef"]) if matching else None,
        }
        write_json(out / "vector-benchmark.json", result)
        write_csv(out / "vector-recall-qps.csv", records)
        return result
    finally:
        index.close()
        baseline.close()


async def graph_microbenchmark(
    graph: Graph, queries: Sequence[str], out: Path, seed: int, size: int = 1000
) -> dict[str, Any]:
    nodes, edges = await graph.snapshot()
    if len(nodes) < size:
        raise ValueError(f"graph contains {len(nodes)} nodes, but benchmark requires {size}")
    sample = sorted((row["id"] for row in nodes), key=lambda name: stable_hash(seed, name))[:size]
    store = graph.rag.chunk_entity_relation_graph
    lookup, one, two, local = [], [], [], []
    for entity in sample:
        start = time.perf_counter()
        if await store.get_node(entity) is None:
            raise ValueError("sample entity could not be reloaded")
        lookup.append(time.perf_counter() - start)
        start = time.perf_counter()
        await graph.neighborhood([entity], 1, limit=10_000_000)
        one.append(time.perf_counter() - start)
        start = time.perf_counter()
        await graph.neighborhood([entity], 2, limit=10_000_000)
        two.append(time.perf_counter() - start)
    for query in queries:
        start = time.perf_counter()
        await graph.retrieve(query, "local")
        local.append(time.perf_counter() - start)
    result = {
        "nodes": len(nodes),
        "edges": len(edges),
        "sampled_entities": sample,
        "lookup": latency(lookup),
        "one_hop": latency(one),
        "two_hop": latency(two),
        "local": latency(local),
        "client_peak_rss_bytes": peak_process_bytes(),
    }
    write_json(out / "graph-benchmark.json", result)
    return result


async def bulk_load_graph(
    graph: Graph,
    out: Path,
    *,
    batch_size: int = 1000,
) -> dict[str, Any]:
    """Persist a graph snapshot in a fresh workspace without generating new extractions."""
    from dataclasses import replace

    from .graph import open_graph
    from .providers import Providers

    if batch_size < 1:
        raise ValueError("bulk-load batch must be positive")
    nodes, edges = await graph.snapshot()
    providers = graph.providers
    settings = replace(providers.settings, graph_vector_store="nano")
    key = uuid.uuid4().hex
    destination = out / f"bulk-{key[:16]}"
    started = time.perf_counter()
    async with (
        Providers(settings, providers.cache, providers.budget.path, uncached=True) as isolated,
        open_graph(
            destination,
            key,
            graph.catalog,
            isolated,
            providers.tokenizer(context=True),
            vector=None,
        ) as copy,
    ):
        setup_seconds = time.perf_counter() - started
        store = copy.rag.chunk_entity_relation_graph
        started = time.perf_counter()
        for i in range(0, len(nodes), batch_size):
            records = [
                (
                    node["id"],
                    {
                        field: value
                        for field, value in node.items()
                        if field not in {"id", "chunk_ids"}
                    },
                )
                for node in nodes[i : i + batch_size]
            ]
            await store.upsert_nodes_batch(records)
        nodes_seconds = time.perf_counter() - started
        started = time.perf_counter()
        for i in range(0, len(edges), batch_size):
            records_edges = [
                (
                    edge["source"],
                    edge["target"],
                    {
                        field: value
                        for field, value in edge.items()
                        if field not in {"source", "target", "chunk_ids"}
                    },
                )
                for edge in edges[i : i + batch_size]
            ]
            await store.upsert_edges_batch(records_edges)
        edges_seconds = time.perf_counter() - started
        started = time.perf_counter()
        await store.index_done_callback()
        persist_seconds = time.perf_counter() - started
        reloaded_nodes, reloaded_edges = await copy.snapshot()
        if len(reloaded_nodes) != len(nodes) or len(reloaded_edges) != len(edges):
            raise ValueError("graph bulk load changed the node or edge count")
    return {
        "backend": settings.graph_store,
        "setup_seconds": setup_seconds,
        "node_load_seconds": nodes_seconds,
        "edge_load_seconds": edges_seconds,
        "persistence_seconds": persist_seconds,
        "client_peak_rss_bytes": peak_process_bytes(),
        "storage_local_bytes": sum(
            path.stat().st_size for path in destination.rglob("*") if path.is_file()
        ),
        "sampled_nodes": len(nodes),
        "sampled_edges": len(edges),
        "workspace": str(destination),
    }
