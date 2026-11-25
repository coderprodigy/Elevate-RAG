"""`hgrag bench …` commands: E4 Level A vector and graph micro-benchmarks, and a health check.

Level B (end to end) is a normal evaluation run with another LightRAG backend, e.g.
``hgrag eval run --pipeline lr_local --experiment e4_db --graph-backend qdrant_neo4j --limit 300``.
"""

import json
import random
from dataclasses import asdict
from typing import Annotated

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from hgrag.bench.graph_bench import benchmark_graph_store
from hgrag.bench.graph_stores import GRAPH_STORES, make_graph_store
from hgrag.bench.memory import CONTAINERS, docker_memory_mb, process_peak_rss_mb
from hgrag.bench.vector_bench import VectorBenchSettings, benchmark_store
from hgrag.bench.vector_stores import VECTOR_STORES, make_vector_store
from hgrag.config import PROJECT_ROOT, load_config
from hgrag.data.io import dataset_files, load_queries
from hgrag.graph.build import default_extractor
from hgrag.graph.lightrag_adapter import GraphLocation
from hgrag.graph.quality import load_graph
from hgrag.index.build import default_chunker
from hgrag.index.models import load_embedder
from hgrag.index.paths import IndexPaths

bench_app = typer.Typer(no_args_is_help=True, help="Database benchmarks (E4 Level A).")
console = Console()
OUT_DIR = PROJECT_ROOT / "reports" / "bench"


def _memory_probe(store: str):  # type: ignore[no-untyped-def]
    container = CONTAINERS.get(store)
    return (lambda: docker_memory_mb(container)) if container else process_peak_rss_mb


@bench_app.command("vector")
def vector(
    dataset: Annotated[
        str, typer.Option(help="Dataset whose chunk embeddings to use.")
    ] = "multihop_rag",
    embedder: Annotated[str, typer.Option(help="Embedder whose vectors to use.")] = "bge-small",
    store: Annotated[
        list[str] | None, typer.Option("--store", "-s", help=f"{VECTOR_STORES}")
    ] = None,
    n_queries: Annotated[int, typer.Option(help="Query vectors (embedded questions).")] = 200,
) -> None:
    """Matched-recall comparison of vector stores on the project's own embeddings."""
    config = load_config()
    paths = IndexPaths.of(config, dataset, default_chunker(config, dataset))
    vectors = np.load(paths.embeddings(embedder))
    query_file = paths.embedding_dir(embedder) / "bench_queries.npy"
    if query_file.is_file():
        queries = np.load(query_file)
    else:
        questions = [q.question for q in load_queries(dataset_files(config, dataset))]
        picked = random.Random(config.seed).sample(questions, min(n_queries, len(questions)))
        model = load_embedder(embedder, config.retrieval.embedders[embedder])
        queries = model.embed_queries(picked)
        np.save(query_file, queries)
    settings = VectorBenchSettings()
    reports = []
    for name in store or ["faiss-flat", "faiss-hnsw"]:
        report = benchmark_store(
            make_vector_store(name, config.retrieval.hnsw),
            vectors,
            queries[:n_queries],
            settings,
            memory_probe=_memory_probe(name),
        )
        reports.append(report)
    title = f"Vector stores @ recall@{settings.k} ≥ {settings.target_recall}"
    table = Table(title=f"{title} ({dataset}, {len(vectors):,} vectors)")
    for column in (
        "store",
        "ef",
        "recall",
        "p50 ms",
        "p95 ms",
        "QPS 1/8/32",
        "build s",
        "filtered R",
        "write ms",
        "MB",
    ):
        table.add_column(column, justify="left" if column == "store" else "right")
    for r in reports:
        table.add_row(
            r.store,
            str(r.matched_ef),
            f"{r.matched_recall:.3f}{'' if r.reached_target else ' ✗'}",
            f"{r.latency_ms['p50']:.2f}",
            f"{r.latency_ms['p95']:.2f}",
            "/".join(f"{r.qps[c]:.0f}" for c in settings.concurrency),
            f"{r.build_seconds:.1f}",
            f"{r.filtered_recall:.3f}",
            f"{r.write_to_searchable_ms:.0f}",
            f"{r.memory_mb:.0f}" if r.memory_mb is not None else "-",
        )
    console.print(table)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"vector_{dataset}_{embedder}.json"
    out.write_text(json.dumps([asdict(r) for r in reports], indent=2), encoding="utf-8")
    console.print(f"Wrote {out} (includes recall-vs-QPS curves)")


@bench_app.command("graph")
def graph(
    dataset: Annotated[str, typer.Option(help="Use this dataset's built graph.")] = "multihop_rag",
    store: Annotated[
        list[str] | None, typer.Option("--store", "-s", help=f"{GRAPH_STORES}")
    ] = None,
    synthetic: Annotated[
        int | None, typer.Option(help="Random graph with N nodes instead.")
    ] = None,
    embedder: Annotated[str | None, typer.Option(help="Embedder of the built graph.")] = None,
) -> None:
    """Bulk load, lookup and 1/2-hop latency per graph store (answers checked vs NetworkX)."""
    config = load_config()
    if synthetic:
        rng = random.Random(config.seed)
        nodes = [f"entity-{i}" for i in range(synthetic)]
        edges = [(rng.choice(nodes), rng.choice(nodes)) for _ in range(synthetic * 3)]
        label = f"synthetic-{synthetic}"
    else:
        location = GraphLocation(
            dataset,
            default_chunker(config, dataset),
            default_extractor(config),
            embedder or config.retrieval.reference_embedder,
            "local",
        )
        g = load_graph(location.working_dir(config))
        nodes = [str(n) for n in g.nodes]
        edges = [(str(a), str(b)) for a, b in g.edges]
        label = dataset
    reports = [
        benchmark_graph_store(
            make_graph_store(name), nodes, edges, seed=config.seed, memory_probe=_memory_probe(name)
        )
        for name in store or ["networkx"]
    ]
    table = Table(title=f"Graph stores ({label}: {len(nodes):,} nodes, {len(edges):,} edges)")
    for column in (
        "store",
        "load s",
        "lookup p50",
        "1-hop p50/p95",
        "2-hop p50/p95",
        "agree",
        "MB",
    ):
        table.add_column(column, justify="left" if column == "store" else "right")
    for r in reports:
        table.add_row(
            r.store,
            f"{r.load_seconds:.1f}",
            f"{r.lookup_ms['p50']:.2f}",
            f"{r.one_hop_ms['p50']:.2f}/{r.one_hop_ms['p95']:.2f}",
            f"{r.two_hop_ms['p50']:.2f}/{r.two_hop_ms['p95']:.2f}",
            f"{r.agreement:.0%}",
            f"{r.memory_mb:.0f}" if r.memory_mb is not None else "-",
        )
    console.print(table)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"graph_{label}.json"
    out.write_text(json.dumps([asdict(r) for r in reports], indent=2), encoding="utf-8")
    console.print(f"Wrote {out}")


@bench_app.command("health")
def health() -> None:
    """Which database services answer (start them with docker/README.md)."""
    config = load_config()
    probe = np.random.default_rng(0).random((20, 8)).astype(np.float32)
    probe /= np.linalg.norm(probe, axis=1, keepdims=True)
    for name in VECTOR_STORES:
        try:
            s = make_vector_store(name, config.retrieval.hnsw)
            s.build(probe, [0] * len(probe))
            ok = len(s.search(probe[0], 3, 32)) > 0
            s.drop()
            console.print(f"[green]✓[/] {name}" if ok else f"[red]✗[/] {name}: empty result")
        except Exception as error:  # report every store, whatever failed
            console.print(f"[red]✗[/] {name}: {type(error).__name__}: {str(error)[:80]}")
    for name in GRAPH_STORES:
        try:
            g = make_graph_store(name)
            g.load(["a", "b"], [("a", "b")])
            ok = g.neighbours("a", 1) == {"b"}
            g.clear()
            console.print(f"[green]✓[/] {name}" if ok else f"[red]✗[/] {name}: wrong answer")
        except Exception as error:
            console.print(f"[red]✗[/] {name}: {type(error).__name__}: {str(error)[:80]}")
