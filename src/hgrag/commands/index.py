"""`hgrag index …` commands: build indexes and evaluate retrieval (both free, local)."""

import json
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from hgrag.config import load_config
from hgrag.data.io import dataset_files, load_queries
from hgrag.data.loaders import REGISTRY
from hgrag.index.build import build_index, default_chunker
from hgrag.index.models import load_embedder, load_reranker
from hgrag.index.paths import IndexPaths
from hgrag.pipelines.vector import VARIANTS
from hgrag.retrieval.evaluate import KS, evaluate_retrieval
from hgrag.retrieval.searcher import Searcher

index_app = typer.Typer(no_args_is_help=True, help="Indexes: build and evaluate retrieval.")
console = Console()
CORPUS_DATASETS = [name for name in REGISTRY if name != "ragrouter_bench"]

DatasetOption = Annotated[
    list[str] | None, typer.Option("--dataset", "-d", help="Repeatable. Default: all.")
]


@index_app.command("build")
def build(
    dataset: DatasetOption = None,
    chunker: Annotated[str | None, typer.Option(help="Default: per-dataset chunker.")] = None,
    embedder: Annotated[
        list[str] | None, typer.Option("--embedder", "-e", help="Repeatable. Default: reference.")
    ] = None,
    no_embeddings: Annotated[bool, typer.Option("--no-embeddings", help="BM25 only.")] = False,
) -> None:
    """Chunk, map gold chunks, build BM25 and embeddings + HNSW. Skips parts already built."""
    config = load_config()
    embedders = [] if no_embeddings else (embedder or [config.retrieval.reference_embedder])
    for name in dataset or CORPUS_DATASETS:
        manifest = build_index(config, name, chunker=chunker, embedders=embedders)
        console.print(
            f"[green]✓[/] {name}/{manifest.chunker}: {manifest.n_chunks:,} chunks "
            f"(mean {manifest.mean_chunk_tokens} tok), gold for {manifest.gold_questions:,} "
            f"questions, evidence mapped {100 * manifest.evidence_mapped_rate:.1f}%, "
            f"embedders {sorted(manifest.embedders)}"
        )


@index_app.command("evaluate")
def evaluate(
    dataset: DatasetOption = None,
    variant: Annotated[
        list[str] | None, typer.Option("--variant", "-v", help="v1-v4, repeatable. Default: all.")
    ] = None,
    split: Annotated[str, typer.Option(help="dev (tuning) or test.")] = "dev",
    chunker: Annotated[str | None, typer.Option(help="Default: per-dataset chunker.")] = None,
    embedder: Annotated[str | None, typer.Option(help="Default: reference embedder.")] = None,
    reranker: Annotated[str | None, typer.Option(help="Default: reference reranker.")] = None,
    limit: Annotated[int | None, typer.Option(help="Only the first N questions.")] = None,
    rerank_candidates: Annotated[
        int | None, typer.Option(help="Rerank fewer candidates (faster local dev; recorded).")
    ] = None,
) -> None:
    """Retrieval recall/nDCG for V1-V4 on a split. No LLM calls, no cost."""
    config = load_config()
    if rerank_candidates is not None:
        search = config.retrieval.search.model_copy(update={"rerank_candidates": rerank_candidates})
        config = config.model_copy(
            update={"retrieval": config.retrieval.model_copy(update={"search": search})}
        )
    retrieval = config.retrieval
    variants = variant or list(VARIANTS)
    embedder_name = embedder or retrieval.reference_embedder
    reranker_name = reranker or retrieval.reference_reranker
    needs_dense = any(VARIANTS[v].retriever != "bm25" for v in variants)
    needs_rerank = any(VARIANTS[v].rerank for v in variants)
    emb = load_embedder(embedder_name, retrieval.embedders[embedder_name]) if needs_dense else None
    rr = load_reranker(reranker_name, retrieval.rerankers[reranker_name]) if needs_rerank else None

    for name in dataset or CORPUS_DATASETS:
        chunk_name = chunker or default_chunker(config, name)
        paths = IndexPaths.of(config, name, chunk_name)
        gold = json.loads(paths.gold_chunks.read_text(encoding="utf-8"))
        queries = load_queries(dataset_files(config, name), split)  # type: ignore[arg-type]
        queries = queries[:limit] if limit else queries

        table = Table(title=f"{name} / {chunk_name} / {split} (chunk-level)")
        columns = [*[f"R@{k}" for k in KS], "hit@5", "nDCG@10", "doc R@5", "~ctx tok", "n"]
        table.add_column("variant")
        for column in columns:
            table.add_column(column, justify="right")
        for v in variants:
            spec = VARIANTS[v]
            searcher = Searcher(
                config,
                name,
                chunk_name,
                spec,
                embedder=emb if spec.retriever != "bm25" else None,
                reranker=rr if spec.rerank else None,
            )
            report = evaluate_retrieval(searcher, queries, gold, v)
            table.add_row(
                f"{v} ({spec.retriever}{'+rerank' if spec.rerank else ''})",
                *[f"{report.chunk_recall[k]:.3f}" for k in KS],
                f"{report.chunk_hit[5]:.3f}",
                f"{report.chunk_ndcg10:.3f}",
                f"{report.doc_recall[5]:.3f}",
                f"{report.mean_context_tokens:,.0f}",
                str(report.n_queries),
            )
        console.print(table)
