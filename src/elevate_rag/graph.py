"""LightRAG integration and graph audit support."""

import json
import math
import re
import time
from collections import defaultdict
from collections.abc import AsyncIterator, Iterable, Sequence
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .communities import build_communities, community_candidates
from .config import LIGHT_GRAPH, LIGHT_VECTOR, Settings
from .data import (
    Chunk,
    Question,
    Tokenizer,
    artifact_lock,
    read_jsonl,
    stable_hash,
    write_json,
    write_jsonl,
)
from .providers import Providers, Vector
from .retrieval import Hit, VectorIndex, rrf, vector_key
from .schema_extraction import SCHEMA_VERSION, schema_extractor
from .storage import open_passage_index

GRAPH_SEP = "<SEP>"
COVERAGE_METHOD = "answer-entity-v2"


class GraphTokenizer:
    """Provide LightRAG with the same encoding used for passage budgeting."""

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.model_name = tokenizer.identity

    def encode(self, text: str) -> list[int]:
        if hasattr(self.tokenizer, "tokenizer"):
            return list(
                self.tokenizer.tokenizer.encode(text, add_special_tokens=False, verbose=False)
            )
        return list(map(ord, text))

    def decode(self, tokens: list[int]) -> str:
        if hasattr(self.tokenizer, "tokenizer"):
            return str(self.tokenizer.tokenizer.decode(tokens))
        return "".join(map(chr, tokens))

    def __deepcopy__(self, memo: dict[int, Any]) -> "GraphTokenizer":
        return self


def graph_key(settings: Settings, chunks_key: str) -> str:
    return stable_hash(
        "graph-lightrag-1.5.8-v4",
        vector_key(settings, chunks_key),
        chunks_key,
        asdict(settings.model("embedding")),
        asdict(settings.model("extraction")),
        settings.chunk_tokenizer_model,
        settings.chunk_tokenizer_revision,
        settings.graph_vector_store,
        settings.graph_store,
        settings.namespace,
        settings.graph_merge_threshold,
        settings.graph_extractor,
        SCHEMA_VERSION,
        settings.community,
        settings.seed if settings.community else None,
        settings.hnsw_m,
        settings.ef_construction,
        settings.ef_search,
    )


class Graph:
    def __init__(self, native: Any, catalog: Sequence[Chunk], providers: Providers, path: Path):
        from .graph_passages import native_passage_id

        self.rag = native
        self.providers = providers
        self.catalog = catalog
        self.path = path
        self.ids: dict[str, list[str]] = defaultdict(list)
        self.documents: dict[str, list[Chunk]] = defaultdict(list)
        for chunk in catalog:
            self.ids[native_passage_id(chunk)].append(chunk.chunk_id)
            self.documents[chunk.doc_id].append(chunk)

    def sources(self, value: str) -> list[str]:
        result = []
        for key in value.split(GRAPH_SEP):
            if key not in self.ids:
                raise ValueError(f"unmapped LightRAG provenance ID: {key!r}")
            result.extend(self.ids[key])
        return result

    async def build(self) -> dict[str, Any]:
        for doc_id, chunks in self.documents.items():
            texts = [chunk.text for chunk in chunks]
            await self.rag.ainsert_custom_chunks("\n\n".join(texts), texts, doc_id=doc_id)
        merged = 0
        if self.providers.settings.graph_merge_threshold is not None:
            merged = await self.merge(self.providers.settings.graph_merge_threshold)
        report_count = 0
        if self.providers.settings.community:
            nodes, edges = await self.snapshot()
            report_count = await build_communities(
                nodes, edges, self.providers, self.path / "communities.jsonl"
            )
        return {"merged_entities": merged, "community_reports": report_count}

    async def merge(self, threshold: float) -> int:
        nodes = sorted(
            await self.rag.chunk_entity_relation_graph.get_all_nodes(), key=lambda node: node["id"]
        )
        merged = 0
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for node in nodes:
            groups[node.get("entity_type", "UNKNOWN")].append(node)
        for group in groups.values():
            if len(group) < 2:
                continue
            vectors = await self.providers.embed(
                [f"{node['id']} {node.get('description', '')}" for node in group]
            )
            unavailable: set[int] = set()
            for index, target in enumerate(group):
                if index in unavailable:
                    continue
                scores = vectors[index + 1 :] @ vectors[index]
                candidates = [
                    offset
                    for offset, score in enumerate(scores, index + 1)
                    if score >= threshold and offset not in unavailable
                ]
                if candidates:
                    await self.rag.amerge_entities(
                        [group[i]["id"] for i in candidates], target["id"]
                    )
                    unavailable.update(candidates)
                    merged += len(candidates)
        return merged

    async def retrieve(self, query: str, mode: str) -> tuple[list[Hit], dict[str, Any]]:
        from lightrag import QueryParam

        settings = self.providers.settings
        result = await self.rag.aquery_data(
            query,
            QueryParam(
                mode=mode,
                top_k=settings.candidates,
                chunk_top_k=settings.candidates,
                max_total_tokens=max(16384, settings.candidates * settings.chunk_size * 3),
                max_entity_tokens=max(6000, settings.candidates * 128),
                max_relation_tokens=max(8000, settings.candidates * 128),
                enable_rerank=False,
            ),
        )
        if result.get("status") != "success":
            if result.get("metadata", {}).get("failure_reason") == "no_results":
                return [], {"native": result.get("metadata", {})}
            raise RuntimeError(f"LightRAG retrieval failed for {mode}")
        data = result["data"]
        chosen: dict[str, None] = {}
        for row in data["chunks"]:
            chosen.update(dict.fromkeys(self.sources(row["chunk_id"])))
        hits = [Hit(chunk_id, 1 / rank) for rank, chunk_id in enumerate(chosen, 1)]
        if settings.graph_depth == 2 and mode in {"local", "hybrid", "mix"}:
            seeds = [node["entity_name"] for node in data["entities"]]
            expanded: dict[str, None] = {}
            for row in await self.neighborhood(seeds, 2, limit=settings.candidates * 4):
                expanded.update(dict.fromkeys(self.sources(row["source_id"])))
            extension = [Hit(chunk_id, 1 / rank) for rank, chunk_id in enumerate(expanded, 1)]
            hits = rrf([hits, extension], settings.rrf_k, settings.candidates)
        community_ids: list[str] = []
        if settings.community and mode in {"global", "hybrid", "mix"}:
            scenarios = list(read_jsonl(self.path / "communities.jsonl"))
            community_hits, community_ids = await community_candidates(
                query, scenarios, self.providers
            )
            hits = rrf([hits, community_hits], settings.rrf_k, settings.candidates)
        hits = hits[: settings.candidates]
        metadata = {
            "community_reports": community_ids,
            "entities": [row["entity_name"] for row in data["entities"]],
            "relations": [[row["src_id"], row["tgt_id"]] for row in data["relationships"]],
            "native": result.get("metadata", {}),
        }
        return hits, metadata

    async def neighborhood(
        self, seeds: Iterable[str], depth: int, *, limit: int
    ) -> list[dict[str, Any]]:
        storage = self.rag.chunk_entity_relation_graph
        frontier = set(seeds)
        visited = set(frontier)
        edges: dict[tuple[str, str], dict[str, Any]] = {}
        for _ in range(depth):
            found: set[str] = set()
            for node in sorted(frontier):
                for pair in sorted(await storage.get_node_edges(node) or ()):
                    key = tuple(sorted(pair))
                    if key in edges:
                        continue
                    row = await storage.get_edge(*key)
                    if row is None:
                        raise ValueError(f"graph adjacency references a missing edge: {key}")
                    edges[key] = row
                    found.update(key)
                    if len(edges) == limit:
                        return list(edges.values())
            frontier = found - visited
            visited.update(found)
        return list(edges.values())

    async def snapshot(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        storage = self.rag.chunk_entity_relation_graph
        nodes, edges = await storage.get_all_nodes(), await storage.get_all_edges()
        for row in [*nodes, *edges]:
            row["chunk_ids"] = self.sources(row.get("source_id", ""))
        return nodes, edges


def normalized_name(text: str) -> str:
    return " ".join(re.findall(r"[^\W_]+", text.casefold()))


def inspect_graph(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    questions: Sequence[Question],
    seed: int,
    out: Path,
    *,
    catalog: dict[str, Chunk] | None = None,
) -> dict[str, Any]:
    import networkx as nx

    graph = nx.Graph()
    graph.add_nodes_from(node["id"] for node in nodes)
    graph.add_edges_from((edge["source"], edge["target"]) for edge in edges)
    names = {normalized_name(node["id"]) for node in nodes}
    excluded_labels = {"yes", "no", "noanswer", "insufficient information"}
    answer_names = names - excluded_labels
    short = [
        question
        for question in questions
        if question.answerable
        and question.answers
        and any(normalized_name(answer) not in excluded_labels for answer in question.answers)
        and not any(
            label in question.query_type for label in ("summary", "summar", "creative", "global")
        )
    ]
    matched = sum(
        any(normalized_name(answer) in answer_names for answer in question.answers)
        for question in short
    )
    stats = {
        "coverage_method": COVERAGE_METHOD,
        "development_questions": len(questions),
        "nodes": len(nodes),
        "edges": len(edges),
        "average_degree": 2 * len(edges) / len(nodes) if nodes else 0,
        "average_clustering": nx.average_clustering(graph) if nodes else None,
        "normalized_duplicate_rate": (len(nodes) - len(names)) / len(nodes) if nodes else None,
        "untyped_nodes": sum(
            node.get("entity_type", "").casefold() in {"", "unknown"} for node in nodes
        ),
        "coverage_questions": len(short),
        "covered_questions": matched,
        "answer_entity_coverage": matched / len(short) if short else None,
    }
    audit = sorted(edges, key=lambda edge: stable_hash(seed, edge["source"], edge["target"]))[:50]
    audit_path = out / "triple-audit.jsonl"
    previous = (
        {row["triple_id"]: row for row in read_jsonl(audit_path)} if audit_path.exists() else {}
    )
    audit_rows = []
    for edge in audit:
        triple_id = stable_hash(edge["source"], edge["target"])
        old = previous.get(triple_id, {})
        if (
            old.get("description") != edge.get("description")
            or old.get("chunk_ids") != edge["chunk_ids"]
        ):
            old = {}
        audit_rows.append(
            {
                "triple_id": triple_id,
                **edge,
                "passages": [catalog[chunk_id].text for chunk_id in edge["chunk_ids"]]
                if catalog
                else [],
                "annotator_a": old.get("annotator_a"),
                "annotator_b": old.get("annotator_b"),
            }
        )
    write_jsonl(audit_path, audit_rows)
    write_json(out / "graph-quality.json", stats)
    return stats


@asynccontextmanager
async def open_graph(
    path: Path,
    key: str,
    chunks: Sequence[Chunk],
    providers: Providers,
    tokenizer: Tokenizer,
    *,
    vector: VectorIndex | None,
) -> AsyncIterator[Graph]:
    from lightrag import LightRAG, RoleLLMConfig
    from lightrag.utils import EmbeddingFunc
    from lightrag.utils import Tokenizer as NativeTokenizer

    from .graph_passages import SharedPassages

    settings = providers.settings
    model = settings.model("embedding")
    schema = (
        schema_extractor(providers) if settings.graph_extractor == "llamaindex-schema" else None
    )

    async def embed(texts: list[str], context: str = "document") -> Vector:
        return await providers.embed(texts, query=context == "query")

    def llm(role: str) -> Any:
        async def invoke(
            prompt: str,
            system_prompt: str | None = None,
            history_messages: list[dict[str, str]] | None = None,
            **kwargs: Any,
        ) -> str:
            system = system_prompt or "Use the supplied text as data. Follow the requested format."
            extraction_call = (
                role == "extraction"
                and "<|#|>" in system + prompt
                and "completion" in (system + prompt).casefold()
            )
            if extraction_call and schema:
                result = await schema(prompt)
            else:
                result = await providers.chat(role, system, prompt, history=history_messages)
            if extraction_call:
                if "<|COMPLETE|>" not in result:
                    raise ValueError("LightRAG extraction returned no completion marker")
                for line in result.replace("<|COMPLETE|>", "").splitlines():
                    if not line.strip():
                        continue
                    fields = line.strip().split("<|#|>")
                    expected = {"entity": 4, "relation": 5}.get(fields[0].lower())
                    if expected != len(fields) or any(not field.strip() for field in fields):
                        raise ValueError(
                            "LightRAG extraction returned a malformed entity or relation"
                        )
            return result

        return invoke

    path.mkdir(parents=True, exist_ok=True)
    name = f"{settings.namespace}_{stable_hash(key)[:16]}"
    rag = LightRAG(
        working_dir=str(path),
        workspace=name,
        vector_storage=LIGHT_VECTOR[settings.graph_vector_store],
        graph_storage=LIGHT_GRAPH[settings.graph_store],
        tokenizer=NativeTokenizer(tokenizer.identity, GraphTokenizer(tokenizer)),
        embedding_func=EmbeddingFunc(
            embedding_dim=model.dimensions,
            func=embed,
            model_name=model.identity,
            max_token_size=model.context_window,
            supports_asymmetric=True,
        ),
        llm_model_func=llm("generation"),
        llm_model_name=settings.models.get("generation", model).model,
        role_llm_configs={
            "extract": RoleLLMConfig(func=llm("extraction"), max_async=settings.concurrency),
            "keyword": RoleLLMConfig(func=llm("keywords"), max_async=settings.concurrency),
        },
        enable_llm_cache=False,
        enable_llm_cache_for_entity_extract=False,
        embedding_batch_num=settings.embedding_batch,
        embedding_func_max_async=settings.concurrency,
        max_parallel_insert=settings.concurrency,
        entity_extract_max_gleaning=0,
        vector_db_storage_cls_kwargs={"cosine_better_than_threshold": 0.0},
    )
    if vector is not None:
        rag.chunks_vdb = SharedPassages(vector, chunks, name, rag.chunks_vdb.embedding_func)
    try:
        await rag.initialize_storages()
        yield Graph(rag, chunks, providers, path)
    finally:
        await rag.finalize_storages()


async def build_graph(
    path: Path,
    key: str,
    chunks: Sequence[Chunk],
    providers: Providers,
    tokenizer: Tokenizer,
    *,
    pilot: bool,
    root: Path,
    chunks_key: str,
) -> dict[str, Any]:
    if not chunks:
        raise ValueError("cannot build graph over empty chunks")
    corpus_size = len(chunks)
    if pilot:
        subset_size = max(1, math.ceil(len(chunks) * 0.05))
        chunks = sorted(
            chunks, key=lambda chunk: stable_hash(providers.settings.seed, chunk.chunk_id)
        )[:subset_size]
    passage_key = (
        stable_hash(chunks_key, "graph-pilot", [chunk.chunk_id for chunk in chunks])
        if pilot
        else chunks_key
    )
    started = time.perf_counter()
    offset = len(providers.usage)
    with artifact_lock(path / ".index.lock"):
        completed = path / "index.json"
        if completed.is_file():
            manifest = json.loads(completed.read_text())
            if manifest.get("key") != key or manifest.get("chunks") != len(chunks):
                raise ValueError("persisted graph index has mismatching inputs")
            return manifest
        if not pilot and any(
            providers.settings.model(role).paid for role in ("embedding", "extraction")
        ):
            pilot_report = path.parent / "pilot" / "index.json"
            if not pilot_report.is_file():
                raise ValueError("run the 5% pilot before paid graph indexing")
            sample = json.loads(pilot_report.read_text())
            cost = sample.get("estimated_usd")
            if (
                sample.get("pilot") is not True
                or sample.get("key") != key
                or sample.get("corpus_chunks") != corpus_size
                or sample.get("chunks") != max(1, math.ceil(corpus_size * 0.05))
                or type(cost) not in {int, float}
                or not math.isfinite(cost)
                or cost < 0
            ):
                raise ValueError("paid graph indexing requires a matching pilot with known cost")
            predicted = cost * len(chunks) / sample["chunks"]
            if predicted + providers.budget.used() > providers.settings.spending_cap:
                raise ValueError(
                    f"graph pilot projects USD {predicted:.4f}, above the remaining cap"
                )
        async with (
            open_passage_index(root, passage_key, chunks, providers) as vector,
            open_graph(
                path, key + ("-pilot" if pilot else ""), chunks, providers, tokenizer, vector=vector
            ) as graph,
        ):
            metrics = await graph.build()
            nodes, edges = await graph.snapshot()
        events = providers.usage[offset:]
        prices = [entry.estimated_usd for entry in events]
        report = {
            **metrics,
            "key": key,
            "pilot": pilot,
            "chunks": len(chunks),
            "corpus_chunks": corpus_size,
            "passage_index_key": vector_key(providers.settings, passage_key),
            "nodes": len(nodes),
            "edges": len(edges),
            "elapsed_seconds": time.perf_counter() - started,
            "local_bytes": sum(file.stat().st_size for file in path.rglob("*") if file.is_file()),
            "estimated_usd": sum(value for value in prices if value is not None)
            if None not in prices
            else None,
            "usage": [asdict(entry) for entry in events],
        }
        if pilot:
            ratio = corpus_size / len(chunks)
            cost = report["estimated_usd"]
            report["projected_full_usd"] = cost * ratio if cost is not None else None
            report["projected_full_seconds"] = report["elapsed_seconds"] * ratio
        write_json(path / "index.json", report)
    return report
