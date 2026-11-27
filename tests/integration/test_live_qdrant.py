"""Opt-in integration check against a dedicated localhost Qdrant service."""

import asyncio
import os
import uuid
from dataclasses import replace

import httpx
import pytest

from elevate_rag.graph import build_graph, graph_key
from elevate_rag.pipeline import index_parts, open_pipeline
from elevate_rag.providers import Cache, Providers
from elevate_rag.reference_worker import loopback


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("ELEVATE_LIVE_QDRANT") != "1", reason="dedicated local Qdrant is opt-in"
)
def test_qdrant_native_graph_and_dense_share_one_passage_collection(
    tmp_path,
    data_path,
    settings,
):
    endpoint = loopback(os.environ["QDRANT_URL"], {"http", "https"})
    namespace = "elevate_live_" + uuid.uuid4().hex[:10]
    graph_backend = os.environ.get("ELEVATE_LIVE_GRAPH_STORE", "networkx")
    if graph_backend == "neo4j":
        loopback(os.environ["NEO4J_URI"], {"bolt", "bolt+s", "bolt+ssc"})
    elif graph_backend == "memgraph":
        loopback(os.environ["MEMGRAPH_URI"], {"bolt", "bolt+s", "bolt+ssc"})
    else:
        assert graph_backend == "networkx"
    settings = replace(
        settings,
        namespace=namespace,
        vector_store="qdrant",
        graph_vector_store="qdrant",
        graph_store=graph_backend,
    )
    query = "Which beacon belongs to Station Aster?"

    async def check():
        async with Providers(
            settings, Cache(tmp_path / "cache"), tmp_path / "budget.sqlite3"
        ) as providers:
            catalog, chunks_key, _ = index_parts(data_path, tmp_path, providers)
            async with open_pipeline(data_path, tmp_path, providers, ["V2"]) as pipeline:
                before, _ = await pipeline.retrieve(query, "V2")
                assert before and pipeline.vector is not None
                [first] = pipeline.vector.get_vectors([before[0].chunk_id]).values()
                assert len(first) == settings.model("embedding").dimensions
                chunk = catalog[before[0].chunk_id]
                embedding = (await providers.embed([query], query=True))[0]
                assert before[0].chunk_id in {
                    hit.chunk_id
                    for hit in pipeline.vector.search(embedding, 10, source=chunk.source)
                }
            graph = graph_key(settings, chunks_key)
            report = await build_graph(
                tmp_path / "graphs" / graph / "full",
                graph,
                list(catalog.values()),
                providers,
                providers.tokenizer(context=True),
                pilot=False,
                root=tmp_path,
                chunks_key=chunks_key,
            )
            assert report["nodes"] == 40 and report["edges"] == 20
            async with open_pipeline(data_path, tmp_path, providers, ["V2", "mix"]) as pipeline:
                answer = await pipeline.answer(query, mode="mix")
                assert answer.answer.text == "Beacon Aster [1]"
                assert pipeline.graph is not None
                assert pipeline.graph.rag.chunks_vdb.vector is pipeline.vector
                assert pipeline.vector is not None
                import_url = f"{endpoint}/collections/{pipeline.vector.name}"
                result = httpx.get(import_url, timeout=5)
                result.raise_for_status()
                collection = result.json()["result"]
                assert collection["points_count"] == len(catalog)
                assert collection["config"]["hnsw_config"]["m"] == settings.hnsw_m
                assert (
                    collection["config"]["hnsw_config"]["ef_construct"] == settings.ef_construction
                )
                collections = httpx.get(f"{endpoint}/collections", timeout=5)
                collections.raise_for_status()
                names = [row["name"] for row in collections.json()["result"]["collections"]]
                assert sum(name.startswith(namespace) for name in names) == 1
                assert all("lightrag_vdb_chunks_" not in name for name in names)

    asyncio.run(check())
