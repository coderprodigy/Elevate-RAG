"""Opt-in regression test for pgvector and LightRAG's PostgreSQL graph backend."""

import asyncio
import os
import uuid
from dataclasses import replace

import pytest

from elevate_rag.graph import build_graph, graph_key
from elevate_rag.pipeline import index_parts, open_pipeline
from elevate_rag.providers import Cache, Providers


@pytest.mark.integration
@pytest.mark.parametrize(
    ("vector_backend", "graph_backend"),
    [
        pytest.param(
            "pgvector",
            "postgres",
            marks=pytest.mark.skipif(
                os.environ.get("ELEVATE_LIVE_POSTGRES") != "1",
                reason="dedicated local PostgreSQL is opt-in",
            ),
        ),
        pytest.param(
            "milvus",
            "networkx",
            marks=pytest.mark.skipif(
                os.environ.get("ELEVATE_LIVE_MILVUS") != "1",
                reason="dedicated local Milvus is opt-in",
            ),
        ),
    ],
)
def test_database_graph_and_vector_share_passages(
    tmp_path, data_path, settings, vector_backend, graph_backend
):
    if vector_backend == "pgvector":
        from psycopg.conninfo import conninfo_to_dict

        dsn = conninfo_to_dict(os.environ["ELEVATE_PGVECTOR_DSN"])
        assert dsn.get("host") in {"localhost", "127.0.0.1", "::1"}
        assert os.environ.get("POSTGRES_HOST") in {"localhost", "127.0.0.1", "::1"}
    else:
        from elevate_rag.reference_worker import loopback

        loopback(os.environ["MILVUS_URI"], {"http", "https"})
    settings = replace(
        settings,
        namespace="elevate_live_" + uuid.uuid4().hex[:10],
        vector_store=vector_backend,
        graph_vector_store=vector_backend,
        graph_store=graph_backend,
    )
    query = "Which beacon belongs to Station Aster?"

    async def check():
        async with Providers(
            settings, Cache(tmp_path / "cache"), tmp_path / "budget.sqlite3"
        ) as providers:
            catalog, chunks_key, _ = index_parts(data_path, tmp_path, providers)
            async with open_pipeline(data_path, tmp_path, providers, ["V2"]) as pipeline:
                hits, _ = await pipeline.retrieve(query, "V2")
                assert hits and pipeline.vector is not None
                first = catalog[hits[0].chunk_id]
                vector = (await providers.embed([query], query=True))[0]
                assert first.chunk_id in {
                    hit.chunk_id for hit in pipeline.vector.search(vector, 10, source=first.source)
                }
                assert pipeline.vector.size() == len(catalog)
            key = graph_key(settings, chunks_key)
            result = await build_graph(
                tmp_path / "graphs" / key / "full",
                key,
                list(catalog.values()),
                providers,
                providers.tokenizer(context=True),
                pilot=False,
                root=tmp_path,
                chunks_key=chunks_key,
            )
            assert result["edges"] == 20 and result["nodes"] == 40
            async with open_pipeline(data_path, tmp_path, providers, ["V2", "mix"]) as pipeline:
                assert (
                    pipeline.graph is not None
                    and pipeline.graph.rag.chunks_vdb.vector is pipeline.vector
                )
                result = await pipeline.answer(query, mode="mix")
                assert result.answer.text == "Beacon Aster [1]"

    asyncio.run(check())
