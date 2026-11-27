import asyncio
import json
from dataclasses import replace

import pytest

from elevate_rag.config import Model
from elevate_rag.data import Question
from elevate_rag.pipeline import Answer, Context, Output, Passage
from elevate_rag.providers import Cache, Providers
from elevate_rag.routing import Route


def test_budgeted_ragas_sdk_with_mocked_judge_response(tmp_path, settings, monkeypatch):
    monkeypatch.setenv("RAGAS_DO_NOT_TRACK", "true")
    pytest.importorskip("ragas")
    from elevate_rag.ragas_metrics import score_ragas

    judge = Model(
        "openai-compatible",
        "test-judge",
        "1",
        base_url="http://127.0.0.1:8080/v1",
        family="judge-family",
    )
    configuration = replace(
        settings,
        models={
            **settings.models,
            "judge": judge,
            "generation": replace(settings.models["generation"], family="fixture-family"),
        },
    )
    snippet = Passage(1, "c", "d", "Title", "source", "Aster is a station.", 0, 19)
    context = Context([snippet], "query", 10, 20)
    answer = Output(
        Answer("Aster is a station [1]", False, [snippet]),
        Route("R0", "V4", "V4", "fixed"),
        [],
        [],
        context,
        {},
        [],
        0,
    )

    async def check():
        async with Providers(
            configuration, Cache(tmp_path / "cache"), tmp_path / "budget.sqlite3"
        ) as providers:
            calls = []

            async def request(role, model, path, payload):
                assert role == "judge"
                assert path == "chat/completions"
                assert payload["temperature"] == 0
                assert payload["max_tokens"] == judge.max_output
                bodies = [
                    {"reason": "supported", "verdict": 1},
                    {
                        "classifications": [
                            {
                                "statement": "Aster is a station.",
                                "reason": "supported",
                                "attributed": 1,
                            }
                        ]
                    },
                    {"statements": ["Aster is a station."]},
                    {
                        "statements": [
                            {
                                "statement": "Aster is a station.",
                                "reason": "supported",
                                "verdict": 1,
                            }
                        ]
                    },
                ]
                value = bodies[len(calls)]
                calls.append(payload)
                return {
                    "id": f"judge-{len(calls)}",
                    "object": "chat.completion",
                    "created": 0,
                    "model": model.model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": json.dumps(value)},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
                }, 1

            monkeypatch.setattr(providers, "request", request)
            scores = await score_ragas(
                Question("q", "What is Aster?", ("A station",)), answer, providers
            )
            assert scores == pytest.approx(
                {"context_precision": 1, "context_recall": 1, "faithfulness": 1}
            )
            assert len(calls) == 4
            assert all(
                event.role == "judge" and event.provider_reported for event in providers.usage
            )

    asyncio.run(check())


def test_llamaindex_schema_adapter_keeps_supported_edges(tmp_path, settings, monkeypatch):
    pytest.importorskip("llama_index.core")
    from elevate_rag.schema_extraction import chunk_from_extraction_prompt, schema_extractor

    prompt = "instructions\n---Input Text---\n```\nStation Aster has Beacon Aster.\n```\n\n---Output---\n"
    assert chunk_from_extraction_prompt(prompt) == "Station Aster has Beacon Aster."
    with pytest.raises(ValueError, match="changed"):
        chunk_from_extraction_prompt("no passage")

    async def check():
        async with Providers(
            settings, Cache(tmp_path / "cache"), tmp_path / "budget.sqlite3"
        ) as providers:

            async def chat(role, system, user, **kwargs):
                assert role == "extraction"
                data = json.loads(user)
                assert "Station Aster" in data["text"]
                return json.dumps(
                    {
                        "triplets": [
                            {
                                "subject": {
                                    "type": "ARTIFACT",
                                    "name": "Station Aster",
                                    "properties": {"description": "A station"},
                                },
                                "relation": {
                                    "type": "ASSOCIATED_WITH",
                                    "properties": {
                                        "description": "Aster station has its own beacon"
                                    },
                                },
                                "object": {
                                    "type": "ARTIFACT",
                                    "name": "Beacon Aster",
                                    "properties": {"description": "A beacon"},
                                },
                            }
                        ]
                    }
                )

            monkeypatch.setattr(providers, "chat", chat)
            extracted = await schema_extractor(providers)(prompt)
            assert "entity<|#|>Station Aster<|#|>artifact<|#|>A station" in extracted
            assert "relation<|#|>Station Aster<|#|>Beacon Aster<|#|>associated with" in extracted
            assert extracted.endswith("<|COMPLETE|>")

    asyncio.run(check())


def test_community_detection_and_report_candidates(tmp_path, settings):
    pytest.importorskip("igraph")
    from elevate_rag.communities import build_communities, community_candidates
    from elevate_rag.data import read_jsonl

    async def check():
        nodes = [
            {"id": name, "description": f"Station {name}", "chunk_ids": [f"chunk-{name}"]}
            for name in ("Aster", "Birch", "Cedar", "Dahlia")
        ]
        edges = [{"source": "Aster", "target": "Birch"}, {"source": "Cedar", "target": "Dahlia"}]
        async with Providers(
            settings, Cache(tmp_path / "cache"), tmp_path / "budget.sqlite3"
        ) as providers:
            path = tmp_path / "reports.jsonl"
            assert await build_communities(nodes, edges, providers, path) == 2
            reports = list(read_jsonl(path))
            hits, identifiers = await community_candidates("Station Aster", reports, providers)
            assert len(identifiers) == 2
            assert {hit.chunk_id for hit in hits} == {f"chunk-{node['id']}" for node in nodes}

    asyncio.run(check())
