import asyncio
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from elevate_rag.data import read_jsonl, write_jsonl
from elevate_rag.experiments import run
from elevate_rag.interop import ReferenceSettings, export_reference, run_reference
from elevate_rag.reference_worker import llamaindex
from elevate_rag.tracing import LocalTracing


def test_framework_launcher_checks_boundaries_and_freezes_execution(
    tmp_path, data_path, monkeypatch
):
    export = tmp_path / "export"
    destination = tmp_path / "outputs"
    options = tmp_path / "worker.toml"
    settings = """
framework_version = "0.14.26"
chat_url = "http://127.0.0.1:8080/v1"
chat_model = "local-chat"
embedding_url = "http://localhost:8081/v1"
embedding_model = "local-embed"
embedding_dimensions = 8
"""
    options.write_text(settings)
    export_reference(
        data_path, export, "dev", framework="llamaindex-property-graph", limit=2, seed=42
    )
    monkeypatch.setenv("OTHER_SECRET", "sensitive-value")

    def launch(args, *, cwd, env, stdout, stderr, timeout):
        assert args[:2] == [sys.executable, "-I"]
        assert Path(args[2]).name == "reference_worker.py"
        assert "OTHER_SECRET" not in env
        assert env["HF_HUB_OFFLINE"] == "1"
        questions = list(read_jsonl(export / "queries.jsonl"))
        assert set(questions[0]) == {"id", "question"}
        write_jsonl(
            cwd / "native-questions.jsonl",
            (
                {
                    "id": row["id"],
                    "status": "ok",
                    "answer": "Local answer",
                    "contexts": ["Evidence"],
                }
                for row in questions
            ),
        )
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(subprocess, "run", launch)
    result = run_reference(
        data_path, export, destination, Path(sys.executable), options, timeout=30
    )
    assert result["framework_version"] == "0.14.26"
    assert result["framework"] == "llamaindex-property-graph"
    assert (destination / "report.md").exists()
    options.write_text(settings.replace("local-chat", "other-chat"))
    with pytest.raises(ValueError, match="different manifest"):
        run_reference(data_path, export, destination, Path(sys.executable), options, timeout=30)
    options.write_text(settings)
    (export / "queries.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="checksum"):
        run_reference(data_path, export, destination, Path(sys.executable), options, timeout=30)


def test_reference_settings_reject_remote_models():
    with pytest.raises(ValueError, match="loopback"):
        ReferenceSettings(
            "1", "https://api.example.test/v1", "chat", "http://localhost", "embed", 8
        )


def test_native_llamaindex_indexes_and_quotes_local_passages(tmp_path, monkeypatch):
    pytest.importorskip("llama_index.core")
    from contextlib import ExitStack

    original = httpx.Client
    calls: list[str] = []

    def handler(request):
        payload = json.loads(request.content)
        calls.append(request.url.path)
        if request.url.path.endswith("/embeddings"):
            vector = [
                item / 255 for item in hashlib.sha256(payload["input"][0].encode()).digest()[:8]
            ]
            return httpx.Response(200, json={"data": [{"embedding": vector}]})
        instruction = payload["messages"][0]["content"].lower()
        if "triplet" in instruction or "knowledge triple" in instruction:
            response = "(Beacon Aster, is maintained by, Station Aster)"
        elif "synonym" in instruction or "keywords" in instruction:
            response = "Station Aster^Beacon Aster"
        else:
            response = "Station Aster maintains Beacon Aster."
        return httpx.Response(200, json={"choices": [{"message": {"content": response}}]})

    class FixtureClient(original):
        def __init__(self, **kwargs):
            kwargs.setdefault("transport", httpx.MockTransport(handler))
            super().__init__(**kwargs)

    monkeypatch.setattr(httpx, "Client", FixtureClient)
    config = {
        "chat_url": "http://localhost:8080/v1",
        "chat_model": "local-chat",
        "embedding_url": "http://localhost:8081/v1",
        "embedding_model": "local-embed",
        "embedding_dimensions": 8,
        "top_k": 5,
        "context_window": 4096,
        "max_output": 128,
        "request_timeout": 5,
    }
    with ExitStack() as stack:
        query = llamaindex(
            [{"id": "d1", "text": "Station Aster maintains Beacon Aster."}], config, tmp_path, stack
        )
        answer, contexts = query("Which beacon belongs to Station Aster?")
    assert "Beacon Aster" in answer
    assert any("/embeddings" in call for call in calls)
    assert any("Station Aster" in context for context in contexts)


def test_langfuse_exports_metadata_and_redacts_exception(
    tmp_path, data_path, settings, monkeypatch
):
    langfuse = pytest.importorskip("langfuse")
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    client = langfuse.Langfuse
    monkeypatch.setattr(
        langfuse, "Langfuse", lambda **kwargs: client(span_exporter=exporter, **kwargs)
    )
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-local")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-local")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
    with pytest.raises(ValueError, match="loopback"):
        LocalTracing(True)
    monkeypatch.setenv("LANGFUSE_BASE_URL", "http://127.0.0.1:3030")
    folder = asyncio.run(
        run(
            data_path,
            tmp_path / "indexes",
            tmp_path / "results",
            replace(settings, langfuse_tracing=True),
            mode="V1",
            split="dev",
            limit=1,
        )
    )
    assert (folder / "questions.jsonl").exists()
    observations = exporter.get_finished_spans()
    assert {
        "experiment-question",
        "route",
        "retrieve",
        "rerank",
        "assemble-context",
        "generate",
    } <= {observation.name for observation in observations}
    assert all(span.attributes is not None for span in observations)
    payload = str([dict(span.attributes or {}) for span in observations])
    assert "Which beacon belongs" not in payload
    trace = LocalTracing(True)
    with pytest.raises(ValueError, match="sensitive"):
        with trace.observe("guarded", {"stage": "evaluation"}):
            raise ValueError("sensitive payload")
    trace.flush()
    last = exporter.get_finished_spans()[-1]
    assert last.name == "guarded"
    assert "ValueError" in str(dict(last.attributes or {}))
    assert "sensitive payload" not in str(dict(last.attributes or {}))


def test_external_outcomes_can_be_graded_with_the_independent_judge(
    tmp_path,
    data_path,
    settings,
    monkeypatch,
):
    from elevate_rag.config import Model
    from elevate_rag.interop import grade_reference, import_reference
    from elevate_rag.providers import Cache, Providers

    export = tmp_path / "export"
    manifest = export_reference(
        data_path, export, "dev", framework="microsoft-graphrag", limit=None, seed=42
    )
    raw = tmp_path / "native.jsonl"
    write_jsonl(
        raw,
        (
            {"id": key, "status": "ok", "answer": "The external answer", "contexts": ["Evidence"]}
            for key in manifest["qids"]
        ),
    )
    ungraded, scored = tmp_path / "native", tmp_path / "scored"
    import_reference(data_path, export, raw, "3.3.0", ungraded)
    configured = replace(
        settings, models={**settings.models, "judge": Model("fixture", "independent", "1")}
    )

    async def check():
        async with Providers(
            configured, Cache(tmp_path / "cache"), tmp_path / "budget.sqlite3"
        ) as providers:

            async def judge(role, system, user):
                assert role == "judge" and json.loads(user)["references"]
                return '{"grade":"partial"}'

            monkeypatch.setattr(providers, "chat", judge)
            result = await grade_reference(data_path, ungraded, scored, providers)
            assert result == {"questions": 4, "scored": 4}

    asyncio.run(check())
    originals = list(read_jsonl(ungraded / "questions.jsonl"))
    rows = list(read_jsonl(scored / "questions.jsonl"))
    assert all(row["correct"] is None for row in originals)
    assert any(row["metrics"]["judge_score"] == 0.5 for row in rows)
    assert all(row["correct"] is False for row in rows)
