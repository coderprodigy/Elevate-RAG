import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from elevate_rag.config import Model, Settings
from elevate_rag.data import (
    CharacterTokenizer,
    Chunk,
    Document,
    Evidence,
    Question,
    file_hash,
    prepare,
    split_questions,
)
from elevate_rag.evaluation import passages_by_document, retrieval_metrics
from elevate_rag.experiments import plan_run
from elevate_rag.graph import graph_key, inspect_graph
from elevate_rag.interop import export_reference, validate_reference_export
from elevate_rag.pipeline import assemble
from elevate_rag.providers import Budget, BudgetExceeded, Cache, ProviderError, Providers
from elevate_rag.reference_worker import checkpoint_ids
from elevate_rag.retrieval import Hit
from elevate_rag.routing import SVM


def test_split_keeps_duplicate_queries_together_and_preserves_total():
    singles = [Question(str(idx), f"Question {idx}?") for idx in range(20)]
    dev, heldout = split_questions(singles, 42)
    questions = [replace(q, text=dev[0].text.upper()) if q == heldout[0] else q for q in singles]
    new_dev, new_heldout = split_questions(questions, 42)
    normalized_dev = {q.text.casefold() for q in new_dev}
    assert normalized_dev.isdisjoint(q.text.casefold() for q in new_heldout)
    assert len(new_dev) == 4
    assert split_questions(reversed(questions), 42) == (new_dev, new_heldout)


def test_ranked_metrics_deduplicate_hits_and_preserve_indexed_scoring():
    good = Chunk("good", "doc", "Title", "substantial evidence", "src", 0, 20)
    decoy = replace(good, chunk_id="decoy", doc_id="other")
    catalog = {chunk.chunk_id: chunk for chunk in (good, decoy)}
    evidence = [Evidence("doc", 0, 8)]
    unique = retrieval_metrics(evidence, [Hit("good", 1), Hit("decoy", 0)], catalog, [])
    duplicate = retrieval_metrics(
        evidence,
        [Hit("good", 1), Hit("good", 0.5), Hit("decoy", 0)],
        catalog,
        [],
        by_document=passages_by_document(catalog),
    )
    assert duplicate == unique
    assert duplicate["ndcg_at_10"] == 1.0


def test_context_budget_uses_serialized_chat_token_count():
    class TemplateTokenizer(CharacterTokenizer):
        def count_chat(self, system, user):
            return len(user) + 350

    chunk = Chunk("c", "d", "title", "evidence", "s", 0, 8)
    args = {"context_budget": 400, "context_window": 440, "output_limit": 21}
    context = assemble("Who?", [Hit("c", 1)], {"c": chunk}, TemplateTokenizer(), **args)
    assert not context.passages
    assert context.prompt_tokens == len(context.user_prompt) + 350
    with pytest.raises(ValueError, match="exceed"):
        assemble("x" * 100, [], {}, TemplateTokenizer(), **args)


def test_boolean_questions_do_not_dilute_entity_coverage(tmp_path):
    questions = [Question("named", "Who?", ("Ana",)), Question("bool", "Did she?", ("yes",))]
    metrics = inspect_graph([{"id": "ANA", "entity_type": "PERSON"}], [], questions, 1, tmp_path)
    assert metrics["development_questions"] == 2
    assert metrics["coverage_questions"] == 1
    assert metrics["answer_entity_coverage"] == 1


def test_generator_tier_does_not_change_shared_graph_identity():
    reference = Settings.read(Path("configs/reference.toml"))
    changed = replace(reference, context_tokenizer_model="different-chat-tokenizer")
    assert graph_key(reference, "shared-corpus") == graph_key(changed, "shared-corpus")
    other = replace(reference, chunk_tokenizer_model="different-chunk-tokenizer")
    assert graph_key(reference, "shared-corpus") != graph_key(other, "shared-corpus")


def test_paid_reservations_are_serialized_across_clients(tmp_path):
    path = tmp_path / "budget.sqlite3"
    clients = [Budget(path, 1.0) for _ in range(8)]

    def reserve(client):
        try:
            return client.reserve(0.6)
        except BudgetExceeded:
            return None

    with ThreadPoolExecutor(8) as workers:
        keys = [key for key in workers.map(reserve, clients) if key is not None]
    assert len(keys) == 1
    assert clients[0].used() == pytest.approx(0.6)
    clients[0].settle(keys[0], 0.1)
    assert clients[1].reserve(0.6) > keys[0]
    with pytest.raises(ValueError, match="nonnegative"):
        clients[0].reserve(float("nan"))


def test_cache_rejects_nonobject_and_tampered_value(tmp_path):
    cache = Cache(tmp_path)
    cache.put("retrieval", "key", {"hits": []})
    file = tmp_path / "retrieval" / "key.json"
    original = json.loads(file.read_text())
    for data in ([], {**original, "value": {"hits": [1]}}):
        file.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="checksum"):
            cache.get("retrieval", "key")


@pytest.mark.parametrize("bad_index", [True, "0", -1])
def test_embedding_rejects_invalid_provider_indices(tmp_path, settings, bad_index):
    embedding = Model(
        "openai-compatible", "loopback", "frozen", base_url="http://localhost/v1", dimensions=2
    )
    settings = replace(settings, models={**settings.models, "embedding": embedding})

    async def check():
        async with Providers(settings, Cache(tmp_path / "cache"), tmp_path / "budget") as providers:
            await providers.client.aclose()
            providers.client = httpx.AsyncClient(
                transport=httpx.MockTransport(
                    lambda _: httpx.Response(
                        200, json={"data": [{"index": bad_index, "embedding": [1, 0]}]}
                    )
                )
            )
            with pytest.raises(ProviderError, match="indexes"):
                await providers.embed(["x"])

    asyncio.run(check())


def test_embedding_batches_deduplicate_and_keep_order(tmp_path, settings):
    calls = []

    def request(req):
        values = json.loads(req.content)["input"]
        calls.extend(values)
        rows = [
            {"index": i, "embedding": [1, 0] if val == "a" else [0, 1]}
            for i, val in enumerate(values)
        ]
        return httpx.Response(200, json={"data": rows[::-1]})

    embedding = Model(
        "openai-compatible", "loopback", "frozen", base_url="http://localhost/v1", dimensions=2
    )
    settings = replace(settings, models={**settings.models, "embedding": embedding})

    async def check():
        async with Providers(settings, Cache(tmp_path / "cache"), tmp_path / "budget") as providers:
            await providers.client.aclose()
            providers.client = httpx.AsyncClient(transport=httpx.MockTransport(request))
            assert (await providers.embed(["a", "b", "a"])).tolist() == [[1, 0], [0, 1], [1, 0]]
            assert (await providers.embed(["b", "a"])).tolist() == [[0, 1], [1, 0]]

    asyncio.run(check())
    assert calls == ["a", "b"]


def test_native_checkpoint_repairs_torn_tail_but_rejects_duplicate(tmp_path):
    checkpoint = tmp_path / "native-questions.jsonl"
    committed = b'{"id":"first","status":"ok"}\n'
    checkpoint.write_bytes(committed + b'{"id":"second","stat')
    assert checkpoint_ids(checkpoint, {"first", "second"}) == {"first"}
    assert checkpoint.read_bytes() == committed
    checkpoint.write_bytes(committed * 2)
    with pytest.raises(ValueError, match="question ID"):
        checkpoint_ids(checkpoint, {"first", "second"})


def test_native_export_rejects_rehashed_question_tampering(data_path, tmp_path):
    out = tmp_path / "external"
    manifest = export_reference(
        data_path, out, "dev", framework="microsoft-graphrag", limit=1, seed=0
    )
    path = out / "queries.jsonl"
    row = json.loads(path.read_text())
    path.write_text(json.dumps({**row, "question": "different text"}) + "\n")
    manifest["inputs"][path.name] = file_hash(path)
    (out / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="benchmark split and text"):
        validate_reference_export(data_path, out)


def test_real_run_rejects_router_from_overlapping_benchmark(tmp_path):
    data = tmp_path / "corpus"
    manifest = prepare(
        data,
        [Document("doc", "Title", "Evidence", "source")],
        [Question(str(i), f"Question {i}?") for i in range(10)],
        dataset="musique",
        version="test",
        seed=42,
        sources={},
    )
    settings = Settings.read(Path("configs/reference.toml"))
    model = SVM.train(
        [("q1", "Short factual answer", "single-hop"), ("q2", "Long thematic summary", "summary")],
        "query-type",
    )
    args = {"mode": "V4", "router": "R2", "split": "heldout", "limit": 1, "uncached": False}
    dev_hash = manifest["files"]["dev.jsonl"]
    with pytest.raises(ValueError, match="current development split"):
        plan_run(
            data,
            settings,
            svm=replace(model, training_dataset="ragrouter-bench", training_dev_sha256=dev_hash),
            **args,
        )
    valid = replace(model, training_dataset="musique", training_dev_sha256=dev_hash)
    assert len(plan_run(data, settings, svm=valid, **args)[1]) == 1


def test_one_class_development_router_round_trips_as_a_constant(tmp_path):
    model = SVM.train([("a", "Who wrote this?", "single-hop")], "query-type")
    output = tmp_path / "router.json"
    model.save(output)
    restored = SVM.read(output)
    assert restored.labels == ["single-hop"]
    assert restored.tokens == restored.weights == []
    assert restored.route("Summarize the corpus") == "V4"
