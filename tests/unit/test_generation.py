from __future__ import annotations

import pytest

from elevate_rag.generation.context import assemble_context
from elevate_rag.generation.generator import (
    SYSTEM_RAG,
    Generator,
    build_prompt,
    is_refusal,
    parse_citations,
)
from elevate_rag.providers.llm import extractive_answer
from elevate_rag.types import ContextItem
from elevate_rag.utils import Tokenizer

TOK = Tokenizer()


def _chunk(i: int, words: int = 50) -> ContextItem:
    return ContextItem(
        item_id=f"d{i}::c0000",
        kind="chunk",
        text=" ".join(["word"] * words),
        score=1 - i / 100,
        doc_id=f"d{i}",
    )


def _ent(i: int, words: int = 50) -> ContextItem:
    return ContextItem(
        item_id=f"entity:E{i}",
        kind="entity",
        text=" ".join(["ent"] * words),
        source_chunk_ids=[f"d{i}::c0000"],
    )


def test_budget_is_never_exceeded_and_final_k_applies() -> None:
    items = [_chunk(i, 300) for i in range(20)]
    ctx = assemble_context(items, budget=1000, final_k=8, kg_share=0.5, tokenizer=TOK)
    assert ctx.tokens <= 1000
    assert sum(TOK.count(i.text) for i in ctx.items) <= 1000
    assert len(ctx.items) <= 8 and ctx.truncated == 1


def test_kg_share_caps_graph_items_and_unused_budget_flows_to_chunks() -> None:
    kg = [_ent(i, 200) for i in range(10)]
    chunks = [_chunk(i, 200) for i in range(10)]
    ctx = assemble_context(kg + chunks, budget=1000, final_k=10, kg_share=0.3, tokenizer=TOK)
    kg_tokens = sum(TOK.count(i.text) for i in ctx.items if i.kind != "chunk")
    assert kg_tokens <= 300 and ctx.tokens <= 1000
    small_kg = assemble_context(
        [_ent(0, 10), *chunks], budget=1000, final_k=10, kg_share=0.5, tokenizer=TOK
    )
    chunk_tokens = sum(TOK.count(i.text) for i in small_kg.items if i.kind == "chunk")
    assert chunk_tokens > 500  # unused KG share went to chunks


def test_kg_only_modes_get_full_budget() -> None:
    ctx = assemble_context(
        [_ent(i, 100) for i in range(30)], budget=800, final_k=5, kg_share=0.25, tokenizer=TOK
    )
    assert 700 < ctx.tokens <= 800


def test_citations_and_refusals() -> None:
    assert parse_citations("A [1] and [3][9]", 3) == ([1, 3], [9])
    assert is_refusal("Insufficient information.")
    assert is_refusal("insufficient information")
    assert not is_refusal("Kessering [1]")


def test_prompt_marks_passages_untrusted_and_closed_book_has_none() -> None:
    system, user = build_prompt("Q?", [_chunk(1, 3)])
    assert system == SYSTEM_RAG and "untrusted" in system and '<passage id="1"' in user
    system0, user0 = build_prompt("Q?", None)
    assert "passage" not in user0 and "own knowledge" in system0
    _, empty = build_prompt("Q?", [])
    assert "no passages" in empty


async def test_generator_maps_citations_to_chunks(scripted) -> None:
    items = [_chunk(1, 3), _ent(2, 3)]
    gen = Generator(scripted(["Answer [1][2][7]"]))
    ans = await gen.answer("Q?", items, context_tokens=6)
    assert ans.citations == [1, 2] and ans.invalid_citations == [7]
    assert ans.cited_chunk_ids == ["d1::c0000", "d2::c0000"]  # entity → its source chunk
    assert not ans.refused


@pytest.mark.parametrize(
    ("question", "expected_refusal"),
    [
        ("In which year was Velmora founded?", False),
        ("What is the gravitational constant on Jupiter's third moon?", True),
    ],
)
def test_extractive_offline_answer(question: str, expected_refusal: bool) -> None:
    out = extractive_answer(question, ["Velmora was founded in 1712 by Ilse Brandt."])
    assert is_refusal(out) is expected_refusal
    if not expected_refusal:
        assert "1712" in out and out.endswith("[1]")
