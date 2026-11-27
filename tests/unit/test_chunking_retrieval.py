from __future__ import annotations

import itertools

import pytest
from pydantic import ValidationError

from elevate_rag.config import ChunkingConfig
from elevate_rag.indexing.chunk_store import ChunkStore
from elevate_rag.indexing.chunker import Chunker, chunk_id
from elevate_rag.retrieval.lexical import BM25Index, reciprocal_rank_fusion
from elevate_rag.types import Document
from elevate_rag.utils import DataError, Tokenizer

LONG = " ".join(
    f"Sentence number {i} talks about topic {i % 7} in some detail." for i in range(400)
)


def _doc(text: str = LONG, title: str = "T", doc_id: str = "d1") -> Document:
    return Document(doc_id=doc_id, title=title, text=text, source="test")


@pytest.mark.parametrize("size", [64, 256, 512])
def test_chunks_respect_token_cap_and_are_deterministic(size: int) -> None:
    cfg = ChunkingConfig(chunk_tokens=size, overlap_tokens=size // 8)
    a = Chunker(cfg).chunk_document(_doc())
    b = Chunker(cfg).chunk_document(_doc())
    assert [c.model_dump() for c in a] == [c.model_dump() for c in b]
    assert all(c.n_tokens <= size for c in a)
    assert [c.chunk_id for c in a] == [chunk_id("d1", i) for i in range(len(a))]
    assert len(a) > 1


def test_overlap_repeats_trailing_sentences() -> None:
    cfg = ChunkingConfig(chunk_tokens=64, overlap_tokens=20, prepend_title=False)
    chunks = Chunker(cfg).chunk_document(_doc())
    tok = Tokenizer()
    for prev, nxt in itertools.pairwise(chunks):
        assert nxt.start_char < prev.end_char  # overlapping span
        overlap = _doc().text[nxt.start_char : prev.end_char]
        assert tok.count(overlap) <= 20 + 2


def test_offsets_map_back_to_source_and_sentence_boundaries() -> None:
    cfg = ChunkingConfig(chunk_tokens=64, overlap_tokens=0, prepend_title=False)
    d = _doc()
    for c in Chunker(cfg).chunk_document(d):
        assert c.text == d.text[c.start_char : c.end_char].strip()
        assert c.text.endswith(".")  # never cut inside a sentence


def test_title_is_prepended_and_counted() -> None:
    cfg = ChunkingConfig(chunk_tokens=64, overlap_tokens=0)
    chunks = Chunker(cfg).chunk_document(_doc(title="Velmora"))
    assert all(c.text.startswith("Velmora\n") and c.n_tokens <= 64 for c in chunks)


def test_overlong_sentence_is_hard_split() -> None:
    words = " ".join(f"w{i}" for i in range(2000))  # one sentence, no punctuation
    chunks = Chunker(ChunkingConfig(chunk_tokens=128, overlap_tokens=0)).chunk_document(_doc(words))
    assert len(chunks) > 5 and all(c.n_tokens <= 128 for c in chunks)


def test_paragraph_strategy_one_chunk_per_paragraph() -> None:
    text = "First para. Still first.\n\nSecond para here.\n\nThird."
    chunks = Chunker(ChunkingConfig(strategy="paragraph", prepend_title=False)).chunk_document(
        _doc(text)
    )
    assert [c.text for c in chunks] == ["First para. Still first.", "Second para here.", "Third."]


def test_overlap_must_be_smaller_than_chunk() -> None:
    with pytest.raises(ValidationError):
        ChunkingConfig(chunk_tokens=64, overlap_tokens=64)


def test_empty_document_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _doc("   ")


def test_chunk_store_roundtrip_and_duplicates(tmp_path) -> None:
    chunks = Chunker(ChunkingConfig(chunk_tokens=64, overlap_tokens=0)).chunk_document(_doc())
    store = ChunkStore(chunks)
    store.save(tmp_path / "c.jsonl")
    again = ChunkStore.load(tmp_path / "c.jsonl")
    assert again.ids == store.ids and again.doc_chunks("d1")[0] == chunks[0]
    with pytest.raises(DataError):
        ChunkStore([chunks[0], chunks[0]])
    with pytest.raises(DataError):
        ChunkStore.load(tmp_path / "missing.jsonl")


# ------------------------------------------------------------------------------- BM25 / RRF


def test_bm25_ranks_matching_document_first_and_is_deterministic() -> None:
    ids = ["a", "b", "c", "d"]
    texts = [
        "the amber harbour of Velmora",
        "software in Kessering",
        "tidal turbines in the harbour",
        "nothing relevant",
    ]
    idx = BM25Index().build(ids, texts)
    hits = idx.search("amber harbour", 4)
    assert hits[0][0] == "a"
    assert hits == idx.search("amber harbour", 4)
    assert "d" not in [h[0] for h in hits]  # zero-score docs are not returned


def test_bm25_ties_broken_by_id_and_empty_index() -> None:
    idx = BM25Index().build(["z", "y"], ["same text", "same text"])
    assert [h[0] for h in idx.search("same", 2)] == ["y", "z"]
    assert BM25Index().build([], []).search("x", 5) == []
    with pytest.raises(ValueError):
        BM25Index().build(["a"], [])


def test_rrf_matches_formula() -> None:
    fused = reciprocal_rank_fusion([[("a", 9), ("b", 8)], [("b", 1.0), ("c", 0.5)]], k=60)
    scores = dict(fused)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["a"] == pytest.approx(1 / 61)
    assert [d for d, _ in fused] == ["b", "a", "c"]
    assert len(reciprocal_rank_fusion([[("a", 1)], [("b", 1)]], limit=1)) == 1
