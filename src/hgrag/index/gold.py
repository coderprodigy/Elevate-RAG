"""Gold chunk IDs per question, for chunk-level retrieval recall.

* Questions with evidence sentences (MultiHop-RAG, GraphRAG-Bench): the chunks of the gold
  documents that contain an evidence sentence (normalised substring match, else fuzzy
  partial match ≥ ``FUZZY_THRESHOLD``).
* Questions with only gold documents (NQ, HotpotQA, MuSiQue): every chunk of those documents
  (with paragraph chunking that is exactly the gold paragraph).
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from rapidfuzz import fuzz

from hgrag.data.schema import Query
from hgrag.index.chunking import Chunk

FUZZY_THRESHOLD = 90.0
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _norm(text: str) -> str:
    return _NON_ALNUM.sub(" ", text.lower()).strip()


@dataclass(frozen=True)
class GoldMapping:
    gold_chunks: dict[str, list[str]]  # qid → chunk IDs (only questions that have gold)
    evidence_total: int
    evidence_mapped: int

    @property
    def evidence_mapped_rate(self) -> float:
        return self.evidence_mapped / self.evidence_total if self.evidence_total else 1.0


def map_gold_chunks(queries: Iterable[Query], chunks: Sequence[Chunk]) -> GoldMapping:
    by_doc: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        by_doc[chunk.doc_id].append(chunk)
    normalised = {c.chunk_id: _norm(c.text) for c in chunks}

    gold: dict[str, list[str]] = {}
    total = mapped = 0
    for query in queries:
        candidates = [c for doc in query.gold_doc_ids for c in by_doc.get(doc, [])]
        if not candidates:
            continue
        if not query.gold_evidence:
            gold[query.qid] = [c.chunk_id for c in candidates]
            continue
        hits: list[str] = []
        for evidence in query.gold_evidence:
            total += 1
            found = _chunks_containing(_norm(evidence), candidates, normalised)
            mapped += bool(found)
            hits += [cid for cid in found if cid not in hits]
        if hits:
            gold[query.qid] = hits
    return GoldMapping(gold, total, mapped)


def _chunks_containing(
    evidence: str, candidates: Sequence[Chunk], normalised: dict[str, str]
) -> list[str]:
    if not evidence:
        return []
    exact = [c.chunk_id for c in candidates if evidence in normalised[c.chunk_id]]
    if exact:
        return exact
    scored = [
        (fuzz.partial_ratio(evidence, normalised[c.chunk_id]), c.chunk_id) for c in candidates
    ]
    best = max(scored, default=(0.0, ""))
    return [best[1]] if best[0] >= FUZZY_THRESHOLD else []
