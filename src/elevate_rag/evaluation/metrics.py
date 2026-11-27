"""Deterministic evaluation metrics with explicit definitions (FR-E1, FR-E4, FR-E6, FR-E12).

Definitions
-----------
* Answer normalisation: SQuAD v1.1 (lower-case, strip punctuation and articles, collapse spaces).
* EM: max over reference aliases of normalised equality. F1: max over aliases of token F1.
* Recall@k (documents): |gold docs ∩ docs of the top-k retrieved chunks| / |gold docs|.
  Undefined (None) when a question has no gold docs — such questions are excluded from the mean
  and the count of excluded questions is reported, never scored as 0.
* nDCG@10: binary relevance at *document* level; a document counts once, at its first rank.
* Evidence recall@k: share of gold evidence strings found in the top-k chunks (normalised
  containment, or ≥ 0.8 token recall for evidence longer than 12 tokens).
* Refusal: answer text classified as "insufficient information" (generation.is_refusal).
* Failed queries (pipeline error) count as wrong for answer metrics and as zero recall, and are
  reported separately as `error_rate`.
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from collections.abc import Sequence

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize_answer(s: str) -> str:
    s = s.lower().translate(_PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def exact_match(prediction: str, references: Sequence[str]) -> float:
    p = normalize_answer(prediction)
    return float(any(p == normalize_answer(r) for r in references)) if references else 0.0


def token_f1(prediction: str, reference: str) -> float:
    p = normalize_answer(prediction).split()
    r = normalize_answer(reference).split()
    if not p or not r:
        return float(p == r)
    common = Counter(p) & Counter(r)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(r)
    return 2 * precision * recall / (precision + recall)


def max_f1(prediction: str, references: Sequence[str]) -> float:
    return max((token_f1(prediction, r) for r in references), default=0.0)


def short_answer(text: str) -> str:
    """The generator is asked for 'short answer first'; take the first line/sentence and drop
    citation markers so EM/F1 are not penalised for the justification."""
    first = re.split(r"\n|(?<=[.!?])\s", text.strip(), maxsplit=1)[0]
    return re.sub(r"\[\d+\]", "", first).strip()


def docs_of(chunk_ids: Sequence[str]) -> list[str]:
    """Chunk IDs are `<doc_id>::c<n>` (indexing.chunker.chunk_id)."""
    return list(dict.fromkeys(c.split("::c")[0] for c in chunk_ids))


def recall_at_k(
    retrieved_chunk_ids: Sequence[str], gold_doc_ids: Sequence[str], k: int
) -> float | None:
    if not gold_doc_ids:
        return None
    top_docs = set(docs_of(retrieved_chunk_ids)[:k])
    return len(top_docs & set(gold_doc_ids)) / len(set(gold_doc_ids))


def ndcg_at_k(
    retrieved_chunk_ids: Sequence[str], gold_doc_ids: Sequence[str], k: int = 10
) -> float | None:
    if not gold_doc_ids:
        return None
    gold = set(gold_doc_ids)
    docs = docs_of(retrieved_chunk_ids)[:k]
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(docs) if d in gold)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / idcg if idcg else None


def evidence_found(evidence: str, texts: Sequence[str]) -> bool:
    ev = normalize_answer(evidence)
    if not ev:
        return False
    ev_tokens = ev.split()
    for t in texts:
        nt = normalize_answer(t)
        if ev in nt:
            return True
        if len(ev_tokens) > 12:
            have = Counter(nt.split())
            hit = sum(min(c, have[w]) for w, c in Counter(ev_tokens).items())
            if hit / len(ev_tokens) >= 0.8:
                return True
    return False


def evidence_recall(retrieved_texts: Sequence[str], gold_evidence: Sequence[str]) -> float | None:
    if not gold_evidence:
        return None
    return sum(evidence_found(e, retrieved_texts) for e in gold_evidence) / len(gold_evidence)


def mean_defined(values: Sequence[float | None]) -> tuple[float | None, int]:
    """Mean over defined values and the number of undefined ones (never silently zero)."""
    vals = [v for v in values if v is not None]
    return (sum(vals) / len(vals) if vals else None), len(values) - len(vals)
