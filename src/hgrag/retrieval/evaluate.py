"""Retrieval-only evaluation: recall@k / nDCG@10 for V1-V4 without any LLM call (free).

Used for dev-set tuning (chunk size, candidates, final_k) before paying for generation.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from pydantic import BaseModel

from hgrag.data.schema import Query
from hgrag.eval.metrics.retrieval import hit_at_k, ndcg_at_k, recall_at_k
from hgrag.retrieval.searcher import Searcher

KS = (1, 5, 10)


class RetrievalReport(BaseModel):
    retriever: str
    n_queries: int  # questions with gold chunks
    chunk_recall: dict[int, float]
    chunk_hit: dict[int, float]  # any gold chunk in top k (standard for NQ)
    chunk_ndcg10: float
    doc_recall: dict[int, float]
    by_query_type: dict[str, dict[str, float]]  # type → {"chunk_recall@5": ..., "n": ...}
    mean_context_tokens: float


def evaluate_retrieval(
    searcher: Searcher,
    queries: Sequence[Query],
    gold_chunks: dict[str, list[str]],
    label: str,
) -> RetrievalReport:
    rows: list[tuple[Query, dict[str, float]]] = []
    context_tokens: list[int] = []
    for query in queries:
        gold = gold_chunks.get(query.qid)
        if not gold:
            continue
        hits = searcher.search(query.question)
        chunk_ranking = [h.chunk.chunk_id for h in hits]
        doc_ranking = [h.chunk.doc_id for h in hits]
        context_tokens.append(sum(h.chunk.n_tokens for h in hits))
        row = {f"chunk_recall@{k}": recall_at_k(chunk_ranking, gold, k) or 0.0 for k in KS}
        row |= {
            f"doc_recall@{k}": recall_at_k(doc_ranking, query.gold_doc_ids, k) or 0.0 for k in KS
        }
        row |= {f"chunk_hit@{k}": hit_at_k(chunk_ranking, gold, k) or 0.0 for k in KS}
        row["chunk_ndcg@10"] = ndcg_at_k(chunk_ranking, gold, 10) or 0.0
        rows.append((query, row))

    def mean(key: str, subset: Sequence[dict[str, float]]) -> float:
        return round(sum(r[key] for r in subset) / len(subset), 4) if subset else 0.0

    all_rows = [r for _, r in rows]
    by_type: dict[str, list[dict[str, float]]] = defaultdict(list)
    for query, row in rows:
        by_type[query.query_type].append(row)
    return RetrievalReport(
        retriever=label,
        n_queries=len(rows),
        chunk_recall={k: mean(f"chunk_recall@{k}", all_rows) for k in KS},
        chunk_hit={k: mean(f"chunk_hit@{k}", all_rows) for k in KS},
        chunk_ndcg10=mean("chunk_ndcg@10", all_rows),
        doc_recall={k: mean(f"doc_recall@{k}", all_rows) for k in KS},
        by_query_type={
            t: {"n": float(len(r)), "chunk_recall@5": mean("chunk_recall@5", r)}
            for t, r in sorted(by_type.items())
        },
        mean_context_tokens=round(sum(context_tokens) / max(len(context_tokens), 1), 1),
    )
