"""MultiHop-RAG: 609 news articles (2023) and 2,556 queries.

Gold evidence comes as (article title, fact sentence) pairs. Titles are unique, so they map to
article-level ``gold_doc_ids``; the fact sentences become ``gold_evidence`` and are matched to
chunks in Phase 3.
"""

from __future__ import annotations

from typing import Any

from hgrag.data.loaders.base import BuildInputs, BuiltDataset, read_json, seeded_sample
from hgrag.data.schema import Document, Query, QueryType
from hgrag.data.text import clean_text, doc_id_for, qid_for

CORPUS_FILE = "corpus.json"
QUERIES_FILE = "MultiHopRAG.json"
_TYPE_MAP: dict[str, QueryType] = {
    "inference_query": "multi_hop",
    "comparison_query": "comparison",
    "temporal_query": "temporal",
    "null_query": "unanswerable",
}


def build_multihop_rag(inputs: BuildInputs) -> BuiltDataset:
    articles: list[dict[str, Any]] = read_json(inputs.raw_dir / CORPUS_FILE)
    raw_queries: list[dict[str, Any]] = read_json(inputs.raw_dir / QUERIES_FILE)

    documents = [_article(a, inputs.name) for a in articles]
    title_to_id = {a["title"]: d.doc_id for a, d in zip(articles, documents, strict=True)}

    queries: list[Query] = []
    unmatched = 0
    for raw in seeded_sample(raw_queries, inputs.settings.sample_size, inputs.seed):
        evidence = raw["evidence_list"]
        gold_ids = {title_to_id[e["title"]] for e in evidence if e["title"] in title_to_id}
        unmatched += sum(e["title"] not in title_to_id for e in evidence)
        query_type = _TYPE_MAP[raw["question_type"]]
        answerable = query_type != "unanswerable"
        question = clean_text(raw["query"])
        queries.append(
            Query(
                qid=qid_for("mhr", question),
                question=question,
                answers=[raw["answer"]] if answerable else [],
                gold_doc_ids=sorted(gold_ids),
                gold_evidence=[clean_text(e["fact"]) for e in evidence],
                query_type=query_type,
                native_type=raw["question_type"],
                hop_count=len({e["title"] for e in evidence}) or None,
                answerable=answerable,
                meta={"original_answer": raw["answer"]},
            )
        )
    notes = [f"{unmatched} evidence items reference an article title missing from the corpus."]
    return BuiltDataset(documents, queries, notes)


def _article(article: dict[str, Any], source: str) -> Document:
    title, text = clean_text(article["title"]), clean_text(article["body"])
    return Document(
        doc_id=doc_id_for(title, text),
        title=title,
        text=text,
        source=source,
        meta={k: article[k] for k in ("source", "published_at", "category", "url", "author")},
    )
