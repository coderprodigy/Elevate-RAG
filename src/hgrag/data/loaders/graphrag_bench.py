"""GraphRAG-Bench (Xiang et al., "When to use Graphs in RAG"): novel and medical subsets.

Each subset is a separate dataset with its own corpus. Documents are whole books (novel) or one
long guideline text (medical); chunking happens in Phase 3. Task levels map to query types:
Fact Retrieval → single_hop, Complex Reasoning → multi_hop, Contextual Summarize → global,
Creative Generation → creative.
"""

from __future__ import annotations

from typing import Any, Literal

from hgrag.data.loaders.base import BuildInputs, BuiltDataset, read_json, seeded_sample
from hgrag.data.schema import Document, Query, QueryType
from hgrag.data.text import clean_text, doc_id_for

Subset = Literal["novel", "medical"]
_TYPE_MAP: dict[str, QueryType] = {
    "Fact Retrieval": "single_hop",
    "Complex Reasoning": "multi_hop",
    "Contextual Summarize": "global",
    "Creative Generation": "creative",
}
_LEVEL = {
    "Fact Retrieval": 1,
    "Complex Reasoning": 2,
    "Contextual Summarize": 3,
    "Creative Generation": 4,
}


def build_graphrag_bench_novel(inputs: BuildInputs) -> BuiltDataset:
    return _build(inputs, "novel")


def build_graphrag_bench_medical(inputs: BuildInputs) -> BuiltDataset:
    return _build(inputs, "medical")


def _build(inputs: BuildInputs, subset: Subset) -> BuiltDataset:
    corpus: list[dict[str, Any]] = read_json(inputs.raw_dir / f"Datasets/Corpus/{subset}.json")
    raw_queries: list[dict[str, Any]] = read_json(
        inputs.raw_dir / f"Datasets/Questions/{subset}_questions.json"
    )

    documents: list[Document] = []
    name_to_id: dict[str, str] = {}
    for entry in corpus:
        title, text = entry["corpus_name"], clean_text(entry["context"])
        doc = Document(doc_id=doc_id_for(title, text), title=title, text=text, source=inputs.name)
        documents.append(doc)
        name_to_id[title] = doc.doc_id

    queries: list[Query] = []
    for raw in seeded_sample(raw_queries, inputs.settings.sample_size, inputs.seed):
        gold_doc = name_to_id.get(raw["source"])
        queries.append(
            Query(
                qid=f"grb-{raw['id']}",
                question=clean_text(raw["question"]),
                answers=[raw["answer"]],
                gold_doc_ids=[gold_doc] if gold_doc else [],
                gold_evidence=[clean_text(e) for e in raw.get("evidence", [])],
                query_type=_TYPE_MAP[raw["question_type"]],
                native_type=raw["question_type"],
                meta={"level": _LEVEL[raw["question_type"]], "source_doc": raw["source"]},
            )
        )
    linked = sum(bool(q.gold_doc_ids) for q in queries)
    notes = [
        f"{len(documents)} source document(s); {linked}/{len(queries)} questions linked to their "
        "source document. Gold evidence is sentence text, mapped to chunks in Phase 3."
    ]
    return BuiltDataset(documents, queries, notes)
