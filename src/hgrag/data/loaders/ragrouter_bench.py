"""RAGRouter-Bench: 7,727 labelled queries over 4 domains, used only to train/test the R2 router.

Two of its domains (MuSiQue, GraphRAG-Bench medical) overlap with our evaluation datasets. Any
question whose normalised text appears in one of our built datasets is dropped, so the router
never trains on questions we evaluate on. Build this dataset last.
"""

from __future__ import annotations

from typing import Any

from hgrag.data.io import DatasetFiles, load_queries
from hgrag.data.loaders.base import BuildInputs, BuiltDataset, read_json_lines, seeded_sample
from hgrag.data.schema import Query, QueryType
from hgrag.data.text import clean_text, normalize_question

DOMAINS = ("musique", "quality", "ultraDomain_legal", "graphragBench_medical")
_TYPE_MAP: dict[str, QueryType] = {
    "single_hop": "single_hop",
    "multi_hop": "multi_hop",
    "summary": "global",
}


def build_ragrouter_bench(inputs: BuildInputs) -> BuiltDataset:
    seen, checked = _evaluation_questions(inputs)
    raw: list[tuple[str, dict[str, Any]]] = [
        (domain, record)
        for domain in DOMAINS
        for record in read_json_lines(inputs.raw_dir / f"data/{domain}/Question.json")
    ]

    queries: list[Query] = []
    dropped = 0
    for domain, record in seeded_sample(raw, inputs.settings.sample_size, inputs.seed):
        question = clean_text(record["question"])
        if normalize_question(question) in seen:
            dropped += 1
            continue
        queries.append(
            Query(
                qid=f"rr-{record['id']}",
                question=question,
                answers=[str(record["answer"])],
                gold_evidence=[_fact_text(f) for f in record.get("supporting_facts", [])],
                query_type=_TYPE_MAP[record["type"]],
                native_type=record["type"],
                meta={"domain": domain},
            )
        )
    notes = [
        f"Router-only dataset (no corpus). Dropped {dropped} questions that also appear in: "
        f"{', '.join(checked) or 'none (build the other datasets first!)'}."
    ]
    return BuiltDataset(documents=[], queries=queries, notes=notes)


def _fact_text(fact: str | dict[str, Any]) -> str:
    """Supporting facts are either plain sentences or ``{"doc_id": …, "text": …}`` passages."""
    return clean_text(fact if isinstance(fact, str) else str(fact["text"]))


def _evaluation_questions(inputs: BuildInputs) -> tuple[set[str], list[str]]:
    """Normalised questions of every other dataset already built under data/processed."""
    seen: set[str] = set()
    checked: list[str] = []
    for folder in sorted(p for p in inputs.processed_root.iterdir() if p.is_dir()):
        files = DatasetFiles(folder)
        if folder.name == inputs.name or not files.exists():
            continue
        seen |= {normalize_question(q.question) for q in load_queries(files)}
        checked.append(folder.name)
    return seen, checked
