"""Real dataset loaders (FR-D1) and their pinned sources (D-12).

Each loader reads raw files from the download cache, samples questions deterministically and
returns normalised :class:`Document` / :class:`Question` records. Records that fail validation are
collected (with the reason) instead of being dropped silently (FR-D3).
"""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from elevate_rag.types import Document, Question
from elevate_rag.utils import DataError, read_jsonl, stable_hash


@dataclass(frozen=True)
class RemoteFile:
    repo_id: str
    filename: str
    revision: str


@dataclass
class LoadResult:
    documents: list[Document]
    questions: list[Question]
    rejected: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    files: dict[str, RemoteFile]
    loader: Callable[[dict[str, Path], dict[str, Any]], LoadResult]
    defaults: dict[str, Any]
    license: str
    description: str
    router_only: bool = False


# --------------------------------------------------------------------------- helpers


def deterministic_sample(
    items: list[Any], n: int | None, seed: int, key: Callable[[Any], str]
) -> list[Any]:
    """Seeded sample that does not depend on input order."""
    ordered = sorted(items, key=key)
    if n is None or n >= len(ordered):
        return ordered
    return sorted(random.Random(seed).sample(ordered, n), key=key)


def _mk(model: type, rejected: list[dict[str, Any]], raw: Any, **fields: Any) -> Any:
    try:
        return model(**fields)
    except ValidationError as exc:
        rejected.append({"record": str(raw)[:300], "reason": exc.errors()[0]["msg"]})
        return None


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return list(read_jsonl(path))  # several "JSON" files are really JSONL


def doc_id_for(prefix: str, *parts: str) -> str:
    return f"{prefix}-{stable_hash(list(parts), 12)}"


# --------------------------------------------------------------------------- NQ


def load_nq(files: dict[str, Path], p: dict[str, Any]) -> LoadResult:
    rows = _load_json(files["dev"])
    rej: list[dict[str, Any]] = []
    sampled = deterministic_sample(
        rows, p["n_questions"], p["seed"], key=lambda r: str(r["example_id"])
    )
    sampled_ids = {r["example_id"] for r in sampled}
    docs: dict[str, Document] = {}

    def add_doc(r: dict[str, Any]) -> str:
        did = f"nq-{r['idx_gold_in_corpus']}"
        if did not in docs:
            d = _mk(
                Document,
                rej,
                r,
                doc_id=did,
                title="",
                text=r.get("text") or "",
                source="nq_open_gold",
            )
            if d is not None:
                docs[did] = d
        return did

    questions = []
    for r in sampled:
        did = add_doc(r)
        q = _mk(
            Question,
            rej,
            r,
            question_id=f"nq-{r['example_id']}",
            dataset="nq",
            question=r["question"],
            answers=list(r["answers"]),
            query_type="single_hop",
            gold_doc_ids=[did] if did in docs else [],
        )
        if q is not None:
            questions.append(q)
    distractors = [r for r in rows if r["example_id"] not in sampled_ids]
    for r in deterministic_sample(
        distractors, p["n_distractors"], p["seed"] + 1, key=lambda r: str(r["example_id"])
    ):
        add_doc(r)
    return LoadResult(
        list(docs.values()),
        questions,
        rej,
        [
            f"corpus = {len(sampled)} gold passages + other questions' gold passages "
            f"as distractors (deduplicated by idx_gold_in_corpus)"
        ],
    )


# --------------------------------------------------------------------------- HotpotQA


def load_hotpotqa(files: dict[str, Path], p: dict[str, Any]) -> LoadResult:
    import pyarrow.parquet as pq

    rows = pq.read_table(files["validation"]).to_pylist()
    rows = [r for r in rows if r["level"] == "hard"]
    sampled = deterministic_sample(rows, p["n_questions"], p["seed"], key=lambda r: r["id"])
    rej: list[dict[str, Any]] = []
    docs: dict[str, Document] = {}
    questions = []
    for r in sampled:
        title_to_doc = {}
        for title, sents in zip(r["context"]["title"], r["context"]["sentences"], strict=True):
            text = "".join(sents).strip()
            did = doc_id_for("hp", title, text)
            title_to_doc[title] = did
            if did not in docs and (
                d := _mk(
                    Document, rej, title, doc_id=did, title=title, text=text, source="hotpotqa"
                )
            ):
                docs[did] = d
        gold = sorted(
            {title_to_doc[t] for t in r["supporting_facts"]["title"] if t in title_to_doc}
        )
        q = _mk(
            Question,
            rej,
            r["id"],
            question_id=f"hp-{r['id']}",
            dataset="hotpotqa",
            question=r["question"],
            answers=[r["answer"]],
            query_type="comparison" if r["type"] == "comparison" else "multi_hop",
            gold_doc_ids=gold,
            metadata={"hotpot_type": r["type"]},
        )
        if q is not None:
            questions.append(q)
    return LoadResult(
        list(docs.values()),
        questions,
        rej,
        ["corpus = pooled distractor-setting context paragraphs of sampled questions"],
    )


# --------------------------------------------------------------------------- MuSiQue


def load_musique(files: dict[str, Path], p: dict[str, Any]) -> LoadResult:
    rows = list(read_jsonl(files["full_dev"]))
    for r in rows:
        r["answerable"] = r["answerable"] in (True, "True", "true")
    ans = [r for r in rows if r["answerable"]]
    sampled_ans = deterministic_sample(ans, p["n_answerable"], p["seed"], key=lambda r: r["id"])
    taken = {r["id"] for r in sampled_ans}
    # An unanswerable question's answerable twin shares its id; its paragraphs would leak the
    # answer into the pooled corpus, so twins of sampled answerable questions are excluded.
    unans = [r for r in rows if not r["answerable"] and r["id"] not in taken]
    sampled_un = deterministic_sample(
        unans, p["n_unanswerable"], p["seed"] + 1, key=lambda r: r["id"]
    )
    rej: list[dict[str, Any]] = []
    docs: dict[str, Document] = {}
    questions = []
    for r in sampled_ans + sampled_un:
        gold = []
        for para in r["paragraphs"]:
            did = doc_id_for("mq", para["title"], para["paragraph_text"])
            if did not in docs and (
                d := _mk(
                    Document,
                    rej,
                    para["title"],
                    doc_id=did,
                    title=para["title"],
                    text=para["paragraph_text"],
                    source="musique",
                )
            ):
                docs[did] = d
            if para.get("is_supporting") and r["answerable"]:
                gold.append(did)
        hops = int(r["id"][0]) if r["id"][0].isdigit() else None
        qid = f"mq-{r['id']}" + ("" if r["answerable"] else "-unans")
        q = _mk(
            Question,
            rej,
            r["id"],
            question_id=qid,
            dataset="musique",
            question=r["question"],
            answers=[r["answer"], *r.get("answer_aliases", [])] if r["answerable"] else [],
            query_type="multi_hop",
            answerable=r["answerable"],
            gold_doc_ids=sorted(gold),
            metadata={"hops": hops, "reference_answer": r["answer"]},
        )
        if q is not None:
            questions.append(q)
    return LoadResult(
        list(docs.values()),
        questions,
        rej,
        [
            "unanswerable questions exclude twins of sampled answerable ones",
            "corpus = pooled paragraphs; other questions' paragraphs may still contain "
            "an unanswerable question's answer (inherent to pooled corpora)",
        ],
    )


# --------------------------------------------------------------------------- MultiHop-RAG

_MHR_TYPES = {
    "inference_query": "multi_hop",
    "comparison_query": "comparison",
    "temporal_query": "temporal",
    "null_query": "null",
}


def load_multihop_rag(files: dict[str, Path], p: dict[str, Any]) -> LoadResult:
    corpus = _load_json(files["corpus"])
    queries = _load_json(files["queries"])
    rej: list[dict[str, Any]] = []
    docs: dict[str, Document] = {}
    url_to_doc: dict[str, str] = {}
    for a in corpus:
        did = doc_id_for("mhr", a["url"])
        url_to_doc[a["url"]] = did
        d = _mk(
            Document,
            rej,
            a.get("title"),
            doc_id=did,
            title=a.get("title", ""),
            text=a.get("body", ""),
            source=f"multihop_rag:{a.get('source', '')}",
            metadata={
                "published_at": a.get("published_at"),
                "url": a["url"],
                "category": a.get("category"),
            },
        )
        if d is not None:
            docs[did] = d
    questions = []
    for i, q in enumerate(queries):
        qtype = _MHR_TYPES.get(q["question_type"], "other")
        null = qtype == "null"
        ev = q.get("evidence_list", [])
        qq = _mk(
            Question,
            rej,
            q.get("query"),
            question_id=f"mhr-{i:05d}",
            dataset="multihop_rag",
            question=q["query"],
            answers=[] if null else [q["answer"]],
            query_type=qtype,
            answerable=not null,
            gold_doc_ids=sorted({url_to_doc[e["url"]] for e in ev if e["url"] in url_to_doc}),
            gold_evidence=[e["fact"] for e in ev if e.get("fact")],
            metadata={"mhr_type": q["question_type"], "reference_answer": q["answer"]},
        )
        if qq is not None:
            questions.append(qq)
    questions = deterministic_sample(
        questions, p.get("n_questions"), p["seed"], key=lambda q: q.question_id
    )
    return LoadResult(list(docs.values()), questions, rej)


# --------------------------------------------------------------------------- GraphRAG-Bench

_GRB_TYPES = {
    "Fact Retrieval": ("single_hop", 1),
    "Complex Reasoning": ("multi_hop", 2),
    "Contextual Summarize": ("global", 3),
    "Creative Generation": ("creative", 4),
}


def _load_graphrag_bench(
    corpus_path: Path, q_path: Path, name: str, p: dict[str, Any]
) -> LoadResult:
    corpus = _load_json(corpus_path)
    rej: list[dict[str, Any]] = []
    docs = []
    for i, c in enumerate(corpus if isinstance(corpus, list) else [corpus]):
        title = c.get("corpus_name", f"{name}-{i}")
        d = _mk(
            Document,
            rej,
            title,
            doc_id=doc_id_for("grb", name, title, str(i)),
            title=title,
            text=c.get("context", ""),
            source=f"graphrag_bench:{name}",
        )
        if d is not None:
            docs.append(d)
    questions = []
    raw_q = _load_json(q_path)
    id_counts = Counter(q.get("id") for q in raw_q)
    seen: Counter[str] = Counter()
    notes = []
    for q in raw_q:
        qid = f"grb-{q['id']}"
        if id_counts[q["id"]] > 1:  # upstream bug: distinct questions sharing one id
            seen[q["id"]] += 1
            qid = f"{qid}#{seen[q['id']]}"
            notes.append(
                f"source id {q['id']!r} is shared by {id_counts[q['id']]} different "
                f"questions; renamed to {qid} (source order)"
            )
        qtype, level = _GRB_TYPES.get(q.get("question_type", ""), ("other", None))
        ev = q.get("evidence") or []
        qq = _mk(
            Question,
            rej,
            q.get("id"),
            question_id=qid,
            dataset=f"graphrag_bench_{name}",
            question=q["question"],
            answers=[q["answer"]] if q.get("answer") else [],
            query_type=qtype,
            gold_evidence=[e for e in (ev if isinstance(ev, list) else [ev]) if e],
            metadata={
                "level": level,
                "grb_type": q.get("question_type"),
                "source": q.get("source"),
            },
        )
        if qq is not None:
            questions.append(qq)
    questions = deterministic_sample(
        questions, p.get("n_questions"), p["seed"], key=lambda q: q.question_id
    )
    return LoadResult(docs, questions, rej, notes)


def load_grb_medical(files: dict[str, Path], p: dict[str, Any]) -> LoadResult:
    return _load_graphrag_bench(files["corpus"], files["questions"], "medical", p)


def load_grb_novel(files: dict[str, Path], p: dict[str, Any]) -> LoadResult:
    return _load_graphrag_bench(files["corpus"], files["questions"], "novel", p)


# --------------------------------------------------------------------------- RAGRouter-Bench

_RR_TYPES = {"single_hop": "single_hop", "multi_hop": "multi_hop", "summary": "global"}


def _evidence_texts(facts: Any) -> list[str]:
    """RAGRouter-Bench stores evidence either as strings or as {"doc_id", "text"} objects
    (QuALITY, legal and part of MuSiQue). Both normalise to the evidence text."""
    out = []
    for f in facts or []:
        text = f.get("text") if isinstance(f, dict) else f
        if isinstance(text, str) and text.strip():
            out.append(text.strip())
    return out


def load_ragrouter(files: dict[str, Path], p: dict[str, Any]) -> LoadResult:
    rej: list[dict[str, Any]] = []
    questions = []
    for domain, path in sorted(files.items()):
        for r in read_jsonl(path):
            q = _mk(
                Question,
                rej,
                r.get("id"),
                question_id=f"rr-{r['id']}",
                dataset="ragrouter",
                question=r["question"],
                answers=[r["answer"]] if r.get("answer") else [],
                query_type=_RR_TYPES.get(r.get("type", ""), "other"),
                gold_evidence=_evidence_texts(r.get("supporting_facts")),
                metadata={"domain": domain, "rr_type": r.get("type")},
            )
            if q is not None:
                questions.append(q)
    return LoadResult([], questions, rej, ["router-only dataset: no corpus is loaded"])


# --------------------------------------------------------------------------- registry

_HF = {
    "hotpot": ("hotpotqa/hotpot_qa", "1908d6afbbead072334abe2965f91bd2709910ab"),
    "nq": ("florin-hf/nq_open_gold", "194571ad81b79e7c186e13cbe9297812a80dcc48"),
    "musique": ("bdsaglam/musique", "22873a405dd809893b22ada0b499299fb612d2df"),
    "mhr": ("yixuantt/MultiHopRAG", "71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82"),
    "grb": ("GraphRAG-Bench/GraphRAG-Bench", "dc3a111e77dbaf8bbaf51ef331f3cfc9b1b5c546"),
    "rr": ("Chaplain0908/RAGRouter", "abff378d9202caba636aa4d5bb9243db8103f70f"),
}


def _rf(src: str, filename: str) -> RemoteFile:
    repo, rev = _HF[src]
    return RemoteFile(repo, filename, rev)


DATASETS: dict[str, DatasetSpec] = {
    "nq": DatasetSpec(
        "nq",
        {"dev": _rf("nq", "dev_dataset.json")},
        load_nq,
        {"n_questions": 500, "n_distractors": 4500, "seed": 13},
        "NQ: CC BY-SA 3.0 (Google); nq_open_gold card: no licence declared — research use",
        "Natural Questions (open) with gold Wikipedia passages",
    ),
    "hotpotqa": DatasetSpec(
        "hotpotqa",
        {"validation": _rf("hotpot", "distractor/validation-00000-of-00001.parquet")},
        load_hotpotqa,
        {"n_questions": 1000, "seed": 13},
        "CC BY-SA 4.0",
        "HotpotQA distractor dev (all level=hard)",
    ),
    "musique": DatasetSpec(
        "musique",
        {"full_dev": _rf("musique", "musique_full_v1.0_dev.jsonl")},
        load_musique,
        {"n_answerable": 500, "n_unanswerable": 100, "seed": 13},
        "CC BY 4.0 (MuSiQue authors)",
        "MuSiQue-Full dev (answerable + unanswerable)",
    ),
    "multihop_rag": DatasetSpec(
        "multihop_rag",
        {"corpus": _rf("mhr", "corpus.json"), "queries": _rf("mhr", "MultiHopRAG.json")},
        load_multihop_rag,
        {"n_questions": None, "seed": 13},
        "ODC-BY",
        "MultiHop-RAG news corpus, full query set",
    ),
    "graphrag_bench_medical": DatasetSpec(
        "graphrag_bench_medical",
        {
            "corpus": _rf("grb", "Datasets/Corpus/medical.json"),
            "questions": _rf("grb", "Datasets/Questions/medical_questions.json"),
        },
        load_grb_medical,
        {"n_questions": None, "seed": 13},
        "no licence declared on the HF card — check before redistribution",
        "GraphRAG-Bench (Xiang et al.) medical / NCCN guidelines",
    ),
    "graphrag_bench_novel": DatasetSpec(
        "graphrag_bench_novel",
        {
            "corpus": _rf("grb", "Datasets/Corpus/novel.json"),
            "questions": _rf("grb", "Datasets/Questions/novel_questions.json"),
        },
        load_grb_novel,
        {"n_questions": None, "seed": 13},
        "no licence declared on the HF card; corpus is Project Gutenberg text",
        "GraphRAG-Bench (Xiang et al.) Gutenberg novels",
    ),
    "ragrouter": DatasetSpec(
        "ragrouter",
        {
            d: _rf("rr", f"data/{d}/Question.json")
            for d in ("musique", "quality", "ultraDomain_legal", "graphragBench_medical")
        },
        load_ragrouter,
        {"seed": 13},
        "MIT",
        "RAGRouter-Bench query labels (router training/testing only)",
        router_only=True,
    ),
}


def get_spec(name: str) -> DatasetSpec:
    if name not in DATASETS:
        raise DataError(f"unknown dataset {name!r}; known: {sorted(DATASETS)}")
    return DATASETS[name]


def iter_specs(names: Iterable[str] | None = None) -> list[DatasetSpec]:
    return [get_spec(n) for n in (names or DATASETS)]
