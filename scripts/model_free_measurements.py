"""Measurements that need no LLM, embedder download or credentials (real data, real numbers).

1. V1 (BM25) retrieval quality on every prepared dataset's TEST split, with the reference
   chunking (512/64, sentence-aware, title prepended): recall@5/10/50, nDCG@10 (document level),
   evidence recall@10 where gold is text. Bootstrap 95 % CIs.
2. R1 rule-router accuracy against dataset query-type labels (test splits).
3. R2 TF-IDF + linear SVM on RAGRouter-Bench: 5-fold CV and leave-one-domain-out (the setting
   of Bansal & Agarwal 2026, cited by the plan).

These are PARTIAL results (retrieval and routing only — no generation, no judge). They are
written to results/model_free/ with the dataset versions and configuration used.

Usage:  uv run python scripts/model_free_measurements.py [--data-dir data] [--out results/...]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import time
from collections import Counter
from pathlib import Path
from typing import Any

from elevate_rag import __version__
from elevate_rag.config import ChunkingConfig
from elevate_rag.data.prepare import DatasetStore
from elevate_rag.evaluation.metrics import evidence_recall, ndcg_at_k, recall_at_k
from elevate_rag.indexing.chunker import Chunker
from elevate_rag.retrieval.lexical import BM25Index
from elevate_rag.routing.router import classify_rules, cross_validate
from elevate_rag.stats import bootstrap_ci

QA_DATASETS = [
    "nq",
    "hotpotqa",
    "musique",
    "multihop_rag",
    "graphrag_bench_medical",
    "graphrag_bench_novel",
]


def _ci(vals: list[float | None]) -> dict[str, Any] | None:
    v = [x for x in vals if x is not None]
    if not v:
        return None
    c = bootstrap_ci(v)
    return {
        "mean": round(c.estimate, 4),
        "ci95": [round(c.low, 4), round(c.high, 4)],
        "n": c.n,
        "undefined": len(vals) - len(v),
    }


def bm25_retrieval(data_dir: Path, name: str) -> dict[str, Any]:
    ds = DatasetStore.open(data_dir, name)
    ds.verify()
    cfg = ChunkingConfig()
    t0 = time.perf_counter()
    chunks = list(Chunker(cfg).chunk_documents(ds.documents()))
    t_chunk = time.perf_counter() - t0
    idx = BM25Index().build([c.chunk_id for c in chunks], [c.text for c in chunks])
    text = {c.chunk_id: c.text for c in chunks}
    qs = [q for q in ds.questions("test") if q.answerable]
    ev_valid = ds.string_evidence_valid
    rows: dict[str, list[float | None]] = {
        k: [] for k in ("recall@5", "recall@10", "recall@50", "ndcg@10", "evidence_recall@10")
    }
    by_type: dict[str, list[float | None]] = {}
    t0 = time.perf_counter()
    for q in qs:
        hits = [h[0] for h in idx.search(q.question, 50)]
        rows["recall@5"].append(recall_at_k(hits, q.gold_doc_ids, 5))
        rows["recall@10"].append(r10 := recall_at_k(hits, q.gold_doc_ids, 10))
        rows["recall@50"].append(recall_at_k(hits, q.gold_doc_ids, 50))
        rows["ndcg@10"].append(ndcg_at_k(hits, q.gold_doc_ids, 10))
        er = evidence_recall([text[h] for h in hits[:10]], q.gold_evidence) if ev_valid else None
        rows["evidence_recall@10"].append(er)
        by_type.setdefault(q.query_type, []).append(r10 if r10 is not None else er)
    t_query = time.perf_counter() - t0
    return {
        "dataset": name,
        "dataset_version": ds.version,
        "chunks": len(chunks),
        "evidence_verbatim_rate": ds.manifest.get("evidence_verbatim_rate"),
        "string_evidence_recall_valid": ev_valid,
        "test_questions_answerable": len(qs),
        "metrics": {k: _ci(v) for k, v in rows.items()},
        "primary_recall_by_query_type": {t: _ci(v) for t, v in sorted(by_type.items())},
        "seconds": {"chunking": round(t_chunk, 2), "queries": round(t_query, 2)},
    }


def rule_router_accuracy(data_dir: Path, names: list[str]) -> dict[str, Any]:
    out = {}
    for name in names:
        qs = DatasetStore.open(data_dir, name).questions("test")
        pred = [classify_rules(q.question)[0] for q in qs]
        gold = [q.query_type for q in qs]
        correct = [float(p == g) for p, g in zip(pred, gold, strict=True)]
        out[name] = {
            "accuracy": _ci(correct),
            "label_distribution": dict(Counter(gold)),
            "confusion_top": Counter(
                f"{g}->{p}" for g, p in zip(gold, pred, strict=True) if g != p
            ).most_common(5),
        }
    return out


def ragrouter_cv(data_dir: Path) -> dict[str, Any]:
    qs = DatasetStore.open(data_dir, "ragrouter").questions("all")
    texts = [q.question for q in qs]
    labels = [q.query_type for q in qs]
    groups = [q.metadata["domain"] for q in qs]
    five = cross_validate(texts, labels, folds=5)
    lodo = cross_validate(texts, labels, groups=groups)
    return {
        "n": len(qs),
        "labels": dict(Counter(labels)),
        "five_fold": {k: five[k] for k in ("accuracy", "macro_f1")},
        "leave_one_domain_out": {
            "accuracy": lodo["accuracy"],
            "macro_f1": lodo["macro_f1"],
            "folds": lodo["folds"],
        },
        "reference": "Bansal & Agarwal 2026 report 93.2 % accuracy / 0.928 macro-F1 "
        "(TF-IDF + SVM on RAGRouter-Bench); protocol details may differ",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("results/model_free"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    present = [n for n in QA_DATASETS if (args.data_dir / "processed" / n).exists()]
    result = {
        "label": "PARTIAL RESULTS — retrieval (V1 BM25) and routing classification only; "
        "no generation, no LLM judge",
        "created_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "elevate_rag": __version__,
        "python": platform.python_version(),
        "chunking": ChunkingConfig().model_dump(),
        "bm25": {"k1": 0.9, "b": 0.4},
        "bm25_retrieval": [bm25_retrieval(args.data_dir, n) for n in present],
        "r1_rule_router": rule_router_accuracy(args.data_dir, present),
    }
    if (args.data_dir / "processed" / "ragrouter").exists():
        result["r2_tfidf_svm_ragrouter"] = ragrouter_cv(args.data_dir)
    (args.out / "model_free_measurements.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=1)[:6000])


if __name__ == "__main__":
    main()
