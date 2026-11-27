"""Compare evaluation paths using a fixed Natural Questions development workload."""

import argparse
import cProfile
import time
from pathlib import Path
from statistics import median

from elevate_rag.data import Chunk, documents_at, questions_at, stable_hash, write_json
from elevate_rag.evaluation import passages_by_document, retrieval_metrics
from elevate_rag.retrieval import Hit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    catalog = {
        doc.doc_id: Chunk(doc.doc_id, doc.doc_id, doc.title, doc.text, doc.source, 0, len(doc.text))
        for doc in documents_at(args.dataset)
    }
    questions = questions_at(args.dataset, "dev")
    candidates = [Hit(key, 1) for key in list(catalog)[:50]]

    def evaluate(grouped):
        index = passages_by_document(catalog) if grouped else None
        return [
            retrieval_metrics(q.evidence, candidates, catalog, [], by_document=index)
            for q in questions
        ]

    report = {"passages": len(catalog), "development_questions": len(questions)}
    for grouped, label in ((False, "full_scan"), (True, "by_document")):
        seconds = []
        for _ in range(5):
            start = time.perf_counter()
            results = evaluate(grouped)
            seconds.append(time.perf_counter() - start)
        profiler = cProfile.Profile()
        assert profiler.runcall(evaluate, grouped) == results
        profiler.dump_stats(args.out / f"{label}.prof")
        report[label] = {
            "seconds": seconds,
            "median_seconds": median(seconds),
            "score_signature": stable_hash(results),
        }
    assert report["full_scan"]["score_signature"] == report["by_document"]["score_signature"]
    write_json(args.out / "metrics.json", report)


if __name__ == "__main__":
    main()
