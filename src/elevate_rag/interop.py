"""Auditable file boundaries for upstream benchmark tools and standalone frameworks."""

import json
import math
import os
import subprocess
import tomllib
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .data import (
    artifact_lock,
    dataset_manifest,
    documents_at,
    file_hash,
    questions_at,
    read_jsonl,
    write_json,
    write_jsonl,
)
from .evaluation import normalize, token_f1
from .pipeline import CITATION
from .providers import Providers
from .reference_worker import DIST, loopback
from .reporting import report_run

FRAMEWORKS = tuple(DIST)
GRAPH_TASKS = {
    "fact-retrieval": "Fact Retrieval",
    "complex-reasoning": "Complex Reasoning",
    "contextual-summarize": "Contextual Summarize",
    "contextual-summarization": "Contextual Summarize",
    "creative-generation": "Creative Generation",
}


def export_reference(
    dataset: Path, out: Path, split: str, *, framework: str, limit: int | None, seed: int
) -> dict[str, Any]:
    from .experiments import select_questions

    manifest = dataset_manifest(dataset)
    if framework not in FRAMEWORKS:
        raise ValueError("unknown standalone framework")
    questions = select_questions(dataset, split, limit, seed)
    write_jsonl(
        out / "documents.jsonl",
        (
            {"id": doc.doc_id, "title": doc.title, "text": doc.text, "source": doc.source}
            for doc in documents_at(dataset)
        ),
    )
    write_jsonl(
        out / "queries.jsonl",
        ({"id": question.question_id, "question": question.text} for question in questions),
    )
    payload = {
        "framework": framework,
        "dataset": manifest,
        "split": split,
        "qids": [question.question_id for question in questions],
        "inputs": {name: file_hash(out / name) for name in ("documents.jsonl", "queries.jsonl")},
    }
    write_json(out / "manifest.json", payload)
    return payload


def validate_reference_export(dataset: Path, exported: Path) -> dict[str, Any]:
    manifest = json.loads((exported / "manifest.json").read_text())
    if manifest["dataset"] != dataset_manifest(dataset) or manifest["framework"] not in FRAMEWORKS:
        raise ValueError("standalone result setup differs from the normalized benchmark")
    for name in ("documents.jsonl", "queries.jsonl"):
        if file_hash(exported / name) != manifest["inputs"][name]:
            raise ValueError("standalone input checksum mismatch")
    rows = list(read_jsonl(exported / "queries.jsonl"))
    ids = [row["id"] for row in rows]
    if ids != manifest["qids"] or len(set(ids)) != len(ids) or not ids:
        raise ValueError("standalone input question IDs differ from the export manifest")
    splits = (
        ("dev", "heldout")
        if manifest["split"] == "all" and manifest["dataset"]["synthetic"]
        else (manifest["split"],)
    )
    expected = {q.question_id: q.text for split in splits for q in questions_at(dataset, split)}
    if any(
        expected.get(row["id"]) != row["question"] or set(row) != {"id", "question"} for row in rows
    ):
        raise ValueError("standalone input questions do not match the benchmark split and text")
    return manifest


def import_reference(
    dataset: Path, exported: Path, source: Path, version: str, out: Path
) -> dict[str, Any]:
    if not version.strip():
        raise ValueError("record the standalone framework version")
    if out.resolve() == exported.resolve():
        raise ValueError("keep standalone measurements in a distinct directory")
    manifest = validate_reference_export(dataset, exported)
    expected = set(manifest["qids"])
    splits = ("dev", "heldout") if manifest["split"] == "all" else (manifest["split"],)
    questions = {q.question_id: q for split in splits for q in questions_at(dataset, split)}
    rows: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(source):
        key = row["id"]
        if key not in expected or key in rows:
            raise ValueError(f"unknown or duplicated standalone outcome {key}")
        if row.get("status") not in {"ok", "error"}:
            raise ValueError(f"{key}: unknown standalone status")
        text = row.get("answer")
        contexts = row.get("contexts", [])
        if not isinstance(contexts, list) or not all(
            isinstance(context, str) for context in contexts
        ):
            raise ValueError(f"{key}: standalone contexts must be an array of strings")
        if row["status"] == "ok" and (not isinstance(text, str) or not text.strip()):
            raise ValueError(f"{key}: missing standalone answer")
        for metric in ("latency", "estimated_usd", "prompt_tokens", "completion_tokens"):
            value = row.get(metric)
            if value is not None and (
                not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
            ):
                raise ValueError(f"{key}: invalid reported {metric}")
        q = questions[key]
        metrics: dict[str, Any] = {}
        if row["status"] == "error":
            metrics["correct"] = False
        elif q.answers and isinstance(text, str):
            metrics.update(
                exact_match=max(
                    float(normalize(text) == normalize(answer)) for answer in q.answers
                ),
                token_f1=max(token_f1(text, answer) for answer in q.answers),
            )
        rows[key] = {
            "question_id": key,
            "query_type": q.query_type,
            "domain": q.domain,
            "status": row["status"],
            "answer": {"text": text},
            "context": contexts,
            "metrics": metrics,
            "correct": metrics.get("correct"),
            "latency": row.get("latency"),
            "estimated_usd": row.get("estimated_usd"),
            "prompt_tokens": row.get("prompt_tokens"),
            "completion_tokens": row.get("completion_tokens"),
        }
    if set(rows) != expected:
        raise ValueError(
            "standalone comparison must report each aligned question, including failures"
        )
    payload = {
        **manifest,
        "framework_version": version,
        "outputs_sha256": file_hash(source),
        "isolated_external_run": True,
        "run_id": out.name,
    }
    write_json(out / "manifest.json", payload)
    records = [rows[key] for key in manifest["qids"]]
    write_jsonl(out / "questions.jsonl", records)
    report_run(out, records, expected=len(expected), seed=manifest["dataset"]["seed"])
    return payload


def export_graphrag_eval(
    dataset: Path,
    manifest: dict[str, Any],
    outcomes: Sequence[dict[str, Any]],
    out: Path,
) -> dict[str, int]:
    version = dataset_manifest(dataset)
    if not version["dataset"].startswith("graphrag-") or version != manifest["dataset"]:
        raise ValueError("upstream evaluator requires matching Xiang GraphRAG-Bench data")
    questions = {q.question_id: q for q in questions_at(dataset, manifest["split"])}
    payload, unavailable = [], []
    for row in outcomes:
        q = questions[row["question_id"]]
        if row["status"] != "ok" or not q.evidence_text:
            unavailable.append(
                {
                    "question_id": q.question_id,
                    "status": row["status"],
                    "evidence": bool(q.evidence_text),
                }
            )
            continue
        raw_contexts = row["context"]
        payload.append(
            {
                "id": q.question_id,
                "question": q.text,
                "question_type": GRAPH_TASKS.get(q.query_type, q.query_type),
                "context": [
                    item if isinstance(item, str) else item["text"] for item in raw_contexts
                ],
                "answer": q.answers[0] if q.answers else "",
                "evidence": list(q.evidence_text),
                "prediction": CITATION.sub("", row["answer"]["text"]).strip(),
            }
        )
    write_json(out, payload)
    write_json(out.with_name(out.stem + "-unavailable.json"), unavailable)
    return {"exported": len(payload), "unavailable": len(unavailable)}


def attach_graphrag_metrics(run: Path, scored: Path, out: Path) -> dict[str, int]:
    if run.resolve() == out.resolve():
        raise ValueError("write an evaluated copy to preserve the original run artifacts")
    manifest = json.loads((run / "manifest.json").read_text())
    if not manifest["dataset"]["dataset"].startswith("graphrag-"):
        raise ValueError("upstream retrieval metrics require a GraphRAG-Bench run")
    groups = json.loads(scored.read_text())
    scores = {}
    for value in groups.values():
        for entry in value["detailed"]:
            key = entry["id"]
            if key in scores or key not in manifest["qids"]:
                raise ValueError(f"invalid upstream score identifier {key}")
            metrics = entry["metrics"]
            selected = {}
            for source, target in (
                ("evidence_recall", "evidence_recall"),
                ("context_relevancy", "context_relevance"),
            ):
                score = metrics[source]
                selected[target] = score if score is not None and math.isfinite(score) else None
            scores[key] = selected
    records = list(read_jsonl(run / "questions.jsonl"))
    for row in records:
        if row["question_id"] in scores:
            row["metrics"].update(scores[row["question_id"]])
    write_json(
        out / "manifest.json",
        {**manifest, "upstream_metrics": {"source_sha256": file_hash(scored)}},
    )
    write_jsonl(out / "questions.jsonl", records)
    report_run(out, records, expected=len(manifest["qids"]), seed=manifest["dataset"]["seed"])
    return {"scored": len(scores), "total": len(records)}


@dataclass(frozen=True, slots=True)
class ReferenceSettings:
    framework_version: str
    chat_url: str
    chat_model: str
    embedding_url: str
    embedding_model: str
    embedding_dimensions: int
    top_k: int = 8
    context_window: int = 8192
    max_output: int = 512
    request_timeout: float = 60.0
    neo4j_uri: str = ""
    neo4j_database: str = "neo4j"

    def __post_init__(self) -> None:
        if any(
            not value.strip()
            for value in (self.framework_version, self.chat_model, self.embedding_model)
        ):
            raise ValueError("record the native framework version and local model IDs")
        loopback(self.chat_url, {"http", "https"})
        loopback(self.embedding_url, {"http", "https"})
        if self.neo4j_uri:
            loopback(self.neo4j_uri, {"bolt", "bolt+s", "bolt+ssc"})
        if (
            min(self.embedding_dimensions, self.top_k) < 1
            or not 0 < self.max_output < self.context_window
        ):
            raise ValueError("invalid reference worker vector or context settings")
        if not math.isfinite(self.request_timeout) or self.request_timeout <= 0:
            raise ValueError("invalid reference worker request timeout")


def run_reference(
    dataset: Path,
    exported: Path,
    output: Path,
    interpreter: Path,
    settings: Path,
    *,
    timeout: float,
) -> dict[str, Any]:
    export = validate_reference_export(dataset, exported)
    if output.resolve() == exported.resolve():
        raise ValueError("standalone output must differ from the benchmark export")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("invalid standalone wall-clock timeout")
    if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        raise ValueError("provide the executable Python from the isolated framework environment")
    with settings.open("rb") as stream:
        options = ReferenceSettings(**tomllib.load(stream))
    is_neo4j = export["framework"] == "neo4j-graphrag"
    if is_neo4j and (
        not options.neo4j_uri
        or not options.neo4j_database
        or any(not os.environ.get(key) for key in ("NEO4J_USERNAME", "NEO4J_PASSWORD"))
    ):
        raise ValueError(
            "configure a local Neo4j database and supply both credential environment variables"
        )
    worker = Path(__file__).with_name("reference_worker.py")
    execution = {
        "framework": export["framework"],
        "export_sha256": file_hash(exported / "manifest.json"),
        "worker_sha256": file_hash(worker),
        "settings": asdict(options),
        "python": str(interpreter.absolute()),
    }
    output.mkdir(exist_ok=True, parents=True)
    with artifact_lock(output / ".run.lock"):
        manifest = output / "native-execution.json"
        raw = output / "native-questions.jsonl"
        if manifest.exists():
            if json.loads(manifest.read_text()) != execution:
                raise ValueError("native framework run already exists with a different manifest")
        elif raw.exists():
            raise ValueError("native outcome rows have no matching execution manifest")
        else:
            write_json(manifest, execution)
        allowed = {"HOME", "TMPDIR", "PATH", "LANG", "LC_ALL", "TZ", "TIKTOKEN_CACHE_DIR"}
        if is_neo4j:
            allowed.update({"NEO4J_USERNAME", "NEO4J_PASSWORD"})
        environment = {key: value for key, value in os.environ.items() if key in allowed}
        environment.update(
            {
                "HF_HUB_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "LITELLM_LOCAL_MODEL_COST_MAP": "True",
                "LITELLM_TELEMETRY": "False",
                "TOKENIZERS_PARALLELISM": "false",
            }
        )
        with (output / "framework.log").open("ab") as log:
            try:
                result = subprocess.run(
                    [
                        str(interpreter.absolute()),
                        "-I",
                        str(worker),
                        str(exported.resolve()),
                        str(output.resolve()),
                        str(manifest.resolve()),
                    ],
                    cwd=output,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    "native framework run timed out; see framework.log and any saved outcomes"
                ) from exc
        if result.returncode != 0:
            raise RuntimeError(
                f"{export['framework']} worker exited with status {result.returncode}; see {output / 'framework.log'}"
            )
        return import_reference(dataset, exported, raw, options.framework_version, output)


async def grade_reference(
    dataset: Path,
    reference_run: Path,
    out: Path,
    providers: Providers,
) -> dict[str, Any]:
    """Apply the project's correctness judge after a native run has finished."""
    from dataclasses import asdict

    from .evaluation import judge_correctness
    from .providers import REFUSAL

    if reference_run.resolve() == out.resolve():
        raise ValueError("write offline grading to a separate directory")
    manifest = json.loads((reference_run / "manifest.json").read_text())
    if manifest.get("isolated_external_run") is not True or manifest["dataset"] != dataset_manifest(
        dataset
    ):
        raise ValueError("grading requires an aligned isolated reference run")
    if manifest["dataset"]["synthetic"] != providers.settings.synthetic:
        raise ValueError("reference data and judge must agree on synthetic versus real execution")
    questions = {item.question_id: item for item in questions_at(dataset, manifest["split"])}
    rows = list(read_jsonl(reference_run / "questions.jsonl"))
    if [row["question_id"] for row in rows] != manifest["qids"]:
        raise ValueError("reference outcomes do not align with the reference manifest")
    judge = providers.settings.model("judge")
    judged = []
    for row in rows:
        question = questions[row["question_id"]]
        metrics = dict(row["metrics"])
        offset = len(providers.usage)
        label = None
        if row["status"] == "error":
            metrics["correct"] = False
        else:
            text = row["answer"]["text"]
            refused = normalize(text) == REFUSAL
            if question.answerable is False:
                metrics["refusal_accuracy"] = float(refused)
                label = "correct" if refused else "wrong"
            elif question.answerable is True and refused:
                label = "wrong"
            elif question.answers:
                label = await judge_correctness(question, text, providers)
            if label:
                metrics["correct"] = label == "correct"
        metrics["judge_grade"] = label
        metrics["judge_score"] = {"correct": 1.0, "partial": 0.5, "wrong": 0.0}.get(label or "")
        judged.append(
            {
                **row,
                "correct": metrics.get("correct"),
                "metrics": metrics,
                "evaluation_events": list(map(asdict, providers.usage[offset:])),
            }
        )
    write_json(
        out / "manifest.json",
        {
            **manifest,
            "run_id": out.name,
            "judge": asdict(judge),
            "ungraded_sha256": file_hash(reference_run / "questions.jsonl"),
        },
    )
    write_jsonl(out / "questions.jsonl", judged)
    report_run(out, judged, expected=len(manifest["qids"]), seed=manifest["dataset"]["seed"])
    return {"questions": len(judged), "scored": sum(row["correct"] is not None for row in judged)}
