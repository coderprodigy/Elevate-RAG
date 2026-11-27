"""Upstream format adapters and reproducible sampling."""

import gzip
import heapq
import html
import json
import re
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import httpx

from .data import Document, Evidence, Question, file_hash, prepare, stable_hash

MAIN = (
    "natural-questions",
    "hotpotqa",
    "musique",
    "multihop-rag",
    "graphrag-novel",
    "graphrag-medical",
)
DATASETS = (*MAIN, "ragrouter-bench", "ultradomain")
GRAPH_URL = (
    "https://raw.githubusercontent.com/GraphRAG-Bench/GraphRAG-Benchmark/"
    "fdbab5959b18c96532580877ffe27d112bccc0ec/Datasets"
)
MULTIHOP_URL = (
    "https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/"
    "e37cfddd4286ea6c5723f159cff03289381548e4"
)
ROUTER_DOMAINS = ("musique", "quality", "ultraDomain_legal", "graphragBench_medical")
ROUTER_URL = (
    "https://huggingface.co/datasets/Chaplain0908/RAGRouter/resolve/"
    "abff378d9202caba636aa4d5bb9243db8103f70f/data"
)
PUBLIC_FILES = {
    "natural-questions": {
        "simplified-nq-dev.jsonl.gz": (
            "https://huggingface.co/datasets/artefactory/natural-questions/resolve/"
            "749c02daef3e3124c440559868183e6495e41ff5/data/simplified-nq-dev.jsonl.gz"
        )
    },
    "musique": {
        "musique_full_v1.0_dev.jsonl": (
            "https://huggingface.co/datasets/voidful/MuSiQue/resolve/"
            "a7d9f9adf6191604fc67cde318ee1a86fcf7babc/musique_full_v1.0_dev.jsonl"
        )
    },
    "ragrouter-bench": {
        f"{domain}/{filename}": f"{ROUTER_URL}/{domain}/{filename}"
        for domain in ROUTER_DOMAINS
        for filename in ("Corpus.json", "Question.json")
    },
    "hotpotqa": {
        "validation-00000-of-00001.parquet": (
            "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/"
            "1908d6afbbead072334abe2965f91bd2709910ab/distractor/validation-00000-of-00001.parquet"
        )
    },
    "multihop-rag": {
        name: f"{MULTIHOP_URL}/{name}" for name in ("corpus.json", "MultiHopRAG.json")
    },
    **{
        f"graphrag-{domain}": {
            f"{domain}.parquet": f"{GRAPH_URL}/Corpus/{domain}.parquet",
            f"{domain}_questions.parquet": f"{GRAPH_URL}/Questions/{domain}_questions.parquet",
        }
        for domain in ("medical", "novel")
    },
}


def download(url: str, path: Path, checksum: str | None = None) -> str:
    if not url.startswith("https://"):
        raise ValueError("dataset download requires HTTPS")
    if path.exists():
        result = file_hash(path)
        if checksum and result != checksum:
            raise ValueError(f"cached download checksum mismatch: {path}")
        return result
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        pending = Path(handle.name)
        try:
            with httpx.stream("GET", url, follow_redirects=True, timeout=60) as response:
                response.raise_for_status()
                for block in response.iter_bytes():
                    handle.write(block)
            handle.flush()
            result = file_hash(pending)
            if checksum and result != checksum:
                raise ValueError(f"download checksum mismatch: {path}")
            pending.replace(path)
            return result
        finally:
            pending.unlink(missing_ok=True)


def rows(path: Path, *, allow_object_lines: bool = False) -> Iterator[dict[str, Any]]:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        for batch in pq.ParquetFile(path).iter_batches():
            yield from batch.to_pylist()
        return
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        jsonl = ".jsonl" in path.name
        if allow_object_lines and not jsonl:
            first = handle.read(1)
            while first and first.isspace():
                first = handle.read(1)
            jsonl = first == "{"
            handle.seek(0)
        if jsonl:
            for number, line in enumerate(handle, 1):
                if allow_object_lines and not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("expected object")
                    yield row
                except ValueError as exc:
                    raise ValueError(f"{path}:{number}: {exc}") from exc
        else:
            values = json.load(handle)
            if not isinstance(values, list) or any(not isinstance(row, dict) for row in values):
                raise ValueError(f"{path}: expected an array of JSON objects")
            yield from values


def sample(
    source: Iterable[dict[str, Any]], size: int | None, seed: int, key: str
) -> list[dict[str, Any]]:
    def ranking(row: dict[str, Any]) -> str:
        return stable_hash(seed, str(row[key]))

    if size is None:
        return sorted(source, key=ranking)
    ordered = heapq.nsmallest(size, source, key=ranking)
    if len(ordered) < size:
        raise ValueError(f"expected at least {size} eligible questions, found {len(ordered)}")
    return ordered


def make_doc(dataset: str, title: str, content: str, key: str | None = None) -> Document:
    return Document(
        "d-" + stable_hash(dataset, key or title, content), title, content, key or dataset
    )


def distinct_evidence(items: Iterable[Evidence]) -> tuple[Evidence, ...]:
    return tuple(dict.fromkeys(items))


def hotpot(path: Path, seed: int, limit: int | None) -> tuple[list[Document], list[Question]]:
    documents, questions = [], []
    eligible = (
        {**row, "_id": row.get("_id") or row["id"]} for row in rows(path) if row["level"] == "hard"
    )
    for row in sample(eligible, limit, seed, "_id"):
        by_title: dict[str, list[Evidence | None]] = {}
        context = row["context"]
        if isinstance(context, dict):
            context = zip(context["title"], context["sentences"], strict=True)
        for title, sentences in context:
            if title in by_title:
                raise ValueError(f"{row['_id']}: duplicate context title {title}")
            if not sentences or not any(sentence.strip() for sentence in sentences):
                by_title[title] = [None] * len(sentences)
                continue
            document = make_doc("hotpotqa", title, "\n".join(sentences))
            documents.append(document)
            offset = 0
            spans: list[Evidence | None] = []
            for sentence in sentences:
                spans.append(
                    Evidence(document.doc_id, offset, offset + len(sentence)) if sentence else None
                )
                offset += len(sentence) + 1
            by_title[title] = spans
        evidence = []
        supporting = row["supporting_facts"]
        if isinstance(supporting, dict):
            supporting = zip(supporting["title"], supporting["sent_id"], strict=True)
        for title, index in supporting:
            span = by_title[title][index]
            if span is None or index < 0:
                raise ValueError(f"{row['_id']}: empty or invalid supporting fact")
            evidence.append(span)
        questions.append(
            Question(
                str(row["_id"]),
                row["question"],
                (row["answer"],),
                "multi-hop" if row["type"] == "bridge" else row["type"],
                True,
                distinct_evidence(evidence),
                "wikipedia",
            )
        )
    return documents, questions


def musique(
    path: Path, seed: int, answerable: int | None, unanswerable: int | None
) -> tuple[list[Document], list[Question]]:
    by_answerability: dict[bool, list[dict[str, Any]]] = {True: [], False: []}
    for row in rows(path):
        if type(row["answerable"]) is not bool:
            raise ValueError(f"{row['id']}: invalid answerability")
        by_answerability[row["answerable"]].append(row)
    negatives = sample(by_answerability[False], unanswerable, seed, "id")
    negative_ids = {row["id"] for row in negatives}
    # The upstream Full release reuses each source ID for both variants.
    positives = sample(
        (row for row in by_answerability[True] if row["id"] not in negative_ids),
        answerable,
        seed,
        "id",
    )
    documents, questions = [], []
    for row in positives + negatives:
        evidence = []
        for passage in row["paragraphs"]:
            document = make_doc("musique", passage["title"], passage["paragraph_text"])
            documents.append(document)
            if passage["is_supporting"]:
                evidence.append(Evidence(document.doc_id))
        answer_list = (
            tuple(dict.fromkeys([row["answer"], *row.get("answer_aliases", [])]))
            if row["answerable"]
            else ()
        )
        hops = str(row["id"]).partition("hop")[0]
        questions.append(
            Question(
                str(row["id"]),
                row["question"],
                answer_list,
                f"{hops}-hop" if hops.isdigit() else "multi-hop",
                row["answerable"],
                distinct_evidence(evidence),
                "wikipedia",
            )
        )
    return documents, questions


def nq(path: Path, seed: int, limit: int | None) -> tuple[list[Document], list[Question]]:
    def annotations(row: dict[str, Any]) -> Iterator[dict[str, Any]]:
        for annotation in row["annotations"]:
            if annotation["long_answer"]["candidate_index"] >= 0 and (
                annotation["short_answers"] or annotation["yes_no_answer"] in {"YES", "NO"}
            ):
                yield annotation

    selected = sample(
        (row for row in rows(path) if any(annotations(row))), limit, seed, "example_id"
    )
    documents, questions = [], []
    for row in selected:
        tokens = row.get("document_tokens")
        if isinstance(tokens, list):
            plain = [
                "" if token["html_token"] else html.unescape(token["token"]) for token in tokens
            ]
        else:
            plain = [
                "" if token.startswith("<") and token.endswith(">") else html.unescape(token)
                for token in row["document_text"].split(" ")
            ]

        def span_text(span: dict[str, int]) -> str:
            return " ".join(
                token for token in plain[span["start_token"] : span["end_token"]] if token
            )

        title = row.get("document_title") or row["document_url"].split("/")[-1]
        candidates = row["long_answer_candidates"]
        gold = {entry["long_answer"]["candidate_index"] for entry in annotations(row)}
        evidence, answers = [], []
        for index, candidate in enumerate(candidates):
            if (candidate["top_level"] or index in gold) and (content := span_text(candidate)):
                document = make_doc("natural-questions", title, content, row["document_url"])
                documents.append(document)
                if index in gold:
                    evidence.append(Evidence(document.doc_id))
        for entry in annotations(row):
            yes_no = entry["yes_no_answer"]
            answer = (
                yes_no.lower()
                if yes_no in {"YES", "NO"}
                else ", ".join(span_text(short) for short in entry["short_answers"])
            )
            if answer:
                answers.append(answer)
        if not answers or not evidence:
            raise ValueError(
                f"{row['example_id']}: gold short answer or long passage could not be decoded"
            )
        questions.append(
            Question(
                str(row["example_id"]),
                row["question_text"],
                tuple(dict.fromkeys(answers)),
                "single-hop",
                True,
                distinct_evidence(evidence),
                "wikipedia",
            )
        )
    return documents, questions


def multihop(corpus: Path, requests: Path) -> tuple[list[Document], list[Question]]:
    documents = [
        make_doc("multihop-rag", row["title"], row["body"], row["url"]) for row in rows(corpus)
    ]
    by_url = {doc.source: doc for doc in documents}
    if len(by_url) != len(documents):
        raise ValueError("MultiHop-RAG corpus URLs are ambiguous")
    questions = []
    for row in rows(requests):
        query_type = row["question_type"].casefold().removesuffix("_query")
        answerable = query_type != "null"
        evidence, facts = [], []
        for entry in row["evidence_list"]:
            doc = by_url[entry["url"]]
            fact = entry.get("fact", "")
            if fact:
                facts.append(fact)
            offset = doc.text.find(fact) if fact else -1
            evidence.append(
                Evidence(doc.doc_id, offset, offset + len(fact))
                if offset >= 0
                else Evidence(doc.doc_id)
            )
        questions.append(
            Question(
                str(row.get("id") or "q-" + stable_hash(row["query"], row["question_type"])),
                row["query"],
                (row["answer"],) if answerable else (),
                query_type,
                answerable,
                distinct_evidence(evidence),
                "news",
                tuple(facts),
            )
        )
    return documents, questions


def graphrag(corpus: Path, requests: Path, domain: str) -> tuple[list[Document], list[Question]]:
    documents = [
        make_doc(f"graphrag-{domain}", row["corpus_name"], row["context"]) for row in rows(corpus)
    ]
    by_name = {doc.title: doc for doc in documents}
    if len(by_name) != len(documents):
        raise ValueError("GraphRAG-Bench source names are ambiguous")
    questions = []
    for row in rows(requests):
        doc = by_name[row["source"]]
        raw = row.get("evidence")
        if isinstance(raw, str):
            raw = [raw]
        facts = tuple(str(item) for item in (raw or []) if item)
        triples = row.get("evidence_triple")
        if isinstance(triples, str):
            triples = [triples]
        spans = []
        for fact in facts:
            if (offset := doc.text.find(fact)) >= 0:
                spans.append(Evidence(doc.doc_id, offset, offset + len(fact)))
        answer = row["answer"]
        refs = (answer,) if isinstance(answer, str) else tuple(answer)
        questions.append(
            Question(
                row["source"] + ":" + str(row["id"]),
                row["question"],
                refs,
                re.sub(r"[^a-z0-9]+", "-", row["question_type"].lower()).strip("-"),
                True,
                distinct_evidence(spans),
                domain,
                facts,
                tuple(str(item) for item in (triples or []) if item),
            )
        )
    return documents, questions


def ragrouter(root: Path) -> tuple[list[Document], list[Question], list[Path]]:
    documents, questions = [], []
    paths: list[Path] = []
    for domain in ROUTER_DOMAINS:
        corpus, requests = root / domain / "Corpus.json", root / domain / "Question.json"
        paths.extend((corpus, requests))
        indexed = {}
        for row in rows(corpus, allow_object_lines=True):
            key = str(row.get("doc_id", row.get("id")))
            source = f"{domain}:{key}"
            title = row["title"].strip() or source
            text = row["text"] if "text" in row else row["context"]
            document = make_doc(domain, title, text, source)
            if key in indexed:
                raise ValueError(f"{domain}: duplicate corpus ID {key}")
            indexed[key] = document
            documents.append(document)
        for row in rows(requests, allow_object_lines=True):
            reference = row.get("answer")
            support = tuple(map(str, row.get("supporting_facts") or ()))
            evidence = tuple(Evidence(indexed[key].doc_id) for key in support if key in indexed)
            questions.append(
                Question(
                    domain + ":" + str(row.get("question_id", row.get("id"))),
                    row["question"],
                    (reference,) if reference else (),
                    row["query_type"] if "query_type" in row else row["type"],
                    True,
                    evidence,
                    domain,
                    support,
                )
            )
    return documents, questions, paths


def fixture(root: Path, destination: Path) -> dict[str, Any]:
    docs_path, questions_path = root / "documents.jsonl", root / "questions.jsonl"
    return prepare(
        destination,
        (Document(**row) for row in rows(docs_path)),
        (Question.from_row(row) for row in rows(questions_path)),
        dataset="synthetic",
        version="1",
        seed=42,
        synthetic=True,
        sources={path.name: file_hash(path) for path in (docs_path, questions_path)},
    )


def prepare_dataset(
    dataset: str, inputs: list[Path], out: Path, seed: int, version: str
) -> dict[str, Any]:
    expected = (
        2 if dataset.startswith("graphrag-") or dataset in ("multihop-rag", "ultradomain") else 1
    )
    if dataset not in DATASETS or len(inputs) != expected or not version.strip():
        raise ValueError(f"{dataset}: expected {expected} input file(s) or directory and a version")
    if dataset == "hotpotqa":
        docs, questions = hotpot(inputs[0], seed, 1000)
    elif dataset == "musique":
        docs, questions = musique(inputs[0], seed, 500, 100)
    elif dataset == "natural-questions":
        docs, questions = nq(inputs[0], seed, 500)
    elif dataset == "multihop-rag":
        docs, questions = multihop(*inputs)
    elif dataset.startswith("graphrag-"):
        docs, questions = graphrag(inputs[0], inputs[1], dataset.split("-")[1])
    elif dataset == "ragrouter-bench":
        docs, questions, inputs = ragrouter(inputs[0])
    else:
        docs = [Document(**row) for row in rows(inputs[0])]
        questions = [Question.from_row(row) for row in rows(inputs[1])]
    sources = {f"{index}:{path.name}": file_hash(path) for index, path in enumerate(inputs)}
    return prepare(
        out, docs, questions, dataset=dataset, version=version, seed=seed, sources=sources
    )
