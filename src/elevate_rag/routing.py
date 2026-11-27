"""Query-only routers and safe, portable linear SVM artifacts."""

import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .config import MODES
from .data import Question, stable_hash, write_json
from .providers import ProviderError, Providers

if TYPE_CHECKING:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC

ROUTE_MAP = {
    "single-hop": "V4",
    "factual": "V4",
    "fact": "V4",
    "fact-retrieval": "V4",
    "multi-hop": "local",
    "bridge": "local",
    "2-hop": "local",
    "3-hop": "local",
    "4-hop": "local",
    "reasoning": "local",
    "complex-reasoning": "local",
    "inference": "local",
    "comparison": "mix",
    "temporal": "mix",
    "global": "global",
    "summary": "global",
    "contextual-summarize": "global",
    "contextual-summarization": "global",
    "creative-generation": "global",
    "null": "V4",
    "unanswerable": "V4",
}
RULES = {
    "global": [r"\b(?:overall|main themes?|across the corpus|summari[sz]e|common patterns)\b"],
    "mix": [r"\b(?:compare|contrast|both|versus|earliest|latest|before|after)\b"],
    "local": [r"\b(?:related to|relationship between|connected to)\b"],
}
TOKEN_PATTERN = r"(?u)\b\w\w+\b"


def route_for_query_type(label: str) -> str:
    return ROUTE_MAP.get(label.replace("_", "-"), "V4")


def fit_svm(examples: Sequence[tuple[str, str, str]]) -> tuple["TfidfVectorizer", "LinearSVC"]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC

    labels = [label for _, _, label in examples]
    if len(set(labels)) < 2:
        raise ValueError("SVM training needs at least two labels")
    if len({key for key, _, _ in examples}) != len(examples):
        raise ValueError("duplicate router training IDs")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), token_pattern=TOKEN_PATTERN, dtype=np.float64)
    vectors = vectorizer.fit_transform([query for _, query, _ in examples])
    return vectorizer, LinearSVC(random_state=0).fit(vectors, labels)


@dataclass(frozen=True, slots=True)
class Route:
    level: str
    initial: str
    final: str
    reason: str
    confidence: float | None = None
    fallback_reason: str | None = None


def classify_rules(query: str, custom: dict[str, list[str]]) -> tuple[str, str]:
    for mode, patterns in (custom or RULES).items():
        if any(re.search(pattern, query, re.I) for pattern in patterns):
            return mode, "matched configured query pattern"
    entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", query)
    question_words = {"Who", "What", "Which", "Where", "When", "Why", "How", "Is", "Was", "Did"}
    if len(set(entities) - question_words) >= 2:
        return "local", "multiple capitalized entity mentions"
    return "V4", "default factual route"


@dataclass(frozen=True, slots=True)
class SVM:
    labels: list[str]
    tokens: list[str]
    idf: list[float]
    weights: list[list[float]]
    intercept: list[float]
    strategy: str
    training_digest: str
    training_dataset: str | None = None
    training_dev_sha256: str | None = None

    @classmethod
    def train(cls, examples: Sequence[tuple[str, str, str]], strategy: str) -> "SVM":
        if strategy not in {"query-type", "outcome"}:
            raise ValueError("unsupported training-label strategy")
        labels = sorted({label for _, _, label in examples})
        digest = stable_hash(sorted(examples))
        if len(labels) == 1:
            if len({key for key, _, _ in examples}) != len(examples):
                raise ValueError("duplicate router training IDs")
            return cls(labels, [], [], [], [], strategy, digest)
        vectorizer, classifier = fit_svm(examples)
        return cls(
            classifier.classes_.tolist(),
            vectorizer.get_feature_names_out().tolist(),
            vectorizer.idf_.tolist(),
            classifier.coef_.tolist(),
            classifier.intercept_.tolist(),
            strategy,
            digest,
        )

    def predict(self, query: str) -> str:
        if len(self.labels) == 1:
            return self.labels[0]
        words = re.findall(TOKEN_PATTERN, query.lower())
        counts = Counter([*words, *(f"{a} {b}" for a, b in zip(words, words[1:]))])
        features = np.asarray(
            [counts[token] * idf for token, idf in zip(self.tokens, self.idf, strict=True)]
        )
        if (magnitude := np.linalg.norm(features)) > 0:
            features /= magnitude
        scores = np.asarray(self.weights) @ features + np.asarray(self.intercept)
        if len(self.labels) == 2:
            return self.labels[int(scores[0] > 0)]
        return self.labels[int(np.argmax(scores))]

    def route(self, query: str) -> str:
        label = self.predict(query)
        return label if self.strategy == "outcome" else route_for_query_type(label)

    def save(self, path: Path) -> None:
        payload = asdict(self)
        write_json(path, {"schema": 1, "checksum": stable_hash(payload), "model": payload})

    @classmethod
    def read(cls, path: Path) -> "SVM":
        wrapped = json.loads(path.read_text())
        data = wrapped["model"]
        if wrapped.get("schema") != 1 or wrapped.get("checksum") != stable_hash(data):
            raise ValueError("router model failed schema or integrity check")
        model = cls(**data)
        class_count = len(model.labels)
        expected_rows = {1: 0, 2: 1}.get(class_count, class_count)
        if (
            len(set(model.labels)) != len(model.labels)
            or not model.labels
            or (len(model.labels) == 1 and bool(model.tokens))
            or model.strategy not in {"query-type", "outcome"}
            or (model.strategy == "outcome" and set(model.labels) - set(MODES))
            or len(model.idf) != len(model.tokens)
            or len(model.weights) != expected_rows
            or len(model.intercept) != expected_rows
            or any(len(row) != len(model.tokens) for row in model.weights)
            or not all(
                map(
                    math.isfinite,
                    [*model.idf, *model.intercept, *(w for row in model.weights for w in row)],
                )
            )
        ):
            raise ValueError("invalid router weights")
        return model


def classification(actual: Sequence[str], predicted: Sequence[str]) -> dict[str, float | None]:
    if len(actual) != len(predicted):
        raise ValueError("classification alignment error")
    if not actual:
        return {"accuracy": None, "macro_f1": None}
    labels = set(actual) | set(predicted)
    pairs = list(zip(actual, predicted, strict=True))
    f1 = []
    for label in labels:
        tp = sum(a == p == label for a, p in pairs)
        possible = sum(a == label for a in actual) + sum(p == label for p in predicted)
        f1.append(2 * tp / possible)
    return {"accuracy": sum(a == p for a, p in pairs) / len(actual), "macro_f1": sum(f1) / len(f1)}


def cross_validate(
    questions: Sequence[Question],
    by_domain: bool,
    seed: int,
    *,
    outcome_labels: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold, StratifiedKFold

    selected = [q for q in questions if outcome_labels is None or q.question_id in outcome_labels]
    labels = [
        q.query_type if outcome_labels is None else outcome_labels[q.question_id] for q in selected
    ]
    texts = [" ".join(q.text.casefold().split()) for q in selected]
    if by_domain:
        groups = [q.domain for q in selected]
        splits = LeaveOneGroupOut().split(selected, labels, groups=groups)
    else:
        distinct = Counter(label for label, _ in set(zip(labels, texts, strict=True)))
        if min(distinct.values(), default=0) < 5:
            raise ValueError("five-fold stratification requires five distinct questions per label")
        options = {"n_splits": 5, "shuffle": True, "random_state": seed}
        splits = (
            StratifiedKFold(**options).split(selected, labels)
            if len(set(texts)) == len(texts)
            else StratifiedGroupKFold(**options).split(selected, labels, groups=texts)
        )
    results = []
    for number, (train, test) in enumerate(splits, 1):
        test_texts = {texts[i] for i in test}
        examples = [
            (selected[i].question_id, selected[i].text, labels[i])
            for i in train
            if texts[i] not in test_texts
        ]
        train_labels = {label for _, _, label in examples}
        constant = len(train_labels) == 1
        if constant:
            predicted = [next(iter(train_labels))] * len(test)
        else:
            vectorizer, classifier = fit_svm(examples)
            predicted = classifier.predict(
                vectorizer.transform([selected[i].text for i in test])
            ).tolist()
        results.append(
            {
                "fold": number,
                "test_ids": [selected[i].question_id for i in test],
                "training_digest": stable_hash(sorted(examples)),
                "training_size": len(examples),
                "constant_training_class": constant,
                "excluded_duplicate_training_rows": len(train) - len(examples),
                **classification([labels[i] for i in test], predicted),
            }
        )
    return results


async def decide(
    level: str,
    query: str,
    static: str,
    providers: Providers,
    svm: SVM | None = None,
) -> Route:
    reason = "fixed experimental arm"
    mode = static
    if level == "R1":
        mode, reason = classify_rules(query, providers.settings.rules)
    elif level == "R2":
        if svm is None:
            raise ValueError("R2 requires a trained classifier")
        mode, reason = svm.route(query), f"TF-IDF plus linear SVM ({svm.strategy})"
    elif level == "R3":
        try:
            raw = await providers.chat(
                "router",
                'Classify only the supplied question text. Return one JSON object with key "route". '
                "Allowed routes are V4 (factual), local (entity or reasoning), mix (comparison or temporal), "
                "and global (themes and summaries).",
                json.dumps({"question": query}),
            )
            value = json.loads(raw)
            if (
                not isinstance(value, dict)
                or set(value) != {"route"}
                or value["route"] not in {"V4", "local", "mix", "global"}
            ):
                raise ValueError("invalid router decision")
            mode, reason = value["route"], "structured model decision"
        except (ValueError, ProviderError, TypeError) as exc:
            mode, _ = classify_rules(query, providers.settings.rules)
            reason = f"rule fallback after {type(exc).__name__}"
    elif level == "R4":
        mode, reason = providers.settings.initial_route, "configured inexpensive initial path"
    elif level != "R0":
        raise ValueError(f"unknown router level {level}")
    if mode not in MODES:
        raise ValueError(f"unsupported route {mode}")
    return Route(level, mode, mode, reason)


def cheapest_correct(outcomes: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    unknowable = set()
    for row in outcomes:
        if row.get("correct") is not True:
            continue
        key = row["question_id"]
        if row.get("estimated_usd") is None:
            unknowable.add(key)
            continue
        rank = (
            row["estimated_usd"],
            math.inf if row.get("tokens") is None else row["tokens"],
            row["mode"],
        )
        existing = output.get(key)
        if existing is None or rank < existing["rank"]:
            output[key] = {"mode": row["mode"], "rank": rank}
    return {key: value for key, value in output.items() if key not in unknowable}


def calibrate_threshold(
    initial: Sequence[dict[str, Any]],
    fused: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Tune confidence on aligned development outcomes under the one-fallback policy."""
    cheap = {row["question_id"]: row for row in initial}
    mix = {row["question_id"]: row for row in fused}
    if (
        not cheap
        or cheap.keys() != mix.keys()
        or len(cheap) != len(initial)
        or len(mix) != len(fused)
    ):
        raise ValueError("confidence calibration requires unique, aligned development outcomes")
    if any(
        row["status"] != "ok" or type(row.get("correct")) is not bool for row in [*initial, *fused]
    ):
        raise ValueError("confidence calibration requires complete, scored development outcomes")
    scores = [
        row["route"]["confidence"] for row in initial if row["route"]["confidence"] is not None
    ]
    if not scores or not all(math.isfinite(item) for item in scores):
        raise ValueError("confidence calibration requires finite measured scores")
    thresholds = sorted({*scores, math.nextafter(max(scores), math.inf)})
    options = []
    for threshold in thresholds:
        correct, escalated = 0, 0
        for key, row in cheap.items():
            confidence = row["route"]["confidence"]
            use_mix = confidence is None or confidence < threshold or row["answer"]["refused"]
            escalated += int(use_mix)
            correct += int((mix[key] if use_mix else row)["correct"])
        options.append(
            {
                "threshold": threshold,
                "accuracy": correct / len(cheap),
                "fallback_count": escalated,
            }
        )
    return {
        "development_questions": len(cheap),
        "candidates": options,
        "selected": min(
            options, key=lambda row: (-row["accuracy"], row["fallback_count"], row["threshold"])
        ),
    }
