"""Router levels R0-R3 (R4 confidence fallback lives in :mod:`hgrag.router.evaluate`).

| Level | Router              | Decides from                                        |
|-------|---------------------|-----------------------------------------------------|
| R0    | ``StaticRouter``    | nothing: always one path                            |
| R1    | ``RuleRouter``      | keyword patterns + number of named entities         |
| R2    | ``ClassifierRouter``| TF-IDF + linear SVM (or embeddings + logistic reg.) |
| R3    | ``LLMRouter``       | one short LLM classification call                   |

Type-predicting routers map the predicted query type to a path with ``type_to_path``; routers
trained on *outcome* labels predict the path directly.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

from hgrag.config import AppConfig
from hgrag.data.schema import QUERY_TYPES
from hgrag.index.embedders import Embedder
from hgrag.llm.gateway import LLMGateway

_NAMES = re.compile(r"\b[A-Z][\w'\u2019-]*(?:\s+(?:of|the|de|von|van)?\s*[A-Z][\w'\u2019-]*)*")
_QUESTION_WORDS = {
    "Who",
    "What",
    "When",
    "Where",
    "Which",
    "Why",
    "How",
    "In",
    "Is",
    "Are",
    "Was",
    "Did",
    "Do",
    "Does",
    "The",
    "A",
    "An",
}


class Router(Protocol):
    name: str

    def route(self, questions: Sequence[str]) -> list[str]:
        """One path per question."""
        ...


class StaticRouter:
    def __init__(self, path: str) -> None:
        self.name = f"R0-always-{path}"
        self._path = path

    def route(self, questions: Sequence[str]) -> list[str]:
        return [self._path] * len(questions)


def count_named_entities(question: str) -> int:
    names = [m.group(0) for m in _NAMES.finditer(question)]
    return sum(1 for n in names if n.split()[0] not in _QUESTION_WORDS or len(n.split()) > 1)


class RuleRouter:
    """R1: patterns from the design doc ("compare", "both", "overall", "main themes", …)."""

    name = "R1-rules"

    def __init__(self, config: AppConfig) -> None:
        self._rules = config.router.rules
        self._type_to_path = config.router.type_to_path

    def classify(self, question: str) -> str:
        q = f" {question.lower()} "
        if any(p in q for p in self._rules.global_patterns):
            return "global"
        if any(p in q for p in self._rules.comparison_patterns):
            return "comparison"
        if any(p in q for p in self._rules.temporal_patterns):
            return "temporal"
        if count_named_entities(question) >= self._rules.multi_hop_min_entities:
            return "multi_hop"
        return "single_hop"

    def route(self, questions: Sequence[str]) -> list[str]:
        return [self._type_to_path[self.classify(q)] for q in questions]


class ClassifierRouter:
    """R2: TF-IDF + linear SVM ("tfidf") or sentence embeddings + logistic regression ("encoder").

    ``labels`` are query types (mapped to paths) or paths (outcome labels), decided by the caller.
    """

    def __init__(
        self,
        config: AppConfig,
        *,
        kind: str = "tfidf",
        label_space: str = "type",
        embedder: Embedder | None = None,
    ) -> None:
        if kind == "encoder" and embedder is None:
            raise ValueError("the encoder classifier needs an embedder")
        self.name = f"R2-{kind}-{label_space}"
        self._config = config
        self._kind = kind
        self._label_space = label_space
        self._embedder = embedder
        self._model: Any = None

    def fit(self, questions: Sequence[str], labels: Sequence[str]) -> ClassifierRouter:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.svm import LinearSVC

        settings = self._config.router.classifier
        if self._kind == "tfidf":
            self._model = make_pipeline(
                TfidfVectorizer(
                    max_features=settings.max_features,
                    ngram_range=tuple(settings.ngram_range),
                    sublinear_tf=True,
                ),
                LinearSVC(C=settings.c, class_weight="balanced"),
            )
            self._model.fit(list(questions), list(labels))
        else:
            self._model = LogisticRegression(max_iter=2000, class_weight="balanced")
            self._model.fit(self._encode(questions), list(labels))
        return self

    def predict(self, questions: Sequence[str]) -> list[str]:
        if self._model is None:
            raise RuntimeError("fit the classifier first")
        features: Any = list(questions) if self._kind == "tfidf" else self._encode(questions)
        return [str(label) for label in self._model.predict(features)]

    def route(self, questions: Sequence[str]) -> list[str]:
        predicted = self.predict(questions)
        if self._label_space == "path":
            return predicted
        return [self._config.router.type_to_path[label] for label in predicted]

    def _encode(self, questions: Sequence[str]) -> np.ndarray[Any, Any]:
        assert self._embedder is not None
        return self._embedder.embed_queries(list(questions))


LLM_ROUTER_VERSION = "router/v1"
LLM_ROUTER_SYSTEM = f"""Classify the question by what it takes to answer it from a document set.
Labels: {", ".join(t for t in QUERY_TYPES if t not in ("creative", "unanswerable"))}.
- single_hop: one fact from one passage.
- multi_hop: chain facts across passages (e.g. "the founder of the company that made X").
- comparison: compare two or more named things.
- temporal: order events in time or compute time differences.
- global: summarise themes or trends across many documents.
Reply with the label only."""


class LLMRouter:
    """R3: zero-shot LLM classifier (role ``router_llm``), cached like every other call."""

    name = "R3-llm"

    def __init__(
        self, config: AppConfig, gateway: LLMGateway, model_key: str | None = None
    ) -> None:
        self._type_to_path = config.router.type_to_path
        self._gateway = gateway
        self._model_key = model_key

    async def classify(self, question: str) -> str:
        response = await self._gateway.acomplete(
            "router_llm",
            f"Question: {question}",
            system=LLM_ROUTER_SYSTEM,
            max_tokens=10,
            model_key=self._model_key,
        )
        text = response.text.strip().lower()
        return next((t for t in QUERY_TYPES if t in text), "single_hop")

    def estimate_cost(self, questions: Sequence[str]) -> float:
        return sum(
            self._gateway.estimate_cost(
                "router_llm",
                f"Question: {q}",
                system=LLM_ROUTER_SYSTEM,
                max_tokens=10,
                model_key=self._model_key,
            )
            for q in questions
        )

    async def aroute(self, questions: Sequence[str]) -> list[str]:
        import asyncio

        types = await asyncio.gather(*(self.classify(q) for q in questions))
        return [self._type_to_path.get(t, "naive") for t in types]
