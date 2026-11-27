"""Query routers R0–R3 (FR-R0–FR-R3). R4 (confidence fallback) lives in `routing.fallback`
because it needs to execute retrieval; the oracle lives in `routing.oracle` (offline only).

Every decision carries the level, the predicted query type, the chosen mode and human-readable
reasons, so routes are explainable and reproducible.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from elevate_rag.config import RouterConfig
from elevate_rag.evaluation.judge import parse_json_object
from elevate_rag.providers.llm import BaseLLM
from elevate_rag.types import QUERY_TYPES
from elevate_rag.utils import ConfigError, ProviderError, stable_hash


@dataclass
class RouteDecision:
    level: str
    mode: str
    query_type: str | None = None
    reasons: list[str] = field(default_factory=list)
    confidence: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------------------------------ R1 rules

_GLOBAL = re.compile(
    r"\b(overall|main themes?|key themes?|summari[sz]e|summary|overview|in "
    r"general|across the|what are the (main|key|major)|trends?|common "
    r"(patterns|themes)|big picture)\b",
    re.I,
)
_COMPARISON = re.compile(
    r"\b(compare|comparison|both|either|neither|same|similar|difference|"
    r"differ|which (one|of)|(more|less|fewer|greater|larger|smaller|older|"
    r"younger|higher|lower|longer|shorter|bigger)\b.*\bthan|"
    r"\bor\b.*\?$)",
    re.I,
)
_TEMPORAL = re.compile(
    r"\b(before|after|earlier|later|first|last|latest|most recent|prior to|"
    r"since|until|during|timeline|sequence|(19|20)\d\d|january|february|march|"
    r"april|may|june|july|august|september|october|november|december)\b",
    re.I,
)
_BRIDGE = re.compile(
    r"\b(of the|that|which|whose|who (founded|wrote|directed|created|"
    r"developed|owns?|married))\b|'s\b",
    re.I,
)
# "the founder of X", "the spouse of the performer of Y": a relation-of-entity chain is the
# canonical bridge (multi-hop) pattern.
_RELATIONAL_OF = re.compile(
    r"\b(the|its|their|his|her)\s+(founder|co-?founder|director|author|"
    r"creator|developer|producer|performer|composer|spouse|wife|husband|"
    r"father|mother|son|daughter|sibling|brother|sister|capital|owner|"
    r"ceo|president|leader|governor|mayor|birthplace|headquarters|"
    r"publisher|manufacturer|parent company|successor|predecessor)"
    r"\s+of\b",
    re.I,
)
_ENTITY = re.compile(r"(?<!^)(?<![.?!]\s)\b[A-Z][a-zA-Z0-9\-]+(?:\s+[A-Z][a-zA-Z0-9\-]+)*")


def rule_features(question: str) -> dict[str, Any]:
    q = question.strip()
    ents = [m.group(0) for m in _ENTITY.finditer(q)]
    return {
        "global": bool(_GLOBAL.search(q)),
        "comparison": bool(_COMPARISON.search(q)),
        "temporal": bool(_TEMPORAL.search(q)),
        "bridge": len(_BRIDGE.findall(q)),
        "relational_of": len(_RELATIONAL_OF.findall(q)),
        "n_entities": len(set(ents)),
        "question_word": (q.split()[0].lower() if q.split() else ""),
        "n_tokens": len(q.split()),
    }


def classify_rules(question: str) -> tuple[str, list[str]]:
    f = rule_features(question)
    if f["global"]:
        return "global", ["global/thematic cue"]
    if f["comparison"] and f["n_entities"] >= 2:
        return "comparison", [f"comparison cue with {f['n_entities']} entities"]
    if f["temporal"] and f["n_entities"] >= 2:
        return "temporal", [f"temporal cue with {f['n_entities']} entities"]
    if f["relational_of"]:
        return "multi_hop", [f"{f['relational_of']} relation-of-entity chain(s)"]
    if f["bridge"] >= 2 or (f["bridge"] >= 1 and f["n_entities"] >= 2):
        return "multi_hop", [f"{f['bridge']} bridge pattern(s), {f['n_entities']} entities"]
    if f["comparison"]:
        return "comparison", ["comparison cue"]
    return "single_hop", ["no multi-hop/global cue"]


class Router:
    """R0 static, R1 rules, R2 classifier, R3 LLM. (R4 → routing.fallback.)"""

    def __init__(
        self,
        cfg: RouterConfig,
        *,
        classifier: TfidfSvmRouter | None = None,
        llm: BaseLLM | None = None,
    ) -> None:
        self.cfg = cfg
        self.classifier = classifier
        self.llm = llm
        if cfg.level == "classifier" and classifier is None:
            raise ConfigError("router level 'classifier' needs a trained classifier")
        if cfg.level == "llm" and llm is None:
            raise ConfigError("router level 'llm' needs models.router_llm")

    def _mode_for(self, qtype: str) -> str:
        return self.cfg.type_to_mode.get(qtype, self.cfg.type_to_mode.get("other", "vector"))

    async def decide(self, question: str) -> RouteDecision:
        lvl = self.cfg.level
        if lvl == "static":
            return RouteDecision("R0", self.cfg.static_mode, reasons=["static"])
        if lvl == "rules":
            qt, why = classify_rules(question)
            return RouteDecision("R1", self._mode_for(qt), qt, why)
        if lvl == "classifier":
            assert self.classifier is not None
            label, margin = self.classifier.predict_one(question)
            if self.classifier.scheme == "outcome":
                return RouteDecision(
                    "R2", label, None, [f"outcome-label SVM (margin {margin:.2f})"], margin
                )
            return RouteDecision(
                "R2",
                self._mode_for(label),
                label,
                [f"type-label SVM (margin {margin:.2f})"],
                margin,
            )
        if lvl == "llm":
            return await self._decide_llm(question)
        raise ConfigError(f"router level {lvl!r} is executed by routing.fallback, not Router")

    async def _decide_llm(self, question: str) -> RouteDecision:
        assert self.llm is not None
        rule_type, _ = classify_rules(question)
        try:
            res = await self.llm.complete(
                [{"role": "user", "content": LLM_ROUTER_PROMPT.format(question=question)}],
                system=LLM_ROUTER_SYSTEM,
                max_tokens=64,
                task="route",
                hints={"task": "route", "rule_type": rule_type},
            )
            obj = parse_json_object(res.text) or {}
            qt = str(obj.get("query_type", "")).strip().lower()
        except ProviderError as exc:
            return RouteDecision(
                "R3",
                self._mode_for(rule_type),
                rule_type,
                [f"LLM router failed ({exc}); fell back to R1 rules"],
            )
        if qt not in LLM_ROUTER_TYPES:
            return RouteDecision(
                "R3",
                self._mode_for(rule_type),
                rule_type,
                [f"invalid LLM label {qt!r}; fell back to R1 rules"],
            )
        return RouteDecision("R3", self._mode_for(qt), qt, ["LLM classification"])


LLM_ROUTER_TYPES = ("single_hop", "multi_hop", "comparison", "temporal", "global")
LLM_ROUTER_SYSTEM = (
    "You classify questions for a retrieval system. Reply with JSON only: "
    '{"query_type": "single_hop|multi_hop|comparison|temporal|global"}'
)
LLM_ROUTER_PROMPT = """Definitions:
- single_hop: one fact about one entity ("When was X founded?")
- multi_hop: needs a chain of facts ("Where was the founder of X born?")
- comparison: compares two or more entities ("Which is older, X or Y?")
- temporal: about order or timing of events across sources ("Did X happen before Y?")
- global: about themes or summaries across the whole corpus ("What are the main themes?")

Question: {question}
JSON:"""


# ------------------------------------------------------------------------------- R2 classifier


class TfidfSvmRouter:
    """TF-IDF (word 1–2-grams) + linear SVM (Bansal & Agarwal 2026 baseline).

    `scheme="type"` predicts query types (dataset labels); `scheme="outcome"` predicts the
    cheapest correct mode (oracle labels from the dev split)."""

    def __init__(self, scheme: str = "type", c: float = 1.0) -> None:
        if scheme not in ("type", "outcome"):
            raise ConfigError("scheme must be 'type' or 'outcome'")
        self.scheme = scheme
        self.c = c
        self.pipeline: Any = None
        self.meta: dict[str, Any] = {}

    def _make(self) -> Any:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import make_pipeline
        from sklearn.svm import LinearSVC

        return make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=True, min_df=1),
            LinearSVC(C=self.c, class_weight="balanced", random_state=13),
        )

    def fit(self, texts: list[str], labels: list[str]) -> TfidfSvmRouter:
        if len(set(labels)) < 2:
            raise ValueError("router classifier needs at least two label classes")
        import sklearn

        self.pipeline = self._make().fit(texts, labels)
        self.meta = {
            "scheme": self.scheme,
            "labels": sorted(set(labels)),
            "n_train": len(texts),
            "train_hash": stable_hash([texts, labels]),
            "c": self.c,
            "sklearn": sklearn.__version__,
            "created_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        }
        return self

    def predict(self, texts: list[str]) -> list[str]:
        return [str(x) for x in self.pipeline.predict(texts)]

    def predict_one(self, text: str) -> tuple[str, float]:
        dec = self.pipeline.decision_function([text])
        label = str(self.pipeline.predict([text])[0])
        margin = float(abs(dec).max()) if hasattr(dec, "max") else 0.0
        return label, margin

    def save(self, path: Path) -> None:
        import joblib

        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path / "model.joblib")
        (path / "meta.json").write_text(json.dumps(self.meta, indent=2))

    @classmethod
    def load(cls, path: Path) -> TfidfSvmRouter:
        import joblib
        import sklearn

        meta = json.loads((path / "meta.json").read_text())
        if meta.get("sklearn") != sklearn.__version__:
            raise ConfigError(
                f"router model trained with scikit-learn {meta.get('sklearn')}, "
                f"running {sklearn.__version__}; retrain it"
            )
        r = cls(meta["scheme"], meta.get("c", 1.0))
        r.pipeline, r.meta = joblib.load(path / "model.joblib"), meta
        return r


def cross_validate(
    texts: list[str],
    labels: list[str],
    *,
    folds: int = 5,
    groups: list[str] | None = None,
    scheme: str = "type",
) -> dict[str, Any]:
    """5-fold stratified CV, or leave-one-group-out when `groups` (dataset names) is given."""
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold

    splitter: Any = (
        LeaveOneGroupOut()
        if groups is not None
        else StratifiedKFold(folds, shuffle=True, random_state=13)
    )
    split_iter = (
        splitter.split(texts, labels, groups)
        if groups is not None
        else splitter.split(texts, labels)
    )
    per_fold = []
    preds_all: list[str] = [""] * len(texts)
    for train, test in split_iter:
        tr_labels = [labels[i] for i in train]
        if len(set(tr_labels)) < 2:
            continue
        model = TfidfSvmRouter(scheme).fit([texts[i] for i in train], tr_labels)
        preds = model.predict([texts[i] for i in test])
        gold = [labels[i] for i in test]
        for i, p in zip(test, preds, strict=True):
            preds_all[i] = p
        fold = {
            "n": len(test),
            "accuracy": accuracy_score(gold, preds),
            "macro_f1": f1_score(gold, preds, average="macro", zero_division=0),
        }
        if groups is not None:
            fold["held_out"] = groups[test[0]]
        per_fold.append(fold)
    done = [i for i, p in enumerate(preds_all) if p]
    return {
        "folds": per_fold,
        "accuracy": accuracy_score([labels[i] for i in done], [preds_all[i] for i in done]),
        "macro_f1": f1_score(
            [labels[i] for i in done],
            [preds_all[i] for i in done],
            average="macro",
            zero_division=0,
        ),
        "scheme": scheme,
        "protocol": "leave-one-dataset-out" if groups else f"{folds}-fold",
    }


def type_labels_valid(labels: list[str]) -> None:
    bad = sorted(set(labels) - set(QUERY_TYPES))
    if bad:
        raise ConfigError(f"unknown query-type labels: {bad}")
