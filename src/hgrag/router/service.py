"""E3 orchestration: train routers on dev, evaluate on test, per dataset (offline, free).

Training data
* Type labels (label scheme 1): every evaluation dataset's dev questions with their query type,
  plus RAGRouter-Bench (already de-duplicated against our evaluation sets).
* Outcome labels (label scheme 2): the oracle's path on each dataset's dev split.
* Leave-one-dataset-out: when evaluating dataset D, D's own dev questions are excluded from R2
  training, so the router is tested on phrasing it has never seen (risk "router overfits").
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass

from hgrag.config import AppConfig
from hgrag.core.runs import RunContext
from hgrag.data.io import dataset_files, load_queries
from hgrag.data.schema import Query
from hgrag.index.embedders import Embedder
from hgrag.llm.gateway import build_gateway
from hgrag.router.evaluate import (
    RouterResult,
    evaluate_choices,
    fallback_choices,
    oracle_choices,
    tune_fallback_threshold,
)
from hgrag.router.outcomes import OutcomeTable, load_outcomes
from hgrag.router.routers import ClassifierRouter, LLMRouter, RuleRouter, StaticRouter

ROUTER_TRAINING_SET = "ragrouter_bench"


@dataclass(frozen=True)
class DatasetRouting:
    dataset: str
    results: list[RouterResult]
    fallback_threshold: float


def _questions(config: AppConfig, dataset: str, split: str | None = None) -> list[Query]:
    return load_queries(dataset_files(config, dataset), split)  # type: ignore[arg-type]


def type_training_data(
    config: AppConfig, datasets: Sequence[str], exclude: str | None
) -> tuple[list[str], list[str]]:
    questions: list[Query] = []
    for name in [*datasets, ROUTER_TRAINING_SET]:
        if name == exclude or not dataset_files(config, name).exists():
            continue
        split = None if name == ROUTER_TRAINING_SET else "dev"
        questions += _questions(config, name, split)
    return [q.question for q in questions], [q.query_type for q in questions]


def evaluate_dataset(
    config: AppConfig,
    experiment: str,
    dataset: str,
    *,
    datasets: Sequence[str],
    train_tables: dict[str, OutcomeTable],
    embedder: Embedder | None = None,
    llm_model_key: str | None = None,
    use_llm: bool = False,
) -> DatasetRouting:
    router = config.router
    test = load_outcomes(config, experiment, dataset, "test", router.paths)
    questions = {q.qid: q.question for q in _questions(config, dataset, "test")}
    qids = test.qids
    texts = [questions[q] for q in qids]
    results: list[RouterResult] = []

    def add(name: str, paths: Sequence[str], **extra: dict[str, float] | dict[str, int]) -> None:
        results.append(
            evaluate_choices(
                name,
                test,
                dict(zip(qids, paths, strict=True)),
                baseline_path=router.baseline_path,
                **extra,  # type: ignore[arg-type]
            )
        )

    for path in router.paths:  # R0
        add(StaticRouter(path).name, StaticRouter(path).route(texts))
    add(RuleRouter.name, RuleRouter(config).route(texts))  # R1

    # R2, label scheme 1 (query types), leave-one-dataset-out
    type_x, type_y = type_training_data(config, datasets, exclude=dataset)
    kinds = ["tfidf", *(["encoder"] if embedder is not None else [])]
    for kind in kinds if len(set(type_y)) >= 2 else []:  # no training data → R2 skipped
        clf = ClassifierRouter(config, kind=kind, label_space="type", embedder=embedder)
        add(clf.name, clf.fit(type_x, type_y).route(texts))

    # R2, label scheme 2 (outcome labels = oracle path on dev), leave-one-dataset-out
    outcome_x: list[str] = []
    outcome_y: list[str] = []
    for name, table in train_tables.items():
        if name == dataset:
            continue
        dev_questions = {q.qid: q.question for q in _questions(config, name, "dev")}
        for qid, path in oracle_choices(table).items():
            outcome_x.append(dev_questions[qid])
            outcome_y.append(path)
    if len(set(outcome_y)) >= 2:
        clf = ClassifierRouter(config, kind="tfidf", label_space="path")
        add(clf.name, clf.fit(outcome_x, outcome_y).route(texts))

    if use_llm:  # R3
        context = RunContext(
            run_id=f"{experiment}.router.{dataset}", experiment=experiment, phase="router"
        )
        llm = LLMRouter(config, build_gateway(config, context), llm_model_key)
        add(llm.name, asyncio.run(llm.aroute(texts)))

    # R4: threshold tuned on this dataset's dev outcomes (falls back to 0.0 without dev runs)
    fb = router.fallback
    dev = train_tables.get(dataset)
    threshold = (
        tune_fallback_threshold(
            dev, fb.cheap_path, fb.escalate_path, fb.thresholds, router.baseline_path
        )
        if dev is not None
        else 0.0
    )
    choices, extra_tokens, extra_cost = fallback_choices(
        test, fb.cheap_path, fb.escalate_path, threshold
    )
    results.append(
        evaluate_choices(
            f"R4-fallback({fb.cheap_path}→{fb.escalate_path}@{threshold:g})",
            test,
            choices,
            baseline_path=router.baseline_path,
            extra_tokens=extra_tokens,
            extra_cost=extra_cost,
        )
    )

    oracle = oracle_choices(test)
    results.append(evaluate_choices("Oracle", test, oracle, baseline_path=router.baseline_path))
    return DatasetRouting(dataset, results, threshold)


def evaluate_experiment(
    config: AppConfig,
    experiment: str,
    datasets: Sequence[str],
    *,
    embedder: Embedder | None = None,
    use_llm: bool = False,
    llm_model_key: str | None = None,
) -> list[DatasetRouting]:
    train_tables: dict[str, OutcomeTable] = {}
    for name in datasets:
        try:
            train_tables[name] = load_outcomes(config, experiment, name, "dev", config.router.paths)
        except FileNotFoundError:
            continue  # no dev runs for this dataset: outcome labels / R4 tuning skip it
    return [
        evaluate_dataset(
            config,
            experiment,
            name,
            datasets=datasets,
            train_tables=train_tables,
            embedder=embedder,
            llm_model_key=llm_model_key,
            use_llm=use_llm,
        )
        for name in datasets
    ]


@dataclass(frozen=True)
class CVResult:
    kind: str
    folds: int
    accuracy: float
    macro_f1: float
    n: int


def ragrouter_cross_validation(
    config: AppConfig, *, kind: str = "tfidf", embedder: Embedder | None = None
) -> CVResult:
    """5-fold CV of the R2 type classifier on RAGRouter-Bench (plan: RAGRouter-Bench row)."""
    from sklearn.model_selection import StratifiedKFold

    from hgrag.eval.metrics.router import macro_f1, routing_accuracy

    questions = _questions(config, ROUTER_TRAINING_SET)
    x = [q.question for q in questions]
    y = [q.query_type for q in questions]
    folds = config.router.classifier.cv_folds
    predicted: list[str] = [""] * len(x)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.seed)
    for train_idx, test_idx in splitter.split(x, y):
        clf = ClassifierRouter(config, kind=kind, label_space="type", embedder=embedder)
        clf.fit([x[i] for i in train_idx], [y[i] for i in train_idx])
        for i, label in zip(test_idx, clf.predict([x[i] for i in test_idx]), strict=True):
            predicted[i] = label
    return CVResult(
        kind=kind,
        folds=folds,
        accuracy=round(routing_accuracy(predicted, y), 4),
        macro_f1=round(macro_f1(predicted, y), 4),
        n=len(x),
    )
