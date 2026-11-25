import json

import pytest

from hgrag.config import AppConfig
from hgrag.data.io import dataset_files, write_jsonl
from hgrag.data.schema import Query
from hgrag.router.evaluate import (
    evaluate_choices,
    fallback_choices,
    oracle_choices,
    tune_fallback_threshold,
)
from hgrag.router.outcomes import OutcomeTable, PathOutcome, load_outcomes
from hgrag.router.routers import ClassifierRouter, RuleRouter, StaticRouter, count_named_entities
from hgrag.router.service import evaluate_experiment


def _o(correct: float, tokens: int, top: float | None = 1.0, refused: bool = False) -> PathOutcome:
    return PathOutcome(
        correct=correct, tokens=tokens, cost_usd=tokens / 1e6, top_score=top, refused=refused
    )


# q1: naive right (cheap). q2: only mix right. q3: nobody right.
TABLE = OutcomeTable(
    dataset="toy",
    split="test",
    outcomes={
        "q1": {"naive": _o(1, 100, top=5.0), "local": _o(1, 300), "mix": _o(1, 500)},
        "q2": {"naive": _o(0, 100, top=-3.0), "local": _o(0, 300), "mix": _o(1, 500)},
        "q3": {"naive": _o(0, 100, top=-3.0), "local": _o(0, 300), "mix": _o(0, 500)},
    },
)


def test_cost_order_and_oracle() -> None:
    assert TABLE.cost_order() == ["naive", "local", "mix"]
    assert oracle_choices(TABLE) == {"q1": "naive", "q2": "mix", "q3": "naive"}


def test_evaluate_choices_hand_computed() -> None:
    always_mix = evaluate_choices(
        "mix", TABLE, dict.fromkeys(TABLE.qids, "mix"), baseline_path="mix"
    )
    assert always_mix.accuracy == pytest.approx(2 / 3, abs=1e-3)
    assert always_mix.regret == 0.0 and always_mix.tokens_saved_vs_baseline == 0.0
    always_naive = evaluate_choices(
        "naive", TABLE, dict.fromkeys(TABLE.qids, "naive"), baseline_path="mix"
    )
    assert always_naive.accuracy == pytest.approx(1 / 3, abs=1e-3)
    assert always_naive.regret == pytest.approx(1 / 3, abs=1e-3)
    assert always_naive.tokens_saved_vs_baseline == pytest.approx(0.8)  # 100 vs 500 tokens


def test_fallback_escalates_on_low_confidence_and_pays_for_both() -> None:
    choices, extra_tokens, _ = fallback_choices(TABLE, "naive", "mix", threshold=0.0)
    assert choices == {"q1": "naive", "q2": "mix", "q3": "mix"}
    result = evaluate_choices("R4", TABLE, choices, baseline_path="mix", extra_tokens=extra_tokens)
    assert result.accuracy == pytest.approx(2 / 3, abs=1e-3)  # matches always-mix ...
    assert result.tokens_per_query == pytest.approx(
        (100 + 600 + 600) / 3, abs=0.1
    )  # ... for fewer tokens
    assert tune_fallback_threshold(TABLE, "naive", "mix", [-10.0, 0.0, 10.0], "mix") == 0.0


def test_rule_router(config: AppConfig) -> None:
    rules = RuleRouter(config)
    assert rules.classify("What are the main themes across these articles?") == "global"
    assert (
        rules.classify("Which is older, the Marlow Institute or Kessel University?") == "comparison"
    )
    assert rules.classify("How many years after 1891 was it completed?") == "temporal"
    assert (
        rules.classify("Where was the founder of Marlow Institute born in Tallis county?")
        == "multi_hop"
    )
    assert rules.classify("who sings does he love me") == "single_hop"
    assert count_named_entities("Did Elena Vosk meet Piotr Anand?") == 2
    assert StaticRouter("mix").route(["a", "b"]) == ["mix", "mix"]


def test_classifier_router_learns_types(config: AppConfig) -> None:
    x = ["summarise the main themes", "overall trends across documents"] * 5 + [
        "who founded the institute",
        "what year was it built",
    ] * 5
    y = ["global"] * 10 + ["single_hop"] * 10
    router = ClassifierRouter(config, kind="tfidf").fit(x, y)
    assert router.predict(["main themes overall"]) == ["global"]
    assert router.route(["who built it"]) == [config.router.type_to_path["single_hop"]]
    with pytest.raises(ValueError, match="embedder"):
        ClassifierRouter(config, kind="encoder")


def _write_run(
    config: AppConfig, dataset: str, pipeline: str, split: str, rows: list[dict[str, object]]
) -> None:
    folder = config.paths.runs_dir / "e2" / f"{dataset}__{pipeline}__{split}"
    folder.mkdir(parents=True)
    (folder / "scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (folder / "predictions.jsonl").write_text("")


def test_offline_e3_evaluation_end_to_end(config: AppConfig) -> None:
    queries = [
        Query(qid=f"q{i}", question=q, answers=["a"], query_type=t, native_type=t, split=s)
        for i, (q, t, s) in enumerate(
            [
                ("who founded the institute", "single_hop", "dev"),
                ("summarise the main themes", "global", "dev"),
                ("who built the tower", "single_hop", "test"),
                ("overall themes across articles", "global", "test"),
            ]
        )
    ]
    write_jsonl(dataset_files(config, "toy").queries, queries)
    for pipeline in config.router.paths.values():
        for split, ids in (("dev", ["q0", "q1"]), ("test", ["q2", "q3"])):
            good = {"v4": {"q0", "q2"}, "lr_global": {"q1", "q3"}}.get(pipeline, set())
            rows = [
                {
                    "qid": q,
                    "correct": float(q in good),
                    "tokens": 100 if pipeline == "v4" else 400,
                    "cost_usd": 0.0001,
                    "refused": False,
                }
                for q in ids
            ]
            _write_run(config, "toy", pipeline, split, rows)

    table = load_outcomes(config, "e2", "toy", "test", config.router.paths)
    assert table.qids == ["q2", "q3"]
    (report,) = evaluate_experiment(config, "e2", ["toy"])
    by_name = {r.router: r for r in report.results}
    assert by_name["Oracle"].accuracy == 1.0
    assert by_name["R0-always-naive"].accuracy == 0.5
    assert by_name["R1-rules"].accuracy == 1.0  # "overall" → global path, else naive
    assert any(name.startswith("R4-fallback") for name in by_name)
    assert not any(
        name.startswith("R2") for name in by_name
    )  # LODO: no other dataset to learn from
