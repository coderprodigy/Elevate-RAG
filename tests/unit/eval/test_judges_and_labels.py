import asyncio
from collections.abc import Callable
from pathlib import Path

import pytest

from hgrag.config import AppConfig, ProviderSpec
from hgrag.core.runs import RunContext
from hgrag.data.schema import Query
from hgrag.eval.batch import read_models
from hgrag.eval.faithfulness import score_evidence_recall, score_faithfulness
from hgrag.eval.human_labels import build_label_sample, cohen_kappa
from hgrag.eval.judge import AnswerJudge, Judgment, judge_predictions, parse_verdict
from hgrag.eval.llm_json import JSONExtractionError, extract_json
from hgrag.eval.metrics.router import macro_f1, regret, routing_accuracy, tokens_saved
from hgrag.eval.pairwise import compare, win_rates
from hgrag.llm.gateway import LLMGateway, build_gateway
from hgrag.llm.types import LLMRequest, ProviderResult
from hgrag.pipelines.base import ContextUnit, Prediction


class ScriptedProvider:
    """Replies with ``script(request)``; lets tests play the judge."""

    def __init__(self, script: Callable[[LLMRequest], str]) -> None:
        self._script = script
        self.calls = 0

    async def complete(self, request: LLMRequest) -> ProviderResult:
        self.calls += 1
        return ProviderResult(text=self._script(request), input_tokens=50, output_tokens=10)


def _gateway(config: AppConfig, provider: ScriptedProvider) -> LLMGateway:
    def factory(_: ProviderSpec) -> ScriptedProvider:
        return provider

    return build_gateway(config, RunContext(run_id="judge-test"), provider_factory=factory)  # type: ignore[arg-type]


def _query(qid: str = "q1") -> Query:
    return Query(
        qid=qid, question="Who?", answers=["Walt Disney"], query_type="multi_hop", native_type="x"
    )


def _prediction(
    qid: str = "q1", answer: str = "Disney", refused: bool = False, **kw: object
) -> Prediction:
    return Prediction(
        qid=qid,
        answer=answer,
        refused=refused,
        model_key="m",
        input_tokens=1,
        output_tokens=1,
        cost_usd=0.0,
        latency_ms=1.0,
        cache_hit=False,
        status="ok",
        **kw,  # type: ignore[arg-type]
    )


# --- JSON + verdict parsing ------------------------------------------------------------------


def test_extract_json_handles_fences_and_prose() -> None:
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! {"verdict": "wrong"} hope that helps') == {"verdict": "wrong"}
    assert extract_json("[1, 2]") == [1, 2]
    with pytest.raises(JSONExtractionError):
        extract_json("no json here")


def test_parse_verdict_json_and_fallback() -> None:
    assert parse_verdict('{"verdict": "Partial", "reason": "vague"}') == ("partial", "vague")
    assert parse_verdict("The answer is correct.")[0] == "correct"
    with pytest.raises(ValueError):
        parse_verdict("no idea")


# --- answer judge ----------------------------------------------------------------------------


def test_judge_scores_and_skips_refusals(config: AppConfig, tmp_path: Path) -> None:
    provider = ScriptedProvider(lambda _: '{"verdict": "partial", "reason": "incomplete"}')
    judge = AnswerJudge(_gateway(config, provider), model_key="fake-small")
    pending = [(_query("q1"), _prediction("q1")), (_query("q2"), _prediction("q2", refused=True))]
    assert judge.estimate_cost(*pending[1]) == 0.0  # refusal: no call

    outcome = asyncio.run(judge_predictions(judge, pending, tmp_path / "judgments.jsonl"))
    judgments = {j.qid: j for j in read_models(tmp_path / "judgments.jsonl", Judgment)}
    assert outcome.n_new == 2 and provider.calls == 1
    assert judgments["q1"].score == 0.5 and judgments["q1"].judge_model == "fake-small"
    assert judgments["q2"].verdict == "wrong" and judgments["q2"].judge_model == "rule"


# --- pairwise --------------------------------------------------------------------------------


def _prefers(word: str) -> Callable[[LLMRequest], str]:
    def script(request: LLMRequest) -> str:
        text = request.messages[-1].content
        first = text.split("Answer 2:")[0]
        return '{"winner": "1"}' if word in first else '{"winner": "2"}'

    return script


def test_pairwise_counts_a_win_only_when_both_orders_agree(config: AppConfig) -> None:
    consistent = _gateway(config, ScriptedProvider(_prefers("GOOD")))
    result = asyncio.run(
        compare(consistent, "q", "GOOD answer", "weak", "comprehensiveness", model_key="fake-small")
    )
    assert result.winner == "a" and result.first_order == result.swapped_order == "a"

    biased = _gateway(config, ScriptedProvider(lambda _: '{"winner": "1"}'))  # always picks slot 1
    result = asyncio.run(compare(biased, "q", "x", "y", "diversity", model_key="fake-small"))
    assert (result.first_order, result.swapped_order, result.winner) == ("a", "b", "tie")
    assert win_rates([result]) == {"a": 0.0, "b": 0.0, "tie": 1.0}


# --- faithfulness ----------------------------------------------------------------------------


def test_faithfulness_counts_supported_claims(config: AppConfig) -> None:
    def script(request: LLMRequest) -> str:
        if "Split the answer" in (request.system or ""):
            return '{"claims": ["A was built in 1900", "A is red", "A is old"]}'
        return '{"supported": [true, false, true]}'

    gateway = _gateway(config, ScriptedProvider(script))
    context = [ContextUnit(unit_id="c1", kind="chunk", text="A was built in 1900. It is old.")]
    score = asyncio.run(
        score_faithfulness(gateway, "q", _prediction(contexts=context), model_key="fake-small")
    )
    assert (score.n_checked, score.n_supported) == (3, 2)
    assert score.score == pytest.approx(2 / 3)

    no_context = asyncio.run(
        score_faithfulness(gateway, "q", _prediction(), model_key="fake-small")
    )
    assert no_context.score is None


# --- router metrics, kappa, label sample -----------------------------------------------------


def test_router_metrics_hand_computed() -> None:
    gold = ["naive", "naive", "local", "global"]
    pred = ["naive", "local", "local", "naive"]
    assert routing_accuracy(pred, gold) == 0.5
    # F1: naive = 2·1/(2+1+1) = 0.5, local = 2/(2+1) = 0.667, global = 0 → mean 0.389
    assert macro_f1(pred, gold) == pytest.approx((0.5 + 2 / 3 + 0) / 3)
    assert regret(0.8, 0.7) == pytest.approx(0.1)
    assert tokens_saved(700, 1000) == pytest.approx(0.3)


def test_cohen_kappa_hand_computed() -> None:
    # observed 0.75; expected 0.5·0.25 + 0.5·0.75 = 0.5 → kappa 0.5
    assert cohen_kappa(["c", "c", "w", "w"], ["c", "w", "w", "w"]) == pytest.approx(0.5)
    assert cohen_kappa(["c", "w"], ["c", "w"]) == 1.0


def test_label_sample_is_blind_deterministic_and_skips_refusals() -> None:
    queries = {f"q{i}": _query(f"q{i}") for i in range(6)}
    predictions = [_prediction(f"q{i}", refused=(i == 0)) for i in range(6)]
    runs = [("run_a", queries, predictions), ("run_b", queries, predictions)]
    sample = build_label_sample(runs, n=5, seed=1)
    assert len(sample) == 5 and sample == build_label_sample(runs, n=5, seed=1)
    assert all(item.qid != "q0" for item in sample)
    assert "run" not in sample[0].answer  # labellers see the answer only


def test_evidence_recall_checks_gold_evidence(config: AppConfig) -> None:
    gateway = _gateway(config, ScriptedProvider(lambda _: '{"supported": [true, false]}'))
    query = _query().model_copy(update={"gold_evidence": ["fact one", "fact two"]})
    context = [ContextUnit(unit_id="c1", kind="chunk", text="fact one")]
    score = asyncio.run(
        score_evidence_recall(gateway, query, _prediction(contexts=context), model_key="fake-small")
    )
    assert score.metric == "evidence_recall" and score.score == 0.5
    no_evidence = asyncio.run(
        score_evidence_recall(
            gateway, _query(), _prediction(contexts=context), model_key="fake-small"
        )
    )
    assert no_evidence.score is None
