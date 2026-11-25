import math

import pytest

from hgrag.eval.metrics.efficiency import percentile, usd_per_1k_queries
from hgrag.eval.metrics.qa import exact_match, normalize_answer, token_f1
from hgrag.eval.metrics.retrieval import hit_at_k, ndcg_at_k, recall_at_k
from hgrag.eval.stats import (
    bootstrap_ci,
    holm,
    mcnemar,
    min_detectable_effect,
    paired_bootstrap,
)
from hgrag.generation.answers import clean_answer, is_refusal, strip_citations

# --- answers ---------------------------------------------------------------------------------


def test_refusal_detection_and_cleanup() -> None:
    assert is_refusal("Insufficient information.")
    assert is_refusal("insufficient  information")
    assert not is_refusal("Walt Disney")
    assert clean_answer('Answer: "Walt Disney" ') == "Walt Disney"
    assert strip_citations("Walt Disney [1, 3].") == "Walt Disney."


# --- QA metrics (SQuAD normalisation) --------------------------------------------------------


def test_normalize_answer_drops_articles_punctuation_case() -> None:
    assert normalize_answer("The  Eiffel Tower!") == "eiffel tower"


def test_exact_match_takes_any_reference() -> None:
    assert exact_match("the Marlow Institute", ["Marlow Institute"]) == 1.0
    assert exact_match("Marlow", ["Marlow Institute"]) == 0.0
    assert exact_match("11 years", ["11", "11 years"]) == 1.0


def test_token_f1_hand_computed() -> None:
    # pred "walt disney company" vs ref "walt disney": overlap 2, P = 2/3, R = 1 → F1 = 0.8
    assert token_f1("Walt Disney Company", ["Walt Disney"]) == pytest.approx(0.8)
    assert token_f1("nothing", ["Walt Disney"]) == 0.0
    assert token_f1("x", []) == 0.0


# --- retrieval metrics -----------------------------------------------------------------------


def test_recall_at_k_dedupes_and_handles_no_gold() -> None:
    ranked = ["d1", "d1", "d3", "d2"]  # duplicate counts once
    assert recall_at_k(ranked, {"d2", "d3"}, k=2) == 0.5  # top-2 unique: d1, d3
    assert recall_at_k(ranked, {"d2", "d3"}, k=3) == 1.0
    assert recall_at_k(ranked, set(), k=5) is None


def test_hit_at_k() -> None:
    assert hit_at_k(["x", "a"], {"a", "b", "c"}, k=2) == 1.0  # one of three gold is enough
    assert hit_at_k(["x", "a"], {"a"}, k=1) == 0.0
    assert hit_at_k(["x"], set(), k=1) is None


def test_ndcg_hand_computed() -> None:
    # gold {a, b}; ranking [x, a, b]: DCG = 1/log2(3) + 1/log2(4); IDCG = 1 + 1/log2(3)
    expected = (1 / math.log2(3) + 0.5) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(["x", "a", "b"], {"a", "b"}, 10) == pytest.approx(expected)
    assert ndcg_at_k(["a", "b"], {"a", "b"}, 10) == pytest.approx(1.0)


def test_percentile_and_cost() -> None:
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([10], 95) == 10
    assert usd_per_1k_queries([0.001, 0.003]) == pytest.approx(2.0)


# --- statistics ------------------------------------------------------------------------------


def test_bootstrap_ci_brackets_the_mean_and_is_deterministic() -> None:
    values = [1.0] * 70 + [0.0] * 30
    ci = bootstrap_ci(values)
    assert ci.estimate == pytest.approx(0.7)
    assert ci.low < 0.7 < ci.high and ci.high - ci.low < 0.25
    assert bootstrap_ci(values) == ci


def test_mcnemar_exact_and_chi_square() -> None:
    a = [True] * 0 + [False] * 5
    b = [True] * 5
    exact = mcnemar(a, b)  # 0 vs 5 discordant: p = 2 * (1/2)^5
    assert exact.exact and exact.p_value == pytest.approx(0.0625)

    a = [True] * 10 + [False] * 30
    b = [False] * 10 + [True] * 30
    approx = mcnemar(a, b)  # (|10-30| - 1)^2 / 40 = 9.025
    assert not approx.exact
    assert approx.p_value == pytest.approx(math.erfc(math.sqrt(9.025 / 2)))
    assert mcnemar([True], [True]).p_value == 1.0


def test_paired_bootstrap_detects_a_clear_difference() -> None:
    a = [1.0] * 80 + [0.0] * 20
    b = [1.0] * 50 + [0.0] * 50
    result = paired_bootstrap(a, b)
    assert result.mean_diff == pytest.approx(0.3)
    assert result.low > 0 and result.p_value < 0.01
    assert paired_bootstrap(a, a).p_value == 1.0


def test_holm_hand_computed() -> None:
    # sorted 0.01, 0.03, 0.04 → 0.03, 0.06, max(0.06, 0.04) = 0.06
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_min_detectable_effect() -> None:
    # (1.96 + 0.8416) * sqrt(0.2 / 400) ≈ 0.0626
    assert min_detectable_effect(400, 0.2) == pytest.approx(0.0626, abs=1e-3)
