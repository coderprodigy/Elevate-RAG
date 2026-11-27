from __future__ import annotations

import pytest

from elevate_rag.evaluation import metrics as M
from elevate_rag.stats import (
    bootstrap_ci,
    cohen_kappa,
    holm_bonferroni,
    mcnemar,
    mde_paired_proportions,
    paired_bootstrap,
    power_check,
)


def test_normalization_em_f1() -> None:
    assert M.normalize_answer("The  Royal, Survey Academy!") == "royal survey academy"
    assert M.exact_match("royal survey academy", ["The Royal Survey Academy"]) == 1.0
    assert M.exact_match("x", []) == 0.0
    assert M.token_f1("Tamsin Okoro", "Okoro") == pytest.approx(2 / 3)
    assert M.max_f1("Okoro", ["nobody", "Tamsin Okoro"]) == pytest.approx(2 / 3)
    assert M.token_f1("", "") == 1.0 and M.token_f1("a", "b") == 0.0


def test_short_answer_strips_justification_and_citations() -> None:
    assert M.short_answer("Kessering [2]. Because the passage says so [1].") == "Kessering ."
    assert M.short_answer("1712\nIt was founded then [1]") == "1712"


def test_recall_and_ndcg_document_level() -> None:
    ranked = ["d1::c0000", "d1::c0001", "d3::c0000", "d2::c0000"]
    assert M.recall_at_k(ranked, ["d2", "d3"], 2) == 0.5  # top-2 *documents* are d1, d3
    assert M.recall_at_k(ranked, ["d2", "d3"], 3) == 1.0
    assert M.recall_at_k(ranked, [], 5) is None  # undefined, not zero
    assert M.ndcg_at_k(ranked, ["d1"]) == 1.0
    assert 0 < M.ndcg_at_k(ranked, ["d2"]) < 1
    assert M.ndcg_at_k(ranked, []) is None


def test_evidence_recall_containment_and_fuzzy() -> None:
    texts = ["Basal cell carcinoma (BCC) is the most common type of skin cancer in adults."]
    assert M.evidence_recall(texts, ["most common type of skin cancer"]) == 1.0
    near = "basal cell carcinoma bcc is the most common type of skin cancer in adults today"
    assert M.evidence_recall(texts, [near]) == 1.0  # 13/14 tokens present ≥ 0.8
    far = "basal cell carcinoma is the most common type of skin cancer in many adult patients"
    assert M.evidence_recall(texts, [far]) == 0.0  # 10/14 < 0.8
    assert M.evidence_recall(texts, ["melanoma spreads quickly"]) == 0.0
    assert M.evidence_recall(texts, []) is None


def test_mean_defined_reports_undefined_count() -> None:
    assert M.mean_defined([1.0, None, 0.0]) == (0.5, 1)
    assert M.mean_defined([None]) == (None, 1)


# ------------------------------------------------------------------------------- stats


def test_bootstrap_ci_brackets_mean_and_is_seeded() -> None:
    vals = [1, 0, 1, 1, 0, 1, 0, 1, 1, 1] * 10
    ci = bootstrap_ci(vals)
    assert ci.low <= ci.estimate == 0.7 <= ci.high
    assert ci == bootstrap_ci(vals)
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_mcnemar_exact_and_chi2() -> None:
    a = [1] * 10 + [0] * 0 + [1] * 50
    b = [0] * 10 + [1] * 0 + [1] * 50
    r = mcnemar(a, b)
    assert r.test == "mcnemar_exact" and r.p_value == pytest.approx(2 * 0.5**10)
    assert r.diff == pytest.approx(10 / 60)
    big_a = [1] * 40 + [0] * 10 + [1] * 50
    big_b = [0] * 40 + [1] * 10 + [1] * 50
    r2 = mcnemar(big_a, big_b)
    assert r2.test == "mcnemar_chi2_cc" and r2.p_value < 0.001
    assert mcnemar([1, 0], [1, 0]).p_value == 1.0
    with pytest.raises(ValueError):
        mcnemar([1, 0], [1])


def test_paired_bootstrap_detects_shift() -> None:
    a = [0.9, 0.8, 0.85, 0.95, 0.9] * 20
    b = [x - 0.2 for x in a]
    r = paired_bootstrap(a, b)
    assert r.diff == pytest.approx(0.2) and r.ci_low > 0 and r.p_value < 0.01


def test_holm_bonferroni() -> None:
    out = holm_bonferroni([0.01, 0.04, 0.03])
    assert [round(p, 4) for p, _ in out] == [0.03, 0.06, 0.06]
    assert [rej for _, rej in out] == [True, False, False]


def test_cohen_kappa_known_values() -> None:
    assert cohen_kappa(list("aabb"), list("aabb")) == 1.0
    assert cohen_kappa(list("abab"), list("abba")) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        cohen_kappa(["a"], [])


def test_mde_and_power_check() -> None:
    mde = mde_paired_proportions(500, 0.2)
    assert 0.03 < mde < 0.06  # plan: "about 3–5 accuracy points" at n=500
    pc = power_check([True, False, True, True], [True, True, False, True], n_test=400)
    assert pc["dev_discordant_rate"] == 0.5 and pc["mde"] > 0
    with pytest.raises(ValueError):
        mde_paired_proportions(0, 0.2)
