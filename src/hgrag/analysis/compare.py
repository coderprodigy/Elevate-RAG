"""Paired system comparisons against a reference pipeline (plan "Statistics").

Same questions for every system → paired tests: McNemar on correct/incorrect (``correct == 1``)
and a paired bootstrap on the continuous primary score. p-values are Holm-corrected across all
comparisons in the table; differences below the minimum detectable effect are labelled
"no clear difference" rather than claimed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from hgrag.analysis.collect import RunRecord
from hgrag.eval.stats import holm, mcnemar, min_detectable_effect, paired_bootstrap


@dataclass(frozen=True)
class Comparison:
    dataset: str
    system: str
    reference: str
    query_type: str  # "all" or one type
    n: int
    system_accuracy: float
    reference_accuracy: float
    diff: float
    diff_low: float
    diff_high: float
    p_mcnemar: float
    p_holm: float
    mde: float
    verdict: str


def _verdict(diff: float, p_holm: float, mde: float) -> str:
    if p_holm < 0.05 and abs(diff) >= mde:
        return "better" if diff > 0 else "worse"
    return "no clear difference"


def compare_to_reference(
    runs: Sequence[RunRecord], reference: str, *, by_type: bool = True
) -> list[Comparison]:
    rows: list[tuple[str, str, str, list[float], list[float]]] = []
    by_dataset: dict[str, dict[str, RunRecord]] = {}
    for run in runs:
        by_dataset.setdefault(run.dataset, {})[run.pipeline] = run
    for dataset, systems in sorted(by_dataset.items()):
        if reference not in systems:
            continue
        ref = systems[reference].by_qid
        for name, run in sorted(systems.items()):
            if name == reference:
                continue
            mine = run.by_qid
            shared = sorted(ref.keys() & mine.keys())
            groups = {"all": shared}
            if by_type:
                for qid in shared:
                    groups.setdefault(mine[qid].query_type, []).append(qid)
            for group, qids in groups.items():
                if len(qids) >= 2:
                    rows.append(
                        (
                            dataset,
                            name,
                            group,
                            [mine[q].correct for q in qids],
                            [ref[q].correct for q in qids],
                        )
                    )

    raw = []
    for dataset, name, group, a, b in rows:
        test = mcnemar([x >= 1.0 for x in a], [y >= 1.0 for y in b])
        boot = paired_bootstrap(a, b) if a != b else None
        discordant = (test.a_only + test.b_only) / len(a)
        raw.append((dataset, name, group, a, b, test.p_value, boot, discordant))
    adjusted = holm([r[5] for r in raw]) if raw else []

    comparisons = []
    for (dataset, name, group, a, b, p, boot, discordant), p_adj in zip(raw, adjusted, strict=True):
        diff = sum(a) / len(a) - sum(b) / len(b)
        mde = min_detectable_effect(len(a), max(discordant, 1 / len(a)))
        comparisons.append(
            Comparison(
                dataset=dataset,
                system=name,
                reference=reference,
                query_type=group,
                n=len(a),
                system_accuracy=round(sum(a) / len(a), 4),
                reference_accuracy=round(sum(b) / len(b), 4),
                diff=round(diff, 4),
                diff_low=round(boot.low if boot else 0.0, 4),
                diff_high=round(boot.high if boot else 0.0, 4),
                p_mcnemar=round(p, 5),
                p_holm=round(p_adj, 5),
                mde=round(mde, 4),
                verdict=_verdict(diff, p_adj, mde),
            )
        )
    return comparisons
