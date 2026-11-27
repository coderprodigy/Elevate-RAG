"""Automated validation of the LLM judges (D-36): calibration on GroUSE plus inter-judge agreement.

Replaces the human agreement gate. Two independent checks must both pass before judge-based
accuracies are reported as *validated*:

1. **Calibration** on GroUSE (Muller et al., COLING 2025; MIT licence), a published
   meta-evaluation benchmark for grounded-QA judges whose expected verdicts were written by
   experts. Each test gives a question, a ground-truth answer, a candidate answer and the scores
   a good judge must give. Our judge grades *correctness against the reference*, so it is tested
   on GroUSE's completeness criterion:

   * completeness ``==5`` → the candidate states the full reference answer → must be ``correct``
   * completeness ``<5``  → part of it is missing → must be ``partial`` or ``wrong``
   * completeness ``==1`` → none of it → must be ``wrong``

   Tests with an undefined completeness (refusal cases; the refusal rule decides those without
   an LLM) and the low-faithfulness tests (complete but with unsupported extra claims, which
   reference-based correctness does not penalise; RAGAS faithfulness measures that) are
   excluded. That leaves 63 of GroUSE's 144 test cases.
2. **Agreement** of two judges from different model families on the run's own answers
   (Cohen's kappa, computed per run in `harness.judge_agreement`).

Both thresholds follow common practice: kappa >= 0.6 is "substantial" agreement (Landis & Koch),
the level the plan required against humans; 80 % calibration pass rate.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from elevate_rag.evaluation.judge import JUDGE_VERSION, CorrectnessJudge
from elevate_rag.providers.llm import BaseLLM
from elevate_rag.utils import DataError, file_sha256, write_json

log = logging.getLogger(__name__)

GROUSE = {
    "name": "GroUSE",
    "repo_id": "illuin/grouse",
    "revision": "caaaae0dacf5126279ae59609d26926e124b8032",
    "filename": "test.jsonl",
    "sha256": "222bcf8224ec5dda1561d235140a3ca4ce944905cf6472262a9f5505da073aed",
    "license": "MIT",
    "paper": "https://arxiv.org/abs/2409.06595",
}
KAPPA_MIN = 0.6
CALIBRATION_MIN = 0.8


@dataclass(frozen=True)
class CalibrationCase:
    case_id: str
    test_type: str
    question: str
    reference: str
    answer: str
    expected: str  # correct | not_correct | wrong


def download_grouse(dest_dir: Path, *, timeout_s: float = 60.0) -> Path:
    """Fetch GroUSE's test split at the pinned revision; verified against its SHA-256."""
    import httpx

    dest = dest_dir / "grouse" / GROUSE["revision"] / GROUSE["filename"]
    if dest.exists() and file_sha256(dest) == GROUSE["sha256"]:
        return dest
    url = (
        f"https://huggingface.co/datasets/{GROUSE['repo_id']}/resolve/"
        f"{GROUSE['revision']}/{GROUSE['filename']}"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = httpx.get(url, follow_redirects=True, timeout=timeout_s)
    r.raise_for_status()
    part = dest.with_name(dest.name + ".part")
    part.write_bytes(r.content)
    if file_sha256(part) != GROUSE["sha256"]:
        part.unlink()
        raise DataError(f"GroUSE {GROUSE['filename']}: checksum mismatch after download")
    part.replace(dest)
    return dest


def expected_verdict(conditions: dict[str, Any]) -> str | None:
    """Map a GroUSE test's conditions to the verdict a reference-based correctness judge must
    give, or None when the test does not apply to such a judge (see module docstring)."""
    if conditions.get("faithfulness_condition") == "==0":
        return None
    return {"==5": "correct", "<5": "not_correct", "==1": "wrong"}.get(
        str(conditions.get("completeness_condition"))
    )


def load_cases(path: Path) -> list[CalibrationCase]:
    cases = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        exp = expected_verdict(row.get("conditions", {}))
        if exp is None:
            continue
        test_type = str(row.get("metadata", {}).get("test_type", "")).strip()
        cases.append(
            CalibrationCase(
                case_id=f"grouse-{i:03d}",
                test_type=re.sub(r"\s+\d+$", "", test_type),
                question=row["input"],
                reference=row["expected_output"],
                answer=row["actual_output"],
                expected=exp,
            )
        )
    if not cases:
        raise DataError(f"{path}: no applicable GroUSE cases")
    return cases


def passes(expected: str, verdict: str) -> bool:
    if expected == "correct":
        return verdict == "correct"
    if expected == "wrong":
        return verdict == "wrong"
    return verdict in ("partial", "wrong")


async def calibrate(
    llm: BaseLLM, cases: list[CalibrationCase], *, concurrency: int = 4
) -> dict[str, Any]:
    judge = CorrectnessJudge(llm)
    sem = asyncio.Semaphore(concurrency)

    async def one(c: CalibrationCase) -> dict[str, Any]:
        async with sem:
            v = await judge.judge(
                c.question, [c.reference], c.answer, answerable=True, refused=False
            )
        return {
            "case_id": c.case_id,
            "test_type": c.test_type,
            "expected": c.expected,
            "verdict": v.verdict,
            "passed": passes(c.expected, v.verdict),
        }

    items = list(await asyncio.gather(*(one(c) for c in cases)))
    by_type: dict[str, list[bool]] = defaultdict(list)
    for it in items:
        by_type[it["test_type"]].append(it["passed"])
    n_pass = sum(it["passed"] for it in items)
    rate = n_pass / len(items)
    return {
        "benchmark": {k: GROUSE[k] for k in ("name", "repo_id", "revision", "sha256", "license")},
        "judge": calibration_key(llm.cfg.model, llm.cfg.revision),
        "model": llm.cfg.model,
        "revision": llm.cfg.revision,
        "judge_version": JUDGE_VERSION,
        "created_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "n": len(items),
        "passed": n_pass,
        "pass_rate": rate,
        "threshold": CALIBRATION_MIN,
        "calibrated": rate >= CALIBRATION_MIN,
        "invalid": sum(it["verdict"] == "invalid" for it in items),
        "by_test_type": {
            t: {"n": len(v), "pass_rate": sum(v) / len(v)} for t, v in sorted(by_type.items())
        },
        "items": items,
    }


def calibration_key(model: str, revision: str | None) -> str:
    """File-system-safe id of one judge (model + revision + judge prompt version)."""
    raw = f"{model}@{revision or 'unpinned'}@{JUDGE_VERSION}"
    return re.sub(r"[^A-Za-z0-9._@-]+", "_", raw)


def write_calibration(out_dir: Path, result: dict[str, Any]) -> Path:
    path = out_dir / f"{result['judge']}.json"
    write_json(path, result)
    return path


def load_calibration(
    out_dir: Path | None, model: str, revision: str | None
) -> dict[str, Any] | None:
    if out_dir is None:
        return None
    path = out_dir / f"{calibration_key(model, revision)}.json"
    if not path.exists():
        return None
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def judge_validation(
    agreement: dict[str, Any] | None,
    calibrations: dict[str, dict[str, Any] | None],
) -> dict[str, Any]:
    """Status of the judge-based numbers in one run: `validated` only when both judges passed
    calibration and agreed with kappa >= KAPPA_MIN on this run's answers."""
    reasons = []
    if not calibrations:
        reasons.append("no LLM judge configured")
    for role, cal in calibrations.items():
        if cal is None:
            reasons.append(f"{role} not calibrated (run `elevate-rag judge calibrate`)")
        elif not cal.get("calibrated"):
            reasons.append(f"{role} GroUSE pass rate {cal['pass_rate']:.2f} < {CALIBRATION_MIN}")
    if "judge2" not in calibrations:
        reasons.append("no second judge (models.judge2) to measure agreement")
    elif agreement is None or agreement.get("kappa") is None:
        reasons.append("judge agreement undefined (no answers judged by both)")
    elif agreement["kappa"] < KAPPA_MIN:
        reasons.append(f"judge kappa {agreement['kappa']:.2f} < {KAPPA_MIN}")
    return {
        "status": "validated" if not reasons else "unvalidated",
        "reasons": reasons,
        "kappa": agreement.get("kappa") if agreement else None,
        "kappa_min": KAPPA_MIN,
        "calibration": {
            role: (
                {"pass_rate": c["pass_rate"], "n": c["n"], "calibrated": c["calibrated"]}
                if c
                else None
            )
            for role, c in calibrations.items()
        },
        "calibration_min": CALIBRATION_MIN,
    }


def revalidate(
    summary: dict[str, Any], config: dict[str, Any], cal_dir: Path | None
) -> dict[str, Any] | None:
    """Recompute a finished run's judge validation from the calibration files present now
    (e.g. a calibration written after the run, or runs copied back from a cluster)."""
    models = config.get("models") or {}
    cals = {
        role: load_calibration(cal_dir, m["model"], m.get("revision"))
        for role in ("judge", "judge2")
        if isinstance(m := models.get(role), dict)
    }
    if not cals and summary.get("judge_validation") is None:
        return None
    return judge_validation(summary.get("judge_agreement"), cals)
