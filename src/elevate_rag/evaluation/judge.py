"""LLM judges (FR-E5, FR-E7) with validated structured output.

* Correctness: correct / partial / wrong against the reference answers. For *unanswerable*
  questions no judge is used: correct ⇔ the system refused (refusal accuracy, FR-E6).
* Pairwise (global questions): comprehensiveness, diversity, empowerment, overall. Each pair is
  judged twice with the answer order swapped; a side wins only if both orders agree, otherwise
  the criterion is a tie (plan protocol rule 4).

Unparseable judge output is retried once and then recorded as `invalid` — reported as missing,
never as wrong.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from elevate_rag.providers.llm import BaseLLM
from elevate_rag.utils import ProviderError

JUDGE_VERSION = "judge-v1"
VERDICTS = ("correct", "partial", "wrong")
CRITERIA = ("comprehensiveness", "diversity", "empowerment", "overall")

CORRECTNESS_SYSTEM = (
    "You are a strict grader for question answering. Compare the candidate answer with the "
    "reference answer(s). Judge meaning, not wording: paraphrases and extra correct detail are "
    "fine. 'correct' = states the reference answer; 'partial' = contains part of it or is "
    "ambiguous between it and something else; 'wrong' = different answer, no answer, or a "
    'refusal. Reply with JSON only: {"verdict": "correct|partial|wrong", "reason": "..."}'
)
CORRECTNESS_PROMPT = """Question: {question}
Reference answer(s): {references}
Candidate answer: {answer}

JSON:"""

PAIRWISE_SYSTEM = (
    "You compare two answers to the same question. Criteria: comprehensiveness (covers all "
    "aspects), diversity (varied perspectives and insights), empowerment (helps the reader "
    'understand and make informed judgements), overall. For each criterion answer "A", "B" '
    'or "tie". Reply with JSON only: {"comprehensiveness": ..., "diversity": ..., '
    '"empowerment": ..., "overall": ...}'
)
PAIRWISE_PROMPT = """Question: {question}

Answer A:
{a}

Answer B:
{b}

JSON:"""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_json_object(text: str) -> dict[str, Any] | None:
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


@dataclass
class Verdict:
    verdict: str  # correct | partial | wrong | invalid
    reason: str = ""
    judged_by: str = ""

    @property
    def score(self) -> float | None:
        return {"correct": 1.0, "partial": 0.5, "wrong": 0.0}.get(self.verdict)


class CorrectnessJudge:
    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm

    async def judge(
        self, question: str, references: list[str], answer: str, *, answerable: bool, refused: bool
    ) -> Verdict:
        if not answerable:
            return Verdict("correct" if refused else "wrong", "unanswerable: refusal rule", "rule")
        if refused or not answer.strip():
            return Verdict("wrong", "system refused or gave no answer", "rule")
        prompt = CORRECTNESS_PROMPT.format(
            question=question, references=" | ".join(references), answer=answer
        )
        for attempt in range(2):
            try:
                res = await self.llm.complete(
                    [
                        {
                            "role": "user",
                            "content": prompt
                            + ("" if attempt == 0 else "\n(Return valid JSON only.)"),
                        }
                    ],
                    system=CORRECTNESS_SYSTEM,
                    max_tokens=200,
                    task="judge",
                    hints={"task": "judge", "answer": answer, "references": references},
                )
            except ProviderError as exc:
                return Verdict("invalid", f"judge call failed: {exc}"[:300], self.llm.cfg.model)
            obj = parse_json_object(res.text)
            if obj and str(obj.get("verdict", "")).lower() in VERDICTS:
                return Verdict(
                    str(obj["verdict"]).lower(), str(obj.get("reason", ""))[:500], res.model
                )
        return Verdict("invalid", f"unparseable judge output: {res.text[:200]}", self.llm.cfg.model)


class PairwiseJudge:
    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm

    async def _once(self, question: str, a: str, b: str) -> dict[str, str] | None:
        res = await self.llm.complete(
            [{"role": "user", "content": PAIRWISE_PROMPT.format(question=question, a=a, b=b)}],
            system=PAIRWISE_SYSTEM,
            max_tokens=200,
            task="pairwise",
            hints={"task": "pairwise"},
        )
        obj = parse_json_object(res.text)
        if not obj:
            return None
        out = {}
        for c in CRITERIA:
            v = str(obj.get(c, "")).strip().upper()
            out[c] = {"A": "A", "B": "B"}.get(v, "tie")
        return out

    async def compare(self, question: str, answer_1: str, answer_2: str) -> dict[str, str]:
        """Returns per criterion "1", "2" or "tie" — a win requires agreement in both orders."""
        first = await self._once(question, answer_1, answer_2)
        second = await self._once(question, answer_2, answer_1)
        if first is None or second is None:
            return dict.fromkeys(CRITERIA, "invalid")
        out = {}
        for c in CRITERIA:
            w1 = {"A": "1", "B": "2"}.get(first[c], "tie")
            w2 = {"A": "2", "B": "1"}.get(second[c], "tie")
            out[c] = w1 if w1 == w2 else "tie"
        return out
