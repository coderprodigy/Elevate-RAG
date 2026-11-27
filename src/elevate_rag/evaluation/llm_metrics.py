"""LLM-graded RAG metrics following the RAGAS definitions (FR-E2).

Implemented natively on the project's provider stack (cached, budgeted, recorded) rather than
through the `ragas` package, whose LangChain-based LLM wrappers would bypass the ledger and cache;
see DECISIONS D-19. Definitions (Es et al., RAGAS):

* faithfulness = |answer claims supported by the context| / |answer claims|
* context_recall = |reference-answer statements attributable to the context| / |statements|
* context_precision = mean over relevant positions k of precision@k (retrieved chunks judged
  useful for producing the reference answer)

Each returns None (missing, not 0) when it is undefined, e.g. a refusal has no claims.
"""

from __future__ import annotations

from typing import Any

from elevate_rag.evaluation.judge import parse_json_object
from elevate_rag.providers.llm import BaseLLM
from elevate_rag.utils import ProviderError

SYSTEM = (
    "You are a careful fact-checking assistant. Reply with JSON only, exactly in the "
    "requested schema."
)

FAITHFULNESS_PROMPT = """Split the ANSWER into short standalone factual claims, then mark each
claim as supported (1) or not supported (0) by the CONTEXT alone.
Return {{"claims": [{{"claim": "...", "supported": 0 or 1}}]}}.

CONTEXT:
{context}

QUESTION: {question}
ANSWER: {answer}
JSON:"""

RECALL_PROMPT = """Split the REFERENCE answer into short statements, then mark each statement as
attributable (1) or not (0) to the CONTEXT.
Return {{"statements": [{{"statement": "...", "attributed": 0 or 1}}]}}.

CONTEXT:
{context}

QUESTION: {question}
REFERENCE: {reference}
JSON:"""

PRECISION_PROMPT = """For each numbered PASSAGE, say whether it is useful (1) or not (0) for
arriving at the REFERENCE answer to the QUESTION.
Return {{"useful": [0 or 1 for each passage, in order]}}.

QUESTION: {question}
REFERENCE: {reference}

{passages}
JSON:"""


async def _ask(llm: BaseLLM, prompt: str, task: str) -> dict[str, Any] | None:
    try:
        res = await llm.complete(
            [{"role": "user", "content": prompt}],
            system=SYSTEM,
            max_tokens=1200,
            task=task,
            hints={"task": task},
        )
    except ProviderError:
        return None
    return parse_json_object(res.text)


async def faithfulness(
    llm: BaseLLM, question: str, answer: str, contexts: list[str]
) -> float | None:
    if not answer.strip() or not contexts:
        return None
    obj = await _ask(
        llm,
        FAITHFULNESS_PROMPT.format(context="\n\n".join(contexts), question=question, answer=answer),
        "faithfulness",
    )
    claims = (obj or {}).get("claims")
    if not isinstance(claims, list) or not claims:
        return None
    return sum(
        1 for c in claims if isinstance(c, dict) and c.get("supported") in (1, "1", True)
    ) / len(claims)


async def context_recall(
    llm: BaseLLM, question: str, reference: str, contexts: list[str]
) -> float | None:
    if not reference.strip():
        return None
    if not contexts:
        return 0.0
    obj = await _ask(
        llm,
        RECALL_PROMPT.format(context="\n\n".join(contexts), question=question, reference=reference),
        "context_recall",
    )
    st = (obj or {}).get("statements")
    if not isinstance(st, list) or not st:
        return None
    return sum(
        1 for s in st if isinstance(s, dict) and s.get("attributed") in (1, "1", True)
    ) / len(st)


async def context_precision(
    llm: BaseLLM, question: str, reference: str, contexts: list[str]
) -> float | None:
    if not reference.strip() or not contexts:
        return None
    passages = "\n\n".join(f"PASSAGE {i}:\n{c}" for i, c in enumerate(contexts, start=1))
    obj = await _ask(
        llm,
        PRECISION_PROMPT.format(question=question, reference=reference, passages=passages),
        "context_precision",
    )
    useful = (obj or {}).get("useful")
    if not isinstance(useful, list) or len(useful) != len(contexts):
        return None
    rel = [1 if u in (1, "1", True) else 0 for u in useful]
    if not any(rel):
        return 0.0
    hits, total = 0, 0.0
    for k, r in enumerate(rel, start=1):
        if r:
            hits += 1
            total += hits / k
    return total / sum(rel)
