"""RAGAS collection metrics evaluated through the project's budgeted judge adapter."""

import json
import math
import os
import time
from dataclasses import asdict
from typing import Any

import httpx

from .data import Question, stable_hash
from .pipeline import CITATION, Output
from .providers import Providers


async def score_ragas(question: Question, output: Output, providers: Providers) -> dict[str, float]:
    os.environ["RAGAS_DO_NOT_TRACK"] = "true"
    from instructor import Mode
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.metrics.collections import ContextPrecision, ContextRecall, Faithfulness

    model = providers.settings.model("judge")
    if model.provider != "openai-compatible":
        raise ValueError("RAGAS requires an OpenAI-compatible judge")
    contexts = [passage.text for passage in output.context.passages]
    if not contexts:
        return {}

    async def transport(request: httpx.Request) -> httpx.Response:
        if request.url.path != "/v1/chat/completions":
            raise ValueError("RAGAS attempted an unsupported judge operation")
        payload = json.loads(request.content)
        payload.update(model=model.model, temperature=0, max_tokens=model.max_output)
        if model.seed is not None:
            payload["seed"] = model.seed
        key = stable_hash("ragas-0.4.4", asdict(model), payload)
        cache = providers.settings.cache_judge and not providers.uncached
        started = time.perf_counter()
        if cache and (body := providers.cache.get("ragas", key)) is not None:
            attempts, cached = 0, True
        else:
            body, attempts = await providers.request("judge", model, "chat/completions", payload)
            cached = False
            if cache:
                providers.cache.put("ragas", key, body)
        counts = providers.usage_counts(body, "chat/completions")
        providers.record(
            "judge",
            model,
            counts,
            started,
            cached=cached,
            reported=None not in counts,
            attempts=attempts,
        )
        return httpx.Response(200, json=body)

    client = AsyncOpenAI(
        api_key="local-transport",
        base_url="https://elevate.invalid/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(transport)),
    )
    values: dict[str, float] = {}

    def read(result: Any) -> float:
        score = float(result.value)
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("RAGAS returned a score outside [0, 1]")
        return score

    async with client:
        llm = llm_factory(model.model, client=client, mode=Mode.JSON, max_retries=1)
        if question.answers and question.answerable:
            values["context_precision"] = read(
                await ContextPrecision(llm=llm).ascore(
                    user_input=question.text,
                    reference=question.answers[0],
                    retrieved_contexts=contexts,
                )
            )
            values["context_recall"] = read(
                await ContextRecall(llm=llm).ascore(
                    user_input=question.text,
                    reference=question.answers[0],
                    retrieved_contexts=contexts,
                )
            )
        if not output.answer.refused:
            values["faithfulness"] = read(
                await Faithfulness(llm=llm).ascore(
                    user_input=question.text,
                    response=CITATION.sub("", output.answer.text),
                    retrieved_contexts=contexts,
                )
            )
    return values
