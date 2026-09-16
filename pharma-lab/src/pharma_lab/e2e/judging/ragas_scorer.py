"""RAGAS metrics for grounded answers (spec §7)."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Protocol

from openai import AsyncOpenAI
from pharma_agent.infrastructure.settings import Settings
from ragas._analytics import get_userid
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, FactualCorrectness, Faithfulness

from pharma_lab.e2e.judging.structured import (
    JUDGE_MODEL,
    JUDGE_REASONING,
    judge_llm_settings,
)

# Reasoning tokens count against the completion budget; RAGAS defaults to 1024.
MAX_COMPLETION_TOKENS = 16_000


class ScoreResult(Protocol):
    @property
    def value(self) -> object: ...


class FaithfulnessMetric(Protocol):
    async def ascore(
        self, user_input: str, response: str, retrieved_contexts: list[str]
    ) -> ScoreResult: ...


class FactualMetric(Protocol):
    async def ascore(self, response: str, reference: str) -> ScoreResult: ...


class RelevancyMetric(Protocol):
    async def ascore(self, user_input: str, response: str) -> ScoreResult: ...


@dataclass(frozen=True)
class RagasScores:
    faithfulness: float | None
    factual_correctness: float | None
    answer_relevancy: float | None


def _number(result: ScoreResult) -> float | None:
    value = result.value
    return float(value) if isinstance(value, int | float) else None


class RagasScorer:
    def __init__(
        self,
        faithfulness: FaithfulnessMetric,
        factual: FactualMetric,
        relevancy: RelevancyMetric,
    ) -> None:
        self._faithfulness = faithfulness
        self._factual = factual
        self._relevancy = relevancy

    async def score(
        self,
        *,
        question: str,
        answer: str,
        contexts: list[str],
        reference: str | None,
    ) -> RagasScores:
        faithfulness = await self._faithfulness.ascore(
            user_input=question, response=answer, retrieved_contexts=contexts
        )
        factual = (
            await self._factual.ascore(response=answer, reference=reference)
            if reference
            else None
        )
        relevancy = await self._relevancy.ascore(user_input=question, response=answer)
        return RagasScores(
            faithfulness=_number(faithfulness),
            factual_correctness=None if factual is None else _number(factual),
            answer_relevancy=_number(relevancy),
        )


def _resolve_ragas_user_id() -> None:
    """Resolve RAGAS's cached analytics id once, without its file-handle leak.

    ragas 0.4.3 builds an analytics event (even with RAGAS_DO_NOT_TRACK) whose
    default id comes from `ragas._analytics.get_userid`, which reads its id file with
    `json.load(open(path))` and never closes it (ragas/_analytics.py line 86). The
    result is cached, so resolving it here is the only place the ResourceWarning can
    occur. Remove this once ragas closes the file.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        get_userid()


def build_ragas_scorer(settings: Settings) -> RagasScorer:
    """RAGAS with gpt-5-mini on the backend LLM endpoint and the local embedder."""
    os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
    _resolve_ragas_user_id()
    judge = judge_llm_settings(settings)
    llm = llm_factory(
        JUDGE_MODEL,
        client=AsyncOpenAI(
            api_key=judge.default.api_key,
            base_url=judge.default.base_url,
            timeout=judge.timeout_seconds,
            max_retries=judge.max_retries,
        ),
        reasoning_effort=JUDGE_REASONING,
        max_tokens=MAX_COMPLETION_TOKENS,
    )
    embedding = settings.retrieval.embedding
    embeddings = OpenAIEmbeddings(
        client=AsyncOpenAI(
            api_key=embedding.api_key,
            base_url=embedding.base_url,
            timeout=embedding.timeout_seconds,
            max_retries=embedding.max_retries,
        ),
        model=embedding.model,
    )
    return RagasScorer(
        Faithfulness(llm=llm),
        FactualCorrectness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
    )
