"""The E2E configurations: five pipeline variants and a closed-book baseline."""

from __future__ import annotations

from enum import StrEnum

from pharma_agent.application.chat.context import PipelineOptions
from pharma_agent.infrastructure.settings import Settings


class E2EConfig(StrEnum):
    FULL = "full"
    ONE_STEP = "one-step"
    NO_JUDGE_REFINE = "no-judge-refine"
    NO_REPHRASE = "no-rephrase"
    NO_RERANK = "no-rerank"
    # The answer model alone, with no retrieval: what the corpus and pipeline add.
    CLOSED_BOOK = "closed-book"


_PIPELINES: dict[E2EConfig, PipelineOptions] = {
    E2EConfig.FULL: PipelineOptions(),
    E2EConfig.ONE_STEP: PipelineOptions(rephrase=False, judge_refine=False),
    E2EConfig.NO_JUDGE_REFINE: PipelineOptions(judge_refine=False),
    E2EConfig.NO_REPHRASE: PipelineOptions(rephrase=False),
    E2EConfig.NO_RERANK: PipelineOptions(),
    E2EConfig.CLOSED_BOOK: PipelineOptions(rephrase=False, judge_refine=False),
}


def pipeline_for(config: E2EConfig) -> PipelineOptions:
    return _PIPELINES[config]


def uses_retrieval(config: E2EConfig) -> bool:
    return config is not E2EConfig.CLOSED_BOOK


def settings_for(base: Settings, config: E2EConfig) -> Settings:
    """Backend settings for one configuration; the environment is never changed."""
    retrieval = base.retrieval
    if config is E2EConfig.NO_RERANK:
        rerank = retrieval.rerank.model_copy(update={"protocol": "none"})
        retrieval = retrieval.model_copy(update={"rerank": rerank})
    langfuse = base.langfuse.model_copy(update={"public_key": None, "secret_key": None})
    return base.model_copy(update={"retrieval": retrieval, "langfuse": langfuse})
