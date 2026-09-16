from dataclasses import dataclass

from pharma_agent.domain.conversation.models import ConversationContext
from pharma_agent.domain.guardrail.service import GuardrailService
from pharma_agent.domain.llm.port import LlmPort
from pharma_agent.domain.retrieval.service import RetrievalService
from pharma_agent.domain.shared.clock import Clock


@dataclass(frozen=True)
class TurnDeps:
    """Ports a chat turn needs. Built once by the composition root."""

    llm: LlmPort
    guardrail: GuardrailService
    retrieval: RetrievalService
    clock: Clock


@dataclass(frozen=True)
class PipelineOptions:
    """Steps of the agent loop. Only the evaluation harness turns any of them off."""

    rephrase: bool = True
    judge_refine: bool = True


@dataclass(frozen=True)
class TurnContext:
    """LangGraph runtime context for one turn (not checkpointed)."""

    deps: TurnDeps
    conversation: ConversationContext
    pipeline: PipelineOptions = PipelineOptions()
