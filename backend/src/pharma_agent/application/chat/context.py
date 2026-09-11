from dataclasses import dataclass

from pharma_agent.domain.conversation.models import ConversationContext
from pharma_agent.domain.guardrail.service import GuardrailService
from pharma_agent.domain.llm.port import LlmPort
from pharma_agent.domain.retrieval.service import RetrievalService
from pharma_agent.domain.shared.clock import Clock
from pharma_agent.domain.skill.ports import SkillCatalog


@dataclass(frozen=True)
class TurnDeps:
    """Ports a chat turn needs. Built once by the composition root."""

    llm: LlmPort
    guardrail: GuardrailService
    retrieval: RetrievalService
    skills: SkillCatalog
    clock: Clock


@dataclass(frozen=True)
class TurnContext:
    """LangGraph runtime context for one turn (not checkpointed)."""

    deps: TurnDeps
    conversation: ConversationContext
