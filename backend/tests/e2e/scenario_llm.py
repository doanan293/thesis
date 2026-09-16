"""FakeLlm whose behaviour is chosen by a marker in the user's question.

- `[e2e:blocked]`: the guardrail classifies the question as an attack; the turn ends `blocked`.
- `[e2e:timeout]`: the guardrail call hangs past the turn deadline; the turn ends `timeout`.
- anything else: a grounded answer that cites `[1]`.
"""

import asyncio
from collections.abc import Sequence
from typing import TypeVar

from pydantic import BaseModel

from pharma_agent.domain.agent.schemas import (
    Audience,
    Intent,
    JudgeDecision,
    JudgeOutcome,
    Language,
    RephraseResult,
)
from pharma_agent.domain.conversation.models import ConversationSummary
from pharma_agent.domain.guardrail.models import LlmGuardVerdict
from pharma_agent.domain.llm.models import ChatMessage, LlmRole, LlmUsage
from tests.fakes import FakeLlm

T = TypeVar("T", bound=BaseModel)

BLOCKED_MARKER = "[e2e:blocked]"
TIMEOUT_MARKER = "[e2e:timeout]"
STANDALONE_QUERY = "liều dùng paracetamol cho người lớn"
GROUNDED_ANSWER = "Người lớn uống 500 mg mỗi 4–6 giờ, tối đa 4 g mỗi ngày [1]."


class ScenarioLlm(FakeLlm):
    def __init__(self, *, hang_seconds: float = 3600.0) -> None:
        super().__init__()
        self.stream_text = GROUNDED_ANSWER
        self._hang_seconds = hang_seconds

    async def structured(
        self, role: LlmRole, messages: Sequence[ChatMessage], schema: type[T]
    ) -> tuple[T, LlmUsage]:
        self.calls.append((role, list(messages)))
        question = messages[-1].content if messages else ""
        if role is LlmRole.GUARDRAIL and TIMEOUT_MARKER in question:
            await asyncio.sleep(self._hang_seconds)
        response = _scripted(role, question)
        if not isinstance(response, schema):
            raise AssertionError(
                f"scenario returns {type(response).__name__} for role {role.value}, "
                f"node asked for {schema.__name__}"
            )
        return response, LlmUsage(prompt_tokens=100, completion_tokens=10)


def _scripted(role: LlmRole, question: str) -> BaseModel:
    if role is LlmRole.GUARDRAIL:
        blocked = BLOCKED_MARKER in question
        return LlmGuardVerdict(
            is_attack=blocked, in_scope=not blocked, reason="e2e scenario"
        )
    if role is LlmRole.REPHRASE:
        return RephraseResult(
            standalone_query=STANDALONE_QUERY,
            audience=Audience.GENERAL_PUBLIC,
            language=Language.VI,
            intent=Intent.PHARMA_QUESTION,
        )
    if role is LlmRole.JUDGE:
        return JudgeDecision(
            decision=JudgeOutcome.ANSWER, gaps=[], reason="e2e scenario"
        )
    if role is LlmRole.SUMMARIZER:
        return ConversationSummary(summary="Người dùng hỏi về liều paracetamol.")
    raise AssertionError(f"no e2e scenario for role {role.value}")
