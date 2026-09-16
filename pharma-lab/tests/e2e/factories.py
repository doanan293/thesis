"""Fakes shared by the E2E tests."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import TypeVar

from pharma_agent.domain.llm.models import ChatMessage, LlmRole, LlmUsage, StreamDelta
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class ScriptedLlm:
    """Returns scripted structured results per role and a fixed streamed answer."""

    def __init__(self, answer: str = "Trả lời [1].") -> None:
        self.answer = answer
        self.scripts: dict[LlmRole, list[BaseModel | Exception]] = {}
        self.calls: list[tuple[LlmRole, list[ChatMessage]]] = []

    def script(self, role: LlmRole, *responses: BaseModel | Exception) -> None:
        self.scripts.setdefault(role, []).extend(responses)

    async def structured(
        self, role: LlmRole, messages: Sequence[ChatMessage], schema: type[T]
    ) -> tuple[T, LlmUsage]:
        self.calls.append((role, list(messages)))
        response = self.scripts[role].pop(0)
        if isinstance(response, Exception):
            raise response
        assert isinstance(response, schema)
        return response, LlmUsage(prompt_tokens=10, completion_tokens=2)

    async def stream(
        self, role: LlmRole, messages: Sequence[ChatMessage]
    ) -> AsyncIterator[StreamDelta]:
        self.calls.append((role, list(messages)))
        yield StreamDelta(text=self.answer)
        yield StreamDelta(usage=LlmUsage(prompt_tokens=100, completion_tokens=20))
