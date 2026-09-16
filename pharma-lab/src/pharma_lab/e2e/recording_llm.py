"""Count calls and tokens per LLM role for one turn."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import TypeVar

from pharma_agent.domain.llm.models import ChatMessage, LlmRole, LlmUsage, StreamDelta
from pharma_agent.domain.llm.port import LlmPort
from pydantic import BaseModel

from pharma_lab.e2e.records import TokenUsage

T = TypeVar("T", bound=BaseModel)


class RecordingLlm:
    def __init__(self, inner: LlmPort) -> None:
        self._inner = inner
        self._usage: dict[str, TokenUsage] = {}

    def _record(self, role: LlmRole, usage: LlmUsage) -> None:
        current = self._usage.get(role.value, TokenUsage())
        self._usage[role.value] = TokenUsage(
            calls=current.calls + 1,
            prompt_tokens=current.prompt_tokens + usage.prompt_tokens,
            completion_tokens=current.completion_tokens + usage.completion_tokens,
        )

    async def structured(
        self, role: LlmRole, messages: Sequence[ChatMessage], schema: type[T]
    ) -> tuple[T, LlmUsage]:
        result, usage = await self._inner.structured(role, messages, schema)
        self._record(role, usage)
        return result, usage

    async def stream(
        self, role: LlmRole, messages: Sequence[ChatMessage]
    ) -> AsyncIterator[StreamDelta]:
        usage = LlmUsage()
        async for delta in self._inner.stream(role, messages):
            if delta.usage is not None:
                usage = delta.usage
            yield delta
        self._record(role, usage)

    def usage_by_role(self) -> dict[str, TokenUsage]:
        return dict(self._usage)
