import asyncio
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime
from typing import TypeVar

from pydantic import BaseModel

from pharma_agent.application.chat.context import TurnDeps
from pharma_agent.domain.guardrail.service import GuardrailService
from pharma_agent.domain.llm.models import ChatMessage, LlmRole, LlmUsage, StreamDelta
from pharma_agent.domain.llm.port import LlmError
from pharma_agent.domain.retrieval.models import Chunk, Hit, HydrateStrategy, Query
from pharma_agent.domain.retrieval.ports import Reranker, RetrievalError
from pharma_agent.domain.retrieval.service import RetrievalConfig, RetrievalService
from pharma_agent.domain.shared.clock import FixedClock
from pharma_agent.domain.skill.models import Skill, SkillMetadata

T = TypeVar("T", bound=BaseModel)

NOW = datetime(2026, 9, 11, 12, 0, tzinfo=UTC)


class FakeLlm:
    """Scripted LLM: one queue of structured responses per role, one streamed answer."""

    def __init__(self) -> None:
        self.structured_responses: dict[LlmRole, list[BaseModel | Exception]] = {}
        self.stream_text = (
            "Paracetamol người lớn 500 mg mỗi 4-6 giờ [1], tối đa 4 g/ngày [1]."
        )
        self.stream_error: Exception | None = None
        self.stream_delay: float = 0.0
        self.calls: list[tuple[LlmRole, list[ChatMessage]]] = []

    def script(self, role: LlmRole, *responses: BaseModel | Exception) -> None:
        self.structured_responses.setdefault(role, []).extend(responses)

    async def structured(
        self, role: LlmRole, messages: Sequence[ChatMessage], schema: type[T]
    ) -> tuple[T, LlmUsage]:
        self.calls.append((role, list(messages)))
        queue = self.structured_responses.get(role) or []
        if not queue:
            raise AssertionError(f"no scripted response for role {role.value}")
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        assert isinstance(item, schema), (
            f"scripted {type(item).__name__} but node asked for {schema.__name__}"
        )
        return item, LlmUsage(prompt_tokens=100, completion_tokens=10)

    async def stream(
        self, role: LlmRole, messages: Sequence[ChatMessage]
    ) -> AsyncIterator[StreamDelta]:
        self.calls.append((role, list(messages)))
        if self.stream_delay:
            await asyncio.sleep(self.stream_delay)
        if self.stream_error is not None:
            raise self.stream_error
        text = self.stream_text
        for start in range(0, len(text), 7):
            yield StreamDelta(text=text[start : start + 7])
        yield StreamDelta(usage=LlmUsage(prompt_tokens=800, completion_tokens=60))

    def calls_for(self, role: LlmRole) -> list[list[ChatMessage]]:
        return [messages for r, messages in self.calls if r is role]


class FakeRetriever:
    def __init__(self, *rounds: list[Hit] | Exception) -> None:
        self.rounds = list(rounds)
        self.calls: list[list[Query]] = []

    async def search_many(
        self, queries: Sequence[Query], top_k: int
    ) -> list[list[Hit]]:
        self.calls.append(list(queries))
        if not self.rounds:
            return [[] for _ in queries]
        current = self.rounds.pop(0)
        if isinstance(current, Exception):
            raise current
        return [
            [h.model_copy(update={"matched_queries": [q.text]}) for h in current]
            for q in queries
        ]


class FakeReranker:
    def __init__(self) -> None:
        self.received: list[list[str]] = []

    async def rerank(self, query: str, hits: Sequence[Hit], top_n: int) -> list[Hit]:
        self.received.append([h.chunk_id for h in hits])
        ordered = sorted(hits, key=lambda h: h.fusion_score, reverse=True)[:top_n]
        return [
            h.model_copy(update={"rerank_score": round(1.0 - i * 0.1, 2)})
            for i, h in enumerate(ordered)
        ]


class FakeHydrator:
    async def hydrate(self, hit: Hit, strategy: HydrateStrategy) -> list[Chunk]:
        if strategy is HydrateStrategy.SEARCH_ONLY:
            return []
        return [
            Chunk(
                chunk_id=hit.chunk_id,
                section_id=hit.section_id,
                chunk_index=hit.chunk_index,
                text=hit.chunk_text,
            )
        ]


class FakeSkillCatalog:
    def __init__(self, *skills: Skill) -> None:
        self.skills = list(skills)

    async def list_catalog(
        self, user_id: str | None, limit: int
    ) -> list[SkillMetadata]:
        return [s.metadata() for s in self.skills if s.enabled][:limit]

    async def get_by_names(
        self, user_id: str | None, names: Sequence[str]
    ) -> list[Skill]:
        wanted = set(names)
        return [s for s in self.skills if s.enabled and s.name in wanted]


MONOGRAPH_SKILL_MD = """---
name: drug-monograph
description: Tra cứu chuyên luận thuốc. Dùng khi hỏi liều, chỉ định, chống chỉ định của một thuốc.
---

# Tra cứu chuyên luận thuốc

## Khi tìm kiếm

- Tìm mục Liều dùng của chuyên luận.

## Khi trả lời

- Ghi liều kèm đơn vị và khoảng cách dùng.
"""


def monograph_skill() -> Skill:
    return Skill.from_markdown(MONOGRAPH_SKILL_MD, directory_name="drug-monograph")


def build_deps(
    llm: FakeLlm,
    retriever: FakeRetriever,
    catalog: FakeSkillCatalog | None = None,
    reranker: Reranker | None = None,
) -> TurnDeps:
    return TurnDeps(
        llm=llm,
        guardrail=GuardrailService(llm),
        retrieval=RetrievalService(
            retriever,
            reranker if reranker is not None else FakeReranker(),
            FakeHydrator(),
            RetrievalConfig(candidate_k=5, rerank_top_n=3),
        ),
        skills=catalog or FakeSkillCatalog(monograph_skill()),
        clock=FixedClock(NOW),
    )


__all__ = [
    "FakeHydrator",
    "FakeLlm",
    "FakeReranker",
    "FakeRetriever",
    "FakeSkillCatalog",
    "LlmError",
    "RetrievalError",
    "build_deps",
    "monograph_skill",
]
