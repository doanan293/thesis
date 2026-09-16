"""Fakes shared by the E2E tests."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Sequence
from typing import TypeVar

from pharma_agent.domain.llm.models import ChatMessage, LlmRole, LlmUsage, StreamDelta
from pharma_agent.domain.retrieval.models import Chunk, Hit, HydrateStrategy, Query
from pharma_agent.domain.retrieval.service import RetrievalConfig, RetrievalService
from pharma_agent.infrastructure.retrieval.llama_cpp_reranker import NoopReranker
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class ScriptedLlm:
    """Returns scripted structured results per role and a fixed streamed answer."""

    def __init__(
        self, answer: str = "Trả lời [1].", stream_error: Exception | None = None
    ) -> None:
        self.answer = answer
        self.stream_error = stream_error
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
        if self.stream_error is not None:
            raise self.stream_error
        yield StreamDelta(text=self.answer)
        yield StreamDelta(usage=LlmUsage(prompt_tokens=100, completion_tokens=20))


NAMESPACE = uuid.UUID("0f5c1d0e-7a8b-4c3d-9e2f-1a2b3c4d5e6f")
RELEASE = uuid.uuid5(NAMESPACE, "release")


def make_hit(
    label: str, section_key: str, *, ordinal: int = 1, score: float = 0.5
) -> Hit:
    return Hit(
        chunk_version_id=uuid.uuid5(NAMESPACE, label),
        release_id=RELEASE,
        collection_id=uuid.uuid5(NAMESPACE, "collection"),
        document_key="drug:paracetamol",
        section_key=section_key,
        section_revision_id=uuid.uuid5(NAMESPACE, section_key),
        ordinal=ordinal,
        hydrate_strategy=HydrateStrategy.CHUNK_WINDOW,
        source="Dược thư Quốc gia Việt Nam",
        title="Paracetamol",
        section="Liều dùng",
        start_page=1,
        end_page=1,
        context_header="Paracetamol > Liều dùng",
        chunk_text=f"text of {label}",
        embedding_text=f"text of {label}",
        kind="prose",
        table_key=None,
        fusion_score=score,
    )


class FakeRetriever:
    def __init__(self, hits: list[Hit]) -> None:
        self.hits = hits
        self.queries: list[list[str]] = []

    async def search_many(
        self, queries: Sequence[Query], top_k: int
    ) -> list[list[Hit]]:
        self.queries.append([query.text for query in queries])
        return [list(self.hits[:top_k]) for _ in queries]


class FakeHydrator:
    async def hydrate(self, hit: Hit, strategy: HydrateStrategy) -> list[Chunk]:
        return [
            Chunk(
                chunk_version_id=hit.chunk_version_id,
                section_revision_id=hit.section_revision_id,
                ordinal=hit.ordinal,
                text=hit.chunk_text,
                kind=hit.kind,
            )
        ]


def retrieval_service(retriever: FakeRetriever) -> RetrievalService:
    return RetrievalService(
        retriever,
        NoopReranker(),
        FakeHydrator(),
        RetrievalConfig(candidate_k=5, rerank_top_n=3),
    )
