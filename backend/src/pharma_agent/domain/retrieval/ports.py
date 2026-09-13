from collections.abc import Sequence
from typing import Protocol
from uuid import UUID

from pharma_agent.domain.retrieval.models import (
    Chunk,
    ChunkRecord,
    Hit,
    HydrateStrategy,
    Query,
)
from pharma_agent.domain.shared.errors import DomainError


class RetrievalError(DomainError):
    code = "RETRIEVAL_ERROR"


class Retriever(Protocol):
    async def search_many(
        self, queries: Sequence[Query], top_k: int
    ) -> list[list[Hit]]:
        """One ranked hit list per query, same order as `queries`. Raises RetrievalError."""
        ...


class Reranker(Protocol):
    async def rerank(self, query: str, hits: Sequence[Hit], top_n: int) -> list[Hit]:
        """Return up to top_n hits sorted by rerank_score desc. Raises RetrievalError."""
        ...


class Hydrator(Protocol):
    async def hydrate(self, hit: Hit, strategy: HydrateStrategy) -> list[Chunk]:
        """Return neighbouring chunks for the strategy (empty for SEARCH_ONLY). Raises RetrievalError."""
        ...


type ChunkKey = tuple[UUID, UUID]
"""(release_id, chunk_version_id): one chunk version as published in one release."""


class CorpusReader(Protocol):
    """Read side of the corpus schema used by retrieval."""

    async def current_releases(
        self, collection_keys: Sequence[str]
    ) -> dict[UUID, UUID]:
        """collection_id -> current release_id for the keys that exist and have a current release. Raises RetrievalError."""
        ...

    async def load_chunks(self, keys: Sequence[ChunkKey]) -> list[ChunkRecord]:
        """Records for the keys that exist, in no particular order. Raises RetrievalError."""
        ...

    async def section_chunks(
        self,
        release_id: UUID,
        section_revision_id: UUID,
        *,
        around: int | None,
        radius: int,
    ) -> list[Chunk]:
        """Chunks of one section revision in one release, ordered by ordinal.

        All of them when `around` is None, otherwise those with ordinal within around ± radius.
        Raises RetrievalError.
        """
        ...
