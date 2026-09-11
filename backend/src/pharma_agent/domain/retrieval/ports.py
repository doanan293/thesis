from collections.abc import Sequence
from typing import Protocol

from pharma_agent.domain.retrieval.models import Chunk, Hit, HydrateStrategy, Query
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
