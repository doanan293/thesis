from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from pharma_agent.domain.retrieval.models import (
    Hit,
    HydrateStrategy,
    Query,
    RetrievedItem,
)
from pharma_agent.domain.retrieval.ports import (
    Hydrator,
    Reranker,
    RetrievalError,
    Retriever,
)


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    candidate_k: int = 30
    rerank_top_n: int = 8


class SearchResult(BaseModel):
    items: list[RetrievedItem] = Field(default_factory=list)
    queries: list[Query] = Field(default_factory=list)
    rerank_failed: bool = False
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


class RetrievalService:
    """Pure orchestration over the retrieval ports: search many, dedupe, rerank, hydrate."""

    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
        hydrator: Hydrator,
        config: RetrievalConfig,
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._hydrator = hydrator
        self._config = config

    async def search(self, queries: Sequence[Query], rerank_query: str) -> SearchResult:
        try:
            per_query = await self._retriever.search_many(
                queries, self._config.candidate_k
            )
        except RetrievalError as exc:
            return SearchResult(queries=list(queries), error=str(exc))

        merged = _dedupe(per_query)
        rerank_failed = False
        try:
            ranked = await self._reranker.rerank(
                rerank_query, merged, self._config.rerank_top_n
            )
        except RetrievalError:
            rerank_failed = True
            ranked = sorted(merged, key=lambda h: h.fusion_score, reverse=True)[
                : self._config.rerank_top_n
            ]

        items: list[RetrievedItem] = []
        for hit in ranked:
            chunks = []
            if hit.hydrate_strategy is not HydrateStrategy.SEARCH_ONLY:
                try:
                    chunks = await self._hydrator.hydrate(hit, hit.hydrate_strategy)
                except RetrievalError:
                    chunks = []
            items.append(RetrievedItem(hit=hit, chunks=chunks))
        return SearchResult(
            items=items, queries=list(queries), rerank_failed=rerank_failed
        )


def _dedupe(per_query: list[list[Hit]]) -> list[Hit]:
    by_chunk: dict[str, Hit] = {}
    for hits in per_query:
        for hit in hits:
            existing = by_chunk.get(hit.chunk_id)
            if existing is None:
                by_chunk[hit.chunk_id] = hit
                continue
            queries = list(existing.matched_queries)
            queries.extend(q for q in hit.matched_queries if q not in queries)
            by_chunk[hit.chunk_id] = existing.model_copy(
                update={
                    "fusion_score": max(existing.fusion_score, hit.fusion_score),
                    "matched_queries": queries,
                }
            )
    return list(by_chunk.values())
