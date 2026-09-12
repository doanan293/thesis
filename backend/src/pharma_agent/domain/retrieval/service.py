from collections.abc import Mapping, Sequence

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
    rerank_candidates: int = Field(default=40, ge=1)


class SearchResult(BaseModel):
    items: list[RetrievedItem] = Field(default_factory=list)
    queries: list[Query] = Field(default_factory=list)
    rerank_failed: bool = False
    rerank_scored: int = 0
    rerank_reused: int = 0
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

    async def search(
        self,
        queries: Sequence[Query],
        rerank_query: str,
        known_scores: Mapping[str, float] | None = None,
    ) -> SearchResult:
        """Search every query, keep a capped candidate pool, rerank it and hydrate the top hits.

        `known_scores` maps chunk_id to a rerank score already computed for the same
        `rerank_query` (an earlier round of the same run); those chunks are not scored again.
        """
        try:
            per_query = await self._retriever.search_many(
                queries, self._config.candidate_k
            )
        except RetrievalError as exc:
            return SearchResult(queries=list(queries), error=str(exc))

        candidates = _candidate_pool(per_query, self._config.rerank_candidates)
        known = known_scores or {}
        reused = [
            hit.model_copy(update={"rerank_score": known[hit.chunk_id]})
            for hit in candidates
            if hit.chunk_id in known
        ]
        fresh = [hit for hit in candidates if hit.chunk_id not in known]
        rerank_failed = False
        try:
            scored = (
                await self._reranker.rerank(rerank_query, fresh, len(fresh))
                if fresh
                else []
            )
            ranked = sorted([*reused, *scored], key=lambda h: h.score, reverse=True)[
                : self._config.rerank_top_n
            ]
        except RetrievalError:
            rerank_failed = True
            reused, fresh = [], []
            ranked = sorted(candidates, key=lambda h: h.fusion_score, reverse=True)[
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
            items=items,
            queries=list(queries),
            rerank_failed=rerank_failed,
            rerank_scored=len(fresh),
            rerank_reused=len(reused),
        )


def _candidate_pool(per_query: list[list[Hit]], limit: int) -> list[Hit]:
    """Merge duplicates across queries, then pick up to `limit` chunks round-robin by rank.

    Round-robin keeps every query's best hits in the pool when several queries run at once;
    merged fields (best fusion score, matched queries in query order) cover every list.
    """
    merged = {hit.chunk_id: hit for hit in _dedupe(per_query)}
    selected: list[str] = []
    depth = max((len(hits) for hits in per_query), default=0)
    for rank in range(depth):
        for hits in per_query:
            if len(selected) >= limit:
                return [merged[chunk_id] for chunk_id in selected]
            if rank < len(hits) and hits[rank].chunk_id not in selected:
                selected.append(hits[rank].chunk_id)
    return [merged[chunk_id] for chunk_id in selected]


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
