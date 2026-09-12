from collections.abc import Sequence

from pharma_agent.domain.retrieval.models import (
    Chunk,
    Hit,
    HydrateStrategy,
    Query,
    QueryOrigin,
)
from pharma_agent.domain.retrieval.ports import RetrievalError
from pharma_agent.domain.retrieval.service import RetrievalConfig, RetrievalService


def hit(
    chunk_id: str,
    fusion: float,
    strategy: HydrateStrategy = HydrateStrategy.CHUNK_WINDOW,
) -> Hit:
    return Hit(
        chunk_id=chunk_id,
        section_id="s",
        chunk_index=1,
        hydrate_strategy=strategy,
        source="src",
        title="T",
        section="S",
        start_page=1,
        end_page=1,
        context_header="T > S",
        chunk_text=f"text {chunk_id}",
        embedding_text=f"T > S\n\ntext {chunk_id}",
        fusion_score=fusion,
    )


class FakeRetriever:
    def __init__(self, results: list[list[Hit]] | Exception) -> None:
        self.results = results
        self.calls: list[tuple[list[Query], int]] = []

    async def search_many(
        self, queries: Sequence[Query], top_k: int
    ) -> list[list[Hit]]:
        self.calls.append((list(queries), top_k))
        if isinstance(self.results, Exception):
            raise self.results
        return [
            [h.model_copy(update={"matched_queries": [q.text]}) for h in hits]
            for q, hits in zip(queries, self.results, strict=True)
        ]


class FakeReranker:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.received: list[Hit] = []
        self.calls = 0

    async def rerank(self, query: str, hits: Sequence[Hit], top_n: int) -> list[Hit]:
        if self.fail:
            raise RetrievalError("rerank down")
        self.received = list(hits)
        self.calls += 1
        scored = [
            h.model_copy(update={"rerank_score": 1.0 / (i + 1)})
            for i, h in enumerate(reversed(list(hits)))
        ]
        return sorted(scored, key=lambda h: h.rerank_score or 0, reverse=True)[:top_n]


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


def queries(*texts: str) -> list[Query]:
    return [Query(text=t, origin=QueryOrigin.INITIAL) for t in texts]


async def test_search_dedupes_across_queries_then_reranks_and_hydrates() -> None:
    retriever = FakeRetriever(
        [[hit("a", 0.9), hit("b", 0.5)], [hit("b", 0.7), hit("c", 0.4)]]
    )
    reranker = FakeReranker()
    service = RetrievalService(
        retriever,
        reranker,
        FakeHydrator(),
        RetrievalConfig(candidate_k=5, rerank_top_n=2),
    )

    result = await service.search(queries("q1", "q2"), rerank_query="q1")

    assert retriever.calls[0][1] == 5
    assert sorted(h.chunk_id for h in reranker.received) == ["a", "b", "c"]
    b = next(h for h in reranker.received if h.chunk_id == "b")
    assert b.fusion_score == 0.7 and b.matched_queries == ["q1", "q2"]
    assert len(result.items) == 2
    assert all(item.chunks for item in result.items)
    assert result.rerank_failed is False and result.error is None


async def test_rerank_failure_keeps_fusion_order() -> None:
    retriever = FakeRetriever([[hit("a", 0.9), hit("b", 0.5), hit("c", 0.1)]])
    service = RetrievalService(
        retriever,
        FakeReranker(fail=True),
        FakeHydrator(),
        RetrievalConfig(candidate_k=5, rerank_top_n=2),
    )
    result = await service.search(queries("q1"), rerank_query="q1")
    assert [i.hit.chunk_id for i in result.items] == ["a", "b"]
    assert result.rerank_failed is True


async def test_retriever_failure_is_reported_not_raised() -> None:
    service = RetrievalService(
        FakeRetriever(RetrievalError("qdrant down")),
        FakeReranker(),
        FakeHydrator(),
        RetrievalConfig(),
    )
    result = await service.search(queries("q1"), rerank_query="q1")
    assert result.items == [] and result.error == "qdrant down"


async def test_search_only_hits_are_not_hydrated() -> None:
    retriever = FakeRetriever([[hit("a", 0.9, HydrateStrategy.SEARCH_ONLY)]])
    service = RetrievalService(
        retriever, FakeReranker(), FakeHydrator(), RetrievalConfig()
    )
    result = await service.search(queries("q1"), rerank_query="q1")
    assert result.items[0].chunks == []


async def test_candidate_pool_interleaves_queries_by_rank_and_caps_new_chunks() -> None:
    retriever = FakeRetriever(
        [
            [hit("a1", 0.9), hit("a2", 0.8), hit("a3", 0.7)],
            [hit("b1", 0.6), hit("a1", 0.5), hit("b2", 0.4)],
        ]
    )
    reranker = FakeReranker()
    service = RetrievalService(
        retriever,
        reranker,
        FakeHydrator(),
        RetrievalConfig(candidate_k=3, rerank_top_n=2, rerank_candidates=4),
    )

    result = await service.search(queries("q1", "q2"), rerank_query="q")

    assert [h.chunk_id for h in reranker.received] == ["a1", "b1", "a2", "a3"]
    assert reranker.received[0].matched_queries == ["q1", "q2"]
    assert (result.rerank_scored, result.rerank_reused) == (4, 0)


async def test_known_scores_are_reused_instead_of_rescored() -> None:
    retriever = FakeRetriever([[hit("a", 0.9), hit("b", 0.5), hit("c", 0.1)]])
    reranker = FakeReranker()
    service = RetrievalService(
        retriever,
        reranker,
        FakeHydrator(),
        RetrievalConfig(candidate_k=5, rerank_top_n=3),
    )

    result = await service.search(
        queries("q1"), rerank_query="q1", known_scores={"b": 0.75}
    )

    assert [h.chunk_id for h in reranker.received] == ["a", "c"]
    assert [(i.hit.chunk_id, i.hit.rerank_score) for i in result.items] == [
        ("c", 1.0),
        ("b", 0.75),
        ("a", 0.5),
    ]
    assert (result.rerank_scored, result.rerank_reused) == (2, 1)


async def test_reranker_is_not_called_when_every_candidate_is_known() -> None:
    reranker = FakeReranker(fail=True)
    service = RetrievalService(
        FakeRetriever([[hit("a", 0.9), hit("b", 0.5)]]),
        reranker,
        FakeHydrator(),
        RetrievalConfig(candidate_k=5, rerank_top_n=2),
    )

    result = await service.search(
        queries("q1"), rerank_query="q1", known_scores={"a": 0.2, "b": 0.6}
    )

    assert reranker.calls == 0 and result.rerank_failed is False
    assert [i.hit.chunk_id for i in result.items] == ["b", "a"]
