from datetime import UTC, datetime

from pharma_agent.domain.agent.budget import BudgetLimits
from pharma_agent.domain.agent.run import AgentRun
from pharma_agent.domain.retrieval.models import (
    Chunk,
    Hit,
    HydrateStrategy,
    RetrievedItem,
)
from pharma_agent.domain.retrieval.service import SearchResult

NOW = datetime(2026, 9, 11, 12, 0, tzinfo=UTC)


def make_hit(
    chunk_id: str,
    *,
    section_id: str = "sec-1",
    chunk_index: int = 0,
    strategy: HydrateStrategy = HydrateStrategy.CHUNK_WINDOW,
    fusion: float = 0.5,
    rerank: float | None = None,
    text: str = "paracetamol 500 mg",
    title: str = "Paracetamol",
    section: str = "Liều dùng",
    table_id: str = "",
) -> Hit:
    return Hit(
        chunk_id=chunk_id,
        section_id=section_id,
        chunk_index=chunk_index,
        hydrate_strategy=strategy,
        source="duoc_thu",
        title=title,
        section=section,
        start_page=10,
        end_page=11,
        context_header=f"{title} > {section}",
        chunk_text=text,
        embedding_text=f"{title} > {section}\n\n{text}",
        table_id=table_id,
        fusion_score=fusion,
        rerank_score=rerank,
        matched_queries=["q1"],
    )


def make_item(
    chunk_id: str, *, rerank: float = 0.8, text: str = "paracetamol 500 mg"
) -> RetrievedItem:
    hit = make_hit(chunk_id, rerank=rerank, text=text)
    return RetrievedItem(
        hit=hit,
        chunks=[
            Chunk(
                chunk_id=chunk_id,
                section_id=hit.section_id,
                chunk_index=hit.chunk_index,
                text=text,
            )
        ],
    )


def search_result(*chunk_ids: str, error: str | None = None) -> SearchResult:
    return SearchResult(items=[make_item(c) for c in chunk_ids], error=error)


def make_run(
    query: str = "Paracetamol liều người lớn?",
    *,
    max_llm_calls: int = 10,
    max_search_rounds: int = 3,
    max_tokens: int = 40_000,
) -> AgentRun:
    return AgentRun.start(
        user_id="u1",
        original_query=query,
        limits=BudgetLimits(
            max_llm_calls=max_llm_calls,
            max_search_rounds=max_search_rounds,
            max_tokens=max_tokens,
        ),
        now=NOW,
        run_id="run-1",
    )
