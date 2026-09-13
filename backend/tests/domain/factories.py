import uuid
from datetime import UTC, datetime

from pharma_agent.domain.agent.budget import BudgetLimits
from pharma_agent.domain.agent.run import AgentRun
from pharma_agent.domain.conversation.models import Citation
from pharma_agent.domain.retrieval.models import (
    Chunk,
    Hit,
    HydrateStrategy,
    RetrievedItem,
)
from pharma_agent.domain.retrieval.service import SearchResult

NOW = datetime(2026, 9, 11, 12, 0, tzinfo=UTC)
TEST_NAMESPACE = uuid.UUID("5f0c7a52-3d4e-4b8a-9c21-6e7f8a9b0c1d")
RELEASE_ID = uuid.uuid5(TEST_NAMESPACE, "release")
COLLECTION_ID = uuid.uuid5(TEST_NAMESPACE, "collection")
SECTION_KEY = "drug:paracetamol:lieu-luong-va-cach-dung"
SOURCE = "Dược thư Quốc gia Việt Nam"


def chunk_uuid(label: str) -> uuid.UUID:
    """Stable chunk version id for a readable test label."""
    return uuid.uuid5(TEST_NAMESPACE, f"chunk:{label}")


def revision_uuid(section_key: str) -> uuid.UUID:
    return uuid.uuid5(TEST_NAMESPACE, f"revision:{section_key}")


def make_hit(
    label: str,
    *,
    section_key: str = SECTION_KEY,
    ordinal: int = 1,
    strategy: HydrateStrategy = HydrateStrategy.CHUNK_WINDOW,
    fusion: float = 0.5,
    rerank: float | None = None,
    text: str = "paracetamol 500 mg",
    title: str = "Paracetamol",
    section: str = "Liều dùng",
    kind: str = "prose",
    table_key: str | None = None,
    start_page: int | None = 10,
    end_page: int | None = 11,
) -> Hit:
    return Hit(
        chunk_version_id=chunk_uuid(label),
        release_id=RELEASE_ID,
        collection_id=COLLECTION_ID,
        document_key="drug:paracetamol",
        section_key=section_key,
        section_revision_id=revision_uuid(section_key),
        ordinal=ordinal,
        hydrate_strategy=strategy,
        source=SOURCE,
        title=title,
        section=section,
        start_page=start_page,
        end_page=end_page,
        context_header=f"{title} > {section}",
        chunk_text=text,
        embedding_text=f"{title} > {section}\n\n{text}",
        kind=kind,
        table_key=table_key,
        fusion_score=fusion,
        rerank_score=rerank,
        matched_queries=["q1"],
    )


def make_chunk(
    label: str,
    *,
    ordinal: int,
    text: str,
    section_key: str = SECTION_KEY,
    kind: str = "prose",
    table_key: str | None = None,
) -> Chunk:
    return Chunk(
        chunk_version_id=chunk_uuid(label),
        section_revision_id=revision_uuid(section_key),
        ordinal=ordinal,
        text=text,
        kind=kind,
        table_key=table_key,
        start_page=10,
        end_page=11,
    )


def chunk_of(hit: Hit) -> Chunk:
    """The chunk a hit was found on, as a hydrator would return it."""
    return Chunk(
        chunk_version_id=hit.chunk_version_id,
        section_revision_id=hit.section_revision_id,
        ordinal=hit.ordinal,
        text=hit.chunk_text,
        kind=hit.kind,
        table_key=hit.table_key,
        start_page=hit.start_page,
        end_page=hit.end_page,
    )


def make_item(
    label: str, *, rerank: float = 0.8, text: str = "paracetamol 500 mg"
) -> RetrievedItem:
    hit = make_hit(label, rerank=rerank, text=text)
    return RetrievedItem(hit=hit, chunks=[chunk_of(hit)])


def make_citation(label: str, *, index: int = 1) -> Citation:
    return Citation(
        index=index,
        chunk_version_id=chunk_uuid(label),
        release_id=RELEASE_ID,
        strategy=HydrateStrategy.CHUNK_WINDOW,
        block_chunk_version_ids=[chunk_uuid(label)],
        source=SOURCE,
        title="Paracetamol",
        section="Liều dùng",
        start_page=10,
        end_page=11,
        snippet="paracetamol 500 mg",
    )


def search_result(*labels: str, error: str | None = None) -> SearchResult:
    return SearchResult(items=[make_item(label) for label in labels], error=error)


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
