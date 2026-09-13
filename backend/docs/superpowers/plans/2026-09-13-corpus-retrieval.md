# Corpus Retrieval Implementation Plan (Plan 3 of 10)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retrieval runs on the corpus platform: Qdrant returns chunk version ids and scores filtered to the current release of every scoped collection, Postgres supplies the chunk content and hydration, citations and audit rows carry immutable chunk version and release ids, and `/health` reports whether the corpus is ready.

**Architecture:** The domain keeps the `Retriever`, `Reranker`, `Hydrator` ports and `RetrievalService`; `Hit`, `Chunk` and `Citation` switch to corpus identities, and a new `CorpusReader` port (with the `ChunkRecord` value it returns) describes the read side of the corpus schema. Infrastructure implements it with `PostgresCorpusReader` and `PostgresHydrator` (SQLAlchemy async over the P2 tables), rewrites `QdrantHybridRetriever` (modes `hybrid`, `dense`, `bm25`) to filter by release and load chunks through the reader, and exposes `build_retrieval_service(settings, *, database=None, embedder=None) -> RetrievalStack`, which `build_application`, the container health check and the CLI reuse. Audit moves to release ids and chunk version ids in migration `0006`.

**Tech Stack:** Python 3.12, pydantic 2, SQLAlchemy 2 async + psycopg, Alembic, qdrant-client 1.19 (server `qdrant/qdrant:v1.19.1`), LangGraph (unchanged), pytest + pytest-asyncio + testcontainers.

**Spec:** `backend/docs/superpowers/specs/2026-09-13-corpus-platform-design.md` §9 (retrieval), §8.3 (Qdrant layout and query filter), §6.2 (tables read here), §12 integration row (search, hydrate, publish/rollback). Names follow `backend/docs/superpowers/plans/2026-09-13-plans-overview.md` §3.1–§3.4. Builds on P1 (`pharma_agent.domain.corpus`, `pharma_agent.domain.shared.text`) and P2 (schema `corpus`, `ImportKnowledgeBundle`, `ReleaseService`, `QdrantVectorIndex`, `open_corpus_services`, fixture bundle, `FakeEmbedder`), written in `backend/docs/superpowers/plans/2026-09-13-corpus-domain.md` and `2026-09-13-corpus-store-import.md`.

## Global Constraints

- The environment is development only. Postgres and Qdrant may be reset; no data backfill or backward compatibility. Migration `0006` deletes existing audit rows and empties `messages.citations`.
- Python 3.12, uv, shared `ruff.toml`, strict `pyrefly.toml` with `unused-ignore = true`, pytest `filterwarnings = ["error"]`. Lint and type errors are fixed in code; never add rule ignores, `# noqa`, `# type: ignore` or new `# pyrefly: ignore`.
- Every task ends green on: `uv run ruff check src tests && uv run ruff format --check src tests && uv run pyrefly check --min-severity warn && uv run pytest -q`. Tasks touching Postgres or Qdrant also run `uv run pytest -q -m integration` (all commands from `backend/`).
- Layering stays enforced by `tests/architecture/test_layering.py`: the domain imports no framework and no outer layer; `api/` never imports `pharma_agent.domain`.
- Prefer established libraries over custom code. No feature flag or "fake mode" in production code; fakes live under `tests/`.
- One commit per task, conventional message, ending with the session attribution trailer of the executing session.
- Product copy is Vietnamese; the agent never adds medical disclaimers.
- Plan-specific values:
  - `Hit`, `Chunk`, `Citation` fields exactly as overview §3.4. `page_label` returns `""` when both pages are `None`.
  - `CorpusReader` lives in `pharma_agent.domain.retrieval.ports`; `ChunkRecord` in `pharma_agent.domain.retrieval.models`. A chunk key is the tuple `(release_id, chunk_version_id)`.
  - `CorpusReader.current_releases(collection_keys)` takes collection **keys** (settings hold keys) and returns `dict[collection_id, release_id]` for the collections that exist and have a current release.
  - Qdrant query filter (payload keys from overview §3.3): `Filter(should=[Filter(must=[collection_id == str(cid), release_ids == str(rid)]) for each scoped collection])`, set on both prefetches in `hybrid` mode and always as `query_filter`; `with_payload=["collection_id"]` (no text in Qdrant).
  - Retrieval modes, `RetrievalSettings.mode: Literal["hybrid", "dense", "bm25"] = "hybrid"`: `hybrid` fuses dense and BM25 prefetches with RRF; `dense` queries `dense_vector` only; `bm25` runs one sparse `query_points` with `models.Document(text=query.text, model=BM25_MODEL_NAME)` on `bm25_sparse_vector` and never calls the embedder (evaluation baseline).
  - `build_retrieval_service(settings, *, database: Database | None = None, embedder: Embedder | None = None)`: an injected embedder (`pharma_agent.domain.corpus.ports.Embedder`) replaces `OpenAiEmbedder` and no embedding client is created; P4 injects a cached query embedder and P7 reuses it.
  - One alias for reads and writes: Task 2 changes P2's `open_corpus_services(settings, *, embedder=None, alias: str | None = None)` so the alias passed to `QdrantVectorIndex(..., alias=...)` defaults to `settings.retrieval.qdrant_collection`; tests never assume `chunks_current` beyond that setting's default.
  - Qdrant collection metadata checked by `verify_collection`: `{"embedding_model": settings.retrieval.embedding.model, "dims": settings.retrieval.embedding.dimension}`.
  - Snippets: `make_snippet(chunk_text, 200)` for evidence event items and `Citation.snippet` (`SNIPPET_CHARS = 200` in `pharma_agent.domain.agent.citations`); `make_snippet(chunk_text, 300)` for `EvidenceSet.summary_view` and audit hit snippets.
  - `Hit` keeps `embedding_text` (loaded from `chunk_versions.embedding_text`) and the reranker keeps scoring it exactly as today; `llama_cpp_reranker.py` is not changed.
  - Health: check name `corpus` replaces the old `qdrant` check; `Container.health_reasons = {"corpus": "CORPUS_NOT_READY"}`; `HealthResponse` gains `reasons: dict[str, str]` listing the reason of each failed check that declares one.
  - Audit: `retrieval_runs.release_ids jsonb` maps `str(collection_id) -> str(release_id)` of the hits of that query; `retrieval_hits` gets `chunk_version_id uuid`, `section_key text`, `table_key text null` (replacing `chunk_id`, `section_id`, `table_id`); no FK to `corpus`.
  - Test ids: `tests/domain/factories.py::chunk_uuid(label)` and `revision_uuid(label)` build deterministic UUIDs so tests keep readable labels.
  - P1 names used (`2026-09-13-corpus-domain.md`): `pharma_agent.domain.shared.text.make_snippet`; `pharma_agent.domain.corpus.bundle` records `KnowledgeBundle`, `DocumentRecord`, `SectionRecord`, `BlockRecord`, `SourceInfo`, `DocumentKind`, `BlockKind`, `RetrievalMode`, `GlossaryEntry`, `ColloquialMappingRecord`; `identity.section_revision_id(section_key, blocks)`; `chunking.chunk_section(document, section, glossary, mappings)`, `CHUNKER_VERSION`, `ChunkDraft` (with `context_header` and `embedding_text`); `hydrate.hydrate_strategy_for(section)`, `section_char_count(section)`.
  - P2 names used (`2026-09-13-corpus-store-import.md`, overview §3.3): table classes in `pharma_agent.infrastructure.persistence.postgres.corpus_tables` with the spec §6.2 columns plus `chunk_versions.context_header text not null` (P2 stores `ChunkDraft.context_header`, so the reader reads it and never rebuilds it); port `pharma_agent.domain.corpus.ports.Embedder` (`model`, `dimension`, `embed`) and `OpenAiEmbedder.model`/`.dimension`; `pharma_agent.infrastructure.corpus_factory.open_corpus_services(settings, *, embedder: Embedder | None = None, alias: str = "chunks_current")` yielding `CorpusServices(importer, releases, index, embedder)` (Task 2 changes this to `alias: str | None = None`, where `None` means `settings.retrieval.qdrant_collection`); `ImportKnowledgeBundle.__call__(bundle, *, publish) -> ImportReport`; `ReleaseService.rollback(collection_key) -> Release`; `QdrantVectorIndex(client, *, model, dimension, alias="chunks_current")` with attributes `collection_name` and `alias`, whose `ensure_collection()` (run by every import) creates `chunks_<model_slug(model)>` with metadata `{"embedding_model", "dims"}`, the keyword payload indexes and the given `alias` when missing (never moved); test fixtures `migrated_dsn` (`tests/postgres.py`) reset with `TRUNCATE corpus.collections, corpus.embedding_cache CASCADE` as P2's tests do, `qdrant_url` (session) and `qdrant_client` (collections wiped) from `tests/qdrant.py`; `tests.corpus_fixtures.small_bundle`, `COLLECTION_KEY` (`"formulary"`), `DOSAGE_SECTION`; `tests.fakes.FakeEmbedder`, `fake_vector`, `FAKE_EMBEDDING_MODEL` (`"fake-embedding-4d"`), `FAKE_EMBEDDING_DIMENSION` (`4`).

---

## File Structure

```text
backend/
  src/pharma_agent/
    domain/retrieval/models.py          Hit, Chunk on corpus ids; page_label(); ChunkRecord (+ to_hit)      # Task 1, 2
    domain/retrieval/ports.py           CorpusReader port, ChunkKey                                       # Task 1, 2
    domain/retrieval/evidence.py        ordinal ordering, UUID keys, text_chunk_version_ids, make_snippet  # Task 2
    domain/retrieval/service.py         known_scores keyed by chunk_version_id                            # Task 2
    domain/retrieval/audit.py           hit records map new Hit fields (Task 2); release ids + UUID records (Task 3)
    domain/agent/citations.py           citations_from builds the new Citation; SNIPPET_CHARS              # Task 2
    domain/conversation/models.py       Citation on chunk version, release and strategy                   # Task 2
    application/chat/nodes.py           evidence items {index, source, title, section, pages, snippet}; JSON citations  # Task 2
    infrastructure/retrieval/postgres_corpus.py   PostgresCorpusReader (Task 1), section_chunks + PostgresHydrator (Task 2)
    infrastructure/retrieval/qdrant_adapter.py    release-filtered QdrantHybridRetriever, modes hybrid/dense/bm25; QdrantHydrator/hit_from_point removed (Task 2); verify_corpus (Task 4)
    infrastructure/observability/langfuse_retrieval.py   rerank output keyed by chunk_version_id           # Task 2
    infrastructure/settings.py          qdrant_collection, collections, mode "bm25"; collection_alias removed  # Task 2
    infrastructure/composition.py       RetrievalStack, build_retrieval_service(database, embedder), build_application reuse  # Task 2
    infrastructure/corpus_factory.py    (P2) import alias defaults to settings.retrieval.qdrant_collection  # Task 2
    infrastructure/container.py         passes Database to composition (Task 2); AuditContext (Task 3); corpus health check (Task 4)
    infrastructure/persistence/postgres/tables.py                     audit columns                       # Task 3
    infrastructure/persistence/postgres/conversation_repository.py    JSON citations (Task 2); release ids (Task 3)
    infrastructure/persistence/postgres/migrations/versions/0006_audit_release_ids.py                     # Task 3
    api/schemas.py, api/routers/health.py   HealthResponse.reasons                                        # Task 4
    cli.py                              probe hit, page labels, JSON citations, check via build_retrieval_service (Task 2); corpus check (Task 4)
  tests/
    corpus_rows.py                      seed a release straight into corpus tables for adapter tests      # Task 1
    infrastructure/test_postgres_corpus.py        reader (Task 1), section_chunks + hydrator (Task 2)
    domain/factories.py, fakes.py       UUID-based make_hit/make_item/make_chunk, fakes on new fields     # Task 2
    domain/test_evidence.py, test_citations.py, test_retrieval_service.py, test_audit.py, test_conversation.py   # Task 2 (test_audit again in Task 3)
    application/test_chat_graph.py, application/test_checkpoint.py, test_cli.py                         # Task 2 (test_cli again in Task 4)
    infrastructure/test_qdrant_adapter.py         reader-based retriever (Task 2), verify_corpus (Task 4)
    infrastructure/test_qdrant_integration.py     release filter in hybrid and bm25 modes on real Qdrant (P2 qdrant_client)  # Task 2
    infrastructure/test_llama_cpp_reranker.py, test_langfuse_tracing.py, test_composition.py, test_settings.py   # Task 2
    infrastructure/test_conversation_repository.py  (Task 2 citations; Task 3 audit), test_feedback_repository.py, api/test_e2e_postgres.py, infrastructure/test_migrations.py   # Task 3
    api/test_health_api.py, infrastructure/test_container.py                                             # Task 4
    infrastructure/test_corpus_retrieval_integration.py   import → search → hydrate → publish/rollback   # Task 5
  .env.example                                            retrieval settings                              # Task 2
  README.md                                               corpus import and health                        # Task 4
docker-compose.yml, .env.example (repo root)              corpus comments                                 # Task 4
```

Task order keeps the suite green: Task 1 adds the reader beside the old code; Task 2 is the atomic switch of `Hit`/`Chunk`/`Citation` with every consumer (audit rows keep their old columns through a field mapping); Task 3 moves audit to release ids; Task 4 adds the health check and docs; Task 5 proves the whole path on real Postgres and Qdrant.

### Task 1: `CorpusReader` port and `PostgresCorpusReader`

**Files:**
- Modify: `backend/src/pharma_agent/domain/retrieval/models.py` (imports at lines 1-3; append `ChunkRecord` after `RetrievedItem`, end of file)
- Modify: `backend/src/pharma_agent/domain/retrieval/ports.py` (imports at lines 1-5; append `ChunkKey` and `CorpusReader` at end of file)
- Create: `backend/src/pharma_agent/infrastructure/retrieval/postgres_corpus.py`
- Create: `backend/tests/corpus_rows.py`
- Test: `backend/tests/infrastructure/test_postgres_corpus.py`

**Interfaces:**
- Consumes (P1): `DocumentRecord`, `SectionRecord`, `BlockRecord`, `SourceInfo`, `DocumentKind`, `BlockKind`, `GlossaryEntry`, `ColloquialMappingRecord` from `pharma_agent.domain.corpus.bundle`; `section_revision_id(section_key, blocks)`; `chunk_section(document, section, glossary, mappings) -> list[ChunkDraft]`; `CHUNKER_VERSION`; `hydrate_strategy_for(section)`; `section_char_count(section)`.
- Consumes (P2): `CollectionTable`, `DocumentTable`, `SectionTable`, `SectionRevisionTable`, `ChunkVersionTable`, `ReleaseTable`, `ReleaseChunkTable` in `pharma_agent.infrastructure.persistence.postgres.corpus_tables`, columns as spec C §6.2 plus `chunk_versions.context_header` (JSONB `term_annotations` is a list of `TermAnnotation.model_dump(mode="json")`, `colloquial` is `ColloquialMapping.model_dump(mode="json")` or `NULL`, `hydrate_strategy` stores `HydrateStrategy.value`); fixture `migrated_dsn` from `tests/postgres.py`.
- Produces:
  - `pharma_agent.domain.retrieval.models.ChunkRecord` (frozen pydantic model): `chunk_version_id: UUID`, `release_id: UUID`, `collection_id: UUID`, `document_key: str`, `section_key: str`, `section_revision_id: UUID`, `ordinal: int`, `hydrate_strategy: HydrateStrategy`, `source: str`, `title: str`, `section: str`, `start_page: int | None`, `end_page: int | None`, `context_header: str`, `chunk_text: str`, `embedding_text: str`, `kind: str`, `table_key: str | None`, `colloquial_mapping: ColloquialMapping | None = None`, `term_annotations: list[TermAnnotation] = []`.
  - `pharma_agent.domain.retrieval.ports`: `type ChunkKey = tuple[UUID, UUID]` (release_id, chunk_version_id); `class CorpusReader(Protocol)` with `async def current_releases(self, collection_keys: Sequence[str]) -> dict[UUID, UUID]` and `async def load_chunks(self, keys: Sequence[ChunkKey]) -> list[ChunkRecord]` (Task 2 adds `section_chunks`).
  - `pharma_agent.infrastructure.retrieval.postgres_corpus.PostgresCorpusReader(sessions: async_sessionmaker[AsyncSession])`.
  - Test helpers: `tests.corpus_rows.reset_corpus(database: Database) -> None`; `tests.corpus_rows.seed_release(sessions, *, collection_key, document, sections, glossary=(), mappings=(), publish=True) -> SeededRelease` with `SeededRelease(collection_id, release_id, drafts: dict[str, list[ChunkDraft]], revisions: dict[str, UUID])`; builders `paracetamol_document()`, `dosage_section(blocks=3, *, key=DOSAGE_KEY, ordinal=1, markers=None)`, `table_section()`, `leaflet_document()`, `leaflet_section()`, `long_paragraph(marker)`, constants `DOSAGE_KEY`, `TABLE_SECTION_KEY`, `GLOSSARY`, `MAPPINGS`, `NOW`.

- [ ] **Step 1: Add the seeding helper**

Create `backend/tests/corpus_rows.py`:

```python
"""Insert a corpus release straight into the P2 tables, for adapter tests that must not
depend on the import pipeline. Chunks come from the real chunker so ids and headers match."""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import text, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pharma_agent.domain.corpus.bundle import (
    BlockKind,
    BlockRecord,
    ColloquialMappingRecord,
    DocumentKind,
    DocumentRecord,
    GlossaryEntry,
    SectionRecord,
    SourceInfo,
)
from pharma_agent.domain.corpus.chunking import CHUNKER_VERSION, ChunkDraft, chunk_section
from pharma_agent.domain.corpus.hydrate import hydrate_strategy_for, section_char_count
from pharma_agent.domain.corpus.identity import section_revision_id
from pharma_agent.infrastructure.persistence.postgres.corpus_tables import (
    ChunkVersionTable,
    CollectionTable,
    DocumentTable,
    ReleaseChunkTable,
    ReleaseTable,
    SectionRevisionTable,
    SectionTable,
)
from pharma_agent.infrastructure.persistence.postgres.database import Database

NOW = datetime(2026, 9, 13, 8, 0, tzinfo=UTC)
DOSAGE_KEY = "drug:paracetamol:lieu-luong-va-cach-dung"
TABLE_SECTION_KEY = "drug:paracetamol:bang-lieu-theo-can-nang"
LEAFLET_KEY = "leaflet:ankhang:thuoc-giam-dau:panadol:thong-tin-chi-tiet"
GLOSSARY = [GlossaryEntry(term="paracetamol", vietnamese_expansions=["acetaminophen"])]
MAPPINGS = [
    ColloquialMappingRecord(
        key="panadol",
        aliases=["thuốc hạ sốt"],
        product_names=["Panadol"],
        section_keys=[DOSAGE_KEY],
    )
]


@dataclass(frozen=True)
class SeededRelease:
    collection_id: uuid.UUID
    release_id: uuid.UUID
    drafts: dict[str, list[ChunkDraft]]
    revisions: dict[str, uuid.UUID]


async def reset_corpus(database: Database) -> None:
    """Empty schema `corpus` in the shared migrated database, as P2's tests do."""
    async with database.engine.begin() as connection:
        await connection.execute(
            text("TRUNCATE corpus.collections, corpus.embedding_cache CASCADE")
        )


def long_paragraph(marker: str) -> str:
    """About 2,500 characters: one chunk on its own, too long to merge with a neighbour."""
    sentence = "Liều dùng kéo dài cho người lớn theo đúng phác đồ điều trị. "
    return f"Mã đoạn {marker}. " + sentence * 40


def paracetamol_document() -> DocumentRecord:
    return DocumentRecord(
        key="drug:paracetamol",
        kind=DocumentKind.DRUG_MONOGRAPH,
        title="Paracetamol",
        source=SourceInfo(title="Dược thư Quốc gia Việt Nam"),
    )


def dosage_section(
    blocks: int = 3,
    *,
    key: str = DOSAGE_KEY,
    ordinal: int = 1,
    markers: Sequence[str] | None = None,
) -> SectionRecord:
    labels = list(markers) if markers is not None else [f"kxdoan{i}" for i in range(1, blocks + 1)]
    return SectionRecord(
        key=key,
        document_key="drug:paracetamol",
        heading="Liều lượng và cách dùng",
        context_path=["Liều lượng và cách dùng"],
        ordinal=ordinal,
        start_page=812,
        end_page=813,
        blocks=[
            BlockRecord(kind=BlockKind.PROSE, markdown=long_paragraph(label), start_page=812, end_page=813)
            for label in labels
        ],
    )


def table_section() -> SectionRecord:
    return SectionRecord(
        key=TABLE_SECTION_KEY,
        document_key="drug:paracetamol",
        heading="Bảng liều theo cân nặng",
        context_path=["Liều lượng và cách dùng", "Bảng liều theo cân nặng"],
        ordinal=2,
        blocks=[
            BlockRecord(
                kind=BlockKind.TABLE,
                markdown="| Cân nặng | Liều |\n| --- | --- |\n| 10 kg | 150 mg |",
                table_key="bang-1",
            )
        ],
    )


def leaflet_document() -> DocumentRecord:
    return DocumentRecord(
        key="leaflet:ankhang:thuoc-giam-dau:panadol",
        kind=DocumentKind.LEAFLET,
        title="Panadol",
        source=SourceInfo(
            title="Nhà thuốc An Khang",
            url="https://www.nhathuocankhang.com/thuoc-giam-dau/panadol",
        ),
    )


def leaflet_section() -> SectionRecord:
    return SectionRecord(
        key=LEAFLET_KEY,
        document_key="leaflet:ankhang:thuoc-giam-dau:panadol",
        heading="Thông tin chi tiết",
        context_path=["Thông tin chi tiết"],
        ordinal=1,
        blocks=[BlockRecord(kind=BlockKind.PROSE, markdown="Panadol chứa paracetamol 500 mg.")],
    )


async def seed_release(
    sessions: async_sessionmaker[AsyncSession],
    *,
    collection_key: str,
    document: DocumentRecord,
    sections: Sequence[SectionRecord],
    glossary: Sequence[GlossaryEntry] = (),
    mappings: Sequence[ColloquialMappingRecord] = (),
    publish: bool = True,
) -> SeededRelease:
    collection_id, document_id, release_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    drafts: dict[str, list[ChunkDraft]] = {}
    revisions: dict[str, uuid.UUID] = {}
    async with sessions.begin() as session:
        session.add(
            CollectionTable(
                id=collection_id,
                key=collection_key,
                title=collection_key,
                owner_user_id=None,
                visibility="private",
                current_release_id=None,
                created_at=NOW,
                updated_at=NOW,
            )
        )
        await session.flush()
        session.add(
            DocumentTable(
                id=document_id,
                collection_id=collection_id,
                key=document.key,
                kind=document.kind.value,
                title=document.title,
                source_title=document.source.title,
                source_url=document.source.url,
                attributes=dict(document.attributes),
            )
        )
        session.add(
            ReleaseTable(
                id=release_id,
                collection_id=collection_id,
                number=1,
                status="ready",
                bundle_digest="seed",
                chunker_version=CHUNKER_VERSION,
                embedding_model="fake-embedding-4d",
                stats={},
                created_at=NOW,
                ready_at=NOW,
                published_at=NOW if publish else None,
            )
        )
        await session.flush()
        for section in sections:
            section_id = uuid.uuid4()
            revision_id = section_revision_id(section.key, section.blocks)
            chunks = chunk_section(document, section, glossary, mappings)
            drafts[section.key] = chunks
            revisions[section.key] = revision_id
            session.add(
                SectionTable(
                    id=section_id,
                    document_id=document_id,
                    key=section.key,
                    heading=section.heading,
                    context_path=list(section.context_path),
                    ordinal=section.ordinal,
                    retrieval_mode=section.retrieval.value,
                )
            )
            await session.flush()
            session.add(
                SectionRevisionTable(
                    id=revision_id,
                    section_id=section_id,
                    blocks=[block.model_dump(mode="json") for block in section.blocks],
                    start_page=section.start_page,
                    end_page=section.end_page,
                    char_count=section_char_count(section),
                )
            )
            await session.flush()
            session.add_all(
                ChunkVersionTable(
                    id=draft.chunk_version_id,
                    section_revision_id=revision_id,
                    ordinal=draft.ordinal,
                    kind=draft.kind.value,
                    chunk_text=draft.chunk_text,
                    context_header=draft.context_header,
                    embedding_text=draft.embedding_text,
                    embedding_text_sha256=draft.embedding_text_sha256,
                    start_page=draft.start_page,
                    end_page=draft.end_page,
                    table_key=draft.table_key,
                    term_annotations=[a.model_dump(mode="json") for a in draft.term_annotations],
                    colloquial=draft.colloquial.model_dump(mode="json") if draft.colloquial is not None else None,
                    chunker_version=CHUNKER_VERSION,
                    created_at=NOW,
                )
                for draft in chunks
            )
            await session.flush()
            session.add_all(
                ReleaseChunkTable(
                    release_id=release_id,
                    chunk_version_id=draft.chunk_version_id,
                    section_id=section_id,
                    section_revision_id=revision_id,
                    ordinal=draft.ordinal,
                    hydrate_strategy=hydrate_strategy_for(section).value,
                )
                for draft in chunks
            )
            await session.flush()
        if publish:
            await session.execute(
                update(CollectionTable)
                .where(CollectionTable.id == collection_id)
                .values(current_release_id=release_id)
            )
    return SeededRelease(collection_id=collection_id, release_id=release_id, drafts=drafts, revisions=revisions)
```

- [ ] **Step 2: Write the failing reader tests**

Create `backend/tests/infrastructure/test_postgres_corpus.py`:

```python
import uuid
from collections.abc import AsyncIterator

import pytest

from pharma_agent.domain.corpus.hydrate import hydrate_strategy_for
from pharma_agent.infrastructure.persistence.postgres.database import Database
from pharma_agent.infrastructure.retrieval.postgres_corpus import PostgresCorpusReader
from tests.corpus_rows import (
    DOSAGE_KEY,
    GLOSSARY,
    MAPPINGS,
    dosage_section,
    leaflet_document,
    leaflet_section,
    paracetamol_document,
    reset_corpus,
    seed_release,
    table_section,
)

pytestmark = pytest.mark.integration


@pytest.fixture
async def database(migrated_dsn: str) -> AsyncIterator[Database]:
    db = Database(migrated_dsn, pool_size=2)
    await reset_corpus(db)
    yield db
    await db.dispose()


async def test_current_releases_maps_only_published_collections(database: Database) -> None:
    published = await seed_release(
        database.sessions,
        collection_key="formulary",
        document=paracetamol_document(),
        sections=[dosage_section()],
    )
    await seed_release(
        database.sessions,
        collection_key="drafts",
        document=leaflet_document(),
        sections=[leaflet_section()],
        publish=False,
    )
    reader = PostgresCorpusReader(database.sessions)

    assert await reader.current_releases(["formulary", "drafts", "missing"]) == {
        published.collection_id: published.release_id
    }
    assert await reader.current_releases([]) == {}


async def test_load_chunks_returns_display_fields_for_release_chunk_pairs(
    database: Database,
) -> None:
    section = dosage_section()
    seeded = await seed_release(
        database.sessions,
        collection_key="formulary",
        document=paracetamol_document(),
        sections=[section, table_section()],
        glossary=GLOSSARY,
        mappings=MAPPINGS,
    )
    first, second = seeded.drafts[DOSAGE_KEY][0], seeded.drafts[DOSAGE_KEY][1]
    reader = PostgresCorpusReader(database.sessions)

    records = await reader.load_chunks(
        [
            (seeded.release_id, first.chunk_version_id),
            (seeded.release_id, second.chunk_version_id),
            (uuid.uuid4(), first.chunk_version_id),
        ]
    )

    by_id = {record.chunk_version_id: record for record in records}
    assert set(by_id) == {first.chunk_version_id, second.chunk_version_id}
    record = by_id[first.chunk_version_id]
    assert (record.release_id, record.collection_id) == (seeded.release_id, seeded.collection_id)
    assert (record.document_key, record.section_key, record.ordinal) == (
        "drug:paracetamol",
        DOSAGE_KEY,
        first.ordinal,
    )
    assert record.section_revision_id == seeded.revisions[DOSAGE_KEY]
    assert record.hydrate_strategy is hydrate_strategy_for(section)
    assert (record.source, record.title, record.section) == (
        "Dược thư Quốc gia Việt Nam",
        "Paracetamol",
        "Liều lượng và cách dùng",
    )
    assert record.context_header == first.context_header
    assert record.chunk_text == first.chunk_text
    assert record.embedding_text == first.embedding_text
    assert (record.kind, record.table_key, record.start_page, record.end_page) == (
        first.kind.value,
        first.table_key,
        first.start_page,
        first.end_page,
    )
    assert record.term_annotations == first.term_annotations
    assert record.colloquial_mapping == first.colloquial
    assert await reader.load_chunks([]) == []
```

- [ ] **Step 3: Run the tests and confirm they fail**

Run: `cd backend && uv run pytest -q -m integration tests/infrastructure/test_postgres_corpus.py`
Expected: collection error `ModuleNotFoundError: No module named 'pharma_agent.infrastructure.retrieval.postgres_corpus'`.

- [ ] **Step 4: Add `ChunkRecord` and the `CorpusReader` port**

In `backend/src/pharma_agent/domain/retrieval/models.py`, add `from uuid import UUID` to the imports and append:

```python
class ChunkRecord(BaseModel):
    """One chunk version as published in one release, with the fields a search hit shows."""

    model_config = ConfigDict(frozen=True)

    chunk_version_id: UUID
    release_id: UUID
    collection_id: UUID
    document_key: str
    section_key: str
    section_revision_id: UUID
    ordinal: int
    hydrate_strategy: HydrateStrategy
    source: str
    title: str
    section: str
    start_page: int | None
    end_page: int | None
    context_header: str
    chunk_text: str
    embedding_text: str
    kind: str
    table_key: str | None
    colloquial_mapping: ColloquialMapping | None = None
    term_annotations: list[TermAnnotation] = Field(default_factory=list)
```

In `backend/src/pharma_agent/domain/retrieval/ports.py`, replace the imports with:

```python
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
```

and append:

```python
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
```

- [ ] **Step 5: Implement `PostgresCorpusReader`**

Create `backend/src/pharma_agent/infrastructure/retrieval/postgres_corpus.py`:

```python
"""Read side of schema `corpus` for retrieval (spec C §9). One indexed query per call and no
cache, so a publish or rollback applies to the very next search."""

from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import select, tuple_
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pharma_agent.domain.retrieval.models import (
    ChunkRecord,
    ColloquialMapping,
    HydrateStrategy,
    TermAnnotation,
)
from pharma_agent.domain.retrieval.ports import ChunkKey, RetrievalError
from pharma_agent.infrastructure.persistence.postgres.corpus_tables import (
    ChunkVersionTable,
    CollectionTable,
    DocumentTable,
    ReleaseChunkTable,
    SectionTable,
)


def _record(
    link: ReleaseChunkTable,
    chunk: ChunkVersionTable,
    section: SectionTable,
    document: DocumentTable,
) -> ChunkRecord:
    return ChunkRecord(
        chunk_version_id=link.chunk_version_id,
        release_id=link.release_id,
        collection_id=document.collection_id,
        document_key=document.key,
        section_key=section.key,
        section_revision_id=link.section_revision_id,
        ordinal=link.ordinal,
        hydrate_strategy=HydrateStrategy(link.hydrate_strategy),
        source=document.source_title,
        title=document.title,
        section=section.heading,
        start_page=chunk.start_page,
        end_page=chunk.end_page,
        context_header=chunk.context_header,
        chunk_text=chunk.chunk_text,
        embedding_text=chunk.embedding_text,
        kind=chunk.kind,
        table_key=chunk.table_key,
        colloquial_mapping=ColloquialMapping.model_validate(chunk.colloquial)
        if chunk.colloquial is not None
        else None,
        term_annotations=[
            TermAnnotation.model_validate(item) for item in chunk.term_annotations
        ],
    )


class PostgresCorpusReader:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def current_releases(
        self, collection_keys: Sequence[str]
    ) -> dict[UUID, UUID]:
        if not collection_keys:
            return {}
        query = select(CollectionTable.id, CollectionTable.current_release_id).where(
            CollectionTable.key.in_(list(collection_keys)),
            CollectionTable.current_release_id.is_not(None),
        )
        try:
            async with self._sessions() as session:
                rows = (await session.execute(query)).tuples().all()
        except SQLAlchemyError as exc:
            raise RetrievalError(f"cannot read current releases: {exc}") from exc
        return {
            collection_id: release_id
            for collection_id, release_id in rows
            if release_id is not None
        }

    async def load_chunks(self, keys: Sequence[ChunkKey]) -> list[ChunkRecord]:
        if not keys:
            return []
        query = (
            select(ReleaseChunkTable, ChunkVersionTable, SectionTable, DocumentTable)
            .join(
                ChunkVersionTable,
                ChunkVersionTable.id == ReleaseChunkTable.chunk_version_id,
            )
            .join(SectionTable, SectionTable.id == ReleaseChunkTable.section_id)
            .join(DocumentTable, DocumentTable.id == SectionTable.document_id)
            .where(
                tuple_(
                    ReleaseChunkTable.release_id, ReleaseChunkTable.chunk_version_id
                ).in_(list(keys))
            )
        )
        try:
            async with self._sessions() as session:
                rows = (await session.execute(query)).tuples().all()
        except SQLAlchemyError as exc:
            raise RetrievalError(f"cannot load {len(keys)} chunks: {exc}") from exc
        return [_record(link, chunk, section, document) for link, chunk, section, document in rows]
```

- [ ] **Step 6: Run the tests and confirm they pass**

Run: `cd backend && uv run pytest -q -m integration tests/infrastructure/test_postgres_corpus.py`
Expected: `2 passed`.

- [ ] **Step 7: Run the full check**

Run: `cd backend && uv run ruff check src tests && uv run ruff format --check src tests && uv run pyrefly check --min-severity warn && uv run pytest -q && uv run pytest -q -m integration`
Expected: ruff and pyrefly report no findings (run `uv run ruff format src tests` first if the format check lists the new files); both pytest runs pass.

- [ ] **Step 8: Commit**

```bash
git add backend/src/pharma_agent/domain/retrieval/models.py backend/src/pharma_agent/domain/retrieval/ports.py backend/src/pharma_agent/infrastructure/retrieval/postgres_corpus.py backend/tests/corpus_rows.py backend/tests/infrastructure/test_postgres_corpus.py
git commit -m "feat(retrieval): read current releases and chunk records from the corpus schema"
```

The commit message ends with the session attribution trailer.

### Task 2: Switch retrieval to chunk versions and releases

This is the one change that cannot be split: every consumer of `Hit`, `Chunk` and `Citation` moves together, the Qdrant adapter stops reading payload text, and hydration moves to Postgres. Audit rows keep their current columns in this task (a field mapping in `audit.py`); Task 3 replaces them.

**Files:**
- Modify: `backend/src/pharma_agent/domain/retrieval/models.py` (`Hit` lines 42-88, `Chunk` lines 91-101; add `page_label`, `ChunkRecord.to_hit`)
- Modify: `backend/src/pharma_agent/domain/retrieval/ports.py` (add `CorpusReader.section_chunks`)
- Modify: `backend/src/pharma_agent/domain/retrieval/evidence.py` (whole file)
- Modify: `backend/src/pharma_agent/domain/retrieval/service.py` (imports lines 1-17, `search` lines 53-104, helpers lines 107-145)
- Modify: `backend/src/pharma_agent/domain/retrieval/audit.py` (hit mapping lines 39-62)
- Modify: `backend/src/pharma_agent/domain/conversation/models.py` (`Citation` lines 16-26)
- Modify: `backend/src/pharma_agent/domain/agent/citations.py` (imports lines 1-5, `citations_from` lines 52-74)
- Modify: `backend/src/pharma_agent/application/chat/nodes.py` (imports lines 15-16 and 35; `answer_node` lines 233-279)
- Modify: `backend/src/pharma_agent/cli.py` (imports lines 3-14, `_run_turn` lines 50-74, `_check` lines 108-162, `_PROBE_HIT` lines 165-178)
- Modify: `backend/src/pharma_agent/infrastructure/observability/langfuse_retrieval.py` (output lines 35-40)
- Modify: `backend/src/pharma_agent/infrastructure/retrieval/qdrant_adapter.py` (whole file)
- Modify: `backend/src/pharma_agent/infrastructure/retrieval/postgres_corpus.py` (add `section_chunks`, `PostgresHydrator`)
- Modify: `backend/src/pharma_agent/infrastructure/settings.py` (`RetrievalSettings` lines 108-117)
- Modify: `backend/src/pharma_agent/infrastructure/composition.py` (whole file)
- Modify: `backend/src/pharma_agent/infrastructure/container.py` (`_qdrant_check` lines 64-72, `AuditContext` line 103, `build_application` call lines 143-148, check registration lines 166-168)
- Modify: `backend/src/pharma_agent/infrastructure/persistence/postgres/conversation_repository.py` (`_message_row` line 80)
- Modify: `backend/src/pharma_agent/infrastructure/corpus_factory.py` (P2: `open_corpus_services` alias default)
- Modify: `backend/.env.example` (retrieval block lines 12-14)
- Modify tests: `backend/tests/domain/factories.py` (whole file), `backend/tests/fakes.py` (imports, `FakeReranker`, `FakeHydrator`; P2's `FakeEmbedder` block stays), `backend/tests/domain/test_evidence.py` (whole file), `backend/tests/domain/test_citations.py` (lines 1-4, 36-49), `backend/tests/domain/test_retrieval_service.py` (lines 1-86 and assertions), `backend/tests/domain/test_audit.py` (lines 1-5, 29-53), `backend/tests/domain/test_conversation.py` (lines 13, 64-72), `backend/tests/application/test_checkpoint.py` (end of second test), `backend/tests/application/test_chat_graph.py` (lines 26, 100-103, 148), `backend/tests/infrastructure/test_llama_cpp_reranker.py` (factories import and the three `chunk_id` assertions; the prompt assertion stays), `backend/tests/infrastructure/test_langfuse_tracing.py` (lines 30, 151-153), `backend/tests/infrastructure/test_qdrant_adapter.py` (whole file), `backend/tests/infrastructure/test_qdrant_integration.py` (whole file), `backend/tests/infrastructure/test_postgres_corpus.py` (append), `backend/tests/infrastructure/test_composition.py` (lines 30-31, 43, 74), `backend/tests/infrastructure/test_settings.py` (lines 10-12), `backend/tests/infrastructure/test_conversation_repository.py` (lines 26, 66-74, 166)

**Interfaces:**
- Consumes: Task 1 `ChunkRecord`, `ChunkKey`, `CorpusReader`, `PostgresCorpusReader`; P1 `make_snippet(text, max_chars)`; P2 `pharma_agent.domain.corpus.ports.Embedder`, `OpenAiEmbedder.model`/`.dimension`, `tests.fakes.FakeEmbedder`, `fake_vector`, fixture `qdrant_client` (`tests/qdrant.py`).
- Produces:
  - `page_label(start_page: int | None, end_page: int | None) -> str` in `pharma_agent.domain.retrieval.models`; `Hit.page_label` delegates to it.
  - `Hit` (keeping `embedding_text`), `Chunk` fields per overview §3.4; `Chunk.is_table` is `kind == "table"`; `ChunkRecord.to_hit(*, fusion_score: float, query_text: str) -> Hit`.
  - `CorpusReader.section_chunks(self, release_id: UUID, section_revision_id: UUID, *, around: int | None, radius: int) -> list[Chunk]` (ordered by ordinal; all chunks when `around is None`).
  - `Evidence.text_chunk_version_ids() -> list[UUID]` (ids of the chunks `text()` joins, in the same order); `EvidenceSet.rerank_scores() -> dict[UUID, float]`; `EvidenceSet.summary_view(snippet_chars: int = 300)`.
  - `RetrievalService.search(queries, rerank_query, known_scores: Mapping[UUID, float] | None = None)`.
  - `Citation(index, chunk_version_id, release_id, strategy, block_chunk_version_ids, source, title, section, start_page, end_page, snippet)`; `SNIPPET_CHARS = 200` and `citations_from(numbered, used) -> list[Citation]` in `pharma_agent.domain.agent.citations`.
  - Evidence event items `{"index", "source", "title", "section", "start_page", "end_page", "snippet"}`; citations event items are `Citation.model_dump(mode="json")`.
  - `QdrantHybridRetriever(client: Any, embedder: Embedder, reader: CorpusReader, *, collection: str, scope: Sequence[str], mode: Literal["hybrid", "dense", "bm25"] = "hybrid", prefetch_k: int = 50, rrf_k: int = 2, max_concurrent: int = 3)` with `Embedder` from `pharma_agent.domain.corpus.ports` (the adapter's local protocol is removed); `bm25` sends one sparse query and never calls the embedder; `release_scope_filter(releases: Mapping[UUID, UUID]) -> models.Filter`; `QdrantHybridRetriever.verify_collection(*, embedding_model: str, dimension: int) -> None`; constants `COLLECTION_ID_KEY = "collection_id"`, `RELEASE_IDS_KEY = "release_ids"`. `QdrantHydrator` and `hit_from_point` are deleted.
  - `PostgresHydrator(reader: CorpusReader, *, window: int = 1)`.
  - `RetrievalSettings.qdrant_collection: str = "chunks_current"`, `RetrievalSettings.collections: list[str] = ["formulary"]`, `RetrievalSettings.mode: Literal["hybrid", "dense", "bm25"] = "hybrid"`.
  - `open_corpus_services(settings, *, embedder: Embedder | None = None, alias: str | None = None)`: `None` means `settings.retrieval.qdrant_collection`, so imports write the alias retrieval reads; an explicit alias (the E2E server's `e2e_chunks_current`) still wins.
  - `RetrievalStack` dataclass (`settings: RetrievalSettings`, `service: RetrievalService`, `retriever: QdrantHybridRetriever`, `embedder: Embedder`, `reranker: Reranker`, `reader: CorpusReader`, `qdrant: AsyncQdrantClient`, `embed_client: AsyncOpenAI | None`, `database: Database`, `owns_database: bool`, `async def aclose(self) -> None`); `build_retrieval_service(settings: Settings, *, database: Database | None = None, embedder: Embedder | None = None) -> RetrievalStack` (an injected embedder means no embedding client is built); `Application(settings, deps, runner, retrieval: RetrievalStack)`; `build_application(settings, *, checkpointer=None, skills=None, tracer=None, database: Database | None = None) -> Application`.
  - Test helpers in `tests/domain/factories.py`: `TEST_NAMESPACE`, `RELEASE_ID`, `COLLECTION_ID`, `SECTION_KEY`, `chunk_uuid(label)`, `revision_uuid(section_key)`, `make_hit(label, *, section_key, ordinal, strategy, fusion, rerank, text, title, section, kind, table_key, start_page, end_page)`, `make_chunk(label, *, ordinal, text, section_key, kind, table_key)`, `chunk_of(hit)`, `make_item(label, *, rerank, text)`, `make_citation(label, *, index)`.

- [ ] **Step 1: Rewrite the test factories and fakes**

Replace `backend/tests/domain/factories.py` with:

```python
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
```

In `backend/tests/fakes.py` (as extended by P2 with `import hashlib`, `FAKE_EMBEDDING_*`, `fake_vector` and `FakeEmbedder`, which all stay), add `import uuid` directly after `import hashlib`, add `from tests.domain.factories import chunk_of` as the last import line, and replace `FakeReranker` and `FakeHydrator` with:

```python
class FakeReranker:
    def __init__(self) -> None:
        self.received: list[list[uuid.UUID]] = []

    async def rerank(self, query: str, hits: Sequence[Hit], top_n: int) -> list[Hit]:
        self.received.append([h.chunk_version_id for h in hits])
        ordered = sorted(hits, key=lambda h: h.fusion_score, reverse=True)[:top_n]
        return [
            h.model_copy(update={"rerank_score": round(1.0 - i * 0.1, 2)})
            for i, h in enumerate(ordered)
        ]


class FakeHydrator:
    async def hydrate(self, hit: Hit, strategy: HydrateStrategy) -> list[Chunk]:
        if strategy is HydrateStrategy.SEARCH_ONLY:
            return []
        return [chunk_of(hit)]
```

(P2 also appends `FakeEmbedder` to `tests/fakes.py`; keep it.)

- [ ] **Step 2: Rewrite the domain tests**

Replace `backend/tests/domain/test_evidence.py` with:

```python
from pharma_agent.domain.retrieval.evidence import EvidenceSet
from pharma_agent.domain.retrieval.models import (
    ColloquialMapping,
    HydrateStrategy,
    Query,
    QueryOrigin,
    RetrievedItem,
    TermAnnotation,
    page_label,
)
from pharma_agent.domain.shared.text import make_snippet
from tests.domain.factories import chunk_uuid, make_chunk, make_hit


def test_query_normalizes_whitespace_and_case() -> None:
    assert (
        Query(text="  Liều   PARACETAMOL ", origin=QueryOrigin.INITIAL).normalized
        == "liều paracetamol"
    )


def test_page_label_handles_missing_pages() -> None:
    assert page_label(10, 11) == "trang 10-11"
    assert page_label(10, 10) == "trang 10"
    assert page_label(None, 12) == "trang 12"
    assert page_label(None, None) == ""
    assert make_hit("c1", start_page=None, end_page=None).page_label == ""


def test_term_hints_collect_aliases_products_and_annotations() -> None:
    hit = make_hit("c1").model_copy(
        update={
            "colloquial_mapping": ColloquialMapping(
                key="paracetamol",
                aliases=["thuốc hạ sốt"],
                product_names=["Panadol", "Efferalgan"],
            ),
            "term_annotations": [
                TermAnnotation(term="APAP", vi=["acetaminophen"], en=["acetaminophen"])
            ],
        }
    )
    assert hit.term_hints() == [
        "thuốc hạ sốt",
        "Panadol",
        "Efferalgan",
        "APAP",
        "acetaminophen",
    ]


def test_merge_assigns_stable_refs_and_keeps_best_score() -> None:
    evidence = EvidenceSet()
    first = evidence.merge(
        [
            RetrievedItem(hit=make_hit("c1", rerank=0.2)),
            RetrievedItem(hit=make_hit("c2", rerank=0.9)),
        ]
    )
    assert [e.ref for e in first] == ["E2", "E1"]  # sorted by score desc, refs stable
    evidence.supersede_all()
    again = evidence.merge(
        [
            RetrievedItem(
                hit=make_hit("c1", rerank=0.95).model_copy(
                    update={"matched_queries": ["q2"]}
                )
            )
        ]
    )
    assert [e.ref for e in again] == ["E1"]
    e1 = next(e for e in evidence.items if e.ref == "E1")
    assert e1.hit.rerank_score == 0.95
    assert e1.hit.matched_queries == ["q1", "q2"]
    assert e1.superseded is False
    assert next(e for e in evidence.items if e.ref == "E2").superseded is True
    assert [e.ref for e in evidence.active()] == ["E1"]


def test_pack_downgrades_strategy_instead_of_truncating() -> None:
    long_chunks = [make_chunk(f"c{i}", ordinal=i, text="x" * 100) for i in range(5)]
    hit = make_hit(
        "c2",
        ordinal=2,
        strategy=HydrateStrategy.FULL_SECTION,
        rerank=0.9,
        text="y" * 50,
    )
    evidence = EvidenceSet()
    evidence.merge([RetrievedItem(hit=hit, chunks=long_chunks)])

    full = evidence.pack(max_chars=1000)
    assert full[0].applied_strategy is HydrateStrategy.FULL_SECTION
    assert len(full[0].text()) > 400

    window = evidence.pack(max_chars=350)
    assert window[0].applied_strategy is HydrateStrategy.CHUNK_WINDOW
    assert [c.ordinal for c in window[0].chunks] == [1, 2, 3]

    search_only = evidence.pack(max_chars=80)
    assert search_only[0].applied_strategy is HydrateStrategy.SEARCH_ONLY
    assert search_only[0].text() == "y" * 50
    assert search_only[0].text_chunk_version_ids() == [chunk_uuid("c2")]

    assert evidence.pack(max_chars=10) == []


def test_context_view_numbers_sources_and_puts_tables_first() -> None:
    table = make_chunk(
        "t1", ordinal=3, text="| liều | mg |", kind="table", table_key="tbl-1"
    )
    body = make_chunk("c1", ordinal=1, text="Người lớn 500 mg.")
    hit = make_hit(
        "c1", ordinal=1, strategy=HydrateStrategy.FULL_SECTION, rerank=0.8
    )
    evidence = EvidenceSet()
    evidence.merge([RetrievedItem(hit=hit, chunks=[body, table])])
    packed = evidence.pack(max_chars=1000)
    text, numbered = evidence.context_view(packed)
    assert numbered[0][0] == 1 and numbered[0][1].ref == "E1"
    assert text.startswith("[1] Paracetamol > Liều dùng (trang 10-11)")
    assert text.index("| liều | mg |") < text.index("Người lớn 500 mg.")
    assert numbered[0][1].text_chunk_version_ids() == [
        chunk_uuid("t1"),
        chunk_uuid("c1"),
    ]


def test_context_view_omits_page_label_without_pages() -> None:
    evidence = EvidenceSet()
    evidence.merge(
        [RetrievedItem(hit=make_hit("c1", rerank=0.8, start_page=None, end_page=None))]
    )
    text, _ = evidence.context_view(evidence.pack(max_chars=1000))
    assert text.startswith("[1] Paracetamol > Liều dùng\n")


def test_summary_view_lists_refs_snippets_and_hints() -> None:
    text = "Người lớn uống 500 mg mỗi 4 đến 6 giờ, tối đa 4 g mỗi ngày. " * 10
    hit = make_hit("c1", rerank=0.7, text=text).model_copy(
        update={
            "colloquial_mapping": ColloquialMapping(
                key="paracetamol", product_names=["Panadol"]
            )
        }
    )
    evidence = EvidenceSet()
    evidence.merge([RetrievedItem(hit=hit)])
    view = evidence.summary_view()
    assert view.startswith("E1 | Paracetamol > Liều dùng | trang 10-11 | ")
    assert make_snippet(text, 300) in view
    assert "gợi ý thuật ngữ: Panadol" in view
    assert make_snippet(text, 40) in evidence.summary_view(snippet_chars=40)


def test_rerank_scores_include_superseded_evidence() -> None:
    evidence = EvidenceSet()
    evidence.merge(
        [
            RetrievedItem(hit=make_hit("c1", rerank=0.8)),
            RetrievedItem(hit=make_hit("c2")),
        ]
    )
    evidence.supersede_all()
    assert evidence.rerank_scores() == {chunk_uuid("c1"): 0.8}
```

In `backend/tests/domain/test_citations.py`, replace the imports (lines 1-4) with:

```python
from pharma_agent.domain.agent.citations import (
    SNIPPET_CHARS,
    CitationSanitizer,
    citations_from,
)
from pharma_agent.domain.retrieval.evidence import EvidenceSet
from pharma_agent.domain.retrieval.models import HydrateStrategy, RetrievedItem
from pharma_agent.domain.shared.text import make_snippet
from tests.domain.factories import RELEASE_ID, SOURCE, chunk_uuid, make_hit, make_item
```

and replace `test_citations_from_numbered_evidence` (lines 36-49) with:

```python
def test_citations_from_numbered_evidence() -> None:
    evidence = EvidenceSet()
    evidence.merge(
        [
            make_item("c1", rerank=0.9),
            RetrievedItem(
                hit=make_hit(
                    "c2", rerank=0.8, table_key="t1", start_page=None, end_page=None
                )
            ),
        ]
    )
    packed = evidence.pack(10_000)
    _, numbered = evidence.context_view(packed)
    citations = citations_from(numbered, used=[2, 1])
    assert [c.index for c in citations] == [2, 1]
    second, first = citations
    assert (
        second.chunk_version_id,
        second.strategy,
        second.block_chunk_version_ids,
        second.start_page,
        second.end_page,
    ) == (chunk_uuid("c2"), HydrateStrategy.SEARCH_ONLY, [chunk_uuid("c2")], None, None)
    assert (first.chunk_version_id, first.release_id, first.strategy) == (
        chunk_uuid("c1"),
        RELEASE_ID,
        HydrateStrategy.CHUNK_WINDOW,
    )
    assert first.block_chunk_version_ids == [chunk_uuid("c1")]
    assert (first.source, first.title, first.section, first.start_page) == (
        SOURCE,
        "Paracetamol",
        "Liều dùng",
        10,
    )
    assert first.snippet == make_snippet("paracetamol 500 mg", SNIPPET_CHARS)
```

In `backend/tests/domain/test_retrieval_service.py`, replace lines 1-86 (imports, `hit`, the fakes and `queries`) with:

```python
import uuid
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
from tests.domain.factories import chunk_of, chunk_uuid, make_hit


def hit(
    label: str,
    fusion: float,
    strategy: HydrateStrategy = HydrateStrategy.CHUNK_WINDOW,
) -> Hit:
    return make_hit(label, fusion=fusion, strategy=strategy, text=f"text {label}")


def ids(*labels: str) -> list[uuid.UUID]:
    return [chunk_uuid(label) for label in labels]


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
        return [chunk_of(hit)]


def queries(*texts: str) -> list[Query]:
    return [Query(text=t, origin=QueryOrigin.INITIAL) for t in texts]
```

and change these assertions in the same file:

```python
# test_search_dedupes_across_queries_then_reranks_and_hydrates
    assert sorted(h.chunk_version_id for h in reranker.received) == sorted(
        ids("a", "b", "c")
    )
    b = next(h for h in reranker.received if h.chunk_version_id == chunk_uuid("b"))
# test_rerank_failure_keeps_fusion_order
    assert [i.hit.chunk_version_id for i in result.items] == ids("a", "b")
# test_candidate_pool_interleaves_queries_by_rank_and_caps_new_chunks
    assert [h.chunk_version_id for h in reranker.received] == ids("a1", "b1", "a2", "a3")
# test_known_scores_are_reused_instead_of_rescored
    result = await service.search(
        queries("q1"), rerank_query="q1", known_scores={chunk_uuid("b"): 0.75}
    )

    assert [h.chunk_version_id for h in reranker.received] == ids("a", "c")
    assert [(i.hit.chunk_version_id, i.hit.rerank_score) for i in result.items] == [
        (chunk_uuid("c"), 1.0),
        (chunk_uuid("b"), 0.75),
        (chunk_uuid("a"), 0.5),
    ]
# test_reranker_is_not_called_when_every_candidate_is_known
    result = await service.search(
        queries("q1"),
        rerank_query="q1",
        known_scores={chunk_uuid("a"): 0.2, chunk_uuid("b"): 0.6},
    )

    assert reranker.calls == 0 and result.rerank_failed is False
    assert [i.hit.chunk_version_id for i in result.items] == ids("b", "a")
```

In `backend/tests/domain/test_audit.py`, replace line 1 and line 5 with:

```python
from pharma_agent.domain.retrieval.audit import audit_from_run
from pharma_agent.domain.retrieval.models import Query, QueryOrigin, RetrievedItem
from pharma_agent.domain.retrieval.service import SearchResult
from tests.domain.factories import NOW, chunk_uuid, make_citation, make_hit, make_run
```

and lines 29-53 with:

```python
    citations = [make_citation("c2")]
    records = audit_from_run(run, citations, snippet_chars=5)

    assert [(r.round, r.query_text) for r in records] == [
        (1, "paracetamol liều"),
        (2, "paracetamol quá liều"),
        (2, "paracetamol trẻ em"),
    ]
    first = records[0]
    assert [(h.rank, h.chunk_id, h.cited) for h in first.hits] == [
        (1, str(chunk_uuid("c2")), True),
        (2, str(chunk_uuid("c1")), False),
    ]
    assert first.hits[0].rerank_score == 0.9 and first.hits[0].snippet == "parac"
    assert [h.chunk_id for h in records[1].hits] == [str(chunk_uuid("c3"))]
    assert records[2].hits == []
```

In `backend/tests/domain/test_conversation.py`, change the factories import (line 13) to `from tests.domain.factories import NOW, make_citation, make_run`, remove `Citation` from the `pharma_agent.domain.conversation.models` import, and replace the `citation = Citation(...)` block (lines 64-72) with `citation = make_citation("c1")`.

In `backend/tests/application/test_checkpoint.py`, change the factories import to `from tests.domain.factories import NOW, chunk_uuid, make_run, search_result` and append to `test_agent_run_round_trips_under_strict_allowlist`:

```python
    assert restored.evidence.items[0].hit.chunk_version_id == chunk_uuid("c1")
    assert restored.evidence.items[0].chunks[0].section_revision_id == (
        run.evidence.items[0].chunks[0].section_revision_id
    )
```

- [ ] **Step 3: Run the domain tests and confirm they fail**

Run: `cd backend && uv run pytest -q tests/domain tests/application/test_checkpoint.py`
Expected: collection errors (`ValidationError` for unknown `chunk_version_id` fields in `make_hit`, `ImportError: cannot import name 'page_label'`, `ImportError: cannot import name 'SNIPPET_CHARS'`).

- [ ] **Step 4: Implement the domain changes**

Replace `backend/src/pharma_agent/domain/retrieval/models.py` with (the `ChunkRecord` fields are unchanged from Task 1; `to_hit` is new):

```python
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class HydrateStrategy(StrEnum):
    FULL_SECTION = "full_section"
    CHUNK_WINDOW = "chunk_window"
    SEARCH_ONLY = "search_only"


class QueryOrigin(StrEnum):
    INITIAL = "initial"
    REFINED = "refined"


class Query(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    origin: QueryOrigin

    @property
    def normalized(self) -> str:
        return " ".join(self.text.lower().split())


class TermAnnotation(BaseModel):
    term: str
    vi: list[str] = Field(default_factory=list)
    en: list[str] = Field(default_factory=list)


class ColloquialMapping(BaseModel):
    key: str = ""
    aliases: list[str] = Field(default_factory=list)
    visual_sign: str = ""
    product_names: list[str] = Field(default_factory=list)


def page_label(start_page: int | None, end_page: int | None) -> str:
    """Vietnamese page label; empty when the source has no page numbers (leaflets)."""
    if start_page is None and end_page is None:
        return ""
    if start_page is None or end_page is None or start_page == end_page:
        return f"trang {start_page if start_page is not None else end_page}"
    return f"trang {start_page}-{end_page}"


class Hit(BaseModel):
    """One chunk version found by search, as published in one release, plus scores."""

    chunk_version_id: UUID
    release_id: UUID
    collection_id: UUID
    document_key: str
    section_key: str
    section_revision_id: UUID
    ordinal: int
    hydrate_strategy: HydrateStrategy
    source: str
    title: str
    section: str
    start_page: int | None
    end_page: int | None
    context_header: str
    chunk_text: str
    # What was embedded and what the reranker scores, unchanged from today.
    embedding_text: str
    kind: str
    table_key: str | None
    colloquial_mapping: ColloquialMapping | None = None
    term_annotations: list[TermAnnotation] = Field(default_factory=list)
    fusion_score: float = 0.0
    rerank_score: float | None = None
    matched_queries: list[str] = Field(default_factory=list)

    @property
    def score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.fusion_score

    @property
    def page_label(self) -> str:
        return page_label(self.start_page, self.end_page)

    def term_hints(self) -> list[str]:
        hints: list[str] = []
        if self.colloquial_mapping is not None:
            hints.extend(self.colloquial_mapping.aliases)
            hints.extend(self.colloquial_mapping.product_names)
        for annotation in self.term_annotations:
            hints.append(annotation.term)
            hints.extend(annotation.vi)
            hints.extend(annotation.en)
        seen: set[str] = set()
        unique: list[str] = []
        for hint in hints:
            key = hint.strip().lower()
            if key and key not in seen:
                seen.add(key)
                unique.append(hint.strip())
        return unique


class Chunk(BaseModel):
    """One chunk of a section revision in the release a hit came from."""

    chunk_version_id: UUID
    section_revision_id: UUID
    ordinal: int
    text: str
    kind: str
    table_key: str | None = None
    start_page: int | None = None
    end_page: int | None = None

    @property
    def is_table(self) -> bool:
        return self.kind == "table"


class RetrievedItem(BaseModel):
    hit: Hit
    chunks: list[Chunk] = Field(default_factory=list)


class ChunkRecord(BaseModel):
    """One chunk version as published in one release, with the fields a search hit shows."""

    model_config = ConfigDict(frozen=True)

    chunk_version_id: UUID
    release_id: UUID
    collection_id: UUID
    document_key: str
    section_key: str
    section_revision_id: UUID
    ordinal: int
    hydrate_strategy: HydrateStrategy
    source: str
    title: str
    section: str
    start_page: int | None
    end_page: int | None
    context_header: str
    chunk_text: str
    embedding_text: str
    kind: str
    table_key: str | None
    colloquial_mapping: ColloquialMapping | None = None
    term_annotations: list[TermAnnotation] = Field(default_factory=list)

    def to_hit(self, *, fusion_score: float, query_text: str) -> Hit:
        return Hit.model_validate(
            {
                **self.model_dump(),
                "fusion_score": fusion_score,
                "matched_queries": [query_text],
            }
        )
```

In `backend/src/pharma_agent/domain/retrieval/ports.py`, add to `CorpusReader`:

```python
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
```

Replace `backend/src/pharma_agent/domain/retrieval/evidence.py` with:

```python
from uuid import UUID

from pydantic import BaseModel, Field

from pharma_agent.domain.retrieval.models import (
    Chunk,
    Hit,
    HydrateStrategy,
    RetrievedItem,
)
from pharma_agent.domain.shared.text import make_snippet


class Evidence(BaseModel):
    ref: str
    hit: Hit
    chunks: list[Chunk] = Field(default_factory=list)
    applied_strategy: HydrateStrategy
    superseded: bool = False

    @property
    def score(self) -> float:
        return self.hit.score

    def ordered_chunks(self) -> list[Chunk]:
        tables = [c for c in self.chunks if c.is_table]
        body = [c for c in self.chunks if not c.is_table]
        return sorted(tables, key=lambda c: c.ordinal) + sorted(
            body, key=lambda c: c.ordinal
        )

    def _uses_hit_text(self) -> bool:
        return self.applied_strategy is HydrateStrategy.SEARCH_ONLY or not self.chunks

    def text(self) -> str:
        if self._uses_hit_text():
            return self.hit.chunk_text
        return "\n\n".join(c.text for c in self.ordered_chunks())

    def text_chunk_version_ids(self) -> list[UUID]:
        """Chunk versions whose text `text()` joins, in the same order."""
        if self._uses_hit_text():
            return [self.hit.chunk_version_id]
        return [c.chunk_version_id for c in self.ordered_chunks()]

    def char_count(self) -> int:
        return len(self.text())


class EvidenceSet(BaseModel):
    items: list[Evidence] = Field(default_factory=list)

    def active(self) -> list[Evidence]:
        return sorted(
            (e for e in self.items if not e.superseded),
            key=lambda e: e.score,
            reverse=True,
        )

    def rerank_scores(self) -> dict[UUID, float]:
        """Rerank scores already computed in this run, superseded evidence included."""
        return {
            e.hit.chunk_version_id: e.hit.rerank_score
            for e in self.items
            if e.hit.rerank_score is not None
        }

    def supersede_all(self) -> None:
        for evidence in self.items:
            evidence.superseded = True

    def merge(self, retrieved: list[RetrievedItem]) -> list[Evidence]:
        by_chunk = {e.hit.chunk_version_id: e for e in self.items}
        for item in retrieved:
            existing = by_chunk.get(item.hit.chunk_version_id)
            if existing is None:
                evidence = Evidence(
                    ref=f"E{len(self.items) + 1}",
                    hit=item.hit,
                    chunks=list(item.chunks),
                    applied_strategy=item.hit.hydrate_strategy
                    if item.chunks
                    else HydrateStrategy.SEARCH_ONLY,
                )
                self.items.append(evidence)
                by_chunk[item.hit.chunk_version_id] = evidence
                continue
            merged_queries = list(existing.hit.matched_queries)
            merged_queries.extend(
                q for q in item.hit.matched_queries if q not in merged_queries
            )
            best_rerank = _max_optional(
                existing.hit.rerank_score, item.hit.rerank_score
            )
            existing.hit = item.hit.model_copy(
                update={
                    "rerank_score": best_rerank,
                    "fusion_score": max(
                        existing.hit.fusion_score, item.hit.fusion_score
                    ),
                    "matched_queries": merged_queries,
                }
            )
            if item.chunks:
                existing.chunks = list(item.chunks)
                existing.applied_strategy = item.hit.hydrate_strategy
            existing.superseded = False
        return self.active()

    def pack(self, max_chars: int) -> list[Evidence]:
        """Fit active evidence into max_chars by downgrading hydrate strategy, never truncating text."""
        packed: list[Evidence] = []
        remaining = max_chars
        for evidence in self.active():
            fitted = _fit(evidence, remaining)
            if fitted is None:
                continue
            packed.append(fitted)
            remaining -= fitted.char_count()
        return packed

    def summary_view(self, snippet_chars: int = 300) -> str:
        lines: list[str] = []
        for evidence in self.active():
            hit = evidence.hit
            parts = [
                evidence.ref,
                hit.context_header,
                hit.page_label,
                make_snippet(hit.chunk_text, snippet_chars),
            ]
            line = " | ".join(part for part in parts if part)
            hints = hit.term_hints()
            if hints:
                line += f" | gợi ý thuật ngữ: {', '.join(hints[:6])}"
            lines.append(line)
        return "\n".join(lines) if lines else "(không có evidence)"

    def context_view(
        self, packed: list[Evidence]
    ) -> tuple[str, list[tuple[int, Evidence]]]:
        numbered: list[tuple[int, Evidence]] = []
        blocks: list[str] = []
        for index, evidence in enumerate(packed, start=1):
            numbered.append((index, evidence))
            hit = evidence.hit
            label = f" ({hit.page_label})" if hit.page_label else ""
            blocks.append(
                f"[{index}] {hit.context_header}{label}\n{evidence.text()}"
            )
        return "\n\n".join(blocks), numbered


def _max_optional(a: float | None, b: float | None) -> float | None:
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


def _window(evidence: Evidence, radius: int = 1) -> list[Chunk]:
    centre = evidence.hit.ordinal
    return [c for c in evidence.chunks if abs(c.ordinal - centre) <= radius]


def _fit(evidence: Evidence, remaining: int) -> Evidence | None:
    candidates: list[Evidence] = []
    if evidence.applied_strategy is HydrateStrategy.FULL_SECTION:
        candidates.append(evidence)
        candidates.append(
            evidence.model_copy(
                update={
                    "chunks": _window(evidence),
                    "applied_strategy": HydrateStrategy.CHUNK_WINDOW,
                }
            )
        )
    elif evidence.applied_strategy is HydrateStrategy.CHUNK_WINDOW:
        candidates.append(evidence)
    candidates.append(
        evidence.model_copy(
            update={"chunks": [], "applied_strategy": HydrateStrategy.SEARCH_ONLY}
        )
    )
    for candidate in candidates:
        if candidate.char_count() <= remaining:
            return candidate
    return None
```

In `backend/src/pharma_agent/domain/retrieval/service.py`:
- add `from uuid import UUID` to the imports;
- change the `search` signature to `known_scores: Mapping[UUID, float] | None = None` and its docstring line to "`known_scores` maps chunk_version_id to a rerank score already computed for the same";
- in `search`, replace `known[hit.chunk_id]` / `hit.chunk_id in known` / `hit.chunk_id not in known` with `known[hit.chunk_version_id]` / `hit.chunk_version_id in known` / `hit.chunk_version_id not in known`;
- replace `_candidate_pool` and `_dedupe` with:

```python
def _candidate_pool(per_query: list[list[Hit]], limit: int) -> list[Hit]:
    """Merge duplicates across queries, then pick up to `limit` chunks round-robin by rank.

    Round-robin keeps every query's best hits in the pool when several queries run at once;
    merged fields (best fusion score, matched queries in query order) cover every list.
    """
    merged = {hit.chunk_version_id: hit for hit in _dedupe(per_query)}
    selected: list[UUID] = []
    depth = max((len(hits) for hits in per_query), default=0)
    for rank in range(depth):
        for hits in per_query:
            if len(selected) >= limit:
                return [merged[chunk_id] for chunk_id in selected]
            if rank < len(hits) and hits[rank].chunk_version_id not in selected:
                selected.append(hits[rank].chunk_version_id)
    return [merged[chunk_id] for chunk_id in selected]


def _dedupe(per_query: list[list[Hit]]) -> list[Hit]:
    by_chunk: dict[UUID, Hit] = {}
    for hits in per_query:
        for hit in hits:
            existing = by_chunk.get(hit.chunk_version_id)
            if existing is None:
                by_chunk[hit.chunk_version_id] = hit
                continue
            queries = list(existing.matched_queries)
            queries.extend(q for q in hit.matched_queries if q not in queries)
            by_chunk[hit.chunk_version_id] = existing.model_copy(
                update={
                    "fusion_score": max(existing.fusion_score, hit.fusion_score),
                    "matched_queries": queries,
                }
            )
    return list(by_chunk.values())
```

In `backend/src/pharma_agent/domain/conversation/models.py`, add `from uuid import UUID` and `from pharma_agent.domain.retrieval.models import HydrateStrategy` to the imports and replace `Citation` with:

```python
class Citation(BaseModel):
    """A cited evidence block: the chunk version the model read and the blocks around it."""

    model_config = ConfigDict(frozen=True)

    index: int
    chunk_version_id: UUID
    release_id: UUID
    strategy: HydrateStrategy
    block_chunk_version_ids: list[UUID]
    source: str
    title: str
    section: str
    start_page: int | None
    end_page: int | None
    snippet: str
```

In `backend/src/pharma_agent/domain/agent/citations.py`, replace the imports with:

```python
import re
from collections.abc import Iterable, Sequence

from pharma_agent.domain.conversation.models import Citation
from pharma_agent.domain.retrieval.evidence import Evidence
from pharma_agent.domain.shared.text import make_snippet

SNIPPET_CHARS = 200
"""Snippet length for evidence items and citations (spec A §3.3)."""
```

and `citations_from` with:

```python
def citations_from(
    numbered: Sequence[tuple[int, Evidence]], used: Sequence[int]
) -> list[Citation]:
    by_index = dict(numbered)
    citations: list[Citation] = []
    for index in used:
        evidence = by_index.get(index)
        if evidence is None:
            continue
        hit = evidence.hit
        citations.append(
            Citation(
                index=index,
                chunk_version_id=hit.chunk_version_id,
                release_id=hit.release_id,
                strategy=evidence.applied_strategy,
                block_chunk_version_ids=evidence.text_chunk_version_ids(),
                source=hit.source,
                title=hit.title,
                section=hit.section,
                start_page=hit.start_page,
                end_page=hit.end_page,
                snippet=make_snippet(hit.chunk_text, SNIPPET_CHARS),
            )
        )
    return citations
```

In `backend/src/pharma_agent/domain/retrieval/audit.py` (transitional until Task 3), replace `cited_chunks = {citation.chunk_id for citation in citations}` with `cited_chunks = {citation.chunk_version_id for citation in citations}` and the `RetrievalHitRecord(...)` construction with:

```python
                RetrievalHitRecord(
                    rank=rank,
                    chunk_id=str(evidence.hit.chunk_version_id),
                    section_id=evidence.hit.section_key,
                    table_id=evidence.hit.table_key or "",
                    fusion_score=evidence.hit.fusion_score,
                    rerank_score=evidence.hit.rerank_score,
                    hydrate_strategy=evidence.hit.hydrate_strategy.value,
                    cited=evidence.hit.chunk_version_id in cited_chunks,
                    snippet=" ".join(evidence.hit.chunk_text.split())[:snippet_chars],
                )
```

- [ ] **Step 5: Run the domain tests and confirm they pass**

Run: `cd backend && uv run pytest -q tests/domain tests/application/test_checkpoint.py`
Expected: all tests in those paths pass.

- [ ] **Step 6: Update the application, reranker and Langfuse tests**

In `backend/tests/application/test_chat_graph.py`, add `from pharma_agent.domain.shared.text import make_snippet` to the imports and change line 26 to `from tests.domain.factories import chunk_uuid, make_hit`. In `test_grounded_answer_in_one_round`, replace

```python
    evidence_event = next(e for e in events if e.type is EventType.EVIDENCE)
    assert [i["index"] for i in evidence_event.data["items"]] == [1, 2]
    assert tokens(events) == outcome.answer_text and "[1]" in outcome.answer_text
    assert [c.index for c in outcome.citations] == [1]
```

with

```python
    evidence_event = next(e for e in events if e.type is EventType.EVIDENCE)
    assert [i["index"] for i in evidence_event.data["items"]] == [1, 2]
    assert evidence_event.data["items"][0] == {
        "index": 1,
        "source": "Dược thư Quốc gia Việt Nam",
        "title": "Paracetamol",
        "section": "Liều dùng",
        "start_page": 10,
        "end_page": 11,
        "snippet": make_snippet("paracetamol 500 mg", 200),
    }
    assert tokens(events) == outcome.answer_text and "[1]" in outcome.answer_text
    assert [c.index for c in outcome.citations] == [1]
    citations_event = next(e for e in events if e.type is EventType.CITATIONS)
    assert citations_event.data["items"][0]["chunk_version_id"] == str(
        chunk_uuid("c1")
    )
    assert citations_event.data["items"][0]["block_chunk_version_ids"] == [
        str(chunk_uuid("c1"))
    ]
```

and in `test_search_more_then_refine_runs_second_search` replace `assert reranker.received == [["c1"], ["c9"]]` with `assert reranker.received == [[chunk_uuid("c1")], [chunk_uuid("c9")]]`.

In `backend/tests/infrastructure/test_llama_cpp_reranker.py`, change the factories import to `from tests.domain.factories import chunk_uuid, make_hit` and replace:

```python
    assert [h.chunk_id for h in hits] == ["b", "a"]
```
with
```python
    assert [h.chunk_version_id for h in hits] == [chunk_uuid("b"), chunk_uuid("a")]
```

The prompt assertion (`"<Document>: Paracetamol > Liều dùng" in body["prompt"]`, "scores embedding_text, like the evaluation") stays as it is: the reranker still scores `embedding_text`, which `make_hit` builds as `"Paracetamol > Liều dùng\n\n<text>"`.

```python
    assert [h.chunk_id for h in hits] == ["b"] and hits[0].rerank_score == 0.7
```
with
```python
    assert [h.chunk_version_id for h in hits] == [chunk_uuid("b")]
    assert hits[0].rerank_score == 0.7
```

```python
    assert [h.chunk_id for h in hits] == ["b"] and hits[0].rerank_score is None
```
with
```python
    assert [h.chunk_version_id for h in hits] == [chunk_uuid("b")]
    assert hits[0].rerank_score is None
```

In `backend/tests/infrastructure/test_langfuse_tracing.py`, change line 30 to `from tests.domain.factories import chunk_uuid, make_hit` and the expected rerank output to:

```python
    assert json.loads(str(attributes["langfuse.observation.output"])) == [
        {"chunk_version_id": str(chunk_uuid("c1")), "rerank_score": 1.0}
    ]
```

- [ ] **Step 7: Run them and confirm they fail**

Run: `cd backend && uv run pytest -q tests/application/test_chat_graph.py tests/infrastructure/test_llama_cpp_reranker.py tests/infrastructure/test_langfuse_tracing.py`
Expected: failures, e.g. `AttributeError: 'Hit' object has no attribute 'table_id'` from `answer_node` (the run falls back, so the evidence assertions fail) and `AttributeError: 'Hit' object has no attribute 'chunk_id'` in the reranker tests.

- [ ] **Step 8: Implement the application, reranker and Langfuse changes**

In `backend/src/pharma_agent/application/chat/nodes.py`, replace the citations import with

```python
from pharma_agent.domain.agent.citations import (
    SNIPPET_CHARS,
    CitationSanitizer,
    citations_from,
)
```

add `from pharma_agent.domain.shared.text import make_snippet` after the skill imports, and in `answer_node` replace the evidence event block and the citations event with:

```python
    context_text, numbered = run.evidence.context_view(packed)
    if numbered:
        _emit(
            ProgressEvent(
                type=EventType.EVIDENCE,
                data={
                    "items": [
                        {
                            "index": index,
                            "source": e.hit.source,
                            "title": e.hit.title,
                            "section": e.hit.section,
                            "start_page": e.hit.start_page,
                            "end_page": e.hit.end_page,
                            "snippet": make_snippet(e.hit.chunk_text, SNIPPET_CHARS),
                        }
                        for index, e in numbered
                    ]
                },
            )
        )
```

```python
    citations = citations_from(numbered, sanitizer.used)
    _emit(
        ProgressEvent(
            type=EventType.CITATIONS,
            data={"items": [c.model_dump(mode="json") for c in citations]},
        )
    )
```

`llama_cpp_reranker.py` is not changed: `_document_text` keeps scoring `hit.embedding_text`.

In `backend/src/pharma_agent/infrastructure/observability/langfuse_retrieval.py`, replace the `observation.update(output=...)` call with:

```python
            observation.update(
                output=[
                    {
                        "chunk_version_id": str(hit.chunk_version_id),
                        "rerank_score": hit.rerank_score,
                    }
                    for hit in ranked
                ]
            )
```

- [ ] **Step 9: Run them and confirm they pass**

Run: `cd backend && uv run pytest -q tests/application/test_chat_graph.py tests/infrastructure/test_llama_cpp_reranker.py tests/infrastructure/test_langfuse_tracing.py`
Expected: all pass.

- [ ] **Step 10: Write the infrastructure tests**

Replace `backend/tests/infrastructure/test_qdrant_adapter.py` with:

```python
import uuid
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Literal

import pytest
from qdrant_client import models

from pharma_agent.domain.retrieval.models import (
    Chunk,
    ChunkRecord,
    HydrateStrategy,
    Query,
    QueryOrigin,
)
from pharma_agent.domain.retrieval.ports import ChunkKey, RetrievalError
from pharma_agent.infrastructure.retrieval.qdrant_adapter import (
    BM25_MODEL_NAME,
    BM25_SPARSE_VECTOR_NAME,
    COLLECTION_ID_KEY,
    DENSE_VECTOR_NAME,
    RELEASE_IDS_KEY,
    OpenAiEmbedder,
    QdrantHybridRetriever,
    release_scope_filter,
)
from tests.domain.factories import COLLECTION_ID, RELEASE_ID, chunk_uuid, make_hit
from tests.fakes import FakeEmbedder, fake_vector

OTHER_COLLECTION = uuid.uuid5(uuid.NAMESPACE_URL, "other-collection")
OTHER_RELEASE = uuid.uuid5(uuid.NAMESPACE_URL, "other-release")


def record(
    label: str,
    *,
    release_id: uuid.UUID = RELEASE_ID,
    collection_id: uuid.UUID = COLLECTION_ID,
) -> ChunkRecord:
    hit = make_hit(label, strategy=HydrateStrategy.FULL_SECTION, text=f"text {label}")
    fields = hit.model_dump(exclude={"fusion_score", "rerank_score", "matched_queries"})
    return ChunkRecord.model_validate(
        {**fields, "release_id": release_id, "collection_id": collection_id}
    )


class FakeReader:
    def __init__(
        self, releases: dict[uuid.UUID, uuid.UUID], records: Sequence[ChunkRecord]
    ) -> None:
        self.releases = releases
        self.records = list(records)
        self.release_calls: list[list[str]] = []
        self.loaded: list[list[ChunkKey]] = []

    async def current_releases(
        self, collection_keys: Sequence[str]
    ) -> dict[uuid.UUID, uuid.UUID]:
        self.release_calls.append(list(collection_keys))
        return dict(self.releases)

    async def load_chunks(self, keys: Sequence[ChunkKey]) -> list[ChunkRecord]:
        self.loaded.append(list(keys))
        wanted = set(keys)
        return [r for r in self.records if (r.release_id, r.chunk_version_id) in wanted]

    async def section_chunks(
        self,
        release_id: uuid.UUID,
        section_revision_id: uuid.UUID,
        *,
        around: int | None,
        radius: int,
    ) -> list[Chunk]:
        return []


class FakeEmbeddings:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.inputs: list[list[str]] = []

    async def create(self, *, model: str, input: list[str]):
        self.inputs.append(list(input))
        data = [
            SimpleNamespace(index=i, embedding=[float(i)] * self.dimension)
            for i in range(len(input))
        ]
        return SimpleNamespace(data=list(reversed(data)))


async def test_embedder_returns_vectors_in_input_order_and_checks_dimension() -> None:
    client = SimpleNamespace(embeddings=FakeEmbeddings(4))
    embedder = OpenAiEmbedder(client, model="m", dimension=4)
    vectors = await embedder.embed(["a", "b"])
    assert vectors == [[0.0] * 4, [1.0] * 4]
    bad = OpenAiEmbedder(
        SimpleNamespace(embeddings=FakeEmbeddings(3)), model="m", dimension=4
    )
    with pytest.raises(RetrievalError, match="dimension"):
        await bad.embed(["a"])


def point(
    label: str, score: float, collection_id: uuid.UUID = COLLECTION_ID
) -> SimpleNamespace:
    return SimpleNamespace(
        id=str(chunk_uuid(label)),
        score=score,
        payload={COLLECTION_ID_KEY: str(collection_id)},
    )


class FakeQdrant:
    def __init__(
        self, points: list[SimpleNamespace], metadata: dict | None = None
    ) -> None:
        self.points = points
        self.metadata = (
            metadata if metadata is not None else {"embedding_model": "m", "dims": 2}
        )
        self.query_calls: list[dict] = []

    async def query_points(self, **kwargs):
        self.query_calls.append(kwargs)
        return SimpleNamespace(points=self.points)

    async def get_collection(self, collection_name: str):
        return SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(
                    vectors={DENSE_VECTOR_NAME: SimpleNamespace(size=2)}
                ),
                metadata=self.metadata,
            )
        )


def retriever(
    client: FakeQdrant,
    reader: FakeReader,
    mode: Literal["hybrid", "dense", "bm25"] = "hybrid",
    embedder: FakeEmbedder | None = None,
) -> QdrantHybridRetriever:
    return QdrantHybridRetriever(
        client,
        embedder if embedder is not None else FakeEmbedder(),
        reader,
        collection="chunks_current",
        scope=["formulary"],
        mode=mode,
        prefetch_k=7,
        rrf_k=2,
        max_concurrent=2,
    )


def test_release_scope_filter_matches_each_collection_with_its_release() -> None:
    scope = release_scope_filter(
        {COLLECTION_ID: RELEASE_ID, OTHER_COLLECTION: OTHER_RELEASE}
    )
    assert scope.must is None
    branches = scope.should
    assert isinstance(branches, list) and len(branches) == 2
    first = branches[0]
    assert isinstance(first, models.Filter) and isinstance(first.must, list)
    collection, release = first.must
    assert isinstance(collection, models.FieldCondition)
    assert isinstance(release, models.FieldCondition)
    assert (collection.key, release.key) == (COLLECTION_ID_KEY, RELEASE_IDS_KEY)
    assert isinstance(collection.match, models.MatchValue)
    assert collection.match.value == str(COLLECTION_ID)
    assert isinstance(release.match, models.MatchValue)
    assert release.match.value == str(RELEASE_ID)


async def test_hybrid_search_filters_by_release_and_loads_chunks_in_rank_order() -> None:
    client = FakeQdrant(
        [
            point("c2", 0.9),
            point("c1", 0.5),
            point("gone", 0.4),
            point("stray", 0.3, OTHER_COLLECTION),
        ]
    )
    reader = FakeReader({COLLECTION_ID: RELEASE_ID}, [record("c1"), record("c2")])

    hits = await retriever(client, reader).search_many(
        [Query(text="paracetamol", origin=QueryOrigin.INITIAL)], top_k=5
    )

    assert reader.release_calls == [["formulary"]]
    assert reader.loaded == [
        [
            (RELEASE_ID, chunk_uuid("c2")),
            (RELEASE_ID, chunk_uuid("c1")),
            (RELEASE_ID, chunk_uuid("gone")),
        ]
    ]
    assert [(h.chunk_version_id, h.fusion_score) for h in hits[0]] == [
        (chunk_uuid("c2"), 0.9),
        (chunk_uuid("c1"), 0.5),
    ]
    top = hits[0][0]
    assert top.release_id == RELEASE_ID and top.matched_queries == ["paracetamol"]
    assert top.chunk_text == "text c2"
    call = client.query_calls[0]
    assert call["collection_name"] == "chunks_current" and call["limit"] == 5
    assert call["with_payload"] == [COLLECTION_ID_KEY]
    expected_scope = release_scope_filter({COLLECTION_ID: RELEASE_ID})
    assert call["query_filter"] == expected_scope
    dense, sparse = call["prefetch"]
    assert (dense.using, dense.limit, dense.query) == (
        DENSE_VECTOR_NAME,
        7,
        fake_vector("paracetamol"),
    )
    assert dense.filter == expected_scope and sparse.filter == expected_scope
    assert sparse.using == BM25_SPARSE_VECTOR_NAME
    assert isinstance(sparse.query, models.Document)
    assert sparse.query.text == "paracetamol" and sparse.query.model == "Qdrant/bm25"
    assert isinstance(call["query"], models.RrfQuery) and call["query"].rrf.k == 2


async def test_dense_mode_uses_the_same_release_filter() -> None:
    client = FakeQdrant([point("c1", 0.8)])
    reader = FakeReader({COLLECTION_ID: RELEASE_ID}, [record("c1")])
    await retriever(client, reader, mode="dense").search_many(
        [Query(text="x", origin=QueryOrigin.INITIAL)], top_k=3
    )
    call = client.query_calls[0]
    assert "prefetch" not in call
    assert (call["using"], call["query"]) == (DENSE_VECTOR_NAME, fake_vector("x"))
    assert call["query_filter"] == release_scope_filter({COLLECTION_ID: RELEASE_ID})


async def test_bm25_mode_runs_one_sparse_query_without_embedding() -> None:
    client = FakeQdrant([point("c1", 0.8)])
    embedder = FakeEmbedder()
    reader = FakeReader({COLLECTION_ID: RELEASE_ID}, [record("c1")])

    hits = await retriever(client, reader, mode="bm25", embedder=embedder).search_many(
        [Query(text="liều trẻ em", origin=QueryOrigin.INITIAL)], top_k=3
    )

    assert embedder.batches == []
    assert [h.chunk_version_id for h in hits[0]] == [chunk_uuid("c1")]
    call = client.query_calls[0]
    assert "prefetch" not in call and call["using"] == BM25_SPARSE_VECTOR_NAME
    assert isinstance(call["query"], models.Document)
    assert (call["query"].text, call["query"].model) == ("liều trẻ em", BM25_MODEL_NAME)
    assert call["query_filter"] == release_scope_filter({COLLECTION_ID: RELEASE_ID})
    assert call["with_payload"] == [COLLECTION_ID_KEY]


async def test_search_without_a_current_release_fails_before_querying_qdrant() -> None:
    client = FakeQdrant([point("c1", 0.8)])
    with pytest.raises(RetrievalError, match="no current release"):
        await retriever(client, FakeReader({}, [])).search_many(
            [Query(text="x", origin=QueryOrigin.INITIAL)], top_k=3
        )
    assert client.query_calls == []


async def test_malformed_point_is_reported_as_retrieval_error() -> None:
    client = FakeQdrant([SimpleNamespace(id="not-a-uuid", score=0.1, payload={})])
    with pytest.raises(RetrievalError, match="qdrant query failed"):
        await retriever(client, FakeReader({COLLECTION_ID: RELEASE_ID}, [])).search_many(
            [Query(text="x", origin=QueryOrigin.INITIAL)], top_k=1
        )


async def test_verify_collection_checks_dimension_and_metadata() -> None:
    reader = FakeReader({}, [])
    await retriever(FakeQdrant([]), reader).verify_collection(
        embedding_model="m", dimension=2
    )
    with pytest.raises(RetrievalError, match="dimension"):
        await retriever(FakeQdrant([]), reader).verify_collection(
            embedding_model="m", dimension=2560
        )
    with pytest.raises(RetrievalError, match="metadata"):
        await retriever(
            FakeQdrant([], metadata={"embedding_model": "other", "dims": 2}), reader
        ).verify_collection(embedding_model="m", dimension=2)
```

Replace `backend/tests/infrastructure/test_qdrant_integration.py` with:

```python
import uuid
from collections.abc import AsyncIterator, Sequence
from typing import Literal

import pytest
from qdrant_client import AsyncQdrantClient, models

from pharma_agent.domain.retrieval.models import Query, QueryOrigin
from pharma_agent.infrastructure.retrieval.qdrant_adapter import (
    BM25_MODEL_NAME,
    BM25_SPARSE_VECTOR_NAME,
    COLLECTION_ID_KEY,
    DENSE_VECTOR_NAME,
    RELEASE_IDS_KEY,
    QdrantHybridRetriever,
)
from tests.domain.factories import COLLECTION_ID, RELEASE_ID, chunk_uuid
from tests.infrastructure.test_qdrant_adapter import FakeReader, record

pytestmark = pytest.mark.integration

DIM = 4
COLLECTION = "chunks_test"
NEXT_RELEASE = uuid.uuid5(uuid.NAMESPACE_URL, "next-release")
TEXTS = {
    "c0": "người lớn 500 mg mỗi 4 giờ",
    "c1": "trẻ em 10 mg/kg",
    "c2": "tối đa 4 g mỗi ngày",
    "c3": "trẻ em 15 mg/kg bản mới",
}
VECTORS = {
    "c0": [1.0, 0.0, 0.0, 0.0],
    "c1": [0.0, 1.0, 0.0, 0.0],
    "c2": [0.5, 0.5, 0.0, 0.0],
    "c3": [0.0, 0.9, 0.1, 0.0],
}
RELEASES = {
    "c0": [RELEASE_ID, NEXT_RELEASE],
    "c1": [RELEASE_ID],
    "c2": [RELEASE_ID, NEXT_RELEASE],
    "c3": [NEXT_RELEASE],
}


class FixedEmbedder:
    """Every query points at c0 in dense space, so only BM25 can lift the matching chunk."""

    @property
    def model(self) -> str:
        return "fake-embedding-4d"

    @property
    def dimension(self) -> int:
        return DIM

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


@pytest.fixture
async def client(qdrant_client: AsyncQdrantClient) -> AsyncIterator[AsyncQdrantClient]:
    """P2's `qdrant_client` (collections wiped) with one small release-tagged collection."""
    await qdrant_client.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(
                size=DIM, distance=models.Distance.COSINE
            )
        },
        sparse_vectors_config={
            BM25_SPARSE_VECTOR_NAME: models.SparseVectorParams(
                modifier=models.Modifier.IDF
            )
        },
        metadata={"embedding_model": "fake-embedding-4d", "dims": DIM},
    )
    await qdrant_client.create_payload_index(
        collection_name=COLLECTION,
        field_name=COLLECTION_ID_KEY,
        field_schema=models.KeywordIndexParams(
            type=models.KeywordIndexType.KEYWORD, is_tenant=True
        ),
    )
    await qdrant_client.create_payload_index(
        collection_name=COLLECTION,
        field_name=RELEASE_IDS_KEY,
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    await qdrant_client.upsert(
        collection_name=COLLECTION,
        points=[
            models.PointStruct(
                id=str(chunk_uuid(label)),
                vector={
                    DENSE_VECTOR_NAME: VECTORS[label],
                    BM25_SPARSE_VECTOR_NAME: models.Document(
                        text=text, model=BM25_MODEL_NAME
                    ),
                },
                payload={
                    COLLECTION_ID_KEY: str(COLLECTION_ID),
                    RELEASE_IDS_KEY: [str(r) for r in RELEASES[label]],
                },
            )
            for label, text in TEXTS.items()
        ],
        wait=True,
    )
    yield qdrant_client


def retriever_for(
    client: AsyncQdrantClient,
    release_id: uuid.UUID,
    mode: Literal["hybrid", "dense", "bm25"] = "hybrid",
) -> QdrantHybridRetriever:
    reader = FakeReader(
        {COLLECTION_ID: release_id},
        [record(label, release_id=r) for label in TEXTS for r in RELEASES[label]],
    )
    return QdrantHybridRetriever(
        client,
        FixedEmbedder(),
        reader,
        collection=COLLECTION,
        scope=["formulary"],
        mode=mode,
        prefetch_k=10,
        rrf_k=2,
    )


async def test_hybrid_search_returns_only_points_of_the_current_release(
    client: AsyncQdrantClient,
) -> None:
    query = [Query(text="trẻ em mg/kg", origin=QueryOrigin.INITIAL)]
    current = retriever_for(client, RELEASE_ID)
    await current.verify_collection(embedding_model="fake-embedding-4d", dimension=DIM)

    hits = (await current.search_many(query, top_k=4))[0]
    assert {h.chunk_version_id for h in hits} == {
        chunk_uuid("c0"),
        chunk_uuid("c1"),
        chunk_uuid("c2"),
    }
    # BM25 pulls the children-dosage chunk up despite the dense vector pointing at c0.
    assert chunk_uuid("c1") in [h.chunk_version_id for h in hits[:2]]
    assert {h.release_id for h in hits} == {RELEASE_ID}

    next_hits = (await retriever_for(client, NEXT_RELEASE).search_many(query, top_k=4))[0]
    assert {h.chunk_version_id for h in next_hits} == {
        chunk_uuid("c0"),
        chunk_uuid("c2"),
        chunk_uuid("c3"),
    }
    assert {h.release_id for h in next_hits} == {NEXT_RELEASE}


async def test_bm25_mode_matches_terms_inside_the_current_release_only(
    client: AsyncQdrantClient,
) -> None:
    query = [Query(text="trẻ em mg/kg", origin=QueryOrigin.INITIAL)]

    current = (
        await retriever_for(client, RELEASE_ID, "bm25").search_many(query, top_k=4)
    )[0]
    assert current[0].chunk_version_id == chunk_uuid("c1")
    assert chunk_uuid("c2") not in {h.chunk_version_id for h in current}  # no query term
    assert {h.release_id for h in current} == {RELEASE_ID}

    upcoming = (
        await retriever_for(client, NEXT_RELEASE, "bm25").search_many(query, top_k=4)
    )[0]
    assert upcoming[0].chunk_version_id == chunk_uuid("c3")
    assert chunk_uuid("c1") not in {h.chunk_version_id for h in upcoming}
```

Append to `backend/tests/infrastructure/test_postgres_corpus.py` (add `from pharma_agent.domain.retrieval.models import HydrateStrategy` to the imports and `PostgresHydrator` to the `postgres_corpus` import):

```python
async def test_hydrator_reads_full_section_and_window_from_the_hit_release(
    database: Database,
) -> None:
    seeded = await seed_release(
        database.sessions,
        collection_key="formulary",
        document=paracetamol_document(),
        sections=[dosage_section(blocks=5), table_section()],
    )
    reader = PostgresCorpusReader(database.sessions)
    drafts = seeded.drafts[DOSAGE_KEY]
    middle = drafts[len(drafts) // 2]
    [found] = await reader.load_chunks([(seeded.release_id, middle.chunk_version_id)])
    hit = found.to_hit(fusion_score=0.5, query_text="q")
    hydrator = PostgresHydrator(reader, window=1)

    full = await hydrator.hydrate(hit, HydrateStrategy.FULL_SECTION)
    assert [c.chunk_version_id for c in full] == [d.chunk_version_id for d in drafts]
    assert [c.ordinal for c in full] == sorted(d.ordinal for d in drafts)
    assert all(c.section_revision_id == seeded.revisions[DOSAGE_KEY] for c in full)
    assert (full[0].text, full[0].kind) == (drafts[0].chunk_text, drafts[0].kind.value)

    window = await hydrator.hydrate(hit, HydrateStrategy.CHUNK_WINDOW)
    assert [c.ordinal for c in window] == [
        d.ordinal for d in drafts if abs(d.ordinal - middle.ordinal) <= 1
    ]
    assert await hydrator.hydrate(hit, HydrateStrategy.SEARCH_ONLY) == []
    assert (
        await reader.section_chunks(
            uuid.uuid4(), seeded.revisions[DOSAGE_KEY], around=None, radius=0
        )
        == []
    )
```

In `backend/tests/infrastructure/test_composition.py`, add the imports

```python
from pharma_agent.infrastructure.composition import (
    build_application,
    build_retrieval_service,
)
from pharma_agent.infrastructure.persistence.postgres.database import Database
from pharma_agent.infrastructure.retrieval.postgres_corpus import PostgresCorpusReader
from tests.fakes import FakeEmbedder
```

(replacing the single `build_application` import), change `app.retriever` / `app.reranker` to `app.retrieval.retriever` / `app.retrieval.reranker` in all three tests, and append:

```python
async def test_build_retrieval_service_owns_its_database_unless_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHARMA_QDRANT__CHECK_COMPATIBILITY", "false")
    settings = Settings(_env_file=None)

    stack = build_retrieval_service(settings)
    assert isinstance(stack.retriever, QdrantHybridRetriever)
    assert isinstance(stack.reader, PostgresCorpusReader)
    assert stack.owns_database is True
    await stack.aclose()

    shared = Database(settings.postgres.dsn, pool_size=1)
    borrowed = build_retrieval_service(settings, database=shared)
    assert borrowed.database is shared and borrowed.owns_database is False
    await borrowed.aclose()
    await shared.dispose()


async def test_build_retrieval_service_uses_an_injected_embedder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHARMA_QDRANT__CHECK_COMPATIBILITY", "false")
    embedder = FakeEmbedder()

    stack = build_retrieval_service(Settings(_env_file=None), embedder=embedder)

    assert stack.embedder is embedder and stack.embed_client is None
    await stack.aclose()
```

Append to P2's `backend/tests/infrastructure/test_corpus_factory.py`:

```python
async def test_open_corpus_services_writes_the_alias_retrieval_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHARMA_QDRANT__CHECK_COMPATIBILITY", "false")
    monkeypatch.setenv("PHARMA_RETRIEVAL__QDRANT_COLLECTION", "chunks_custom")
    settings = Settings(_env_file=None)

    async with open_corpus_services(settings, embedder=FakeEmbedder()) as services:
        assert isinstance(services.index, QdrantVectorIndex)
        assert services.index.alias == settings.retrieval.qdrant_collection
        assert services.index.alias == "chunks_custom"

    async with open_corpus_services(
        settings, embedder=FakeEmbedder(), alias="e2e_chunks_current"
    ) as services:
        assert isinstance(services.index, QdrantVectorIndex)
        assert services.index.alias == "e2e_chunks_current"
```

In `backend/tests/infrastructure/test_settings.py`, replace the `collection_alias` assertion in `test_defaults_match_spec` with

```python
    assert settings.retrieval.qdrant_collection == "chunks_current"
    assert settings.retrieval.collections == ["formulary"]
    assert settings.retrieval.mode == "hybrid"
```

and in `test_role_override_and_nested_env` add `monkeypatch.setenv("PHARMA_RETRIEVAL__COLLECTIONS", '["formulary", "leaflets"]')` and `monkeypatch.setenv("PHARMA_RETRIEVAL__MODE", "bm25")` before `Settings(...)`, then `assert settings.retrieval.collections == ["formulary", "leaflets"]` and `assert settings.retrieval.mode == "bm25"` at the end.

In `backend/tests/infrastructure/test_conversation_repository.py`, remove `Citation` from the domain import, change line 26 to `from tests.domain.factories import NOW, chunk_uuid, make_citation`, replace the `citation = Citation(...)` block in `turn` with `citation = make_citation("c1")`, and replace `messages[0].citations[0].chunk_id == "c1"` with `messages[0].citations[0].chunk_version_id == chunk_uuid("c1")`.

- [ ] **Step 11: Run the infrastructure tests and confirm they fail**

Run: `cd backend && uv run pytest -q tests/infrastructure/test_qdrant_adapter.py tests/infrastructure/test_composition.py tests/infrastructure/test_settings.py tests/infrastructure/test_corpus_factory.py`
Expected: `ImportError: cannot import name 'COLLECTION_ID_KEY'` and `cannot import name 'build_retrieval_service'`; `test_defaults_match_spec` fails on `qdrant_collection`; the new factory test fails with `AssertionError` because the index alias is still `chunks_current`.

- [ ] **Step 12: Implement the infrastructure changes**

Replace `backend/src/pharma_agent/infrastructure/retrieval/qdrant_adapter.py` with:

```python
"""Qdrant hybrid search over the corpus index (spec C §8.3, §9).

Qdrant holds vectors and routing payload only (`collection_id`, `release_ids`). A search reads
the current release of every scoped collection from Postgres, asks Qdrant for chunk version ids
and fused scores inside those releases, then loads the chunks from Postgres in one query.
"""

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any, Literal
from uuid import UUID

from qdrant_client import models

from pharma_agent.domain.corpus.ports import Embedder
from pharma_agent.domain.retrieval.models import Hit, Query
from pharma_agent.domain.retrieval.ports import ChunkKey, CorpusReader, RetrievalError

DENSE_VECTOR_NAME = "dense_vector"
BM25_SPARSE_VECTOR_NAME = "bm25_sparse_vector"
BM25_MODEL_NAME = "Qdrant/bm25"
COLLECTION_ID_KEY = "collection_id"
RELEASE_IDS_KEY = "release_ids"


class OpenAiEmbedder:
    """Query embeddings through an OpenAI-compatible /v1/embeddings endpoint (llama.cpp here)."""

    def __init__(self, client: Any, *, model: str, dimension: int) -> None:
        self._client = client
        self._model = model
        self._dimension = dimension

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = await self._client.embeddings.create(
                model=self._model, input=list(texts)
            )
        except Exception as exc:
            raise RetrievalError(f"embedding request failed: {exc}") from exc
        ordered = sorted(response.data, key=lambda item: item.index)
        vectors = [list(item.embedding) for item in ordered]
        if len(vectors) != len(texts):
            raise RetrievalError(
                f"embedding returned {len(vectors)} vectors for {len(texts)} inputs"
            )
        for vector in vectors:
            if len(vector) != self._dimension:
                raise RetrievalError(
                    f"embedding dimension {len(vector)} != configured {self._dimension}"
                )
        return vectors


def release_scope_filter(releases: Mapping[UUID, UUID]) -> models.Filter:
    """Points of each scoped collection that belong to that collection's current release."""
    branches: list[models.Condition] = [
        models.Filter(
            must=[
                models.FieldCondition(
                    key=COLLECTION_ID_KEY,
                    match=models.MatchValue(value=str(collection_id)),
                ),
                models.FieldCondition(
                    key=RELEASE_IDS_KEY, match=models.MatchValue(value=str(release_id))
                ),
            ]
        )
        for collection_id, release_id in releases.items()
    ]
    return models.Filter(should=branches)


def _scored_point(point: Any) -> tuple[UUID, UUID, float]:
    """(chunk_version_id, collection_id, fused score) of one Qdrant point."""
    payload = point.payload or {}
    return (
        UUID(str(point.id)),
        UUID(str(payload[COLLECTION_ID_KEY])),
        float(point.score or 0.0),
    )


class QdrantHybridRetriever:
    def __init__(
        self,
        client: Any,
        embedder: Embedder,
        reader: CorpusReader,
        *,
        collection: str,
        scope: Sequence[str],
        mode: Literal["hybrid", "dense", "bm25"] = "hybrid",
        prefetch_k: int = 50,
        rrf_k: int = 2,
        max_concurrent: int = 3,
    ) -> None:
        self._client = client
        self._embedder = embedder
        self._reader = reader
        self._collection = collection
        self._scope = list(scope)
        self._mode = mode
        self._prefetch_k = prefetch_k
        self._rrf_k = rrf_k
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def search_many(
        self, queries: Sequence[Query], top_k: int
    ) -> list[list[Hit]]:
        if not queries:
            return []
        releases = await self._reader.current_releases(self._scope)
        if not releases:
            raise RetrievalError(
                f"no current release for collections {', '.join(self._scope)}"
            )
        # bm25 needs no query vector, so the embedding endpoint is not called at all.
        vectors: list[list[float] | None] = [None] * len(queries)
        if self._mode != "bm25":
            embedded = await self._embedder.embed([q.text for q in queries])
            vectors = [*embedded]
        scope = release_scope_filter(releases)
        try:
            ranked = await asyncio.gather(
                *(
                    self._search_one(q, v, top_k, scope)
                    for q, v in zip(queries, vectors, strict=True)
                )
            )
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(f"qdrant query failed: {exc}") from exc

        per_query: list[list[tuple[ChunkKey, float]]] = [
            [
                ((releases[collection_id], chunk_id), score)
                for chunk_id, collection_id, score in points
                if collection_id in releases
            ]
            for points in ranked
        ]
        keys = list(dict.fromkeys(key for scored in per_query for key, _ in scored))
        records = {
            (record.release_id, record.chunk_version_id): record
            for record in await self._reader.load_chunks(keys)
        }
        return [
            [
                records[key].to_hit(fusion_score=score, query_text=query.text)
                for key, score in scored
                if key in records
            ]
            for query, scored in zip(queries, per_query, strict=True)
        ]

    async def _search_one(
        self,
        query: Query,
        vector: list[float] | None,
        top_k: int,
        scope: models.Filter,
    ) -> list[tuple[UUID, UUID, float]]:
        async with self._semaphore:
            if self._mode == "bm25":
                response = await self._client.query_points(
                    collection_name=self._collection,
                    query=models.Document(text=query.text, model=BM25_MODEL_NAME),
                    using=BM25_SPARSE_VECTOR_NAME,
                    query_filter=scope,
                    limit=top_k,
                    with_payload=[COLLECTION_ID_KEY],
                )
            elif vector is None:
                raise RetrievalError(f"{self._mode} search needs a query vector")
            elif self._mode == "hybrid":
                response = await self._client.query_points(
                    collection_name=self._collection,
                    prefetch=[
                        models.Prefetch(
                            query=vector,
                            using=DENSE_VECTOR_NAME,
                            filter=scope,
                            limit=self._prefetch_k,
                        ),
                        models.Prefetch(
                            query=models.Document(
                                text=query.text, model=BM25_MODEL_NAME
                            ),
                            using=BM25_SPARSE_VECTOR_NAME,
                            filter=scope,
                            limit=self._prefetch_k,
                        ),
                    ],
                    query=models.RrfQuery(rrf=models.Rrf(k=self._rrf_k)),
                    query_filter=scope,
                    limit=top_k,
                    with_payload=[COLLECTION_ID_KEY],
                )
            else:
                response = await self._client.query_points(
                    collection_name=self._collection,
                    query=vector,
                    using=DENSE_VECTOR_NAME,
                    query_filter=scope,
                    limit=top_k,
                    with_payload=[COLLECTION_ID_KEY],
                )
        return [_scored_point(point) for point in response.points]

    async def verify_collection(self, *, embedding_model: str, dimension: int) -> None:
        """Raise RetrievalError unless the collection was built for this embedding model."""
        try:
            info = await self._client.get_collection(collection_name=self._collection)
        except Exception as exc:
            raise RetrievalError(
                f"cannot read collection {self._collection}: {exc}"
            ) from exc
        vectors = info.config.params.vectors
        params = vectors[DENSE_VECTOR_NAME] if isinstance(vectors, dict) else vectors
        size = int(getattr(params, "size", 0))
        if size != dimension:
            raise RetrievalError(
                f"collection {self._collection} dense vector dimension {size} != configured {dimension}"
            )
        metadata = info.config.metadata or {}
        if (
            metadata.get("embedding_model") != embedding_model
            or metadata.get("dims") != dimension
        ):
            raise RetrievalError(
                f"collection {self._collection} metadata {metadata!r} does not match "
                f"embedding model {embedding_model} with {dimension} dims"
            )
```

In `backend/src/pharma_agent/infrastructure/retrieval/postgres_corpus.py`, extend the imports to

```python
from pharma_agent.domain.retrieval.models import (
    Chunk,
    ChunkRecord,
    ColloquialMapping,
    Hit,
    HydrateStrategy,
    TermAnnotation,
)
from pharma_agent.domain.retrieval.ports import ChunkKey, CorpusReader, RetrievalError
```

add to `PostgresCorpusReader`:

```python
    async def section_chunks(
        self,
        release_id: UUID,
        section_revision_id: UUID,
        *,
        around: int | None,
        radius: int,
    ) -> list[Chunk]:
        query = (
            select(ReleaseChunkTable, ChunkVersionTable)
            .join(
                ChunkVersionTable,
                ChunkVersionTable.id == ReleaseChunkTable.chunk_version_id,
            )
            .where(
                ReleaseChunkTable.release_id == release_id,
                ReleaseChunkTable.section_revision_id == section_revision_id,
            )
            .order_by(ReleaseChunkTable.ordinal)
        )
        if around is not None:
            query = query.where(
                ReleaseChunkTable.ordinal.between(around - radius, around + radius)
            )
        try:
            async with self._sessions() as session:
                rows = (await session.execute(query)).tuples().all()
        except SQLAlchemyError as exc:
            raise RetrievalError(
                f"cannot hydrate section revision {section_revision_id}: {exc}"
            ) from exc
        return [
            Chunk(
                chunk_version_id=chunk.id,
                section_revision_id=link.section_revision_id,
                ordinal=link.ordinal,
                text=chunk.chunk_text,
                kind=chunk.kind,
                table_key=chunk.table_key,
                start_page=chunk.start_page,
                end_page=chunk.end_page,
            )
            for link, chunk in rows
        ]
```

and append:

```python
class PostgresHydrator:
    """`Hydrator` reading the release that produced the hit: the whole section revision
    (`full_section`) or the chunks within ordinal ± window (`chunk_window`)."""

    def __init__(self, reader: CorpusReader, *, window: int = 1) -> None:
        self._reader = reader
        self._window = window

    async def hydrate(self, hit: Hit, strategy: HydrateStrategy) -> list[Chunk]:
        if strategy is HydrateStrategy.SEARCH_ONLY:
            return []
        around = hit.ordinal if strategy is HydrateStrategy.CHUNK_WINDOW else None
        return await self._reader.section_chunks(
            hit.release_id, hit.section_revision_id, around=around, radius=self._window
        )
```

In `backend/src/pharma_agent/infrastructure/settings.py`, replace the first line of `RetrievalSettings` (`collection_alias: str = ...`) with:

```python
    # Alias of the physical collection chunks_<embedding model slug> (spec C §8.3).
    qdrant_collection: str = "chunks_current"
    # Corpus collections the agent searches; each one needs a current release.
    collections: list[str] = Field(default_factory=lambda: ["formulary"])
```

and change the `mode` line to:

```python
    # bm25 = sparse only with no query embedding (evaluation baseline).
    mode: Literal["hybrid", "dense", "bm25"] = "hybrid"
```

In P2's `backend/src/pharma_agent/infrastructure/corpus_factory.py`, change the `alias` parameter of `open_corpus_services` to `alias: str | None = None`, replace the last docstring sentence with

```python
    `alias` is the Qdrant alias imports write. It defaults to
    `settings.retrieval.qdrant_collection`, the alias retrieval reads, so both sides always
    agree; the E2E server passes its own alias.
```

build the index as

```python
        index = QdrantVectorIndex(
            qdrant,
            model=embedder.model,
            dimension=embedder.dimension,
            alias=alias if alias is not None else settings.retrieval.qdrant_collection,
        )
```

and remove the `CURRENT_ALIAS` import if nothing else in the module uses it.

Replace `backend/src/pharma_agent/infrastructure/composition.py` with:

```python
"""The only module that knows concrete adapters: the retrieval stack, TurnDeps and the runner."""

import os
from dataclasses import dataclass

from langfuse import get_client
from langgraph.checkpoint.base import BaseCheckpointSaver
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient

from pharma_agent.application.chat.context import TurnDeps
from pharma_agent.application.chat.graph import build_chat_graph
from pharma_agent.application.chat.runner import ChatTurnRunner
from pharma_agent.application.tracing import TurnTracer
from pharma_agent.domain.corpus.ports import Embedder
from pharma_agent.domain.guardrail.service import GuardrailService
from pharma_agent.domain.retrieval.ports import CorpusReader, Reranker
from pharma_agent.domain.retrieval.service import RetrievalConfig, RetrievalService
from pharma_agent.domain.shared.clock import SystemClock
from pharma_agent.domain.skill.ports import SkillCatalog
from pharma_agent.infrastructure.llm.openai_adapter import (
    OpenAiLlmAdapter,
    default_client_factory,
    langfuse_client_factory,
)
from pharma_agent.infrastructure.observability.langfuse_retrieval import (
    LangfuseTracedReranker,
)
from pharma_agent.infrastructure.openai_client import build_async_openai
from pharma_agent.infrastructure.persistence.postgres.database import Database
from pharma_agent.infrastructure.retrieval.llama_cpp_reranker import build_reranker
from pharma_agent.infrastructure.retrieval.postgres_corpus import (
    PostgresCorpusReader,
    PostgresHydrator,
)
from pharma_agent.infrastructure.retrieval.qdrant_adapter import (
    OpenAiEmbedder,
    QdrantHybridRetriever,
)
from pharma_agent.infrastructure.settings import RetrievalSettings, Settings
from pharma_agent.infrastructure.skills.filesystem_catalog import FileSystemSkillCatalog


@dataclass
class RetrievalStack:
    """Retrieval over Postgres (schema `corpus`) and Qdrant. Public entry point for the
    seed-pipeline evaluation (spec C §4); release everything with `aclose`."""

    settings: RetrievalSettings
    service: RetrievalService
    retriever: QdrantHybridRetriever
    embedder: Embedder
    reranker: Reranker
    reader: CorpusReader
    qdrant: AsyncQdrantClient
    embed_client: AsyncOpenAI | None
    database: Database
    owns_database: bool

    async def aclose(self) -> None:
        await self.qdrant.close()
        if self.embed_client is not None:
            await self.embed_client.close()
        close_reranker = getattr(self.reranker, "aclose", None)
        if close_reranker is not None:
            await close_reranker()
        if self.owns_database:
            await self.database.dispose()


@dataclass
class Application:
    settings: Settings
    deps: TurnDeps
    runner: ChatTurnRunner
    retrieval: RetrievalStack

    async def aclose(self) -> None:
        await self.retrieval.aclose()


def _export_langfuse_environment(settings: Settings) -> None:
    """Langfuse's OpenAI integration reads its credentials from the environment."""
    if settings.langfuse.enabled:
        os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse.public_key or "")
        os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse.secret_key or "")
        os.environ.setdefault("LANGFUSE_HOST", settings.langfuse.host)


def build_retrieval_service(
    settings: Settings,
    *,
    database: Database | None = None,
    embedder: Embedder | None = None,
) -> RetrievalStack:
    """Build retrieval from settings.

    Pass `database` to reuse a pool the caller disposes, and `embedder` to replace the
    OpenAI-compatible query embedder (the evaluation injects a cached one).
    """
    _export_langfuse_environment(settings)
    retrieval_settings = settings.retrieval
    db = (
        database
        if database is not None
        else Database(
            settings.postgres.dsn,
            pool_size=settings.postgres.pool_size,
            echo=settings.postgres.echo,
        )
    )
    embed_client: AsyncOpenAI | None = None
    if embedder is not None:
        query_embedder = embedder
    else:
        embed_client = build_async_openai(
            api_key=retrieval_settings.embedding.api_key,
            base_url=retrieval_settings.embedding.base_url,
            timeout=retrieval_settings.embedding.timeout_seconds,
            max_retries=retrieval_settings.embedding.max_retries,
            traced=settings.langfuse.enabled,
        )
        query_embedder = OpenAiEmbedder(
            embed_client,
            model=retrieval_settings.embedding.model,
            dimension=retrieval_settings.embedding.dimension,
        )
    qdrant = AsyncQdrantClient(
        url=settings.qdrant.url,
        api_key=settings.qdrant.api_key,
        timeout=int(settings.qdrant.timeout_seconds),
        check_compatibility=settings.qdrant.check_compatibility,
    )
    reader = PostgresCorpusReader(db.sessions)
    retriever = QdrantHybridRetriever(
        qdrant,
        query_embedder,
        reader,
        collection=retrieval_settings.qdrant_collection,
        scope=retrieval_settings.collections,
        mode=retrieval_settings.mode,
        prefetch_k=retrieval_settings.prefetch_k,
        rrf_k=retrieval_settings.rrf_k,
        max_concurrent=retrieval_settings.max_concurrent_searches,
    )
    reranker = build_reranker(retrieval_settings.rerank)
    if settings.langfuse.enabled:
        reranker = LangfuseTracedReranker(
            reranker,
            get_client(public_key=settings.langfuse.public_key),
            protocol=retrieval_settings.rerank.protocol,
            model=retrieval_settings.rerank.model,
        )
    service = RetrievalService(
        retriever,
        reranker,
        PostgresHydrator(reader, window=retrieval_settings.hydrate_window),
        RetrievalConfig(
            candidate_k=retrieval_settings.candidate_k,
            rerank_top_n=retrieval_settings.rerank.top_n,
            rerank_candidates=retrieval_settings.rerank.max_candidates,
        ),
    )
    return RetrievalStack(
        settings=retrieval_settings,
        service=service,
        retriever=retriever,
        embedder=query_embedder,
        reranker=reranker,
        reader=reader,
        qdrant=qdrant,
        embed_client=embed_client,
        database=db,
        owns_database=database is None,
    )


def build_application(
    settings: Settings,
    *,
    checkpointer: BaseCheckpointSaver | None = None,
    skills: SkillCatalog | None = None,
    tracer: TurnTracer | None = None,
    database: Database | None = None,
) -> Application:
    _export_langfuse_environment(settings)
    client_factory = (
        langfuse_client_factory if settings.langfuse.enabled else default_client_factory
    )
    llm = OpenAiLlmAdapter(settings.llm, client_factory=client_factory)
    retrieval = build_retrieval_service(settings, database=database)
    deps = TurnDeps(
        llm=llm,
        guardrail=GuardrailService(llm),
        retrieval=retrieval.service,
        skills=skills
        if skills is not None
        else FileSystemSkillCatalog(settings.skills_dir),
        clock=SystemClock(),
    )
    runner = ChatTurnRunner(
        build_chat_graph(checkpointer=checkpointer),
        deps,
        settings.budget,
        tracer=tracer,
    )
    return Application(settings=settings, deps=deps, runner=runner, retrieval=retrieval)
```

In `backend/src/pharma_agent/infrastructure/container.py`:
- replace `_qdrant_check` with

```python
def _qdrant_check(agent: Application, settings: Settings) -> HealthCheck:
    async def check() -> bool:
        try:
            await agent.retrieval.retriever.verify_collection(
                embedding_model=settings.retrieval.embedding.model,
                dimension=settings.retrieval.embedding.dimension,
            )
        except RetrievalError:
            return False
        return True

    return check
```

- change `corpus_version=settings.retrieval.collection_alias,` to `corpus_version=settings.retrieval.qdrant_collection,` (removed in Task 3);
- pass `database=database` to `build_application(...)`;
- change the registration to `container.health_checks["qdrant"] = _qdrant_check(agent, settings)`.

In `backend/src/pharma_agent/infrastructure/persistence/postgres/conversation_repository.py`, `_message_row` stores `citations=[citation.model_dump(mode="json") for citation in message.citations],` (UUIDs must be JSON strings in JSONB).

In `backend/src/pharma_agent/cli.py`:
- imports: add `from uuid import UUID`; change the models import to `from pharma_agent.domain.retrieval.models import Hit, HydrateStrategy, page_label` and the composition import to `from pharma_agent.infrastructure.composition import build_application, build_retrieval_service`;
- in `_run_turn`, emit `"citations": [c.model_dump(mode="json") for c in outcome.citations],` and print citations with

```python
    for citation in outcome.citations:
        label = page_label(citation.start_page, citation.end_page)
        pages = f" ({label})" if label else ""
        typer.echo(f"[{citation.index}] {citation.title} > {citation.section}{pages}")
```

- replace `check`'s docstring with `"""Kiểm tra cấu hình LLM, Qdrant, embedding và reranker."""` and `_check` with

```python
async def _check(settings: Settings) -> list[tuple[str, bool, str]]:
    results: list[tuple[str, bool, str]] = [
        (
            "llm",
            settings.llm.configured,
            "api key configured"
            if settings.llm.configured
            else "missing PHARMA_LLM__DEFAULT__API_KEY",
        )
    ]
    retrieval = build_retrieval_service(settings)
    embedding = settings.retrieval.embedding
    try:
        try:
            await retrieval.retriever.verify_collection(
                embedding_model=embedding.model, dimension=embedding.dimension
            )
            results.append(
                (
                    "qdrant",
                    True,
                    f"{settings.retrieval.qdrant_collection} {embedding.model} dimension {embedding.dimension}",
                )
            )
        except Exception as exc:
            results.append(("qdrant", False, str(exc)))
        try:
            vectors = await retrieval.embedder.embed(["kiểm tra"])
            results.append(
                ("embedding", True, f"{embedding.model} -> {len(vectors[0])} dims")
            )
        except Exception as exc:
            results.append(("embedding", False, str(exc)))
        try:
            ranked = await retrieval.reranker.rerank(
                "liều paracetamol", [_PROBE_HIT], top_n=1
            )
            results.append(
                (
                    "rerank",
                    True,
                    f"{settings.retrieval.rerank.protocol} score {ranked[0].rerank_score}",
                )
            )
        except Exception as exc:
            results.append(("rerank", False, str(exc)))
    finally:
        await retrieval.aclose()
    return results


_PROBE_ID = UUID(int=0)
_PROBE_HIT = Hit(
    chunk_version_id=_PROBE_ID,
    release_id=_PROBE_ID,
    collection_id=_PROBE_ID,
    document_key="probe",
    section_key="probe",
    section_revision_id=_PROBE_ID,
    ordinal=1,
    hydrate_strategy=HydrateStrategy.SEARCH_ONLY,
    source="probe",
    title="Paracetamol",
    section="Liều dùng",
    start_page=None,
    end_page=None,
    context_header="Paracetamol > Liều dùng",
    chunk_text="Người lớn 500 mg",
    embedding_text="Paracetamol > Liều dùng\n\nNgười lớn 500 mg",
    kind="prose",
    table_key=None,
)
```

In `backend/.env.example`, replace

```text
# Retrieval (defaults match the best corpus-pipeline run)
PHARMA_QDRANT__URL=http://localhost:6333
PHARMA_RETRIEVAL__COLLECTION_ALIAS=thesis_chunks_qwen3_embedding_4b_fp16
```

with

```text
# Retrieval. Corpus text lives in Postgres schema `corpus`; Qdrant holds the index.
# Load data with: uv run pharma-agent corpus import <bundle_dir> --collection formulary --publish
PHARMA_QDRANT__URL=http://localhost:6333
PHARMA_RETRIEVAL__QDRANT_COLLECTION=chunks_current
# PHARMA_RETRIEVAL__COLLECTIONS=["formulary"]
```

- [ ] **Step 13: Run the whole suite**

Run: `cd backend && uv run pytest -q && uv run pytest -q -m integration`
Expected: both runs pass (unit suite includes `tests/test_cli.py`, whose output still contains `[1] Paracetamol > Liều dùng`; integration includes the rewritten Qdrant tests for `hybrid` and `bm25`, the hydrator test and `test_conversation_repository.py`).

- [ ] **Step 14: Run the full check**

Run: `cd backend && uv run ruff check src tests && uv run ruff format --check src tests && uv run pyrefly check --min-severity warn`
Expected: no findings, and `grep -rn "collection_alias\|QdrantHydrator\|hit_from_point\|chunk_index" src tests` prints nothing. The old audit names (`chunk_id`, `table_id`, `corpus_version`) remain only in `domain/retrieval/audit.py`, `persistence/postgres/tables.py`, `conversation_repository.py`, `container.py`, migration `0001` and the audit/repository tests, which Task 3 changes.

- [ ] **Step 15: Commit**

```bash
git add backend/src/pharma_agent/domain backend/src/pharma_agent/application/chat/nodes.py backend/src/pharma_agent/cli.py backend/src/pharma_agent/infrastructure/retrieval backend/src/pharma_agent/infrastructure/observability/langfuse_retrieval.py backend/src/pharma_agent/infrastructure/settings.py backend/src/pharma_agent/infrastructure/composition.py backend/src/pharma_agent/infrastructure/corpus_factory.py backend/src/pharma_agent/infrastructure/container.py backend/src/pharma_agent/infrastructure/persistence/postgres/conversation_repository.py backend/.env.example backend/tests
git commit -m "refactor(retrieval): identify hits, chunks and citations by chunk version and release"
```

The commit message ends with the session attribution trailer.

### Task 3: Audit with release ids and chunk version ids (migration `0006`)

**Files:**
- Create: `backend/src/pharma_agent/infrastructure/persistence/postgres/migrations/versions/0006_audit_release_ids.py`
- Modify: `backend/src/pharma_agent/infrastructure/persistence/postgres/tables.py` (`RetrievalRunTable.corpus_version` line 139; `RetrievalHitTable` lines 156-158)
- Modify: `backend/src/pharma_agent/domain/retrieval/audit.py` (whole file)
- Modify: `backend/src/pharma_agent/infrastructure/persistence/postgres/conversation_repository.py` (`AuditContext` lines 31-35; run row in `append_turn` lines 189-199)
- Modify: `backend/src/pharma_agent/infrastructure/container.py` (`AuditContext(...)` in `open_container`)
- Test: `backend/tests/domain/test_audit.py` (whole file), `backend/tests/infrastructure/test_conversation_repository.py` (imports, `repository`, `AUDIT`, audit assertions), `backend/tests/infrastructure/test_feedback_repository.py` (two `AuditContext(...)` calls), `backend/tests/api/test_e2e_postgres.py` (`AuditContext(...)` call), `backend/tests/infrastructure/test_migrations.py` (append)

**Interfaces:**
- Consumes: Task 2 `Hit.collection_id`, `Hit.release_id`, `Hit.section_key`, `Hit.table_key`, `Citation.chunk_version_id`; P1 `make_snippet`; P2 revision `0005`.
- Produces:
  - `RetrievalHitRecord(rank: int, chunk_version_id: UUID, section_key: str, table_key: str | None, fusion_score: float, rerank_score: float | None, hydrate_strategy: str, cited: bool, snippet: str)`.
  - `RetrievalRunRecord(round: int, query_text: str, release_ids: dict[str, str] = {}, hits: list[RetrievalHitRecord] = [])`; `release_ids` maps `str(collection_id)` to `str(release_id)` of that query's hits (empty when the query found nothing).
  - `audit_from_run(run, citations, *, snippet_chars: int = 300)` with snippets from `make_snippet`.
  - `AuditContext(embedding_model: str, retriever_config: dict[str, Any] = {})`.
  - Table columns: `retrieval_runs.release_ids jsonb not null default '{}'`; `retrieval_hits.chunk_version_id uuid not null`, `section_key text not null`, `table_key text null`; no FK to schema `corpus`.

- [ ] **Step 1: Write the failing tests**

Replace `backend/tests/domain/test_audit.py` with:

```python
from pharma_agent.domain.retrieval.audit import audit_from_run
from pharma_agent.domain.retrieval.models import Query, QueryOrigin, RetrievedItem
from pharma_agent.domain.retrieval.service import SearchResult
from pharma_agent.domain.shared.text import make_snippet
from tests.domain.factories import (
    COLLECTION_ID,
    NOW,
    RELEASE_ID,
    SECTION_KEY,
    chunk_uuid,
    make_citation,
    make_hit,
    make_run,
)


def result_for(query: str, *hits: tuple[str, float]) -> SearchResult:
    return SearchResult(
        items=[
            RetrievedItem(
                hit=make_hit(label, rerank=score).model_copy(
                    update={"matched_queries": [query]}
                )
            )
            for label, score in hits
        ]
    )


def test_one_record_per_query_per_round_with_ranked_hits_and_citations() -> None:
    run = make_run()
    q1 = Query(text="paracetamol liều", origin=QueryOrigin.INITIAL)
    run.record_search([q1], result_for(q1.text, ("c1", 0.4), ("c2", 0.9)), now=NOW)
    q2 = Query(text="paracetamol quá liều", origin=QueryOrigin.REFINED)
    q3 = Query(text="paracetamol trẻ em", origin=QueryOrigin.REFINED)
    run.record_search([q2, q3], result_for(q2.text, ("c3", 0.7)), now=NOW)

    records = audit_from_run(run, [make_citation("c2")], snippet_chars=12)

    assert [(r.round, r.query_text) for r in records] == [
        (1, "paracetamol liều"),
        (2, "paracetamol quá liều"),
        (2, "paracetamol trẻ em"),
    ]
    first = records[0]
    assert first.release_ids == {str(COLLECTION_ID): str(RELEASE_ID)}
    assert [(h.rank, h.chunk_version_id, h.cited) for h in first.hits] == [
        (1, chunk_uuid("c2"), True),
        (2, chunk_uuid("c1"), False),
    ]
    top = first.hits[0]
    assert (top.section_key, top.table_key, top.rerank_score) == (SECTION_KEY, None, 0.9)
    assert top.snippet == make_snippet("paracetamol 500 mg", 12)
    assert [h.chunk_version_id for h in records[1].hits] == [chunk_uuid("c3")]
    assert records[2].hits == [] and records[2].release_ids == {}


def test_failed_search_still_produces_a_record() -> None:
    run = make_run()
    q1 = Query(text="x", origin=QueryOrigin.INITIAL)
    run.record_search([q1], SearchResult(error="qdrant down"), now=NOW)
    records = audit_from_run(run, [])
    assert len(records) == 1 and records[0].hits == []
```

In `backend/tests/infrastructure/test_conversation_repository.py`, change the factories import to

```python
from tests.domain.factories import (
    COLLECTION_ID,
    NOW,
    RELEASE_ID,
    SECTION_KEY,
    chunk_uuid,
    make_citation,
)
```

replace `repository` and `AUDIT` with

```python
def repository(database: Database) -> PostgresConversationRepository:
    return PostgresConversationRepository(
        database.sessions,
        AuditContext(
            embedding_model="qwen3-embedding:4b-fp16",
            retriever_config={"mode": "hybrid", "rrf_k": 2},
        ),
    )
```

```python
AUDIT = [
    RetrievalRunRecord(
        round=1,
        query_text="paracetamol liều",
        release_ids={str(COLLECTION_ID): str(RELEASE_ID)},
        hits=[
            RetrievalHitRecord(
                rank=1,
                chunk_version_id=chunk_uuid("c1"),
                section_key=SECTION_KEY,
                table_key=None,
                fusion_score=0.5,
                rerank_score=0.9,
                hydrate_strategy="chunk_window",
                cited=True,
                snippet="x",
            ),
        ],
    )
]
```

and in `test_append_turn_is_atomic_and_increments_turn_count` replace the `async with database.sessions() as session:` block at the end with

```python
    async with database.sessions() as session:
        run = (await session.execute(select(RetrievalRunTable))).scalar_one()
        assert run.release_ids == {str(COLLECTION_ID): str(RELEASE_ID)}
        assert run.round == 1 and run.embedding_model == "qwen3-embedding:4b-fp16"
        hit = (await session.execute(select(RetrievalHitTable))).scalar_one()
        assert hit.cited is True and hit.rerank_score == 0.9
        assert (hit.chunk_version_id, hit.section_key, hit.table_key) == (
            chunk_uuid("c1"),
            SECTION_KEY,
            None,
        )
```

In `backend/tests/infrastructure/test_feedback_repository.py`, change both `AuditContext(corpus_version="t", embedding_model="t")` to `AuditContext(embedding_model="t")`. In `backend/tests/api/test_e2e_postgres.py`, change `AuditContext(corpus_version="test", embedding_model="test")` to `AuditContext(embedding_model="test")`.

Append to `backend/tests/infrastructure/test_migrations.py`:

```python
def _audit_column_names(connection: Connection, table: str) -> set[str]:
    return {column["name"] for column in inspect(connection).get_columns(table)}


def _audit_referred_tables(connection: Connection, table: str) -> set[str]:
    return {fk["referred_table"] for fk in inspect(connection).get_foreign_keys(table)}


async def test_audit_tables_reference_releases_and_chunk_versions(
    migrated_dsn: str,
) -> None:
    engine = create_async_engine(migrated_dsn)
    async with engine.connect() as connection:
        runs = await connection.run_sync(_audit_column_names, "retrieval_runs")
        hits = await connection.run_sync(_audit_column_names, "retrieval_hits")
        hit_references = await connection.run_sync(
            _audit_referred_tables, "retrieval_hits"
        )
    await engine.dispose()
    assert "release_ids" in runs and "corpus_version" not in runs
    assert {"chunk_version_id", "section_key", "table_key"} <= hits
    assert not {"chunk_id", "section_id", "table_id"} & hits
    assert hit_references == {"retrieval_runs"}  # never a FK into schema corpus
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `cd backend && uv run pytest -q tests/domain/test_audit.py && uv run pytest -q -m integration tests/infrastructure/test_migrations.py tests/infrastructure/test_conversation_repository.py`
Expected: `ValidationError` for `RetrievalHitRecord` / `release_ids` in `test_audit.py`; `TypeError: AuditContext.__init__() missing 1 required positional argument: 'corpus_version'` in the repository tests; `assert 'release_ids' in runs` fails in the migration test.

- [ ] **Step 3: Implement the migration, tables, records and repository**

Create `backend/src/pharma_agent/infrastructure/persistence/postgres/migrations/versions/0006_audit_release_ids.py`:

```python
"""Retrieval audit references corpus releases and chunk versions (spec C §9).

Existing audit rows and stored citations point at position-based chunk ids that no longer
exist, so they are discarded (development data only). There is deliberately no foreign key
into schema `corpus`: audit must never block `pharma-agent corpus gc`.

Revision ID: 0006
Create Date: 2026-09-13 10:00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _discard_audit_and_citations() -> None:
    op.execute("DELETE FROM retrieval_runs")  # cascades to retrieval_hits
    op.execute("UPDATE messages SET citations = '[]'::jsonb")


def upgrade() -> None:
    _discard_audit_and_citations()
    op.drop_column("retrieval_runs", "corpus_version")
    op.add_column(
        "retrieval_runs",
        sa.Column(
            "release_ids",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
    )
    op.drop_column("retrieval_hits", "chunk_id")
    op.drop_column("retrieval_hits", "section_id")
    op.drop_column("retrieval_hits", "table_id")
    op.add_column(
        "retrieval_hits", sa.Column("chunk_version_id", sa.Uuid(), nullable=False)
    )
    op.add_column("retrieval_hits", sa.Column("section_key", sa.Text(), nullable=False))
    op.add_column("retrieval_hits", sa.Column("table_key", sa.Text(), nullable=True))


def downgrade() -> None:
    _discard_audit_and_citations()
    op.drop_column("retrieval_hits", "table_key")
    op.drop_column("retrieval_hits", "section_key")
    op.drop_column("retrieval_hits", "chunk_version_id")
    op.add_column(
        "retrieval_hits",
        sa.Column("table_id", sa.Text(), server_default="", nullable=False),
    )
    op.add_column("retrieval_hits", sa.Column("section_id", sa.Text(), nullable=False))
    op.add_column("retrieval_hits", sa.Column("chunk_id", sa.Text(), nullable=False))
    op.drop_column("retrieval_runs", "release_ids")
    op.add_column(
        "retrieval_runs",
        sa.Column("corpus_version", sa.String(length=200), nullable=False),
    )
```

In `backend/src/pharma_agent/infrastructure/persistence/postgres/tables.py`, replace `corpus_version: Mapped[str] = mapped_column(String(200), nullable=False)` in `RetrievalRunTable` with

```python
    # str(collection_id) -> str(release_id). No FK into schema corpus, so gc is never blocked.
    release_ids: Mapped[dict[str, str]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
```

and the `chunk_id`, `section_id`, `table_id` lines of `RetrievalHitTable` with

```python
    chunk_version_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    section_key: Mapped[str] = mapped_column(Text, nullable=False)
    table_key: Mapped[str | None] = mapped_column(Text, nullable=True)
```

Replace `backend/src/pharma_agent/domain/retrieval/audit.py` with:

```python
"""Retrieval audit derived from a finished AgentRun (one record per query per search round)."""

from collections.abc import Sequence
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from pharma_agent.domain.agent.actions import ActionKind
from pharma_agent.domain.agent.run import AgentRun
from pharma_agent.domain.conversation.models import Citation
from pharma_agent.domain.shared.text import make_snippet


class RetrievalHitRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    rank: int
    chunk_version_id: UUID
    section_key: str
    table_key: str | None
    fusion_score: float
    rerank_score: float | None
    hydrate_strategy: str
    cited: bool
    snippet: str


class RetrievalRunRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    round: int
    query_text: str
    # str(collection_id) -> str(release_id) of the releases this query's hits came from.
    release_ids: dict[str, str] = Field(default_factory=dict)
    hits: list[RetrievalHitRecord] = Field(default_factory=list)


def audit_from_run(
    run: AgentRun, citations: Sequence[Citation], *, snippet_chars: int = 300
) -> list[RetrievalRunRecord]:
    cited_chunks = {citation.chunk_version_id for citation in citations}
    records: list[RetrievalRunRecord] = []
    search_actions = [a for a in run.actions.entries if a.kind is ActionKind.SEARCH]
    for round_number, action in enumerate(search_actions, start=1):
        queries = action.payload.get("queries", [])
        for query_text in queries if isinstance(queries, list) else []:
            matching = sorted(
                (e for e in run.evidence.items if query_text in e.hit.matched_queries),
                key=lambda e: e.score,
                reverse=True,
            )
            hits = [
                RetrievalHitRecord(
                    rank=rank,
                    chunk_version_id=evidence.hit.chunk_version_id,
                    section_key=evidence.hit.section_key,
                    table_key=evidence.hit.table_key,
                    fusion_score=evidence.hit.fusion_score,
                    rerank_score=evidence.hit.rerank_score,
                    hydrate_strategy=evidence.hit.hydrate_strategy.value,
                    cited=evidence.hit.chunk_version_id in cited_chunks,
                    snippet=make_snippet(evidence.hit.chunk_text, snippet_chars),
                )
                for rank, evidence in enumerate(matching, start=1)
            ]
            records.append(
                RetrievalRunRecord(
                    round=round_number,
                    query_text=str(query_text),
                    release_ids={
                        str(e.hit.collection_id): str(e.hit.release_id)
                        for e in matching
                    },
                    hits=hits,
                )
            )
    return records
```

In `backend/src/pharma_agent/infrastructure/persistence/postgres/conversation_repository.py`, replace `AuditContext` with

```python
@dataclass(frozen=True)
class AuditContext:
    """Per-deployment audit fields; release ids come from each record."""

    embedding_model: str
    retriever_config: dict[str, Any] = field(default_factory=dict)
```

and in `append_turn` build the run row as

```python
                run_row = RetrievalRunTable(
                    message_id=assistant_id,
                    conversation_id=key,
                    run_id=assistant_message.run_id or "",
                    round=record.round,
                    query_text=record.query_text,
                    release_ids=dict(record.release_ids),
                    embedding_model=self._audit_context.embedding_model,
                    retriever_config=dict(self._audit_context.retriever_config),
                    created_at=assistant_message.created_at,
                )
```

(`RetrievalHitTable(retrieval_run_id=run_row.id, **hit.model_dump())` stays: the record fields now equal the column names and `model_dump()` keeps `UUID` values for the `Uuid` column.)

In `backend/src/pharma_agent/infrastructure/container.py`, build the audit context as

```python
        AuditContext(
            embedding_model=settings.retrieval.embedding.model,
            retriever_config=settings.retrieval.model_dump(
                mode="json", exclude={"embedding": {"api_key"}, "rerank": {"api_key"}}
            ),
        ),
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `cd backend && uv run pytest -q tests/domain/test_audit.py && uv run pytest -q -m integration tests/infrastructure/test_migrations.py tests/infrastructure/test_conversation_repository.py tests/infrastructure/test_feedback_repository.py tests/api/test_e2e_postgres.py`
Expected: all pass, including `test_upgrade_head_matches_the_models` (models and migration agree) and `test_downgrade_then_upgrade_round_trips`.

- [ ] **Step 5: Run the full check**

Run: `cd backend && uv run ruff check src tests && uv run ruff format --check src tests && uv run pyrefly check --min-severity warn && uv run pytest -q && uv run pytest -q -m integration`
Expected: no findings; both runs pass; `grep -rn "corpus_version\|table_id" src tests` matches only migrations `0001` and `0006`.

- [ ] **Step 6: Commit**

```bash
git add backend/src/pharma_agent/infrastructure/persistence/postgres/migrations/versions/0006_audit_release_ids.py backend/src/pharma_agent/infrastructure/persistence/postgres/tables.py backend/src/pharma_agent/domain/retrieval/audit.py backend/src/pharma_agent/infrastructure/persistence/postgres/conversation_repository.py backend/src/pharma_agent/infrastructure/container.py backend/tests/domain/test_audit.py backend/tests/infrastructure/test_conversation_repository.py backend/tests/infrastructure/test_feedback_repository.py backend/tests/api/test_e2e_postgres.py backend/tests/infrastructure/test_migrations.py
git commit -m "feat(audit): record release ids and chunk version ids for retrieval runs"
```

The commit message ends with the session attribution trailer.

### Task 4: Corpus health check, CLI check and docs

**Files:**
- Modify: `backend/src/pharma_agent/infrastructure/retrieval/qdrant_adapter.py` (add `verify_corpus` after `verify_collection`)
- Modify: `backend/src/pharma_agent/infrastructure/container.py` (`Container` dataclass lines 48-58; `_qdrant_check` from Task 2; check registration)
- Modify: `backend/src/pharma_agent/api/schemas.py` (`HealthResponse` lines 18-21)
- Modify: `backend/src/pharma_agent/api/routers/health.py` (lines 19-25)
- Modify: `backend/src/pharma_agent/cli.py` (`_check` qdrant block from Task 2)
- Modify: `backend/README.md` (lines 14-15, 44-46, health row of the endpoint table), `docker-compose.yml` (backend `environment`, after `PHARMA_QDRANT__URL`), `.env.example` at repo root (lines 7-8)
- Test: `backend/tests/infrastructure/test_qdrant_adapter.py` (append), `backend/tests/api/harness.py` (`build_harness` signature and `Container(...)`), `backend/tests/api/test_health_api.py` (whole file), `backend/tests/infrastructure/test_container.py` (whole file), `backend/tests/test_cli.py` (append)

**Interfaces:**
- Consumes: Task 2 `QdrantHybridRetriever.verify_collection`, `CorpusReader.current_releases`, `Application.retrieval`, `build_retrieval_service`.
- Produces:
  - `QdrantHybridRetriever.verify_corpus(*, embedding_model: str, dimension: int) -> None` (raises `RetrievalError` unless the collection matches and every scoped collection has a current release).
  - `pharma_agent.infrastructure.container.CORPUS_NOT_READY = "CORPUS_NOT_READY"`; `Container.health_reasons: dict[str, str]`; health check name `corpus` (replaces `qdrant`).
  - `HealthResponse.reasons: dict[str, str]` (only failed checks that declare a reason).
  - `pharma-agent check` prints a `corpus` line instead of `qdrant`.
  - `tests.api.harness.build_harness(..., health_reasons: dict[str, str] | None = None)`.

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/infrastructure/test_qdrant_adapter.py`:

```python
async def test_verify_corpus_requires_a_current_release_for_every_scoped_collection() -> (
    None
):
    ready = FakeReader({COLLECTION_ID: RELEASE_ID}, [])
    await retriever(FakeQdrant([]), ready).verify_corpus(
        embedding_model="m", dimension=2
    )
    assert ready.release_calls == [["formulary"]]
    with pytest.raises(RetrievalError, match="0 of 1 collections"):
        await retriever(FakeQdrant([]), FakeReader({}, [])).verify_corpus(
            embedding_model="m", dimension=2
        )
    with pytest.raises(RetrievalError, match="metadata"):
        await retriever(FakeQdrant([], metadata={}), ready).verify_corpus(
            embedding_model="m", dimension=2
        )
```

In `backend/tests/api/harness.py`, add the keyword parameter `health_reasons: dict[str, str] | None = None` to `build_harness` (after `health`) and pass `health_reasons=health_reasons or {},` to `Container(...)` after `health_checks=...`.

Replace `backend/tests/api/test_health_api.py` with:

```python
from tests.api.harness import build_harness


async def _down() -> bool:
    return False


async def _up() -> bool:
    return True


async def test_health_ok_and_degraded() -> None:
    ok = build_harness(authenticated=False, health={"postgres": _up})
    async with ok.client() as client:
        response = await client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json() == {
            "status": "ok",
            "agent": True,
            "checks": {"postgres": True},
            "reasons": {},
        }

    degraded = build_harness(
        authenticated=False,
        agent=False,
        health={"postgres": _up, "corpus": _down},
        health_reasons={"corpus": "CORPUS_NOT_READY"},
    )
    async with degraded.client() as client:
        response = await client.get("/api/v1/health")
        assert response.status_code == 503
        assert response.json() == {
            "status": "degraded",
            "agent": False,
            "checks": {"postgres": True, "corpus": False},
            "reasons": {"corpus": "CORPUS_NOT_READY"},
        }


async def test_reasons_list_only_failed_checks() -> None:
    harness = build_harness(
        authenticated=False,
        health={"postgres": _down, "corpus": _up},
        health_reasons={"corpus": "CORPUS_NOT_READY"},
    )
    async with harness.client() as client:
        response = await client.get("/api/v1/health")
    assert response.status_code == 503
    assert response.json()["reasons"] == {}
```

Replace `backend/tests/infrastructure/test_container.py` with:

```python
import pytest

from pharma_agent.infrastructure.container import CORPUS_NOT_READY, open_container
from pharma_agent.infrastructure.settings import Settings

pytestmark = pytest.mark.integration


async def test_container_without_llm_serves_queries_only(
    migrated_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PHARMA_POSTGRES__DSN", migrated_dsn)
    monkeypatch.delenv("PHARMA_LLM__DEFAULT__API_KEY", raising=False)
    async with open_container(Settings(_env_file=None)) as container:
        assert container.chat is None and container.summarizer is None
        assert container.skills is not None and container.feedback is not None
        catalog = await container.skills.list_for_user("a" * 32)
        assert catalog == []
        assert await container.health_checks["postgres"]() is True
        assert container.health_reasons == {}
        assert await container.queries.list_conversations("a" * 32, limit=5) == []


async def test_container_with_llm_builds_chat_and_corpus_check(
    migrated_dsn: str, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("PHARMA_POSTGRES__DSN", migrated_dsn)
    monkeypatch.setenv("PHARMA_LLM__DEFAULT__API_KEY", "sk-test")
    monkeypatch.setenv("PHARMA_QDRANT__CHECK_COMPATIBILITY", "false")
    # Nothing listens on the discard port, so the Qdrant call fails fast.
    monkeypatch.setenv("PHARMA_QDRANT__URL", "http://127.0.0.1:9")
    monkeypatch.setenv("PHARMA_SKILLS_DIR", str(tmp_path))
    async with open_container(Settings(_env_file=None)) as container:
        assert container.chat is not None and container.summarizer is not None
        assert set(container.health_checks) == {"postgres", "corpus"}
        assert container.health_reasons == {"corpus": CORPUS_NOT_READY}
        assert await container.health_checks["corpus"]() is False
```

Append to `backend/tests/test_cli.py` (add `from pharma_agent.domain.retrieval.ports import RetrievalError` and `from pharma_agent.infrastructure.retrieval.llama_cpp_reranker import NoopReranker` to the imports):

```python
def test_check_reports_corpus_embedding_and_rerank(monkeypatch) -> None:
    class Retriever:
        async def verify_corpus(self, *, embedding_model: str, dimension: int) -> None:
            raise RetrievalError(
                "0 of 1 collections in formulary have a current release"
            )

    class Embedder:
        async def embed(self, texts):
            return [[0.0] * 2560 for _ in texts]

    closed: list[bool] = []

    async def aclose() -> None:
        closed.append(True)

    stack = SimpleNamespace(
        retriever=Retriever(), embedder=Embedder(), reranker=NoopReranker(), aclose=aclose
    )
    monkeypatch.setattr(cli, "build_retrieval_service", lambda settings: stack)
    result = CliRunner().invoke(cli.app, ["check"])
    assert result.exit_code == 1
    assert "ERR corpus: CORPUS_NOT_READY: 0 of 1 collections" in result.output
    assert "OK  embedding:" in result.output and "-> 2560 dims" in result.output
    assert "OK  rerank:" in result.output
    assert closed == [True]
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `cd backend && uv run pytest -q tests/infrastructure/test_qdrant_adapter.py tests/api/test_health_api.py tests/test_cli.py`
Expected: `AttributeError: 'QdrantHybridRetriever' object has no attribute 'verify_corpus'`; `TypeError: build_harness() got an unexpected keyword argument 'health_reasons'` (or, after the harness edit, `Container.__init__() got an unexpected keyword argument 'health_reasons'`); the CLI test fails on the missing `corpus` line.

- [ ] **Step 3: Implement**

In `backend/src/pharma_agent/infrastructure/retrieval/qdrant_adapter.py`, add to `QdrantHybridRetriever`:

```python
    async def verify_corpus(self, *, embedding_model: str, dimension: int) -> None:
        """Raise RetrievalError unless the collection matches the embedding settings and every
        scoped corpus collection has a current release (spec C §9, health `corpus`)."""
        await self.verify_collection(embedding_model=embedding_model, dimension=dimension)
        scope = sorted(set(self._scope))
        releases = await self._reader.current_releases(scope)
        if len(releases) < len(scope):
            raise RetrievalError(
                f"{len(releases)} of {len(scope)} collections in {', '.join(scope)} "
                "have a current release"
            )
```

In `backend/src/pharma_agent/infrastructure/container.py`:
- after `HealthCheck = ...`, add

```python
CORPUS_NOT_READY = "CORPUS_NOT_READY"
```

- add to `Container`, after `health_checks`:

```python
    # Reason code /health reports when the named check fails.
    health_reasons: dict[str, str] = field(default_factory=dict)
```

- replace `_qdrant_check` with

```python
def _corpus_check(agent: Application, settings: Settings) -> HealthCheck:
    async def check() -> bool:
        try:
            await agent.retrieval.retriever.verify_corpus(
                embedding_model=settings.retrieval.embedding.model,
                dimension=settings.retrieval.embedding.dimension,
            )
        except RetrievalError as exc:
            logger.warning("corpus not ready: %s", exc)
            return False
        return True

    return check
```

- replace the registration line with

```python
                container.health_checks["corpus"] = _corpus_check(agent, settings)
                container.health_reasons["corpus"] = CORPUS_NOT_READY
```

In `backend/src/pharma_agent/api/schemas.py`, replace `HealthResponse` with:

```python
class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    agent: bool
    checks: dict[str, bool]
    # Reason code of each failed check that declares one, e.g. {"corpus": "CORPUS_NOT_READY"}.
    reasons: dict[str, str] = Field(default_factory=dict)
```

In `backend/src/pharma_agent/api/routers/health.py`, build the body as:

```python
        checks = dict(zip(names, results, strict=True))
        healthy = all(checks.values())
        body = HealthResponse(
            status="ok" if healthy else "degraded",
            agent=container.chat is not None,
            checks=checks,
            reasons={
                name: container.health_reasons[name]
                for name, ok in checks.items()
                if not ok and name in container.health_reasons
            },
        )
```

In `backend/src/pharma_agent/cli.py`, in `_check` replace the `verify_collection` try block with:

```python
        try:
            await retrieval.retriever.verify_corpus(
                embedding_model=embedding.model, dimension=embedding.dimension
            )
            results.append(
                (
                    "corpus",
                    True,
                    f"{settings.retrieval.qdrant_collection} ({embedding.model}, {embedding.dimension} dims), "
                    f"current release for {', '.join(settings.retrieval.collections)}",
                )
            )
        except Exception as exc:
            results.append(("corpus", False, f"CORPUS_NOT_READY: {exc}"))
```

and the `check` docstring with `"""Kiểm tra cấu hình LLM, corpus (Qdrant + release hiện hành), embedding và reranker."""`.

- [ ] **Step 4: Update the docs**

In `backend/README.md`, replace

```text
Cần Qdrant (collection alias `thesis_chunks_qwen3_embedding_4b_fp16` do corpus-pipeline publish),
llama.cpp embedding (cổng 11434) và reranker (cổng 11435) từ `docker-compose.yml` ở repo root.
```

with

````text
Cần Postgres, Qdrant, llama.cpp embedding (cổng 11434) và reranker (cổng 11435) từ
`docker-compose.yml` ở repo root. Corpus nằm trong schema `corpus` của Postgres; Qdrant chỉ là
index (alias `chunks_current`). Nạp một knowledge bundle và publish trước khi hỏi:

```bash
uv run pharma-agent migrate
uv run pharma-agent corpus import <bundle_dir> --collection formulary --publish
```
````

replace

```text
Embedding phải vẫn là qwen3-embedding 4B để
  khớp collection.
```

with

```text
Embedding phải là model đã dùng khi
  import corpus; `/health` báo `CORPUS_NOT_READY` nếu metadata collection không khớp.
- Lần đầu cần nạp corpus vào container: `docker compose cp <bundle_dir> backend:/tmp/bundle` rồi
  `docker compose exec backend pharma-agent corpus import /tmp/bundle --collection formulary --publish`.
```

and the health row with

```text
| `GET /api/v1/health` | Postgres, `corpus` (collection Qdrant khớp model embedding và mọi collection trong `PHARMA_RETRIEVAL__COLLECTIONS` có release hiện hành; nếu không, `reasons.corpus = "CORPUS_NOT_READY"`), trạng thái agent |
```

In `docker-compose.yml`, after the `PHARMA_QDRANT__URL` line of the `x-backend` environment, add:

```yaml
    # Corpus text lives in Postgres schema `corpus`; Qdrant keeps only the index (alias chunks_current).
    # Load it once: docker compose exec backend pharma-agent corpus import <bundle_dir> --collection formulary --publish
```

In the repo-root `.env.example`, replace

```text
# The embedding model must stay qwen3-embedding 4B so query vectors match the Qdrant collection.
```

with

```text
# The embedding model must be the one the corpus was imported with; /health reports CORPUS_NOT_READY otherwise.
```

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `cd backend && uv run pytest -q tests/infrastructure/test_qdrant_adapter.py tests/api/test_health_api.py tests/test_cli.py && uv run pytest -q -m integration tests/infrastructure/test_container.py`
Expected: all pass.

- [ ] **Step 6: Run the full check**

Run: `cd backend && uv run ruff check src tests && uv run ruff format --check src tests && uv run pyrefly check --min-severity warn && uv run pytest -q && uv run pytest -q -m integration`
Expected: no findings; both runs pass. `docker compose config --quiet` (repo root) exits 0.

- [ ] **Step 7: Commit**

```bash
git add backend/src/pharma_agent/infrastructure/retrieval/qdrant_adapter.py backend/src/pharma_agent/infrastructure/container.py backend/src/pharma_agent/api/schemas.py backend/src/pharma_agent/api/routers/health.py backend/src/pharma_agent/cli.py backend/README.md docker-compose.yml .env.example backend/tests/infrastructure/test_qdrant_adapter.py backend/tests/api/harness.py backend/tests/api/test_health_api.py backend/tests/infrastructure/test_container.py backend/tests/test_cli.py
git commit -m "feat(health): report corpus readiness with CORPUS_NOT_READY"
```

Run the commands from the repo root. The commit message ends with the session attribution trailer.

### Task 5: End-to-end retrieval on the corpus platform

This task adds no production code. It proves the spec §12 integration row on real Postgres and Qdrant with both sides built by their real factories: P2's `open_corpus_services(settings, embedder=FakeEmbedder())` imports the fixture bundle, and `build_retrieval_service(settings, embedder=FakeEmbedder())` searches it in `hybrid` and `bm25` modes, hydrates `full_section` and `chunk_window`, and follows publish and rollback. A failure here is a defect in the adapter it points at (Tasks 1-4) or in P2, and is fixed there, never by loosening the test.

**Files:**
- Test: `backend/tests/infrastructure/test_corpus_retrieval_integration.py`

**Interfaces:**
- Consumes (P1): `KnowledgeBundle`, `SectionRecord`, `BlockRecord`, `BlockKind`, `DocumentKind` (`bundle`); `chunk_section(document, section, glossary, mappings)`.
- Consumes (P2): `pharma_agent.infrastructure.corpus_factory.open_corpus_services(settings, *, embedder)` yielding `CorpusServices` with `importer` (`ImportKnowledgeBundle.__call__(bundle, *, publish)`) and `releases` (`ReleaseService.rollback(collection_key)`); `QdrantVectorIndex.ensure_collection()` runs inside the import and creates `chunks_fake_embedding_4d` with its metadata and the alias named by `settings.retrieval.qdrant_collection` (the factory default after Task 2); fixtures `migrated_dsn`, `qdrant_url`, `qdrant_client`; `tests.corpus_fixtures.small_bundle`, `COLLECTION_KEY`, `DOSAGE_SECTION`; `tests.fakes.FakeEmbedder`, `FAKE_EMBEDDING_MODEL`, `FAKE_EMBEDDING_DIMENSION`.
- Consumes (this plan): `build_retrieval_service(settings, *, embedder)`, `RetrievalStack.retriever`/`.reader`/`.service`/`.aclose`, `PostgresHydrator`, `QdrantHybridRetriever.search_many`/`verify_corpus`, `tests.corpus_rows.long_paragraph`, `reset_corpus`.
- Produces: nothing new.

- [ ] **Step 1: Write the integration test**

Create `backend/tests/infrastructure/test_corpus_retrieval_integration.py`:

```python
"""Corpus platform end to end on real Postgres and Qdrant (spec C §12).

P2's factory imports the fixture bundle and the retrieval factory searches it; only the
embedder is replaced, by the deterministic FakeEmbedder, on both sides.
"""

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Literal

import pytest
from qdrant_client import AsyncQdrantClient

from pharma_agent.domain.corpus.bundle import (
    BlockKind,
    BlockRecord,
    DocumentKind,
    KnowledgeBundle,
    SectionRecord,
)
from pharma_agent.domain.corpus.chunking import chunk_section
from pharma_agent.domain.retrieval.models import Hit, HydrateStrategy, Query, QueryOrigin
from pharma_agent.domain.retrieval.ports import RetrievalError
from pharma_agent.infrastructure.composition import (
    RetrievalStack,
    build_retrieval_service,
)
from pharma_agent.infrastructure.corpus_factory import (
    CorpusServices,
    open_corpus_services,
)
from pharma_agent.infrastructure.persistence.postgres.database import Database
from pharma_agent.infrastructure.retrieval.postgres_corpus import PostgresHydrator
from pharma_agent.infrastructure.settings import Settings
from tests.corpus_fixtures import COLLECTION_KEY, DOSAGE_SECTION, small_bundle
from tests.corpus_rows import long_paragraph, reset_corpus
from tests.fakes import FAKE_EMBEDDING_DIMENSION, FAKE_EMBEDDING_MODEL, FakeEmbedder

pytestmark = pytest.mark.integration

LONG_SECTION_SUFFIX = ":lieu-dung-keo-dai"
# Seven blocks of about 2,500 characters: one chunk each, and the section is longer than
# FULL_SECTION_MAX_CHARS, so the hydrate policy picks chunk_window.
FIRST_MARKERS = [f"kxdoan{i}" for i in range(1, 8)]
EDITED_MARKERS = ["kxmoi1", *FIRST_MARKERS[1:]]


def settings_for(
    dsn: str,
    qdrant_url: str,
    mode: Literal["hybrid", "dense", "bm25"] = "hybrid",
    *,
    qdrant_collection: str | None = None,
) -> Settings:
    """Both factories read the Qdrant alias from `retrieval.qdrant_collection`; without an
    override it keeps the settings default."""
    return Settings(
        _env_file=None,
        postgres={"dsn": dsn},
        qdrant={"url": qdrant_url},
        retrieval={
            "mode": mode,
            "collections": [COLLECTION_KEY],
            "prefetch_k": 100,
            "embedding": {
                "model": FAKE_EMBEDDING_MODEL,
                "dimension": FAKE_EMBEDDING_DIMENSION,
            },
            "rerank": {"protocol": "none"},
            **({"qdrant_collection": qdrant_collection} if qdrant_collection else {}),
        },
    )


@dataclass
class Platform:
    corpus: CorpusServices
    retrieval: RetrievalStack
    hydrator: PostgresHydrator
    dsn: str
    qdrant_url: str


@pytest.fixture
async def platform(
    migrated_dsn: str, qdrant_url: str, qdrant_client: AsyncQdrantClient
) -> AsyncIterator[Platform]:
    """Empty corpus schema and Qdrant (`qdrant_client` wipes every collection)."""
    database = Database(migrated_dsn, pool_size=1)
    await reset_corpus(database)
    await database.dispose()
    settings = settings_for(migrated_dsn, qdrant_url)
    async with open_corpus_services(settings, embedder=FakeEmbedder()) as corpus:
        retrieval = build_retrieval_service(settings, embedder=FakeEmbedder())
        try:
            yield Platform(
                corpus=corpus,
                retrieval=retrieval,
                hydrator=PostgresHydrator(
                    retrieval.reader, window=settings.retrieval.hydrate_window
                ),
                dsn=migrated_dsn,
                qdrant_url=qdrant_url,
            )
        finally:
            await retrieval.aclose()


def long_section(bundle: KnowledgeBundle, markers: Sequence[str]) -> SectionRecord:
    document = next(d for d in bundle.documents if d.kind is DocumentKind.DRUG_MONOGRAPH)
    ordinal = 1 + max(s.ordinal for s in bundle.sections if s.document_key == document.key)
    return SectionRecord(
        key=f"{document.key}{LONG_SECTION_SUFFIX}",
        document_key=document.key,
        heading="Liều dùng kéo dài",
        context_path=["Liều dùng kéo dài"],
        ordinal=ordinal,
        blocks=[
            BlockRecord(kind=BlockKind.PROSE, markdown=long_paragraph(marker))
            for marker in markers
        ],
    )


def bundle_with_long_section(markers: Sequence[str]) -> KnowledgeBundle:
    """The fixture bundle plus one long section, edited in memory like P2's
    `with_section_text`. Its chunks have no bundled vectors, so the import embeds them."""
    base = small_bundle()
    return base.model_copy(
        update={"sections": [*base.sections, long_section(base, markers)]}
    )


async def search(stack: RetrievalStack, text: str) -> list[Hit]:
    [hits] = await stack.retriever.search_many(
        [Query(text=text, origin=QueryOrigin.INITIAL)], top_k=10
    )
    return hits


def with_marker(hits: Sequence[Hit], marker: str) -> list[Hit]:
    return [hit for hit in hits if f"Mã đoạn {marker}." in hit.chunk_text]


async def test_imported_bundle_is_searchable_and_hydrates_both_strategies(
    platform: Platform,
) -> None:
    retrieval = platform.retrieval
    with pytest.raises(RetrievalError):
        await retrieval.retriever.verify_corpus(
            embedding_model=FAKE_EMBEDDING_MODEL, dimension=FAKE_EMBEDDING_DIMENSION
        )

    bundle = bundle_with_long_section(FIRST_MARKERS)
    await platform.corpus.importer(bundle, publish=True)

    await retrieval.retriever.verify_corpus(
        embedding_model=FAKE_EMBEDDING_MODEL, dimension=FAKE_EMBEDDING_DIMENSION
    )
    releases = await retrieval.reader.current_releases([COLLECTION_KEY])
    assert len(releases) == 1

    [top] = with_marker(await search(retrieval, "kxdoan4"), "kxdoan4")
    assert releases[top.collection_id] == top.release_id
    assert top.section_key.endswith(LONG_SECTION_SUFFIX)
    assert top.hydrate_strategy is HydrateStrategy.CHUNK_WINDOW

    section = next(s for s in bundle.sections if s.key == top.section_key)
    document = next(d for d in bundle.documents if d.key == section.document_key)
    drafts = chunk_section(document, section, bundle.glossary, bundle.colloquial_mappings)
    draft = next(d for d in drafts if d.chunk_version_id == top.chunk_version_id)
    assert (top.context_header, top.embedding_text) == (
        draft.context_header,
        draft.embedding_text,
    )

    full = await platform.hydrator.hydrate(top, HydrateStrategy.FULL_SECTION)
    assert [c.chunk_version_id for c in full] == [d.chunk_version_id for d in drafts]
    window = await platform.hydrator.hydrate(top, HydrateStrategy.CHUNK_WINDOW)
    assert [c.ordinal for c in window] == [
        d.ordinal for d in drafts if abs(d.ordinal - top.ordinal) <= 1
    ]
    assert len(window) == 3
    assert await platform.hydrator.hydrate(top, HydrateStrategy.SEARCH_ONLY) == []

    dosage = next(s for s in bundle.sections if s.key == DOSAGE_SECTION)
    dosage_hits = await search(retrieval, dosage.blocks[0].markdown[:200])
    assert DOSAGE_SECTION in {hit.section_key for hit in dosage_hits}

    # The composed RetrievalService hydrates through the same PostgresHydrator.
    result = await retrieval.service.search(
        [Query(text="kxdoan4", origin=QueryOrigin.INITIAL)], rerank_query="kxdoan4"
    )
    item = next(i for i in result.items if i.hit.chunk_version_id == top.chunk_version_id)
    assert [c.ordinal for c in item.chunks] == [c.ordinal for c in window]


async def test_bm25_mode_searches_the_imported_release_without_embedding(
    platform: Platform,
) -> None:
    await platform.corpus.importer(
        bundle_with_long_section(FIRST_MARKERS), publish=True
    )
    embedder = FakeEmbedder()
    stack = build_retrieval_service(
        settings_for(platform.dsn, platform.qdrant_url, "bm25"), embedder=embedder
    )
    try:
        hits = await search(stack, "kxdoan6")
    finally:
        await stack.aclose()

    assert with_marker(hits[:1], "kxdoan6")
    assert embedder.batches == []


async def test_custom_qdrant_collection_is_used_for_import_and_search(
    platform: Platform, qdrant_client: AsyncQdrantClient
) -> None:
    settings = settings_for(
        platform.dsn, platform.qdrant_url, qdrant_collection="p3_custom_alias"
    )
    async with open_corpus_services(settings, embedder=FakeEmbedder()) as corpus:
        await corpus.importer(bundle_with_long_section(FIRST_MARKERS), publish=True)

    aliases = (await qdrant_client.get_aliases()).aliases
    assert {a.alias_name: a.collection_name for a in aliases} == {
        "p3_custom_alias": "chunks_fake_embedding_4d"
    }
    stack = build_retrieval_service(settings, embedder=FakeEmbedder())
    try:
        await stack.retriever.verify_corpus(
            embedding_model=FAKE_EMBEDDING_MODEL, dimension=FAKE_EMBEDDING_DIMENSION
        )
        assert with_marker(await search(stack, "kxdoan4"), "kxdoan4")
    finally:
        await stack.aclose()
    # Nothing was written under the default alias, so retrieval on defaults is not ready.
    with pytest.raises(RetrievalError):
        await platform.retrieval.retriever.verify_corpus(
            embedding_model=FAKE_EMBEDDING_MODEL, dimension=FAKE_EMBEDDING_DIMENSION
        )


async def test_publish_and_rollback_switch_search_results(platform: Platform) -> None:
    retrieval = platform.retrieval
    await platform.corpus.importer(
        bundle_with_long_section(FIRST_MARKERS), publish=True
    )
    first = await retrieval.reader.current_releases([COLLECTION_KEY])

    await platform.corpus.importer(
        bundle_with_long_section(EDITED_MARKERS), publish=True
    )
    second = await retrieval.reader.current_releases([COLLECTION_KEY])
    assert second.keys() == first.keys() and second != first

    edited = await search(retrieval, "kxmoi1")
    assert len(with_marker(edited, "kxmoi1")) == 1
    assert {hit.release_id for hit in edited} == set(second.values())
    assert with_marker(await search(retrieval, "kxdoan1"), "kxdoan1") == []
    [unchanged] = with_marker(await search(retrieval, "kxdoan4"), "kxdoan4")
    assert unchanged.release_id in second.values()

    await platform.corpus.releases.rollback(COLLECTION_KEY)
    assert await retrieval.reader.current_releases([COLLECTION_KEY]) == first

    restored = await search(retrieval, "kxdoan1")
    [original] = with_marker(restored, "kxdoan1")
    assert {hit.release_id for hit in restored} == set(first.values())
    assert with_marker(await search(retrieval, "kxmoi1"), "kxmoi1") == []
    window = await platform.hydrator.hydrate(original, HydrateStrategy.CHUNK_WINDOW)
    assert window and all("kxmoi1" not in chunk.text for chunk in window)
```

- [ ] **Step 2: Run the test**

Run: `cd backend && uv run pytest -q -m integration tests/infrastructure/test_corpus_retrieval_integration.py`
Expected: `4 passed`. If it fails, find the cause before changing anything: `TypeError: open_corpus_services() got an unexpected keyword argument 'embedder'` means P2 has not added the injection pinned in overview §3.3 (add it in P2's factory, not here); an empty `with_marker(...)` after import means the release filter or the `chunks_current` alias does not match P2's index payload (overview §3.3); wrong window ordinals mean `section_chunks` or P2's `release_chunks.ordinal` is off; a non-empty `embedder.batches` in the `bm25` test means the adapter still embeds in that mode.

- [ ] **Step 3: Run the full check**

Run: `cd backend && uv run ruff check src tests && uv run ruff format --check src tests && uv run pyrefly check --min-severity warn && uv run pytest -q && uv run pytest -q -m integration`
Expected: no findings; both runs pass.

- [ ] **Step 4: Commit**

```bash
git add backend/tests/infrastructure/test_corpus_retrieval_integration.py
git commit -m "test(retrieval): cover import, hybrid and bm25 search, hydration and rollback end to end"
```

The commit message ends with the session attribution trailer of the executing session.

## Self-Review

### Spec coverage

| Requirement (spec C §9, §8.3, §12; overview §3.1, §3.3, §3.4) | Task |
| --- | --- |
| `Hit`/`Chunk`: `chunk_version_id`, `release_id`, `collection_id`, `document_key`, `section_key`, `section_revision_id`, `ordinal`, per-chunk pages, `None` instead of 0; `embedding_text` kept; `chunk_id`, `section_id`, `chunk_index`, `content_type`, `table_id` removed; `page_label` handles `None` | 2 |
| Port `CorpusReader`: `current_releases`, `load_chunks` (content, `context_header` and `embedding_text` read from `chunk_versions`, never rebuilt) | 1 |
| Port `CorpusReader.section_chunks(release_id, section_revision_id, around, radius)` | 2 |
| `QdrantHybridRetriever` modes `hybrid` (RRF dense + BM25), `dense`, `bm25` (sparse only, no embedding call); filter `collection_id ∈ scope` and per collection `release_ids = current release`; ids + scores from Qdrant, chunks from `load_chunks` in one query, order and scores preserved | 2 (unit + real Qdrant), 5 |
| `PostgresHydrator` replaces `QdrantHydrator`: `full_section` all release chunks by ordinal, `chunk_window` ordinal ± `hydrate_window`, `search_only` empty; `hit_from_point` removed | 2, 5 |
| Reranker unchanged; it keeps scoring `embedding_text` | 2 (only `chunk_id` assertions in its tests move) |
| Scope `retrieval.collections = ["formulary"]`; current release read from Postgres on every search, no cache | 1, 2 |
| Settings: `collection_alias` removed; `qdrant_collection = "chunks_current"`, `collections`, `mode` with `bm25`; `.env.example`, compose and README updated | 2, 4 |
| `verify_collection` checks dims and collection metadata `embedding_model` | 2 |
| `/health` check `corpus` (collection exists, metadata matches, every scoped collection has a current release), `degraded` with `CORPUS_NOT_READY` | 4 |
| Audit: `retrieval_runs.release_ids jsonb`; `retrieval_hits.chunk_version_id uuid`, `section_key text`; no FK to corpus; migration `0006`; `AuditContext`, repository, `RetrievalHitRecord` | 3 |
| `citations_from` builds `Citation(index, chunk_version_id, release_id, strategy, block_chunk_version_ids, source, title, section, start_page, end_page, snippet)`; kept in `messages.citations` JSONB until P6 | 2 |
| `EvidenceSet.summary_view` uses `make_snippet(..., 300)` | 2 |
| Evidence event items `{index, source, title, section, start_page, end_page, snippet}` | 2 |
| `build_retrieval_service(settings, *, database=None, embedder=None) -> RetrievalStack` with `service` and `aclose`; `build_application` reuses it | 2 (injection tested in `test_composition.py`, used in Task 5) |
| Imports and retrieval use one Qdrant alias: `open_corpus_services` defaults `alias` to `settings.retrieval.qdrant_collection` | 2 (unit), 5 (custom alias import + search) |
| CLI `_PROBE_HIT`, citation printing, `check` | 2, 4 |
| Ripple through `langfuse_retrieval.py`, `tests/fakes.py`, `tests/domain/factories.py` and every affected test | 2 |
| Integration: import the fixture bundle through `open_corpus_services` + `FakeEmbedder` → hybrid and bm25 search → hydrate `full_section`/`chunk_window` → publish/rollback change results | 5 |
| `domain/agent/prompts.py`, `application/progress.py` | Checked: no change needed (prompts receive the summary string; `ProgressEvent.data` is `dict[str, Any]`) |
| `gc` keeps cited chunks, `reindex` after deleting the collection (rest of the §12 row) | Not here: P2 owns them |

### Names checked against the overview and the P1/P2 plans

- §3.1 `make_snippet(text, max_chars)`. §3.2 and the P1 plan: `chunk_section(document, section, glossary, mappings)`, `ChunkDraft` (`context_header`, `embedding_text`), `CHUNKER_VERSION`, `section_revision_id(section_key, blocks)`, `hydrate_strategy_for(section)`, `section_char_count(section)`, bundle records and enums.
- §3.3 and the P2 plan: table classes and columns including `chunk_versions.context_header`; payload keys `collection_id`/`release_ids`; vector names; alias `chunks_current` created by `QdrantVectorIndex.ensure_collection()`; `pharma_agent.domain.corpus.ports.Embedder`; `OpenAiEmbedder.model`/`.dimension`; `open_corpus_services(settings, *, embedder, alias)` and `CorpusServices`; `QdrantVectorIndex(client, *, model, dimension, alias)`; `ReleaseService.rollback(collection_key)`; fixtures `migrated_dsn`, `qdrant_url`, `qdrant_client`; `small_bundle`, `COLLECTION_KEY`, `DOSAGE_SECTION`; `FakeEmbedder`, `fake_vector`, `FAKE_EMBEDDING_MODEL`, `FAKE_EMBEDDING_DIMENSION`; revision `0005` → `0006`.
- §3.4 exactly: `Hit` (with `embedding_text`), `Chunk`, `Citation` field lists; evidence item keys; `CorpusReader`; `PostgresCorpusReader` and `PostgresHydrator` in `postgres_corpus`; `RetrievalSettings.qdrant_collection` / `collections` / `mode: Literal["hybrid", "dense", "bm25"]`; `build_retrieval_service(settings, *, database, embedder) -> RetrievalStack`; health check `corpus`, reason `CORPUS_NOT_READY`.
- Names introduced here and used consistently across tasks: `ChunkRecord`, `ChunkRecord.to_hit`, `ChunkKey`, `release_scope_filter`, `COLLECTION_ID_KEY`, `RELEASE_IDS_KEY`, `verify_collection(*, embedding_model, dimension)`, `verify_corpus(*, embedding_model, dimension)`, `Evidence.text_chunk_version_ids`, `SNIPPET_CHARS`, `RetrievalStack` fields, `Container.health_reasons`, `HealthResponse.reasons`, `AuditContext(embedding_model, retriever_config)`, `RetrievalRunRecord.release_ids`, test helpers `chunk_uuid`, `revision_uuid`, `make_hit`, `make_chunk`, `chunk_of`, `make_item`, `make_citation`, `seed_release`, `reset_corpus`.

### Decisions where the spec, overview or code left room

1. `current_releases` takes collection **keys** (settings hold keys; spec wrote `collection_ids`) and returns `collection_id -> release_id`.
2. `load_chunks` takes `(release_id, chunk_version_id)` pairs, because `hydrate_strategy` and `ordinal` live on `release_chunks`, which depend on the release.
3. `retrieval_hits.table_id` becomes `table_key` (nullable) to match `Hit`; the spec does not mention it.
4. `release_ids` in an audit record comes from that query's hits, so it is `{}` for a query with no hits.
5. The old `qdrant` health check is replaced by `corpus` (it already covered the collection check); `HealthResponse` gains `reasons`.
6. A search with no current release raises `RetrievalError`; the turn records a failed search and abstains.
7. `RetrievalStack` carries more fields than the pinned minimum; with an injected embedder `embed_client` is `None`.
8. The adapter drops its local `Embedder` protocol and uses P2's domain port, so an injected evaluation embedder satisfies both the retriever and `RetrievalStack`.
9. This plan follows overview §3.3 for `chunk_versions.context_header` and `open_corpus_services(..., embedder=...)`; if P2 lands without either, Task 5 Step 2 says to fix P2, not the test.
10. P3 owns the default of P2's `alias` parameter because P3 introduces `retrieval.qdrant_collection`; an explicit `alias` argument still overrides it.
