# Corpus Store and Import Implementation Plan (Plan 2 of 10)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Store the corpus in the Postgres schema `corpus` (migration `0005`), import a `knowledge-bundle/v1` into an immutable release with resumable embedding and a derived Qdrant index, and manage releases (list, publish, rollback, gc, reindex) through `pharma-agent corpus ...`.

**Architecture:** The domain gains `pharma_agent.domain.corpus.models` (collections, releases, the row-shaped snapshot of a bundle built with P1's chunker and identity functions, retention rule) and `ports.py` (`CorpusRepository`, `EmbeddingCache`, `VectorIndex`, `Embedder`). The application layer adds `ImportKnowledgeBundle` (spec C §8.2 steps 1–10), `ReleaseService` and shared embedding and indexing helpers that stream vectors through the embedding cache instead of holding them in memory. Infrastructure adds the SQLAlchemy tables and migration, `PostgresCorpusRepository`, `PostgresEmbeddingCache`, `QdrantVectorIndex`, and `open_corpus_services(settings, *, embedder=None)`, which the thin typer sub-app uses and into which tests and the E2E server inject `FakeEmbedder`.

**Tech Stack:** Python 3.12, SQLAlchemy 2.0.52 (asyncio, psycopg 3.3), Alembic 1.19.2, qdrant-client 1.19.0 against Qdrant server `v1.19.1` (server-side BM25 inference through `models.Document`), typer 0.27, pydantic 2, pytest 9 with pytest-asyncio 1.4 and testcontainers 4.15 (`postgres:17-alpine`, `qdrant/qdrant:v1.19.1`).

**Spec:** `backend/docs/superpowers/specs/2026-09-13-corpus-platform-design.md` §4 (code layout), §6.2 (tables), §8 (import, embedding, index, release, gc), §12 (application unit and integration test rows for import, publish/rollback, gc, reindex). Builds on P1 (`backend/docs/superpowers/plans/2026-09-13-corpus-domain.md`) using only the names pinned in `backend/docs/superpowers/plans/2026-09-13-plans-overview.md` §3.1–§3.3.

## Global Constraints

- The environment is development only. Postgres and Qdrant may be reset; no data backfill or backward compatibility is needed.
- Python 3.12, uv, shared `ruff.toml`, strict `pyrefly.toml` with `unused-ignore = true`, pytest `filterwarnings = ["error"]`. Lint and type errors are fixed in code; never add rule ignores, `# noqa`, `# type: ignore` or new `# pyrefly: ignore`.
- Every task ends green on: `uv run ruff check src tests && uv run ruff format --check src tests && uv run pyrefly check --min-severity warn && uv run pytest -q`. Tasks touching Postgres or Qdrant also run `uv run pytest -q -m integration`. All commands run from `backend/`.
- Layering is enforced by `tests/architecture/test_layering.py`: the domain imports no framework (`sqlalchemy`, `qdrant_client`, `fastapi`, `openai`, `httpx`, `langgraph`...) and no outer layer; `api/` never imports `pharma_agent.domain`.
- Prefer established libraries over custom code. No feature flag or "fake mode" in production code; fakes live under `tests/`.
- Commits: one commit per task, conventional message, ending with the session attribution trailer given in the executing session.
- Pinned names used here (overview §3.3): tables `collections`, `documents`, `sections`, `section_revisions`, `chunk_versions`, `releases`, `release_chunks`, `glossary_entries`, `colloquial_mappings`, `embedding_cache` in schema `corpus` with classes `CollectionTable`, `DocumentTable`, `SectionTable`, `SectionRevisionTable`, `ChunkVersionTable`, `ReleaseTable`, `ReleaseChunkTable`, `GlossaryEntryTable`, `ColloquialMappingTable`, `EmbeddingCacheTable`; revision id `0005`; `ImportKnowledgeBundle.__call__(bundle, *, publish) -> ImportReport`; `ReleaseService.list_releases/publish/rollback/gc/reindex`; `CorpusSettings(embed_batch_size=32, embed_max_concurrent=1, gc_keep=2)` as `Settings.corpus`; Qdrant collection `chunks_<model_slug(model)>`, alias `chunks_current`, vectors `dense_vector` and `bm25_sparse_vector`, payload keys `collection_id`, `release_ids`, `document_id`, `section_id`, `section_revision_id`, `kind`, point id `str(chunk_version_id)`; fixture `backend/tests/fixtures/knowledge_bundle_small/`, `tests/corpus_fixtures.py::small_bundle()`, `tests/fakes.py::FakeEmbedder`, fake model `fake-embedding-4d`.
- P1 names consumed (overview §3.2): `KnowledgeBundle`, `BundleManifest`, `BundleCollection`, `BundleGenerator`, `BundleEmbeddingFile`, `DocumentRecord`, `SectionRecord`, `BlockRecord`, `SourceInfo`, `GlossaryEntry`, `ColloquialMappingRecord`, `DocumentKind`, `BlockKind`, `RetrievalMode`, `BundleValidationError` (`.problems`), `model_slug`, `read_bundle`, `write_bundle`, `CORPUS_NAMESPACE`, `canonical_json`, `sha256_hex`, `section_revision_id`, `CHUNKER_VERSION`, `chunk_section`, `ChunkDraft`, `section_char_count`, `hydrate_strategy_for`, `FULL_SECTION_MAX_CHARS`.
- Plan-specific values: SQL rows per multi-row statement `500`; ids per Qdrant retrieve/set-payload request `500`; points per Qdrant upsert `256`; embedding cache write batch for bundled vectors `500`; UUIDs are stored as `uuid` and written to Qdrant payloads as `str(uuid)`; vectors in `embedding_cache.vector` are float32 little-endian bytes (`struct` format `"<{dims}f"`).
- Deterministic identity added by this plan: `collection_id_for(key) = uuid5(CORPUS_NAMESPACE, "collection\x1f" + key)`, `document_id_for(collection_id, key) = uuid5(CORPUS_NAMESPACE, "document\x1f" + str(collection_id) + "\x1f" + key)`, `section_id_for(document_id, key) = uuid5(CORPUS_NAMESPACE, "section\x1f" + str(document_id) + "\x1f" + key)`. Release ids are random `uuid4`.
- `bundle_digest` hashes the knowledge content only (documents, sections, glossary, colloquial mappings as canonical JSON), not the manifest, the generator build id or the `embeddings/` files, so adding precomputed vectors to a bundle is not a content change.

## Decisions taken in this plan

| Topic | Decision | Why |
| --- | --- | --- |
| Step 1 validation | `read_bundle` performs spec C §5.3 at the boundary (CLI now, upload API later). `ImportKnowledgeBundle` receives an already parsed `KnowledgeBundle` and re-checks only what it depends on: section `document_key` references and vector lengths. It raises `CorpusImportError` before any write | File checksums cannot be re-verified on an in-memory model; P1 pins no model-level validator to call |
| `--collection` | The manifest's `collection.key` is the target. `corpus import --collection <key>` must equal it, otherwise exit code 2 | The pinned `__call__` has no collection parameter; a mismatch almost always means the wrong bundle |
| Re-import | Same digest, `CHUNKER_VERSION` and embedding model as the current release → `no_change`. Same triple on a non-current, non-retired release → that release is reused (`building` resumes, `ready` is only published if asked) | Spec "import lại không sinh dữ liệu" and resume after failure |
| `release_ids` payload | Always recomputed from Postgres (non-retired releases containing the chunk) and written with `SetPayload`, never appended in Qdrant | Qdrant has no atomic array append; recomputing is idempotent and survives crashes |
| Chunk placement | `chunk_versions.section_revision_id` is the revision where the version first appeared; the Qdrant payload uses it. The per-release placement lives in `release_chunks` | `chunk_version_id` does not include the revision (P1 §6.1), so one version can appear in several revisions |
| gc and FK `RESTRICT` | Orphan chunk versions and revisions are deleted in batches of 500 inside a savepoint; a batch that hits an `IntegrityError` is retried row by row and rows still blocked are kept and counted | Works for every referencing table (P6's `message_citations` and anything later) without the repository knowing them |
| gc test before P6 | The integration test creates `public.gc_probe_citations(chunk_version_id uuid REFERENCES corpus.chunk_versions(id) ON DELETE RESTRICT)`, references one orphan chunk, runs gc, asserts that chunk stays and the other orphan goes, then drops the table | Proves the real Postgres RESTRICT path P6 relies on, without inventing P6's table early |
| Glossary and mapping keys | `glossary_entries` PK `(release_id, term)`. `colloquial_mappings` gets an extra `position` column with PK `(release_id, position)` | The ORM needs a primary key. P1's `write_bundle`/`read_bundle` reject case-insensitive duplicate glossary terms, but P1 allows `ColloquialMappingRecord.key == ""` for title-only leaflet mappings, so mapping keys repeat |
| `chunk_versions.context_header` | Extra `text not null` column written from `ChunkDraft.context_header` (overview §3.3) | P3's reader reads the header instead of rebuilding it |
| Visibility | Imported collections are created `private` with `owner_user_id = NULL` | Making corpus public is a later explicit decision (spec §13) |
| Alias | `chunks_current` is created only when missing and never moved | Switching embedding model is out of scope (spec §13) |
| Fixture bundle | Generated by `tests/corpus_fixtures.py` through P1's `write_bundle` and committed; a test regenerates into `tmp_path` and compares with the committed copy | Embedding file keys are `sha256(embedding_text)` produced by P1's chunker, so hand-written static files could not be correct; committed files are still shared with seed-pipeline's contract test (spec §12) |
| Metadata for Alembic | New module `persistence/postgres/metadata.py` imports every table module and owns `include_name` (moved from `tables.py`), with `include_schemas=True` in `env.py` | `env.py` and the migration test need both table modules registered without unused imports |

## Interfaces pinned by this plan

```python
# pharma_agent/domain/corpus/models.py
class Visibility(StrEnum): PRIVATE = "private"; PUBLIC = "public"
class ReleaseStatus(StrEnum): BUILDING = "building"; READY = "ready"; RETIRED = "retired"
class CorpusError(DomainError): code = "CORPUS_ERROR"
class CorpusImportError(CorpusError): code = "CORPUS_IMPORT_FAILED"
class CollectionNotFound(CorpusError): code = "COLLECTION_NOT_FOUND"
class ReleaseNotFound(CorpusError): code = "RELEASE_NOT_FOUND"
class ReleaseNotPublishable(CorpusError): code = "RELEASE_NOT_PUBLISHABLE"
class NoEarlierRelease(CorpusError): code = "NO_EARLIER_RELEASE"
class IndexMismatch(CorpusError): code = "CORPUS_INDEX_MISMATCH"
class Collection(BaseModel): id: UUID; key: str; title: str; owner_user_id: UUID | None = None; visibility: Visibility = Visibility.PRIVATE; current_release_id: UUID | None = None
class ReleaseStats(BaseModel): documents, sections, section_revisions, chunks, glossary_entries, colloquial_mappings, embeddings_cached, embeddings_from_bundle, embeddings_computed, points_upserted, points_updated: int = 0
class Release(BaseModel): id: UUID; collection_id: UUID; number: int; status: ReleaseStatus; bundle_digest: str; chunker_version: str; embedding_model: str; stats: ReleaseStats | None = None; created_at: datetime; ready_at / published_at / retired_at: datetime | None = None
class ReleaseSummary(BaseModel): collection_key: str; release: Release; chunk_count: int; current: bool
class Document(BaseModel): id; collection_id; key; kind: DocumentKind; title; source_title; source_url: str | None; attributes: dict[str, str | int | float | bool | None]
class Section(BaseModel): id; document_id; key; heading; context_path: list[str]; ordinal: int; retrieval_mode: RetrievalMode
class SectionRevision(BaseModel): id; section_id; blocks: list[BlockRecord]; start_page: int | None; end_page: int | None; char_count: int
class ChunkVersion(BaseModel): id; section_revision_id; ordinal; kind: BlockKind; chunk_text; context_header; embedding_text; embedding_text_sha256; start_page; end_page; table_key; term_annotations: list[TermAnnotation]; colloquial: ColloquialMapping | None; chunker_version: str
class ReleaseChunk(BaseModel): chunk_version_id; section_id; section_revision_id; ordinal: int; hydrate_strategy: HydrateStrategy
class CorpusSnapshot(BaseModel): collection; bundle_digest; chunker_version; documents; sections; revisions; chunks; release_chunks; glossary: list[GlossaryEntry]; colloquial_mappings: list[ColloquialMappingRecord]
    def embedding_texts(self) -> dict[str, str]: ...   # embedding_text_sha256 -> embedding_text
    def stats(self) -> ReleaseStats: ...               # structure counts only
class IndexItem(BaseModel): chunk_version_id; collection_id; document_id; section_id; section_revision_id; kind: BlockKind; embedding_text; embedding_text_sha256; release_ids: list[UUID]
class PurgeResult(BaseModel): release_chunks_deleted: int; chunk_versions_deleted: int; chunk_versions_kept: int; section_revisions_deleted: int
def collection_id_for(key: str) -> UUID: ...
def document_id_for(collection_id: UUID, key: str) -> UUID: ...
def section_id_for(document_id: UUID, key: str) -> UUID: ...
def bundle_digest(bundle: KnowledgeBundle) -> str: ...
def build_snapshot(bundle: KnowledgeBundle) -> CorpusSnapshot: ...   # raises CorpusImportError
def releases_to_retire(releases: Sequence[Release], current_release_id: UUID | None, keep: int) -> list[Release]: ...

# pharma_agent/domain/corpus/ports.py
class CorpusRepository(Protocol):
    async def get_collection(self, key: str) -> Collection | None: ...
    async def get_release(self, release_id: UUID) -> Release | None: ...
    async def find_release(self, collection_id: UUID, *, bundle_digest: str, chunker_version: str, embedding_model: str) -> Release | None: ...
    async def list_releases(self, collection_key: str | None) -> list[ReleaseSummary]: ...
    async def stage_release(self, snapshot: CorpusSnapshot, *, release_id: UUID, embedding_model: str, at: datetime) -> Release: ...
    async def index_items(self, release_ids: Sequence[UUID]) -> list[IndexItem]: ...
    async def mark_ready(self, release_id: UUID, stats: ReleaseStats, at: datetime) -> None: ...
    async def publish(self, release_id: UUID, at: datetime) -> None: ...
    async def retire(self, release_ids: Sequence[UUID], at: datetime) -> None: ...
    async def purge_retired(self, collection_id: UUID) -> PurgeResult: ...
class EmbeddingCache(Protocol):
    async def missing(self, model: str, hashes: Sequence[str]) -> set[str]: ...
    async def get_many(self, model: str, hashes: Sequence[str]) -> dict[str, list[float]]: ...
    async def put_many(self, model: str, dimension: int, vectors: Mapping[str, Sequence[float]]) -> None: ...
class Embedder(Protocol):
    @property
    def model(self) -> str: ...
    @property
    def dimension(self) -> int: ...
    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...
class VectorIndex(Protocol):
    async def ensure_collection(self) -> str: ...
    async def existing_ids(self, ids: Sequence[UUID]) -> set[UUID]: ...
    async def upsert(self, items: Sequence[IndexItem], vectors: Mapping[str, Sequence[float]]) -> None: ...
    async def set_release_ids(self, items: Sequence[IndexItem]) -> None: ...
    async def delete(self, ids: Sequence[UUID]) -> None: ...
    async def count_release(self, release_id: UUID) -> int: ...

# pharma_agent/application/corpus/indexing.py
class EmbeddingCounts(BaseModel): cached: int; from_bundle: int; computed: int
class EmbeddingResolver: __init__(cache, embedder, *, batch_size: int, max_concurrent: int); async ensure(texts: Mapping[str, str], bundled: Mapping[str, Sequence[float]]) -> EmbeddingCounts
class IndexCounts(BaseModel): upserted: int; updated: int
class IndexWriter: __init__(index, cache, model: str); async write(items: Sequence[IndexItem], *, overwrite: bool = False) -> IndexCounts

# pharma_agent/application/corpus/import_bundle.py
class ImportOutcome(StrEnum): IMPORTED = "imported"; REUSED = "reused"; NO_CHANGE = "no_change"
class ImportReport(BaseModel): outcome: ImportOutcome; collection_key: str; release: Release; published: bool
class ImportKnowledgeBundle: __init__(repository, cache, index, embedder, clock, *, embed_batch_size: int, embed_max_concurrent: int)

# pharma_agent/application/corpus/releases.py
class GcReport(BaseModel): retired: list[UUID]; points_updated: int; points_deleted: int; purge: PurgeResult
class ReindexReport(BaseModel): points_upserted: int; release_points: dict[UUID, int]; skipped: list[UUID]
class ReleaseService: __init__(repository, cache, index, embedder, clock, *, embed_batch_size: int, embed_max_concurrent: int)
    async list_releases(collection_key: str | None) -> list[ReleaseSummary]; async publish(release_id: UUID) -> Release; async rollback(collection_key: str) -> Release
    async gc(collection_key: str, keep: int) -> GcReport; async reindex(collection_key: str) -> ReindexReport

# pharma_agent/infrastructure/corpus_factory.py
@dataclass class CorpusServices: importer: ImportKnowledgeBundle; releases: ReleaseService; index: VectorIndex; embedder: Embedder
@asynccontextmanager async def open_corpus_services(settings: Settings, *, embedder: Embedder | None = None, alias: str = CURRENT_ALIAS) -> AsyncGenerator[CorpusServices]   # injected embedder replaces OpenAiEmbedder; the index follows its model and dimension and uses `alias`

# pharma_agent/infrastructure/retrieval/qdrant_index.py
CURRENT_ALIAS = "chunks_current"
class QdrantVectorIndex: __init__(client: AsyncQdrantClient, *, model: str, dimension: int, alias: str = CURRENT_ALIAS); collection_name: str; alias: str
```

---

## File Structure

```text
backend/
  src/pharma_agent/
    domain/corpus/models.py                         collections, releases, snapshot builder, retention rule        # Task 3
    domain/corpus/ports.py                          CorpusRepository, EmbeddingCache, Embedder, VectorIndex         # Task 3
    application/corpus/__init__.py                  package marker                                                   # Task 6
    application/corpus/indexing.py                  EmbeddingResolver, IndexWriter                                   # Task 6
    application/corpus/import_bundle.py             ImportKnowledgeBundle, ImportReport                              # Task 6
    application/corpus/releases.py                  ReleaseService, GcReport, ReindexReport                          # Task 7
    infrastructure/persistence/postgres/corpus_tables.py     SQLAlchemy tables of schema corpus                     # Task 1
    infrastructure/persistence/postgres/metadata.py          Alembic target metadata and include_name               # Task 1
    infrastructure/persistence/postgres/tables.py            (include_name moved out)                                # Task 1
    infrastructure/persistence/postgres/migrations/env.py    (include_schemas, metadata module)                     # Task 1
    infrastructure/persistence/postgres/migrations/versions/0005_corpus_schema.py   schema corpus                   # Task 1
    infrastructure/persistence/postgres/corpus_repository.py PostgresCorpusRepository                               # Task 4
    infrastructure/persistence/postgres/embedding_cache.py   PostgresEmbeddingCache                                 # Task 4
    infrastructure/retrieval/qdrant_index.py        QdrantVectorIndex                                                # Task 5
    infrastructure/retrieval/qdrant_adapter.py      (OpenAiEmbedder exposes model and dimension)                     # Task 2
    infrastructure/settings.py                      (+ CorpusSettings, Settings.corpus)                              # Task 8
    infrastructure/corpus_factory.py                open_corpus_services(settings, *, embedder=None)                 # Task 8
    cli.py                                          (+ corpus sub-app: import, releases, publish, rollback, gc, reindex)  # Task 8
  tests/
    conftest.py                                     (+ tests.qdrant plugin)                                          # Task 5
    qdrant.py                                       session Qdrant container v1.19.1, clean client fixture           # Task 5
    fakes.py                                        (+ FakeEmbedder, fake_vector)                                    # Task 2
    corpus_fixtures.py                              build_small_bundle, small_bundle, with_section_text, generator   # Task 2
    corpus_memory.py                                InMemoryCorpusRepository, InMemoryEmbeddingCache, InMemoryVectorIndex  # Task 6
    fixtures/knowledge_bundle_small/                generated bundle files (committed)                               # Task 2
    test_corpus_fixtures.py                         fixture is up to date and covers every case                      # Task 2
    domain/corpus/test_corpus_models.py                 snapshot, digest, identity, retention                            # Task 3
    application/test_import_bundle.py               import flow with in-memory adapters                              # Task 6
    application/test_release_service.py             publish, rollback, gc, reindex with in-memory adapters           # Task 7
    infrastructure/test_persistence_metadata.py     corpus tables on Base.metadata, include_name, FK options         # Task 1
    infrastructure/test_migrations.py               (corpus schema in diff and round trip)                           # Task 1
    infrastructure/test_openai_embedder.py          OpenAiEmbedder exposes model and dimension                       # Task 2
    infrastructure/test_corpus_repository.py        repository and cache against Postgres (RESTRICT probe)           # Task 4
    infrastructure/test_qdrant_index.py             QdrantVectorIndex against Qdrant                                 # Task 5
    infrastructure/test_corpus_factory.py           factory wiring                                                   # Task 8
    infrastructure/test_settings.py                 (+ corpus defaults)                                              # Task 8
    infrastructure/test_corpus_integration.py       import → publish/rollback → gc (RESTRICT) → reindex             # Task 9
    test_cli_corpus.py                              corpus commands with in-memory services                          # Task 8
```

---

### Task 1: Schema `corpus`: tables, migration `0005`, Alembic metadata

**Files:**
- Create: `backend/src/pharma_agent/infrastructure/persistence/postgres/corpus_tables.py`
- Create: `backend/src/pharma_agent/infrastructure/persistence/postgres/metadata.py`
- Create: `backend/src/pharma_agent/infrastructure/persistence/postgres/migrations/versions/0005_corpus_schema.py`
- Modify: `backend/src/pharma_agent/infrastructure/persistence/postgres/tables.py:47-49` (remove `include_name`, now in `metadata.py`)
- Modify: `backend/src/pharma_agent/infrastructure/persistence/postgres/migrations/env.py:8,15,24-43`
- Modify: `backend/tests/infrastructure/test_migrations.py` (whole file)
- Test: `backend/tests/infrastructure/test_persistence_metadata.py`

**Interfaces:**
- Consumes: `Base`, `EXTERNALLY_MANAGED_TABLES` from `tables.py`; fixtures `migrated_dsn`, `fresh_database_dsn` from `tests/postgres.py`.
- Produces: `CORPUS_SCHEMA = "corpus"`; the ten table classes pinned in overview §3.3; `metadata.target_metadata: MetaData`; `metadata.include_name(name: str | None, type_: str, parent_names: object) -> bool`; Alembic revision `0005` (down revision `0004`).

- [ ] **Step 1: Write the failing tests**

`backend/tests/infrastructure/test_persistence_metadata.py`:

```python
from pharma_agent.infrastructure.persistence.postgres.corpus_tables import (
    CORPUS_SCHEMA,
)
from pharma_agent.infrastructure.persistence.postgres.metadata import (
    include_name,
    target_metadata,
)

CORPUS_TABLES = {
    "collections",
    "documents",
    "sections",
    "section_revisions",
    "chunk_versions",
    "releases",
    "release_chunks",
    "glossary_entries",
    "colloquial_mappings",
    "embedding_cache",
}


def test_target_metadata_holds_app_and_corpus_tables() -> None:
    corpus = {
        table.name
        for table in target_metadata.tables.values()
        if table.schema == CORPUS_SCHEMA
    }
    public = {
        table.name
        for table in target_metadata.tables.values()
        if table.schema is None
    }
    assert corpus == CORPUS_TABLES
    assert {"user", "conversations", "messages", "skills"} <= public


def test_include_name_manages_default_and_corpus_schemas_only() -> None:
    assert include_name(None, "schema", {}) is True
    assert include_name("corpus", "schema", {}) is True
    assert include_name("langfuse", "schema", {}) is False
    assert include_name("checkpoints", "table", {}) is False
    assert include_name("conversations", "table", {}) is True


def test_current_release_fk_is_deferrable_and_breaks_the_cycle() -> None:
    collections = target_metadata.tables["corpus.collections"]
    (fk,) = [
        fk
        for fk in collections.foreign_key_constraints
        if fk.column_keys == ["current_release_id"]
    ]
    assert fk.deferrable is True and fk.initially == "DEFERRED" and fk.use_alter
    assert fk.name == "fk_collections_current_release_id_releases"


def test_chunk_versions_are_protected_by_restrict() -> None:
    release_chunks = target_metadata.tables["corpus.release_chunks"]
    chunk_versions = target_metadata.tables["corpus.chunk_versions"]
    ondelete = {
        tuple(fk.column_keys): fk.ondelete
        for fk in [
            *release_chunks.foreign_key_constraints,
            *chunk_versions.foreign_key_constraints,
        ]
    }
    assert ondelete[("chunk_version_id",)] == "RESTRICT"
    assert ondelete[("section_revision_id",)] == "RESTRICT"
    assert ondelete[("release_id",)] == "CASCADE"
```

Replace `backend/tests/infrastructure/test_migrations.py` with:

```python
import asyncio

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import inspect
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from pharma_agent.infrastructure.persistence.postgres.alembic_config import (
    alembic_config,
)
from pharma_agent.infrastructure.persistence.postgres.corpus_tables import (
    CORPUS_SCHEMA,
)
from pharma_agent.infrastructure.persistence.postgres.metadata import (
    include_name,
    target_metadata,
)

pytestmark = pytest.mark.integration

EXPECTED_TABLES = {
    "user",
    "oauth_account",
    "conversations",
    "messages",
    "retrieval_runs",
    "retrieval_hits",
}
EXPECTED_CORPUS_TABLES = {
    "collections",
    "documents",
    "sections",
    "section_revisions",
    "chunk_versions",
    "releases",
    "release_chunks",
    "glossary_entries",
    "colloquial_mappings",
    "embedding_cache",
}


def _diff(connection: Connection) -> list[object]:
    context = MigrationContext.configure(
        connection,
        opts={
            "compare_type": True,
            "include_schemas": True,
            "include_name": include_name,
        },
    )
    return list(compare_metadata(context, target_metadata))


def _tables(connection: Connection, schema: str | None = None) -> set[str]:
    return set(inspect(connection).get_table_names(schema=schema))


def _schemas(connection: Connection) -> set[str]:
    return set(inspect(connection).get_schema_names())


async def test_upgrade_head_matches_the_models(migrated_dsn: str) -> None:
    engine = create_async_engine(migrated_dsn)
    async with engine.connect() as connection:
        assert await connection.run_sync(_tables) >= EXPECTED_TABLES
        assert (
            await connection.run_sync(_tables, CORPUS_SCHEMA)
            == EXPECTED_CORPUS_TABLES
        )
        assert await connection.run_sync(_diff) == []
    await engine.dispose()


async def test_downgrade_then_upgrade_round_trips(fresh_database_dsn: str) -> None:
    config = alembic_config(fresh_database_dsn)
    await asyncio.to_thread(command.upgrade, config, "head")
    await asyncio.to_thread(command.downgrade, config, "base")
    engine = create_async_engine(fresh_database_dsn)
    async with engine.connect() as connection:
        assert not (EXPECTED_TABLES & await connection.run_sync(_tables))
        assert CORPUS_SCHEMA not in await connection.run_sync(_schemas)
    await engine.dispose()
    await asyncio.to_thread(command.upgrade, config, "head")
```

- [ ] **Step 2: Run the tests and watch them fail**

Run: `uv run pytest -q tests/infrastructure/test_persistence_metadata.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'pharma_agent.infrastructure.persistence.postgres.corpus_tables'`.

- [ ] **Step 3: Write the tables**

`backend/src/pharma_agent/infrastructure/persistence/postgres/corpus_tables.py`:

```python
"""Tables of the Postgres schema `corpus` (spec 2026-09-13 corpus platform §6.2).

`section_revisions`, `chunk_versions`, `release_chunks`, `glossary_entries`,
`colloquial_mappings` and `embedding_cache` rows are immutable once written; releases only
change status. Chunk versions are protected by `RESTRICT` foreign keys so a citation can never
lose the text it points to.
"""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from pharma_agent.infrastructure.persistence.postgres.tables import Base

CORPUS_SCHEMA = "corpus"


def _created_at() -> Mapped[datetime]:
    return mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class CollectionTable(Base):
    __tablename__ = "collections"
    __table_args__ = (
        UniqueConstraint("key", name="uq_collections_key"),
        CheckConstraint("visibility IN ('private', 'public')", name="visibility"),
        {"schema": CORPUS_SCHEMA},
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    key: Mapped[str] = mapped_column(String(200), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    owner_user_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("user.id", ondelete="CASCADE"), nullable=True
    )
    visibility: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="private"
    )
    current_release_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid,
        ForeignKey(
            "corpus.releases.id",
            use_alter=True,
            deferrable=True,
            initially="DEFERRED",
        ),
        nullable=True,
    )
    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = _created_at()


class DocumentTable(Base):
    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("collection_id", "key", name="uq_documents_collection_key"),
        {"schema": CORPUS_SCHEMA},
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    collection_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("corpus.collections.id", ondelete="CASCADE"), nullable=False
    )
    key: Mapped[str] = mapped_column(String(300), nullable=False)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    source_title: Mapped[str] = mapped_column(Text, nullable=False)
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    attributes: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )


class SectionTable(Base):
    __tablename__ = "sections"
    __table_args__ = (
        UniqueConstraint("document_id", "key", name="uq_sections_document_key"),
        {"schema": CORPUS_SCHEMA},
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    document_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("corpus.documents.id", ondelete="CASCADE"), nullable=False
    )
    key: Mapped[str] = mapped_column(String(400), nullable=False)
    heading: Mapped[str] = mapped_column(Text, nullable=False)
    context_path: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, server_default=text("'{}'::text[]")
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    retrieval_mode: Mapped[str] = mapped_column(String(16), nullable=False)


class SectionRevisionTable(Base):
    __tablename__ = "section_revisions"
    __table_args__ = ({"schema": CORPUS_SCHEMA},)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    section_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("corpus.sections.id", ondelete="CASCADE"), nullable=False
    )
    blocks: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False)
    start_page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    end_page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False)


class ChunkVersionTable(Base):
    __tablename__ = "chunk_versions"
    __table_args__ = (
        Index("ix_chunk_versions_embedding_text_sha256", "embedding_text_sha256"),
        {"schema": CORPUS_SCHEMA},
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    section_revision_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("corpus.section_revisions.id", ondelete="RESTRICT"),
        nullable=False,
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    # From ChunkDraft.context_header, so readers (P3) never rebuild it.
    context_header: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_text_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    start_page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    end_page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    table_key: Mapped[str | None] = mapped_column(String(200), nullable=True)
    term_annotations: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    # none_as_null: Python None is stored as SQL NULL, not as the JSON value null.
    colloquial: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB(none_as_null=True), nullable=True
    )
    chunker_version: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = _created_at()


class ReleaseTable(Base):
    __tablename__ = "releases"
    __table_args__ = (
        UniqueConstraint(
            "collection_id", "number", name="uq_releases_collection_number"
        ),
        CheckConstraint("status IN ('building', 'ready', 'retired')", name="status"),
        {"schema": CORPUS_SCHEMA},
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    collection_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("corpus.collections.id", ondelete="CASCADE"), nullable=False
    )
    number: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    bundle_digest: Mapped[str] = mapped_column(String(64), nullable=False)
    chunker_version: Mapped[str] = mapped_column(String(64), nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(200), nullable=False)
    stats: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB(none_as_null=True), nullable=True
    )
    created_at: Mapped[datetime] = _created_at()
    ready_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    published_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    retired_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class ReleaseChunkTable(Base):
    __tablename__ = "release_chunks"
    __table_args__ = (
        Index(
            "ix_release_chunks_release_revision_ordinal",
            "release_id",
            "section_revision_id",
            "ordinal",
        ),
        Index("ix_release_chunks_chunk_version_id", "chunk_version_id"),
        {"schema": CORPUS_SCHEMA},
    )

    release_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("corpus.releases.id", ondelete="CASCADE"), primary_key=True
    )
    chunk_version_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("corpus.chunk_versions.id", ondelete="RESTRICT"),
        primary_key=True,
    )
    section_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("corpus.sections.id", ondelete="CASCADE"), nullable=False
    )
    section_revision_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("corpus.section_revisions.id", ondelete="RESTRICT"),
        nullable=False,
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    hydrate_strategy: Mapped[str] = mapped_column(String(32), nullable=False)


class GlossaryEntryTable(Base):
    __tablename__ = "glossary_entries"
    __table_args__ = ({"schema": CORPUS_SCHEMA},)

    release_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("corpus.releases.id", ondelete="CASCADE"), primary_key=True
    )
    # P1's bundle validation rejects case-insensitive duplicate terms.
    term: Mapped[str] = mapped_column(Text, primary_key=True)
    data: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)


class ColloquialMappingTable(Base):
    __tablename__ = "colloquial_mappings"
    __table_args__ = ({"schema": CORPUS_SCHEMA},)

    release_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("corpus.releases.id", ondelete="CASCADE"), primary_key=True
    )
    # Bundle order; `key` is not unique (P1 allows "" for title-only leaflet mappings).
    position: Mapped[int] = mapped_column(Integer, primary_key=True)
    key: Mapped[str] = mapped_column(String(300), nullable=False)
    data: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)


class EmbeddingCacheTable(Base):
    __tablename__ = "embedding_cache"
    __table_args__ = ({"schema": CORPUS_SCHEMA},)

    model: Mapped[str] = mapped_column(String(200), primary_key=True)
    embedding_text_sha256: Mapped[str] = mapped_column(String(64), primary_key=True)
    dims: Mapped[int] = mapped_column(Integer, nullable=False)
    vector: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    created_at: Mapped[datetime] = _created_at()
```

`backend/src/pharma_agent/infrastructure/persistence/postgres/metadata.py`:

```python
"""Alembic's view of the database: every table module on one MetaData, and what it manages.

Importing `tables` and `corpus_tables` here registers all tables on `Base.metadata`, so
`migrations/env.py` and the migration test see the complete schema.
"""

from pharma_agent.infrastructure.persistence.postgres import corpus_tables, tables

target_metadata = tables.Base.metadata

# Schemas Alembic owns besides the default one (reported to include_name as None).
MANAGED_SCHEMAS = frozenset({corpus_tables.CORPUS_SCHEMA})


def include_name(name: str | None, type_: str, parent_names: object) -> bool:
    """Alembic `include_name` hook.

    Only the default schema and `corpus` are compared, and tables owned by the LangGraph
    checkpointer are skipped.
    """
    if type_ == "schema":
        return name is None or name in MANAGED_SCHEMAS
    return not (type_ == "table" and name in tables.EXTERNALLY_MANAGED_TABLES)
```

In `backend/src/pharma_agent/infrastructure/persistence/postgres/tables.py`, delete the function at lines 47–49:

```python
def include_name(name: str | None, type_: str, parent_names: object) -> bool:
    """Alembic `include_name` hook: skip tables owned by the LangGraph checkpointer."""
    return not (type_ == "table" and name in EXTERNALLY_MANAGED_TABLES)
```

and keep `EXTERNALLY_MANAGED_TABLES` with its comment.

In `backend/src/pharma_agent/infrastructure/persistence/postgres/migrations/env.py`, replace line 8 and line 15:

```python
from pharma_agent.infrastructure.persistence.postgres.metadata import (
    include_name,
    target_metadata,
)
from pharma_agent.infrastructure.settings import Settings

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)
```

(remove the old `target_metadata = Base.metadata` line) and add `include_schemas=True` to both `context.configure(...)` calls:

```python
def run_migrations_offline() -> None:
    context.configure(
        url=_dsn(),
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True,
        include_schemas=True,
        include_name=include_name,
    )
    with context.begin_transaction():
        context.run_migrations()


def _run_with_connection(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        include_schemas=True,
        include_name=include_name,
    )
    with context.begin_transaction():
        context.run_migrations()
```

- [ ] **Step 4: Run the metadata tests**

Run: `uv run pytest -q tests/infrastructure/test_persistence_metadata.py`
Expected: PASS (4 passed).

- [ ] **Step 5: Write migration `0005`**

`backend/src/pharma_agent/infrastructure/persistence/postgres/migrations/versions/0005_corpus_schema.py`:

```python
"""Corpus schema: collections, documents, sections, immutable section revisions and chunk
versions, releases and the embedding cache (spec 2026-09-13 corpus platform §6.2).

Revision ID: 0005
Create Date: 2026-09-13 12:00:00
"""

from collections.abc import Sequence
from datetime import datetime

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "corpus"


def _created_at(name: str = "created_at") -> sa.Column[datetime]:
    return sa.Column(
        name,
        sa.DateTime(timezone=True),
        server_default=sa.text("now()"),
        nullable=False,
    )


def _optional_at(name: str) -> sa.Column[datetime]:
    return sa.Column(name, sa.DateTime(timezone=True), nullable=True)


def upgrade() -> None:
    op.execute(sa.schema.CreateSchema(SCHEMA))
    op.create_table(
        "collections",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("key", sa.String(length=200), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("owner_user_id", sa.Uuid(), nullable=True),
        sa.Column(
            "visibility", sa.String(length=16), server_default="private", nullable=False
        ),
        sa.Column("current_release_id", sa.Uuid(), nullable=True),
        _created_at(),
        _created_at("updated_at"),
        sa.CheckConstraint(
            "visibility IN ('private', 'public')",
            name=op.f("ck_collections_visibility"),
        ),
        sa.ForeignKeyConstraint(
            ["owner_user_id"],
            ["user.id"],
            name=op.f("fk_collections_owner_user_id_user"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_collections")),
        sa.UniqueConstraint("key", name="uq_collections_key"),
        schema=SCHEMA,
    )
    op.create_table(
        "documents",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("collection_id", sa.Uuid(), nullable=False),
        sa.Column("key", sa.String(length=300), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("source_title", sa.Text(), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=True),
        sa.Column(
            "attributes",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["collection_id"],
            ["corpus.collections.id"],
            name=op.f("fk_documents_collection_id_collections"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_documents")),
        sa.UniqueConstraint("collection_id", "key", name="uq_documents_collection_key"),
        schema=SCHEMA,
    )
    op.create_table(
        "sections",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("key", sa.String(length=400), nullable=False),
        sa.Column("heading", sa.Text(), nullable=False),
        sa.Column(
            "context_path",
            postgresql.ARRAY(sa.Text()),
            server_default=sa.text("'{}'::text[]"),
            nullable=False,
        ),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("retrieval_mode", sa.String(length=16), nullable=False),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["corpus.documents.id"],
            name=op.f("fk_sections_document_id_documents"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_sections")),
        sa.UniqueConstraint("document_id", "key", name="uq_sections_document_key"),
        schema=SCHEMA,
    )
    op.create_table(
        "section_revisions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("section_id", sa.Uuid(), nullable=False),
        sa.Column("blocks", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("start_page", sa.Integer(), nullable=True),
        sa.Column("end_page", sa.Integer(), nullable=True),
        sa.Column("char_count", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["section_id"],
            ["corpus.sections.id"],
            name=op.f("fk_section_revisions_section_id_sections"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_section_revisions")),
        schema=SCHEMA,
    )
    op.create_table(
        "chunk_versions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("section_revision_id", sa.Uuid(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("chunk_text", sa.Text(), nullable=False),
        sa.Column("context_header", sa.Text(), nullable=False),
        sa.Column("embedding_text", sa.Text(), nullable=False),
        sa.Column("embedding_text_sha256", sa.String(length=64), nullable=False),
        sa.Column("start_page", sa.Integer(), nullable=True),
        sa.Column("end_page", sa.Integer(), nullable=True),
        sa.Column("table_key", sa.String(length=200), nullable=True),
        sa.Column(
            "term_annotations",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "colloquial", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column("chunker_version", sa.String(length=64), nullable=False),
        _created_at(),
        sa.ForeignKeyConstraint(
            ["section_revision_id"],
            ["corpus.section_revisions.id"],
            name=op.f("fk_chunk_versions_section_revision_id_section_revisions"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_chunk_versions")),
        schema=SCHEMA,
    )
    op.create_index(
        "ix_chunk_versions_embedding_text_sha256",
        "chunk_versions",
        ["embedding_text_sha256"],
        unique=False,
        schema=SCHEMA,
    )
    op.create_table(
        "releases",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("collection_id", sa.Uuid(), nullable=False),
        sa.Column("number", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("bundle_digest", sa.String(length=64), nullable=False),
        sa.Column("chunker_version", sa.String(length=64), nullable=False),
        sa.Column("embedding_model", sa.String(length=200), nullable=False),
        sa.Column("stats", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        _created_at(),
        _optional_at("ready_at"),
        _optional_at("published_at"),
        _optional_at("retired_at"),
        sa.CheckConstraint(
            "status IN ('building', 'ready', 'retired')",
            name=op.f("ck_releases_status"),
        ),
        sa.ForeignKeyConstraint(
            ["collection_id"],
            ["corpus.collections.id"],
            name=op.f("fk_releases_collection_id_collections"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_releases")),
        sa.UniqueConstraint(
            "collection_id", "number", name="uq_releases_collection_number"
        ),
        schema=SCHEMA,
    )
    op.create_foreign_key(
        op.f("fk_collections_current_release_id_releases"),
        "collections",
        "releases",
        ["current_release_id"],
        ["id"],
        source_schema=SCHEMA,
        referent_schema=SCHEMA,
        deferrable=True,
        initially="DEFERRED",
    )
    op.create_table(
        "release_chunks",
        sa.Column("release_id", sa.Uuid(), nullable=False),
        sa.Column("chunk_version_id", sa.Uuid(), nullable=False),
        sa.Column("section_id", sa.Uuid(), nullable=False),
        sa.Column("section_revision_id", sa.Uuid(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("hydrate_strategy", sa.String(length=32), nullable=False),
        sa.ForeignKeyConstraint(
            ["chunk_version_id"],
            ["corpus.chunk_versions.id"],
            name=op.f("fk_release_chunks_chunk_version_id_chunk_versions"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["release_id"],
            ["corpus.releases.id"],
            name=op.f("fk_release_chunks_release_id_releases"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["section_id"],
            ["corpus.sections.id"],
            name=op.f("fk_release_chunks_section_id_sections"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["section_revision_id"],
            ["corpus.section_revisions.id"],
            name=op.f("fk_release_chunks_section_revision_id_section_revisions"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint(
            "release_id", "chunk_version_id", name=op.f("pk_release_chunks")
        ),
        schema=SCHEMA,
    )
    op.create_index(
        "ix_release_chunks_release_revision_ordinal",
        "release_chunks",
        ["release_id", "section_revision_id", "ordinal"],
        unique=False,
        schema=SCHEMA,
    )
    op.create_index(
        "ix_release_chunks_chunk_version_id",
        "release_chunks",
        ["chunk_version_id"],
        unique=False,
        schema=SCHEMA,
    )
    op.create_table(
        "glossary_entries",
        sa.Column("release_id", sa.Uuid(), nullable=False),
        sa.Column("term", sa.Text(), nullable=False),
        sa.Column("data", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.ForeignKeyConstraint(
            ["release_id"],
            ["corpus.releases.id"],
            name=op.f("fk_glossary_entries_release_id_releases"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("release_id", "term", name=op.f("pk_glossary_entries")),
        schema=SCHEMA,
    )
    op.create_table(
        "colloquial_mappings",
        sa.Column("release_id", sa.Uuid(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("key", sa.String(length=300), nullable=False),
        sa.Column("data", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.ForeignKeyConstraint(
            ["release_id"],
            ["corpus.releases.id"],
            name=op.f("fk_colloquial_mappings_release_id_releases"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "release_id", "position", name=op.f("pk_colloquial_mappings")
        ),
        schema=SCHEMA,
    )
    op.create_table(
        "embedding_cache",
        sa.Column("model", sa.String(length=200), nullable=False),
        sa.Column("embedding_text_sha256", sa.String(length=64), nullable=False),
        sa.Column("dims", sa.Integer(), nullable=False),
        sa.Column("vector", sa.LargeBinary(), nullable=False),
        _created_at(),
        sa.PrimaryKeyConstraint(
            "model", "embedding_text_sha256", name=op.f("pk_embedding_cache")
        ),
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_constraint(
        op.f("fk_collections_current_release_id_releases"),
        "collections",
        schema=SCHEMA,
        type_="foreignkey",
    )
    op.drop_table("embedding_cache", schema=SCHEMA)
    op.drop_table("colloquial_mappings", schema=SCHEMA)
    op.drop_table("glossary_entries", schema=SCHEMA)
    op.drop_table("release_chunks", schema=SCHEMA)
    op.drop_table("releases", schema=SCHEMA)
    op.drop_table("chunk_versions", schema=SCHEMA)
    op.drop_table("section_revisions", schema=SCHEMA)
    op.drop_table("sections", schema=SCHEMA)
    op.drop_table("documents", schema=SCHEMA)
    op.drop_table("collections", schema=SCHEMA)
    op.execute(sa.schema.DropSchema(SCHEMA))
```

- [ ] **Step 6: Run the migration tests against Postgres**

Run: `uv run pytest -q -m integration tests/infrastructure/test_migrations.py`
Expected: PASS (2 passed). If `_diff` reports a difference, fix the migration or the table so both describe the same schema (for example an FK `ondelete` or `deferrable` mismatch), never filter the difference out.

- [ ] **Step 7: Confirm `pharma-agent migrate` still works end to end**

Run: `uv run pytest -q tests/test_cli.py::test_migrate_upgrades_to_head && uv run pytest -q -m integration tests/infrastructure/test_postgres_fixture.py`
Expected: PASS. The CLI command is unchanged; `migrated_dsn` (used by every Postgres test) now reaches `0005` through the same `alembic_config`.

- [ ] **Step 8: Run the full check**

Run: `uv run ruff check src tests && uv run ruff format --check src tests && uv run pyrefly check --min-severity warn && uv run pytest -q && uv run pytest -q -m integration`
Expected: all green.

- [ ] **Step 9: Commit**

```bash
git add src/pharma_agent/infrastructure/persistence/postgres/corpus_tables.py \
  src/pharma_agent/infrastructure/persistence/postgres/metadata.py \
  src/pharma_agent/infrastructure/persistence/postgres/tables.py \
  src/pharma_agent/infrastructure/persistence/postgres/migrations/env.py \
  src/pharma_agent/infrastructure/persistence/postgres/migrations/versions/0005_corpus_schema.py \
  tests/infrastructure/test_persistence_metadata.py \
  tests/infrastructure/test_migrations.py
git commit -m "feat(corpus): add corpus schema tables and migration 0005"
```

The commit message ends with the session attribution trailer.

---

### Task 2: Test support: `FakeEmbedder`, fixture bundle generator and committed fixture

**Files:**
- Modify: `backend/tests/fakes.py` (imports at lines 1-16, new block before `__all__`, `__all__` at lines 174-184)
- Modify: `backend/src/pharma_agent/infrastructure/retrieval/qdrant_adapter.py:29-35` (expose `model` and `dimension`)
- Create: `backend/tests/corpus_fixtures.py`
- Create (generated, committed): `backend/tests/fixtures/knowledge_bundle_small/` (`manifest.json`, `documents.jsonl`, `sections.jsonl`, `glossary.json`, `colloquial_mappings.json`, `embeddings/fake_embedding_4d.jsonl`)
- Test: `backend/tests/test_corpus_fixtures.py`, `backend/tests/infrastructure/test_openai_embedder.py`

**Interfaces:**
- Consumes (P1): `KnowledgeBundle`, `BundleManifest`, `BundleCollection`, `BundleGenerator`, `BundleEmbeddingFile`, `DocumentRecord`, `SectionRecord`, `BlockRecord`, `SourceInfo`, `GlossaryEntry`, `ColloquialMappingRecord`, `DocumentKind`, `BlockKind`, `RetrievalMode`, `model_slug`, `read_bundle`, `write_bundle` (`pharma_agent.domain.corpus.bundle`); `chunk_section` (`chunking`); `sha256_hex` (`identity`); `section_char_count`, `FULL_SECTION_MAX_CHARS` (`hydrate`).
- Produces: `tests.fakes.FAKE_EMBEDDING_MODEL = "fake-embedding-4d"`, `FAKE_EMBEDDING_DIMENSION = 4`, `fake_vector(text: str) -> list[float]`, `FakeEmbedder(*, fail_on_batch: int | None = None)` with `model`, `dimension`, `batches: list[list[str]]`, `async embed(texts) -> list[list[float]]`; `tests.corpus_fixtures.FIXTURE_DIR`, `COLLECTION_KEY`, section key constants, `build_small_bundle() -> KnowledgeBundle`, `small_bundle() -> KnowledgeBundle`, `with_section_text(bundle, section_key, extra) -> KnowledgeBundle`, `regenerate(directory: Path = FIXTURE_DIR) -> None`; `OpenAiEmbedder.model -> str`, `OpenAiEmbedder.dimension -> int`.

Why generated: the embeddings file is keyed by `sha256(embedding_text)`, and `embedding_text` is produced by P1's chunker and enrichment, so a hand-written file could not carry correct keys. The Python data below is the source of truth, `python -m tests.corpus_fixtures` writes it with P1's `write_bundle`, and the committed directory stays available to seed-pipeline's contract test (spec C §12). A test fails whenever the committed copy drifts from the generator.

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_corpus_fixtures.py`:

```python
import struct
from pathlib import Path

import pytest

from pharma_agent.domain.corpus.bundle import (
    BlockKind,
    DocumentKind,
    RetrievalMode,
    read_bundle,
    write_bundle,
)
from pharma_agent.domain.corpus.hydrate import (
    FULL_SECTION_MAX_CHARS,
    section_char_count,
)
from pharma_agent.domain.retrieval.ports import RetrievalError
from tests.corpus_fixtures import (
    BRANDS_SECTION,
    COLLECTION_KEY,
    DOSAGE_SECTION,
    FIXTURE_DIR,
    INTERACTIONS_SECTION,
    LEAFLET_SECTION,
    PHARMACOLOGY_SECTION,
    build_small_bundle,
    small_bundle,
    with_section_text,
)
from tests.fakes import (
    FAKE_EMBEDDING_DIMENSION,
    FAKE_EMBEDDING_MODEL,
    FakeEmbedder,
    fake_vector,
)


def _files(directory: Path) -> list[str]:
    return sorted(
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
    )


def test_committed_fixture_matches_the_generator(tmp_path: Path) -> None:
    target = tmp_path / "bundle"
    target.mkdir()
    write_bundle(build_small_bundle(), target)
    assert read_bundle(target) == small_bundle(), (
        "stale fixture: run `uv run python -m tests.corpus_fixtures` and commit it"
    )
    assert _files(target) == _files(FIXTURE_DIR)


def test_fixture_covers_every_case_the_corpus_tests_need() -> None:
    bundle = small_bundle()
    sections = {section.key: section for section in bundle.sections}
    assert bundle.manifest.collection.key == COLLECTION_KEY
    assert [document.kind for document in bundle.documents] == [
        DocumentKind.DRUG_MONOGRAPH,
        DocumentKind.LEAFLET,
    ]
    assert {block.kind for block in sections[DOSAGE_SECTION].blocks} == {
        BlockKind.PROSE
    }
    assert section_char_count(sections[PHARMACOLOGY_SECTION]) > FULL_SECTION_MAX_CHARS
    assert sections[INTERACTIONS_SECTION].blocks[0].kind is BlockKind.TABLE
    assert sections[BRANDS_SECTION].retrieval is RetrievalMode.INDEX_ONLY
    assert sections[BRANDS_SECTION].blocks[0].kind is BlockKind.INDEX_ENTRIES
    assert [mapping.section_keys for mapping in bundle.colloquial_mappings] == [
        [LEAFLET_SECTION]
    ]
    assert [entry.term for entry in bundle.glossary] == ["NSAID", "G6PD"]
    (embedding_file,) = bundle.manifest.embeddings
    assert (embedding_file.model, embedding_file.dims) == (
        FAKE_EMBEDDING_MODEL,
        FAKE_EMBEDDING_DIMENSION,
    )
    assert embedding_file.file == "embeddings/fake_embedding_4d.jsonl"
    assert len(bundle.embeddings[FAKE_EMBEDDING_MODEL]) >= len(bundle.sections)


def test_with_section_text_changes_only_that_section() -> None:
    bundle = small_bundle()
    edited = with_section_text(bundle, DOSAGE_SECTION, "Ghi chú mới.")
    before = {section.key: section for section in bundle.sections}
    after = {section.key: section for section in edited.sections}
    assert after[DOSAGE_SECTION].blocks[0].markdown.endswith("\n\nGhi chú mới.")
    assert all(after[key] == before[key] for key in before if key != DOSAGE_SECTION)
    with pytest.raises(KeyError):
        with_section_text(bundle, "drug:missing:section", "x")


async def test_fake_embedder_is_deterministic_and_float32_exact() -> None:
    embedder = FakeEmbedder()
    first = await embedder.embed(["a", "b"])
    second = await embedder.embed(["a"])
    assert first[0] == second[0] == fake_vector("a")
    assert len(first[1]) == FAKE_EMBEDDING_DIMENSION
    packed = struct.pack("<4f", *first[1])
    assert list(struct.unpack("<4f", packed)) == first[1]
    assert embedder.batches == [["a", "b"], ["a"]]
    assert (embedder.model, embedder.dimension) == ("fake-embedding-4d", 4)


async def test_fake_embedder_fails_on_the_requested_batch() -> None:
    embedder = FakeEmbedder(fail_on_batch=2)
    await embedder.embed(["a"])
    with pytest.raises(RetrievalError):
        await embedder.embed(["b"])
    assert embedder.batches == [["a"], ["b"]]
```

`backend/tests/infrastructure/test_openai_embedder.py`:

```python
from pharma_agent.infrastructure.retrieval.qdrant_adapter import OpenAiEmbedder


def test_openai_embedder_exposes_model_and_dimension() -> None:
    embedder = OpenAiEmbedder(object(), model="qwen3-embedding:4b-fp16", dimension=2560)
    assert (embedder.model, embedder.dimension) == ("qwen3-embedding:4b-fp16", 2560)
```

- [ ] **Step 2: Run the tests and watch them fail**

Run: `uv run pytest -q tests/test_corpus_fixtures.py tests/infrastructure/test_openai_embedder.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'tests.corpus_fixtures'` and `AttributeError: 'OpenAiEmbedder' object has no attribute 'model'`.

- [ ] **Step 3: Add `model` and `dimension` to `OpenAiEmbedder`**

In `backend/src/pharma_agent/infrastructure/retrieval/qdrant_adapter.py`, directly after `OpenAiEmbedder.__init__` (line 35) add:

```python
    @property
    def model(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension
```

- [ ] **Step 4: Add the fake embedder**

In `backend/tests/fakes.py`, add `import hashlib` as the first import line, and add this block directly above `MONOGRAPH_SKILL_MD`:

```python
FAKE_EMBEDDING_MODEL = "fake-embedding-4d"
FAKE_EMBEDDING_DIMENSION = 4


def fake_vector(text: str) -> list[float]:
    """Deterministic vector from sha256(text).

    Components are k/16 (k in 1..16), so they survive float32 encoding exactly and are never
    all zero.
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [(digest[i] % 16 + 1) / 16 for i in range(FAKE_EMBEDDING_DIMENSION)]


class FakeEmbedder:
    """Implements `pharma_agent.domain.corpus.ports.Embedder`; records every batch."""

    def __init__(self, *, fail_on_batch: int | None = None) -> None:
        self.batches: list[list[str]] = []
        self._fail_on_batch = fail_on_batch

    @property
    def model(self) -> str:
        return FAKE_EMBEDDING_MODEL

    @property
    def dimension(self) -> int:
        return FAKE_EMBEDDING_DIMENSION

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        if self._fail_on_batch is not None and len(self.batches) == self._fail_on_batch:
            raise RetrievalError("fake embedding endpoint unavailable")
        return [fake_vector(text) for text in texts]
```

and extend `__all__`:

```python
__all__ = [
    "FAKE_EMBEDDING_DIMENSION",
    "FAKE_EMBEDDING_MODEL",
    "FakeEmbedder",
    "FakeHydrator",
    "FakeLlm",
    "FakeReranker",
    "FakeRetriever",
    "FakeSkillCatalog",
    "LlmError",
    "RetrievalError",
    "build_deps",
    "fake_vector",
    "monograph_skill",
]
```

- [ ] **Step 5: Write the fixture generator**

`backend/tests/corpus_fixtures.py`:

```python
"""The small knowledge bundle shared by the backend tests and seed-pipeline's contract test.

`build_small_bundle()` is the source of truth. After changing it, or after P1's chunker or
enrichment changes, regenerate the committed copy with `uv run python -m tests.corpus_fixtures`;
`tests/test_corpus_fixtures.py` fails while the committed copy is stale.
"""

import shutil
from pathlib import Path

from pharma_agent.domain.corpus.bundle import (
    BlockKind,
    BlockRecord,
    BundleCollection,
    BundleEmbeddingFile,
    BundleGenerator,
    BundleManifest,
    ColloquialMappingRecord,
    DocumentKind,
    DocumentRecord,
    GlossaryEntry,
    KnowledgeBundle,
    RetrievalMode,
    SectionRecord,
    SourceInfo,
    model_slug,
    read_bundle,
    write_bundle,
)
from pharma_agent.domain.corpus.chunking import chunk_section
from pharma_agent.domain.corpus.identity import sha256_hex
from tests.fakes import FAKE_EMBEDDING_DIMENSION, FAKE_EMBEDDING_MODEL, fake_vector

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "knowledge_bundle_small"
COLLECTION_KEY = "formulary"
COLLECTION_TITLE = "Dược thư Quốc gia Việt Nam (fixture)"

PARACETAMOL = "drug:paracetamol"
LEAFLET = "leaflet:ankhang:thuoc-giam-dau:panadol-extra"
DOSAGE_SECTION = "drug:paracetamol:lieu-luong-va-cach-dung"
PHARMACOLOGY_SECTION = "drug:paracetamol:duoc-ly-va-co-che-tac-dung"
INTERACTIONS_SECTION = "drug:paracetamol:tuong-tac-thuoc"
BRANDS_SECTION = "drug:paracetamol:biet-duoc"
LEAFLET_SECTION = "leaflet:ankhang:thuoc-giam-dau:panadol-extra:cong-dung"

FORMULARY_SOURCE = SourceInfo(title="Dược thư Quốc gia Việt Nam 2022", url=None)

INTERACTIONS_TABLE = """| Thuốc phối hợp | Tương tác | Xử trí |
| --- | --- | --- |
| Warfarin | Tăng tác dụng chống đông khi dùng paracetamol liều cao kéo dài | Theo dõi INR |
| Rượu | Tăng nguy cơ độc tính trên gan | Tránh uống rượu |
| NSAID | Tăng nguy cơ tác dụng phụ trên thận khi phối hợp lâu dài | Hạn chế phối hợp kéo dài |"""

BRAND_INDEX = """Efferalgan (UPSA) - viên sủi 500 mg
Hapacol (DHG Pharma) - gói bột 250 mg
Panadol (GSK) - viên nén 500 mg
Tylenol (Johnson & Johnson) - viên nén 325 mg"""


def _pharmacology_paragraphs(first: int, count: int) -> str:
    return "\n\n".join(
        " ".join(
            f"Đoạn {paragraph}, ý {sentence}: paracetamol ức chế tổng hợp prostaglandin "
            "ở hệ thần kinh trung ương nên hạ sốt và giảm đau, còn tác dụng chống viêm "
            "ngoại vi rất yếu."
            for sentence in range(1, 7)
        )
        for paragraph in range(first, first + count)
    )


def _documents() -> list[DocumentRecord]:
    return [
        DocumentRecord(
            key=PARACETAMOL,
            kind=DocumentKind.DRUG_MONOGRAPH,
            title="Paracetamol",
            source=FORMULARY_SOURCE,
            attributes={"atc": "N02BE01"},
        ),
        DocumentRecord(
            key=LEAFLET,
            kind=DocumentKind.LEAFLET,
            title="Panadol Extra",
            source=SourceInfo(
                title="Nhà thuốc An Khang",
                url="https://www.nhathuocankhang.com/thuoc-giam-dau/panadol-extra",
            ),
            attributes={"category": "thuoc-giam-dau"},
        ),
    ]


def _sections() -> list[SectionRecord]:
    return [
        SectionRecord(
            key=DOSAGE_SECTION,
            document_key=PARACETAMOL,
            heading="Liều lượng và cách dùng",
            context_path=["Liều lượng và cách dùng"],
            ordinal=1,
            start_page=120,
            end_page=120,
            blocks=[
                BlockRecord(
                    kind=BlockKind.PROSE,
                    markdown="Người lớn: uống 500 mg đến 1 g mỗi 4 đến 6 giờ khi cần, "
                    "tối đa 4 g mỗi ngày.",
                    start_page=120,
                    end_page=120,
                ),
                BlockRecord(
                    kind=BlockKind.PROSE,
                    markdown="Trẻ em: 10 đến 15 mg/kg mỗi 4 đến 6 giờ, không quá 5 lần "
                    "trong 24 giờ. Người suy gan cần giảm liều.",
                    start_page=120,
                    end_page=120,
                ),
            ],
        ),
        SectionRecord(
            key=PHARMACOLOGY_SECTION,
            document_key=PARACETAMOL,
            heading="Dược lý và cơ chế tác dụng",
            context_path=["Dược lý và cơ chế tác dụng"],
            ordinal=2,
            start_page=121,
            end_page=124,
            blocks=[
                BlockRecord(
                    kind=BlockKind.PROSE,
                    markdown=_pharmacology_paragraphs(1, 5),
                    start_page=121,
                    end_page=121,
                ),
                BlockRecord(
                    kind=BlockKind.PROSE,
                    markdown=_pharmacology_paragraphs(6, 5),
                    start_page=122,
                    end_page=122,
                ),
                BlockRecord(
                    kind=BlockKind.PROSE,
                    markdown=_pharmacology_paragraphs(11, 5),
                    start_page=123,
                    end_page=123,
                ),
                BlockRecord(
                    kind=BlockKind.PROSE,
                    markdown=_pharmacology_paragraphs(16, 5)
                    + "\n\nNgười thiếu G6PD dùng liều điều trị thường dung nạp tốt; "
                    "quá liều gây hoại tử tế bào gan do tích lũy "
                    "N-acetyl-p-benzoquinon imin.",
                    start_page=124,
                    end_page=124,
                ),
            ],
        ),
        SectionRecord(
            key=INTERACTIONS_SECTION,
            document_key=PARACETAMOL,
            heading="Tương tác thuốc",
            context_path=["Tương tác thuốc"],
            ordinal=3,
            start_page=125,
            end_page=125,
            blocks=[
                BlockRecord(
                    kind=BlockKind.TABLE,
                    markdown=INTERACTIONS_TABLE,
                    start_page=125,
                    end_page=125,
                    table_key="paracetamol-tuong-tac-thuoc",
                )
            ],
        ),
        SectionRecord(
            key=BRANDS_SECTION,
            document_key=PARACETAMOL,
            heading="Biệt dược",
            context_path=["Biệt dược"],
            ordinal=4,
            start_page=126,
            end_page=126,
            retrieval=RetrievalMode.INDEX_ONLY,
            blocks=[
                BlockRecord(
                    kind=BlockKind.INDEX_ENTRIES,
                    markdown=BRAND_INDEX,
                    start_page=126,
                    end_page=126,
                )
            ],
        ),
        SectionRecord(
            key=LEAFLET_SECTION,
            document_key=LEAFLET,
            heading="Công dụng",
            context_path=["Công dụng"],
            ordinal=1,
            blocks=[
                BlockRecord(
                    kind=BlockKind.PROSE,
                    markdown="Panadol Extra chứa paracetamol 500 mg và cafein 65 mg, "
                    "dùng giảm đau đầu, đau răng, đau bụng kinh và hạ sốt.",
                )
            ],
        ),
    ]


def _glossary() -> list[GlossaryEntry]:
    return [
        GlossaryEntry(
            term="NSAID",
            case_sensitive=True,
            vietnamese_expansions=["thuốc chống viêm không steroid"],
            english_expansions=["non-steroidal anti-inflammatory drug"],
            category="drug_class",
            confidence="high",
            source="fixture",
        ),
        GlossaryEntry(
            term="G6PD",
            case_sensitive=True,
            vietnamese_expansions=["glucose-6-phosphat dehydrogenase"],
            english_expansions=["glucose-6-phosphate dehydrogenase"],
            category="enzyme",
            confidence="high",
            source="fixture",
        ),
    ]


def _colloquial_mappings() -> list[ColloquialMappingRecord]:
    return [
        ColloquialMappingRecord(
            key="panadol-extra",
            aliases=["panadol đỏ", "thuốc giảm đau viên đỏ"],
            visual_sign="viên nén dài bao phim màu đỏ",
            product_names=["Panadol Extra"],
            section_keys=[LEAFLET_SECTION],
        )
    ]


def build_small_bundle() -> KnowledgeBundle:
    documents = _documents()
    sections = _sections()
    glossary = _glossary()
    mappings = _colloquial_mappings()
    documents_by_key = {document.key: document for document in documents}
    vectors: dict[str, list[float]] = {}
    for section in sections:
        for draft in chunk_section(
            documents_by_key[section.document_key], section, glossary, mappings
        ):
            vectors[draft.embedding_text_sha256] = fake_vector(draft.embedding_text)
    manifest = BundleManifest(
        schema_version="knowledge-bundle/v1",
        collection=BundleCollection(key=COLLECTION_KEY, title=COLLECTION_TITLE),
        generator=BundleGenerator(
            name="pharma-agent-tests", version="1", build_id="knowledge-bundle-small"
        ),
        source_digests={
            "source_pdf_sha256": sha256_hex("fixture source pdf"),
            "snapshot_sha256": sha256_hex("fixture ankhang snapshot"),
        },
        document_count=len(documents),
        section_count=len(sections),
        files={},
        embeddings=[
            BundleEmbeddingFile(
                model=FAKE_EMBEDDING_MODEL,
                dims=FAKE_EMBEDDING_DIMENSION,
                file=f"embeddings/{model_slug(FAKE_EMBEDDING_MODEL)}.jsonl",
            )
        ],
    )
    return KnowledgeBundle(
        manifest=manifest,
        documents=documents,
        sections=sections,
        glossary=glossary,
        colloquial_mappings=mappings,
        embeddings={FAKE_EMBEDDING_MODEL: vectors},
    )


def small_bundle() -> KnowledgeBundle:
    """The committed fixture, read and validated by P1's `read_bundle`."""
    return read_bundle(FIXTURE_DIR)


def with_section_text(
    bundle: KnowledgeBundle, section_key: str, extra: str
) -> KnowledgeBundle:
    """Copy of `bundle` whose section's first block ends with `extra` (a content change)."""
    if all(section.key != section_key for section in bundle.sections):
        raise KeyError(section_key)
    sections: list[SectionRecord] = []
    for section in bundle.sections:
        if section.key == section_key:
            first, *rest = section.blocks
            edited = first.model_copy(
                update={"markdown": f"{first.markdown}\n\n{extra}"}
            )
            section = section.model_copy(update={"blocks": [edited, *rest]})
        sections.append(section)
    return bundle.model_copy(update={"sections": sections})


def regenerate(directory: Path = FIXTURE_DIR) -> None:
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    write_bundle(build_small_bundle(), directory)


if __name__ == "__main__":
    regenerate()
```

`files={}`, `document_count`, `section_count` and the `embeddings` entry are placeholders: P1's `write_bundle` validates the content, then rebuilds `files`, both counts and the `embeddings` list (from `bundle.embeddings`) before writing `manifest.json`. `BundleManifest` has no validators, so the placeholders construct fine under `strict=True`. `context_path` leaves out the document title because P1's `build_context_header(title, context_path)` already prefixes it.

- [ ] **Step 6: Generate the committed fixture**

Run: `uv run python -m tests.corpus_fixtures && ls -R tests/fixtures/knowledge_bundle_small`
Expected: `manifest.json documents.jsonl sections.jsonl glossary.json colloquial_mappings.json` and `embeddings/fake_embedding_4d.jsonl`. Open `manifest.json` and check `document_count` is 2, `section_count` is 5 and `embeddings` lists `fake-embedding-4d` with `dims` 4.

- [ ] **Step 7: Run the tests**

Run: `uv run pytest -q tests/test_corpus_fixtures.py tests/infrastructure/test_openai_embedder.py`
Expected: PASS (6 passed). If `section_char_count(...) > FULL_SECTION_MAX_CHARS` fails, raise `_pharmacology_paragraphs` counts until it passes and regenerate; do not lower the assertion.

- [ ] **Step 8: Run the full check**

Run: `uv run ruff check src tests && uv run ruff format --check src tests && uv run pyrefly check --min-severity warn && uv run pytest -q`
Expected: all green.

- [ ] **Step 9: Commit**

```bash
git add tests/fakes.py tests/corpus_fixtures.py \
  tests/fixtures/knowledge_bundle_small \
  tests/test_corpus_fixtures.py \
  tests/infrastructure/test_openai_embedder.py \
  src/pharma_agent/infrastructure/retrieval/qdrant_adapter.py
git commit -m "test(corpus): add fake embedder and generated small knowledge bundle fixture"
```

The commit message ends with the session attribution trailer.

---

### Task 3: Domain corpus models and ports

**Files:**
- Create: `backend/src/pharma_agent/domain/corpus/models.py`
- Create: `backend/src/pharma_agent/domain/corpus/ports.py`
- Test: `backend/tests/domain/corpus/test_corpus_models.py`

**Interfaces:**
- Consumes (P1): `KnowledgeBundle`, `BlockKind`, `BlockRecord`, `ColloquialMappingRecord`, `DocumentKind`, `GlossaryEntry`, `RetrievalMode` (`bundle`); `CHUNKER_VERSION`, `chunk_section` (`chunking`); `hydrate_strategy_for`, `section_char_count` (`hydrate`); `CORPUS_NAMESPACE`, `canonical_json`, `section_revision_id`, `sha256_hex` (`identity`). Existing: `HydrateStrategy`, `TermAnnotation`, `ColloquialMapping` (`domain/retrieval/models.py`), `DomainError` (`domain/shared/errors.py`). Tests: `small_bundle`, `with_section_text`, section constants (Task 2), `FakeEmbedder`, `NOW` (`tests/fakes.py`).
- Produces: every name listed for `models.py` and `ports.py` in "Interfaces pinned by this plan" above, with exactly those signatures.

- [ ] **Step 1: Write the failing tests**

`backend/tests/domain/corpus/test_corpus_models.py`:

```python
import uuid

import pytest

from pharma_agent.domain.corpus.bundle import BlockKind
from pharma_agent.domain.corpus.chunking import CHUNKER_VERSION
from pharma_agent.domain.corpus.identity import CORPUS_NAMESPACE
from pharma_agent.domain.corpus.models import (
    CorpusImportError,
    Release,
    ReleaseStatus,
    Visibility,
    build_snapshot,
    bundle_digest,
    collection_id_for,
    document_id_for,
    releases_to_retire,
    section_id_for,
)
from pharma_agent.domain.corpus.ports import Embedder
from pharma_agent.domain.retrieval.models import HydrateStrategy
from tests.corpus_fixtures import (
    BRANDS_SECTION,
    DOSAGE_SECTION,
    INTERACTIONS_SECTION,
    LEAFLET,
    LEAFLET_SECTION,
    PARACETAMOL,
    PHARMACOLOGY_SECTION,
    small_bundle,
    with_section_text,
)
from tests.fakes import FAKE_EMBEDDING_MODEL, NOW, FakeEmbedder


def test_snapshot_maps_the_fixture_to_rows() -> None:
    bundle = small_bundle()
    snapshot = build_snapshot(bundle)

    assert snapshot.collection.id == collection_id_for("formulary")
    assert snapshot.collection.visibility is Visibility.PRIVATE
    assert snapshot.collection.current_release_id is None
    assert [document.key for document in snapshot.documents] == [PARACETAMOL, LEAFLET]
    assert len(snapshot.sections) == len(snapshot.revisions) == 5
    assert len(snapshot.chunks) == len(snapshot.release_chunks) > 5
    assert {chunk.chunker_version for chunk in snapshot.chunks} == {CHUNKER_VERSION}
    assert snapshot.chunker_version == CHUNKER_VERSION
    dosage_section_id = next(s.id for s in snapshot.sections if s.key == DOSAGE_SECTION)
    dosage_revision_id = next(
        r.id for r in snapshot.revisions if r.section_id == dosage_section_id
    )
    # P1 build_context_header: title and context_path lines joined with "\n> ".
    assert {
        chunk.context_header
        for chunk in snapshot.chunks
        if chunk.section_revision_id == dosage_revision_id
    } == {"Paracetamol\n> Liều lượng và cách dùng"}
    assert snapshot.bundle_digest == bundle_digest(bundle)

    section_keys = {section.id: section.key for section in snapshot.sections}
    strategies: dict[str, set[HydrateStrategy]] = {}
    for placement in snapshot.release_chunks:
        strategies.setdefault(section_keys[placement.section_id], set()).add(
            placement.hydrate_strategy
        )
    assert strategies[BRANDS_SECTION] == {HydrateStrategy.SEARCH_ONLY}
    assert strategies[PHARMACOLOGY_SECTION] == {HydrateStrategy.CHUNK_WINDOW}
    for key in (DOSAGE_SECTION, INTERACTIONS_SECTION, LEAFLET_SECTION):
        assert strategies[key] == {HydrateStrategy.FULL_SECTION}

    pharmacology_id = next(
        section.id for section in snapshot.sections if section.key == PHARMACOLOGY_SECTION
    )
    ordinals = [
        placement.ordinal
        for placement in snapshot.release_chunks
        if placement.section_id == pharmacology_id
    ]
    assert len(ordinals) > 1 and ordinals == sorted(ordinals) and ordinals[0] == 1

    chunks = {chunk.id: chunk for chunk in snapshot.chunks}
    kinds = {chunks[p.chunk_version_id].kind for p in snapshot.release_chunks}
    assert {BlockKind.PROSE, BlockKind.TABLE, BlockKind.INDEX_ENTRIES} <= kinds
    leaflet_id = next(s.id for s in snapshot.sections if s.key == LEAFLET_SECTION)
    leaflet_chunks = [
        chunks[p.chunk_version_id]
        for p in snapshot.release_chunks
        if p.section_id == leaflet_id
    ]
    assert all(
        chunk.colloquial is not None and chunk.colloquial.key == "panadol-extra"
        for chunk in leaflet_chunks
    )

    assert set(snapshot.embedding_texts()) == set(bundle.embeddings[FAKE_EMBEDDING_MODEL])
    stats = snapshot.stats()
    assert (stats.documents, stats.sections, stats.section_revisions) == (2, 5, 5)
    assert (stats.glossary_entries, stats.colloquial_mappings) == (2, 1)
    assert stats.chunks == len(snapshot.release_chunks)
    assert stats.embeddings_computed == stats.points_upserted == 0


def test_snapshot_ids_are_deterministic_and_follow_content() -> None:
    first = build_snapshot(small_bundle())
    assert build_snapshot(small_bundle()) == first

    edited = build_snapshot(
        with_section_text(small_bundle(), DOSAGE_SECTION, "Ghi chú mới.")
    )
    assert {s.id for s in edited.sections} == {s.id for s in first.sections}
    assert len({r.id for r in first.revisions} - {r.id for r in edited.revisions}) == 1
    dosage_id = next(s.id for s in first.sections if s.key == DOSAGE_SECTION)
    untouched_before = {
        p.chunk_version_id for p in first.release_chunks if p.section_id != dosage_id
    }
    untouched_after = {
        p.chunk_version_id for p in edited.release_chunks if p.section_id != dosage_id
    }
    assert untouched_before == untouched_after
    assert {p.chunk_version_id for p in first.release_chunks} != {
        p.chunk_version_id for p in edited.release_chunks
    }


def test_bundle_digest_tracks_content_not_embeddings() -> None:
    bundle = small_bundle()
    digest = bundle_digest(bundle)
    assert len(digest) == 64
    assert bundle_digest(bundle.model_copy(update={"embeddings": {}})) == digest
    assert (
        bundle_digest(with_section_text(bundle, DOSAGE_SECTION, "Ghi chú.")) != digest
    )


def test_snapshot_rejects_unknown_document_reference() -> None:
    bundle = small_bundle()
    broken = bundle.model_copy(
        update={
            "sections": [
                bundle.sections[0].model_copy(update={"document_key": "drug:missing"})
            ]
        }
    )
    with pytest.raises(CorpusImportError, match="drug:missing"):
        build_snapshot(broken)


def test_identity_helpers_are_uuid5_in_the_corpus_namespace() -> None:
    collection_id = collection_id_for("formulary")
    document_id = document_id_for(collection_id, PARACETAMOL)
    assert collection_id == uuid.uuid5(CORPUS_NAMESPACE, "collection\x1fformulary")
    assert document_id == uuid.uuid5(
        CORPUS_NAMESPACE, f"document\x1f{collection_id}\x1f{PARACETAMOL}"
    )
    assert section_id_for(document_id, DOSAGE_SECTION) == uuid.uuid5(
        CORPUS_NAMESPACE, f"section\x1f{document_id}\x1f{DOSAGE_SECTION}"
    )


def _release(number: int, status: ReleaseStatus = ReleaseStatus.READY) -> Release:
    return Release(
        id=uuid.uuid4(),
        collection_id=collection_id_for("formulary"),
        number=number,
        status=status,
        bundle_digest="0" * 64,
        chunker_version=CHUNKER_VERSION,
        embedding_model=FAKE_EMBEDDING_MODEL,
        created_at=NOW,
    )


def test_releases_to_retire_keeps_current_and_newest() -> None:
    r1, r3, r4, r5 = _release(1), _release(3), _release(4), _release(5)
    r2 = _release(2, ReleaseStatus.RETIRED)
    releases = [r1, r2, r3, r4, r5]
    assert releases_to_retire(releases, r1.id, keep=2) == [r3]
    assert releases_to_retire(releases, r1.id, keep=0) == [r5, r4, r3]
    assert releases_to_retire(releases, None, keep=10) == []
    with pytest.raises(ValueError, match="keep"):
        releases_to_retire(releases, None, keep=-1)


async def test_fake_embedder_satisfies_the_embedder_port() -> None:
    embedder: Embedder = FakeEmbedder()
    assert (embedder.model, embedder.dimension) == (FAKE_EMBEDDING_MODEL, 4)
    assert len((await embedder.embed(["x"]))[0]) == embedder.dimension
```

- [ ] **Step 2: Run the tests and watch them fail**

Run: `uv run pytest -q tests/domain/corpus/test_corpus_models.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'pharma_agent.domain.corpus.models'`.

- [ ] **Step 3: Write `models.py`**

`backend/src/pharma_agent/domain/corpus/models.py`:

```python
"""Corpus collections, releases and the row-shaped snapshot of a knowledge bundle.

Framework-free. `build_snapshot` turns a `KnowledgeBundle` into the rows of spec C §6.2 using
the chunker, enrichment, hydrate policy and identity functions of this package, so the
repository only stores what the domain computed.
"""

import uuid
from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from pharma_agent.domain.corpus.bundle import (
    BlockKind,
    BlockRecord,
    ColloquialMappingRecord,
    DocumentKind,
    GlossaryEntry,
    KnowledgeBundle,
    RetrievalMode,
)
from pharma_agent.domain.corpus.chunking import CHUNKER_VERSION, chunk_section
from pharma_agent.domain.corpus.hydrate import hydrate_strategy_for, section_char_count
from pharma_agent.domain.corpus.identity import (
    CORPUS_NAMESPACE,
    canonical_json,
    section_revision_id,
    sha256_hex,
)
from pharma_agent.domain.retrieval.models import (
    ColloquialMapping,
    HydrateStrategy,
    TermAnnotation,
)
from pharma_agent.domain.shared.errors import DomainError


class Visibility(StrEnum):
    PRIVATE = "private"
    PUBLIC = "public"


class ReleaseStatus(StrEnum):
    BUILDING = "building"
    READY = "ready"
    RETIRED = "retired"


class CorpusError(DomainError):
    code = "CORPUS_ERROR"


class CorpusImportError(CorpusError):
    code = "CORPUS_IMPORT_FAILED"


class CollectionNotFound(CorpusError):
    code = "COLLECTION_NOT_FOUND"


class ReleaseNotFound(CorpusError):
    code = "RELEASE_NOT_FOUND"


class ReleaseNotPublishable(CorpusError):
    code = "RELEASE_NOT_PUBLISHABLE"


class NoEarlierRelease(CorpusError):
    code = "NO_EARLIER_RELEASE"


class IndexMismatch(CorpusError):
    code = "CORPUS_INDEX_MISMATCH"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class Collection(_Frozen):
    id: uuid.UUID
    key: str
    title: str
    owner_user_id: uuid.UUID | None = None
    visibility: Visibility = Visibility.PRIVATE
    current_release_id: uuid.UUID | None = None


class ReleaseStats(_Frozen):
    documents: int = 0
    sections: int = 0
    section_revisions: int = 0
    chunks: int = 0
    glossary_entries: int = 0
    colloquial_mappings: int = 0
    embeddings_cached: int = 0
    embeddings_from_bundle: int = 0
    embeddings_computed: int = 0
    points_upserted: int = 0
    points_updated: int = 0


class Release(_Frozen):
    id: uuid.UUID
    collection_id: uuid.UUID
    number: int
    status: ReleaseStatus
    bundle_digest: str
    chunker_version: str
    embedding_model: str
    stats: ReleaseStats | None = None
    created_at: datetime
    ready_at: datetime | None = None
    published_at: datetime | None = None
    retired_at: datetime | None = None


class ReleaseSummary(_Frozen):
    collection_key: str
    release: Release
    chunk_count: int
    current: bool


class Document(_Frozen):
    id: uuid.UUID
    collection_id: uuid.UUID
    key: str
    kind: DocumentKind
    title: str
    source_title: str
    source_url: str | None
    attributes: dict[str, str | int | float | bool | None] = Field(
        default_factory=dict
    )


class Section(_Frozen):
    id: uuid.UUID
    document_id: uuid.UUID
    key: str
    heading: str
    context_path: list[str]
    ordinal: int
    retrieval_mode: RetrievalMode


class SectionRevision(_Frozen):
    id: uuid.UUID
    section_id: uuid.UUID
    blocks: list[BlockRecord]
    start_page: int | None
    end_page: int | None
    char_count: int


class ChunkVersion(_Frozen):
    id: uuid.UUID
    section_revision_id: uuid.UUID
    ordinal: int
    kind: BlockKind
    chunk_text: str
    context_header: str
    embedding_text: str
    embedding_text_sha256: str
    start_page: int | None
    end_page: int | None
    table_key: str | None
    term_annotations: list[TermAnnotation] = Field(default_factory=list)
    colloquial: ColloquialMapping | None = None
    chunker_version: str


class ReleaseChunk(_Frozen):
    chunk_version_id: uuid.UUID
    section_id: uuid.UUID
    section_revision_id: uuid.UUID
    ordinal: int
    hydrate_strategy: HydrateStrategy


class CorpusSnapshot(_Frozen):
    """Everything one import writes, computed before touching storage."""

    collection: Collection
    bundle_digest: str
    chunker_version: str
    documents: list[Document]
    sections: list[Section]
    revisions: list[SectionRevision]
    chunks: list[ChunkVersion]
    release_chunks: list[ReleaseChunk]
    glossary: list[GlossaryEntry]
    colloquial_mappings: list[ColloquialMappingRecord]

    def embedding_texts(self) -> dict[str, str]:
        return {chunk.embedding_text_sha256: chunk.embedding_text for chunk in self.chunks}

    def stats(self) -> ReleaseStats:
        return ReleaseStats(
            documents=len(self.documents),
            sections=len(self.sections),
            section_revisions=len(self.revisions),
            chunks=len(self.release_chunks),
            glossary_entries=len(self.glossary),
            colloquial_mappings=len(self.colloquial_mappings),
        )


class IndexItem(_Frozen):
    """What the vector index stores for one chunk version; payload carries no text."""

    chunk_version_id: uuid.UUID
    collection_id: uuid.UUID
    document_id: uuid.UUID
    section_id: uuid.UUID
    section_revision_id: uuid.UUID
    kind: BlockKind
    embedding_text: str
    embedding_text_sha256: str
    release_ids: list[uuid.UUID]


class PurgeResult(_Frozen):
    release_chunks_deleted: int = 0
    chunk_versions_deleted: int = 0
    chunk_versions_kept: int = 0
    section_revisions_deleted: int = 0


def collection_id_for(key: str) -> uuid.UUID:
    return uuid.uuid5(CORPUS_NAMESPACE, f"collection\x1f{key}")


def document_id_for(collection_id: uuid.UUID, key: str) -> uuid.UUID:
    return uuid.uuid5(CORPUS_NAMESPACE, f"document\x1f{collection_id}\x1f{key}")


def section_id_for(document_id: uuid.UUID, key: str) -> uuid.UUID:
    return uuid.uuid5(CORPUS_NAMESPACE, f"section\x1f{document_id}\x1f{key}")


def bundle_digest(bundle: KnowledgeBundle) -> str:
    """sha256 of the knowledge content; manifest and precomputed vectors are excluded."""
    content = {
        "documents": [record.model_dump(mode="json") for record in bundle.documents],
        "sections": [record.model_dump(mode="json") for record in bundle.sections],
        "glossary": [entry.model_dump(mode="json") for entry in bundle.glossary],
        "colloquial_mappings": [
            mapping.model_dump(mode="json") for mapping in bundle.colloquial_mappings
        ],
    }
    return sha256_hex(canonical_json(content))


def build_snapshot(bundle: KnowledgeBundle) -> CorpusSnapshot:
    """Rows for one release of `bundle`. Raises CorpusImportError on broken references."""
    records = {record.key: record for record in bundle.documents}
    problems = [
        f"sections.jsonl: section {section.key} references unknown document "
        f"{section.document_key}"
        for section in bundle.sections
        if section.document_key not in records
    ]
    if problems:
        raise CorpusImportError("; ".join(problems))

    manifest_collection = bundle.manifest.collection
    collection = Collection(
        id=collection_id_for(manifest_collection.key),
        key=manifest_collection.key,
        title=manifest_collection.title,
    )
    documents = [
        Document(
            id=document_id_for(collection.id, record.key),
            collection_id=collection.id,
            key=record.key,
            kind=record.kind,
            title=record.title,
            source_title=record.source.title,
            source_url=record.source.url,
            attributes=dict(record.attributes),
        )
        for record in bundle.documents
    ]
    document_ids = {document.key: document.id for document in documents}

    sections: list[Section] = []
    revisions: list[SectionRevision] = []
    chunks: dict[uuid.UUID, ChunkVersion] = {}
    release_chunks: list[ReleaseChunk] = []
    for record in bundle.sections:
        section_id = section_id_for(document_ids[record.document_key], record.key)
        sections.append(
            Section(
                id=section_id,
                document_id=document_ids[record.document_key],
                key=record.key,
                heading=record.heading,
                context_path=list(record.context_path),
                ordinal=record.ordinal,
                retrieval_mode=record.retrieval,
            )
        )
        revision_id = section_revision_id(record.key, record.blocks)
        revisions.append(
            SectionRevision(
                id=revision_id,
                section_id=section_id,
                blocks=list(record.blocks),
                start_page=record.start_page,
                end_page=record.end_page,
                char_count=section_char_count(record),
            )
        )
        strategy = hydrate_strategy_for(record)
        drafts = chunk_section(
            records[record.document_key],
            record,
            bundle.glossary,
            bundle.colloquial_mappings,
        )
        for draft in drafts:
            if draft.chunk_version_id in chunks:
                continue
            chunks[draft.chunk_version_id] = ChunkVersion(
                id=draft.chunk_version_id,
                section_revision_id=revision_id,
                ordinal=draft.ordinal,
                kind=draft.kind,
                chunk_text=draft.chunk_text,
                context_header=draft.context_header,
                embedding_text=draft.embedding_text,
                embedding_text_sha256=draft.embedding_text_sha256,
                start_page=draft.start_page,
                end_page=draft.end_page,
                table_key=draft.table_key,
                term_annotations=list(draft.term_annotations),
                colloquial=draft.colloquial,
                chunker_version=CHUNKER_VERSION,
            )
            release_chunks.append(
                ReleaseChunk(
                    chunk_version_id=draft.chunk_version_id,
                    section_id=section_id,
                    section_revision_id=revision_id,
                    ordinal=draft.ordinal,
                    hydrate_strategy=strategy,
                )
            )

    return CorpusSnapshot(
        collection=collection,
        bundle_digest=bundle_digest(bundle),
        chunker_version=CHUNKER_VERSION,
        documents=documents,
        sections=sections,
        revisions=revisions,
        chunks=list(chunks.values()),
        release_chunks=release_chunks,
        glossary=list(bundle.glossary),
        colloquial_mappings=list(bundle.colloquial_mappings),
    )


def releases_to_retire(
    releases: Sequence[Release], current_release_id: uuid.UUID | None, keep: int
) -> list[Release]:
    """Spec C §8.5 step 1: not current and not among the `keep` newest live releases."""
    if keep < 0:
        raise ValueError("keep must be zero or positive")
    live = sorted(
        (release for release in releases if release.status is not ReleaseStatus.RETIRED),
        key=lambda release: release.number,
        reverse=True,
    )
    kept = {release.id for release in live[:keep]}
    return [
        release
        for release in live
        if release.id not in kept and release.id != current_release_id
    ]
```

- [ ] **Step 4: Write `ports.py`**

`backend/src/pharma_agent/domain/corpus/ports.py`:

```python
"""Ports the corpus application services depend on (adapters live in infrastructure)."""

import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Protocol

from pharma_agent.domain.corpus.models import (
    Collection,
    CorpusSnapshot,
    IndexItem,
    PurgeResult,
    Release,
    ReleaseStats,
    ReleaseSummary,
)


class CorpusRepository(Protocol):
    async def get_collection(self, key: str) -> Collection | None: ...

    async def get_release(self, release_id: uuid.UUID) -> Release | None: ...

    async def find_release(
        self,
        collection_id: uuid.UUID,
        *,
        bundle_digest: str,
        chunker_version: str,
        embedding_model: str,
    ) -> Release | None:
        """Newest non-retired release of the collection built from the same inputs."""
        ...

    async def list_releases(self, collection_key: str | None) -> list[ReleaseSummary]:
        """Ordered by collection key, then release number descending."""
        ...

    async def stage_release(
        self,
        snapshot: CorpusSnapshot,
        *,
        release_id: uuid.UUID,
        embedding_model: str,
        at: datetime,
    ) -> Release:
        """Spec C §8.2 steps 3-6 in one transaction, idempotent.

        Upserts collection, documents and sections, inserts missing section revisions and
        chunk versions, and creates the release (status `building`, next number) with its
        `release_chunks`, `glossary_entries` and `colloquial_mappings` unless `release_id`
        already exists. Returns the stored release.
        """
        ...

    async def index_items(self, release_ids: Sequence[uuid.UUID]) -> list[IndexItem]:
        """Items for every chunk version in the releases; `release_ids` of each item lists
        the non-retired releases containing it (possibly empty)."""
        ...

    async def mark_ready(
        self, release_id: uuid.UUID, stats: ReleaseStats, at: datetime
    ) -> None: ...

    async def publish(self, release_id: uuid.UUID, at: datetime) -> None:
        """One transaction: collection.current_release_id and first published_at."""
        ...

    async def retire(self, release_ids: Sequence[uuid.UUID], at: datetime) -> None: ...

    async def purge_retired(self, collection_id: uuid.UUID) -> PurgeResult:
        """Spec C §8.5 steps 3-4 for every retired release of the collection."""
        ...


class EmbeddingCache(Protocol):
    async def missing(self, model: str, hashes: Sequence[str]) -> set[str]: ...

    async def get_many(
        self, model: str, hashes: Sequence[str]
    ) -> dict[str, list[float]]: ...

    async def put_many(
        self, model: str, dimension: int, vectors: Mapping[str, Sequence[float]]
    ) -> None:
        """Insert missing entries; existing (model, hash) rows are left untouched."""
        ...


class Embedder(Protocol):
    @property
    def model(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class VectorIndex(Protocol):
    """Derived index of chunk versions for one embedding model."""

    async def ensure_collection(self) -> str:
        """Create the physical collection, payload indexes and alias when missing; verify
        metadata otherwise (raises IndexMismatch). Returns the physical name."""
        ...

    async def existing_ids(self, ids: Sequence[uuid.UUID]) -> set[uuid.UUID]: ...

    async def upsert(
        self, items: Sequence[IndexItem], vectors: Mapping[str, Sequence[float]]
    ) -> None:
        """Write points; `vectors` is keyed by embedding_text_sha256."""
        ...

    async def set_release_ids(self, items: Sequence[IndexItem]) -> None:
        """Overwrite the `release_ids` payload of existing points."""
        ...

    async def delete(self, ids: Sequence[uuid.UUID]) -> None: ...

    async def count_release(self, release_id: uuid.UUID) -> int: ...
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest -q tests/domain/corpus/test_corpus_models.py tests/architecture/test_layering.py`
Expected: PASS (7 passed in the new file; layering stays green because `models.py` and `ports.py` import only pydantic, the standard library and the domain).

- [ ] **Step 6: Run the full check**

Run: `uv run ruff check src tests && uv run ruff format --check src tests && uv run pyrefly check --min-severity warn && uv run pytest -q`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add src/pharma_agent/domain/corpus/models.py \
  src/pharma_agent/domain/corpus/ports.py \
  tests/domain/corpus/test_corpus_models.py
git commit -m "feat(corpus): add corpus snapshot, release models and storage ports"
```

The commit message ends with the session attribution trailer.

---

### Task 4: Postgres adapters: `PostgresCorpusRepository` and `PostgresEmbeddingCache`

**Files:**
- Create: `backend/src/pharma_agent/infrastructure/persistence/postgres/corpus_repository.py`
- Create: `backend/src/pharma_agent/infrastructure/persistence/postgres/embedding_cache.py`
- Test: `backend/tests/infrastructure/test_corpus_repository.py`

**Interfaces:**
- Consumes: tables from Task 1; `Base` (`tables.py`); `Database` (`database.py`); domain models and ports from Task 3; `small_bundle`, `with_section_text`, section constants (Task 2); `FAKE_EMBEDDING_MODEL`, `NOW` (`tests/fakes.py`); fixture `migrated_dsn`.
- Produces: `PostgresCorpusRepository(sessions: async_sessionmaker[AsyncSession])` implementing `CorpusRepository`; `PostgresEmbeddingCache(sessions)` implementing `EmbeddingCache`; `pack_vector(values: Sequence[float]) -> bytes`; `unpack_vector(data: bytes, dimension: int) -> list[float]`; `ROWS_PER_STATEMENT = 500`.

- [ ] **Step 1: Write the failing integration tests**

`backend/tests/infrastructure/test_corpus_repository.py`:

```python
import uuid
from collections.abc import AsyncGenerator
from datetime import timedelta

import pytest
from sqlalchemy import select, text

from pharma_agent.domain.corpus.models import (
    CorpusSnapshot,
    PurgeResult,
    ReleaseStats,
    ReleaseStatus,
    build_snapshot,
)
from pharma_agent.domain.corpus.ports import CorpusRepository, EmbeddingCache
from pharma_agent.infrastructure.persistence.postgres.corpus_repository import (
    PostgresCorpusRepository,
)
from pharma_agent.infrastructure.persistence.postgres.corpus_tables import (
    ChunkVersionTable,
)
from pharma_agent.infrastructure.persistence.postgres.database import Database
from pharma_agent.infrastructure.persistence.postgres.embedding_cache import (
    PostgresEmbeddingCache,
)
from tests.corpus_fixtures import (
    DOSAGE_SECTION,
    LEAFLET_SECTION,
    small_bundle,
    with_section_text,
)
from tests.fakes import FAKE_EMBEDDING_MODEL, NOW

pytestmark = pytest.mark.integration


@pytest.fixture
async def database(migrated_dsn: str) -> AsyncGenerator[Database]:
    db = Database(migrated_dsn, pool_size=2)
    async with db.engine.begin() as connection:
        await connection.execute(
            text("TRUNCATE corpus.collections, corpus.embedding_cache CASCADE")
        )
    yield db
    await db.dispose()


def edited_snapshot() -> CorpusSnapshot:
    bundle = with_section_text(small_bundle(), DOSAGE_SECTION, "Ghi chú liều.")
    return build_snapshot(with_section_text(bundle, LEAFLET_SECTION, "Ghi chú tờ HDSD."))


async def count(database: Database, table: str, where: str = "true") -> int:
    async with database.engine.connect() as connection:
        result = await connection.execute(
            text(f"SELECT count(*) FROM corpus.{table} WHERE {where}")
        )
        return int(result.scalar_one())


async def test_stage_release_writes_every_row_once(database: Database) -> None:
    repository: CorpusRepository = PostgresCorpusRepository(database.sessions)
    snapshot = build_snapshot(small_bundle())
    release_id = uuid.uuid4()

    first = await repository.stage_release(
        snapshot, release_id=release_id, embedding_model=FAKE_EMBEDDING_MODEL, at=NOW
    )
    again = await repository.stage_release(
        snapshot, release_id=release_id, embedding_model=FAKE_EMBEDDING_MODEL, at=NOW
    )

    assert again == first
    assert (first.number, first.status, first.stats) == (1, ReleaseStatus.BUILDING, None)
    assert await repository.get_collection("formulary") == snapshot.collection
    assert await count(database, "documents") == 2
    assert await count(database, "sections") == 5
    assert await count(database, "section_revisions") == 5
    assert await count(database, "chunk_versions") == len(snapshot.chunks)
    assert await count(database, "release_chunks") == len(snapshot.release_chunks)
    assert await count(database, "glossary_entries") == 2
    assert await count(database, "colloquial_mappings") == 1
    assert await count(database, "chunk_versions", "colloquial IS NULL") == sum(
        chunk.colloquial is None for chunk in snapshot.chunks
    )
    assert await count(database, "releases", "stats IS NULL") == 1
    async with database.sessions() as session:
        stored_headers = dict(
            (
                await session.execute(
                    select(ChunkVersionTable.id, ChunkVersionTable.context_header)
                )
            )
            .tuples()
            .all()
        )
    assert stored_headers == {chunk.id: chunk.context_header for chunk in snapshot.chunks}
    assert all(stored_headers.values())
    assert (
        await repository.find_release(
            snapshot.collection.id,
            bundle_digest=snapshot.bundle_digest,
            chunker_version=snapshot.chunker_version,
            embedding_model=FAKE_EMBEDDING_MODEL,
        )
        == first
    )
    assert (
        await repository.find_release(
            snapshot.collection.id,
            bundle_digest="f" * 64,
            chunker_version=snapshot.chunker_version,
            embedding_model=FAKE_EMBEDDING_MODEL,
        )
        is None
    )
    assert await repository.get_release(release_id) == first
    assert await repository.get_release(uuid.uuid4()) is None


async def test_second_release_shares_unchanged_chunk_versions(database: Database) -> None:
    repository = PostgresCorpusRepository(database.sessions)
    old, new = build_snapshot(small_bundle()), edited_snapshot()
    r1 = await repository.stage_release(
        old, release_id=uuid.uuid4(), embedding_model=FAKE_EMBEDDING_MODEL, at=NOW
    )
    r2 = await repository.stage_release(
        new, release_id=uuid.uuid4(), embedding_model=FAKE_EMBEDDING_MODEL, at=NOW
    )

    old_ids = {chunk.id for chunk in old.chunks}
    new_ids = {chunk.id for chunk in new.chunks}
    assert r2.number == 2
    assert await count(database, "chunk_versions") == len(old_ids | new_ids)
    assert await count(database, "release_chunks") == len(old.release_chunks) + len(
        new.release_chunks
    )

    items = {item.chunk_version_id: item for item in await repository.index_items([r2.id])}
    assert set(items) == new_ids
    shared = next(iter(old_ids & new_ids))
    fresh = next(iter(new_ids - old_ids))
    assert items[shared].release_ids == sorted([r1.id, r2.id])
    assert items[fresh].release_ids == [r2.id]

    chunk = next(chunk for chunk in new.chunks if chunk.id == fresh)
    section = next(
        placement.section_id
        for placement in new.release_chunks
        if placement.chunk_version_id == fresh
    )
    document = next(s.document_id for s in new.sections if s.id == section)
    assert items[fresh].collection_id == new.collection.id
    assert items[fresh].document_id == document
    assert items[fresh].section_id == section
    assert items[fresh].section_revision_id == chunk.section_revision_id
    assert items[fresh].kind == chunk.kind
    assert items[fresh].embedding_text == chunk.embedding_text
    assert items[fresh].embedding_text_sha256 == chunk.embedding_text_sha256
    assert await repository.index_items([]) == []


async def test_mark_ready_publish_and_list(database: Database) -> None:
    repository = PostgresCorpusRepository(database.sessions)
    snapshot = build_snapshot(small_bundle())
    release = await repository.stage_release(
        snapshot, release_id=uuid.uuid4(), embedding_model=FAKE_EMBEDDING_MODEL, at=NOW
    )
    stats = snapshot.stats().model_copy(update={"embeddings_computed": 3})
    await repository.mark_ready(release.id, stats, NOW)
    await repository.publish(release.id, NOW)
    await repository.publish(release.id, NOW + timedelta(hours=1))

    stored = await repository.get_release(release.id)
    assert stored is not None
    assert (stored.status, stored.stats, stored.ready_at, stored.published_at) == (
        ReleaseStatus.READY,
        stats,
        NOW,
        NOW,
    )
    collection = await repository.get_collection("formulary")
    assert collection is not None and collection.current_release_id == release.id

    (summary,) = await repository.list_releases("formulary")
    assert (summary.collection_key, summary.current, summary.chunk_count) == (
        "formulary",
        True,
        len(snapshot.release_chunks),
    )
    assert summary.release == stored
    assert [s.release.id for s in await repository.list_releases(None)] == [release.id]
    assert await repository.list_releases("other") == []
    assert isinstance(stats, ReleaseStats)


async def test_purge_keeps_chunk_versions_protected_by_restrict(
    database: Database,
) -> None:
    repository = PostgresCorpusRepository(database.sessions)
    old, new = build_snapshot(small_bundle()), edited_snapshot()
    r1 = await repository.stage_release(
        old, release_id=uuid.uuid4(), embedding_model=FAKE_EMBEDDING_MODEL, at=NOW
    )
    r2 = await repository.stage_release(
        new, release_id=uuid.uuid4(), embedding_model=FAKE_EMBEDDING_MODEL, at=NOW
    )
    await repository.retire([r1.id], NOW)

    old_only = {c.id for c in old.chunks} - {c.id for c in new.chunks}
    assert len(old_only) >= 2
    orphans = [item for item in await repository.index_items([r1.id]) if not item.release_ids]
    assert {item.chunk_version_id for item in orphans} == old_only
    protected = sorted(old_only)[0]

    # Stands in for P6's message_citations: any RESTRICT reference must keep the row.
    async with database.engine.begin() as connection:
        await connection.execute(
            text(
                "CREATE TABLE public.gc_probe_citations (chunk_version_id uuid NOT NULL "
                "REFERENCES corpus.chunk_versions(id) ON DELETE RESTRICT)"
            )
        )
        await connection.execute(
            text("INSERT INTO public.gc_probe_citations VALUES (:id)"),
            {"id": protected},
        )
    try:
        result = await repository.purge_retired(old.collection.id)
        again = await repository.purge_retired(old.collection.id)
    finally:
        async with database.engine.begin() as connection:
            await connection.execute(text("DROP TABLE public.gc_probe_citations"))

    surviving_chunks = [
        c for c in old.chunks if c.id not in old_only or c.id == protected
    ] + [c for c in new.chunks if c.id not in {o.id for o in old.chunks}]
    referenced = {c.section_revision_id for c in surviving_chunks} | {
        p.section_revision_id for p in new.release_chunks
    }
    all_revisions = {r.id for r in old.revisions} | {r.id for r in new.revisions}
    assert result == PurgeResult(
        release_chunks_deleted=len(old.release_chunks),
        chunk_versions_deleted=len(old_only) - 1,
        chunk_versions_kept=1,
        section_revisions_deleted=len(all_revisions - referenced),
    )
    assert await count(database, "chunk_versions", f"id = '{protected}'") == 1
    assert await count(database, "glossary_entries", f"release_id = '{r1.id}'") == 0
    summaries = {s.release.id: s for s in await repository.list_releases("formulary")}
    assert summaries[r1.id].release.status is ReleaseStatus.RETIRED
    assert summaries[r1.id].release.retired_at == NOW
    assert summaries[r1.id].chunk_count == 0
    assert summaries[r2.id].chunk_count == len(new.release_chunks)
    assert again == PurgeResult(chunk_versions_kept=1)


async def test_embedding_cache_round_trips_float32(database: Database) -> None:
    cache: EmbeddingCache = PostgresEmbeddingCache(database.sessions)
    a, b = "a" * 64, "b" * 64
    await cache.put_many(FAKE_EMBEDDING_MODEL, 4, {a: [0.5, 0.25, 1.0, 0.0625]})
    await cache.put_many(FAKE_EMBEDDING_MODEL, 4, {a: [1.0, 1.0, 1.0, 1.0]})

    assert await cache.missing(FAKE_EMBEDDING_MODEL, [a, b]) == {b}
    assert await cache.missing("other-model", [a]) == {a}
    assert await cache.get_many(FAKE_EMBEDDING_MODEL, [a, b]) == {
        a: [0.5, 0.25, 1.0, 0.0625]
    }
    assert await cache.missing(FAKE_EMBEDDING_MODEL, []) == set()
    with pytest.raises(ValueError, match="dimension"):
        await cache.put_many(FAKE_EMBEDDING_MODEL, 4, {b: [0.5]})
```

The second `purge_retired` call runs while the probe table still exists: the protected chunk version is still an orphan candidate and still blocked, so the result is only `chunk_versions_kept=1`. That proves purge is idempotent.

- [ ] **Step 2: Run the tests and watch them fail**

Run: `uv run pytest -q -m integration tests/infrastructure/test_corpus_repository.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'pharma_agent.infrastructure.persistence.postgres.corpus_repository'`.

- [ ] **Step 3: Write the embedding cache**

`backend/src/pharma_agent/infrastructure/persistence/postgres/embedding_cache.py`:

```python
"""Postgres `corpus.embedding_cache`: vectors keyed by (model, sha256(embedding_text))."""

import itertools
import struct
from collections.abc import Mapping, Sequence
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pharma_agent.infrastructure.persistence.postgres.corpus_tables import (
    EmbeddingCacheTable,
)

HASHES_PER_QUERY = 1000
ROWS_PER_STATEMENT = 500


def pack_vector(values: Sequence[float]) -> bytes:
    """float32 little-endian, the same layout as knowledge-bundle vectors."""
    return struct.pack(f"<{len(values)}f", *values)


def unpack_vector(data: bytes, dimension: int) -> list[float]:
    return list(struct.unpack(f"<{dimension}f", data))


class PostgresEmbeddingCache:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def missing(self, model: str, hashes: Sequence[str]) -> set[str]:
        wanted = set(hashes)
        found: set[str] = set()
        async with self._sessions() as session:
            for batch in itertools.batched(sorted(wanted), HASHES_PER_QUERY):
                result = await session.execute(
                    select(EmbeddingCacheTable.embedding_text_sha256).where(
                        EmbeddingCacheTable.model == model,
                        EmbeddingCacheTable.embedding_text_sha256.in_(batch),
                    )
                )
                found.update(result.scalars().all())
        return wanted - found

    async def get_many(self, model: str, hashes: Sequence[str]) -> dict[str, list[float]]:
        vectors: dict[str, list[float]] = {}
        async with self._sessions() as session:
            for batch in itertools.batched(sorted(set(hashes)), HASHES_PER_QUERY):
                result = await session.execute(
                    select(
                        EmbeddingCacheTable.embedding_text_sha256,
                        EmbeddingCacheTable.dims,
                        EmbeddingCacheTable.vector,
                    ).where(
                        EmbeddingCacheTable.model == model,
                        EmbeddingCacheTable.embedding_text_sha256.in_(batch),
                    )
                )
                for sha, dims, vector in result.all():
                    vectors[sha] = unpack_vector(vector, dims)
        return vectors

    async def put_many(
        self, model: str, dimension: int, vectors: Mapping[str, Sequence[float]]
    ) -> None:
        rows: list[dict[str, Any]] = []
        for sha, vector in vectors.items():
            if len(vector) != dimension:
                raise ValueError(
                    f"vector for {sha} has dimension {len(vector)}, expected {dimension}"
                )
            rows.append(
                {
                    "model": model,
                    "embedding_text_sha256": sha,
                    "dims": dimension,
                    "vector": pack_vector(vector),
                }
            )
        if not rows:
            return
        async with self._sessions.begin() as session:
            for batch in itertools.batched(rows, ROWS_PER_STATEMENT):
                await session.execute(
                    insert(EmbeddingCacheTable)
                    .values(list(batch))
                    .on_conflict_do_nothing(
                        index_elements=["model", "embedding_text_sha256"]
                    )
                )
```

- [ ] **Step 4: Write the corpus repository**

`backend/src/pharma_agent/infrastructure/persistence/postgres/corpus_repository.py`:

```python
"""Postgres adapter of `CorpusRepository` over the schema `corpus`."""

import itertools
import uuid
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from sqlalchemy import delete, exists, func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pharma_agent.domain.corpus.bundle import BlockKind
from pharma_agent.domain.corpus.models import (
    ChunkVersion,
    Collection,
    CorpusSnapshot,
    Document,
    IndexItem,
    PurgeResult,
    Release,
    ReleaseChunk,
    ReleaseNotFound,
    ReleaseStats,
    ReleaseStatus,
    ReleaseSummary,
    Section,
    SectionRevision,
    Visibility,
)
from pharma_agent.infrastructure.persistence.postgres.corpus_tables import (
    ChunkVersionTable,
    CollectionTable,
    ColloquialMappingTable,
    DocumentTable,
    GlossaryEntryTable,
    ReleaseChunkTable,
    ReleaseTable,
    SectionRevisionTable,
    SectionTable,
)
from pharma_agent.infrastructure.persistence.postgres.tables import Base

ROWS_PER_STATEMENT = 500


def _collection(row: CollectionTable) -> Collection:
    return Collection(
        id=row.id,
        key=row.key,
        title=row.title,
        owner_user_id=row.owner_user_id,
        visibility=Visibility(row.visibility),
        current_release_id=row.current_release_id,
    )


def _release(row: ReleaseTable) -> Release:
    return Release(
        id=row.id,
        collection_id=row.collection_id,
        number=row.number,
        status=ReleaseStatus(row.status),
        bundle_digest=row.bundle_digest,
        chunker_version=row.chunker_version,
        embedding_model=row.embedding_model,
        stats=ReleaseStats.model_validate(row.stats) if row.stats is not None else None,
        created_at=row.created_at,
        ready_at=row.ready_at,
        published_at=row.published_at,
        retired_at=row.retired_at,
    )


def _document_row(document: Document) -> dict[str, Any]:
    return {
        "id": document.id,
        "collection_id": document.collection_id,
        "key": document.key,
        "kind": document.kind.value,
        "title": document.title,
        "source_title": document.source_title,
        "source_url": document.source_url,
        "attributes": dict(document.attributes),
    }


def _section_row(section: Section) -> dict[str, Any]:
    return {
        "id": section.id,
        "document_id": section.document_id,
        "key": section.key,
        "heading": section.heading,
        "context_path": list(section.context_path),
        "ordinal": section.ordinal,
        "retrieval_mode": section.retrieval_mode.value,
    }


def _revision_row(revision: SectionRevision) -> dict[str, Any]:
    return {
        "id": revision.id,
        "section_id": revision.section_id,
        "blocks": [block.model_dump(mode="json") for block in revision.blocks],
        "start_page": revision.start_page,
        "end_page": revision.end_page,
        "char_count": revision.char_count,
    }


def _chunk_row(chunk: ChunkVersion) -> dict[str, Any]:
    return {
        "id": chunk.id,
        "section_revision_id": chunk.section_revision_id,
        "ordinal": chunk.ordinal,
        "kind": chunk.kind.value,
        "chunk_text": chunk.chunk_text,
        "context_header": chunk.context_header,
        "embedding_text": chunk.embedding_text,
        "embedding_text_sha256": chunk.embedding_text_sha256,
        "start_page": chunk.start_page,
        "end_page": chunk.end_page,
        "table_key": chunk.table_key,
        "term_annotations": [
            annotation.model_dump(mode="json") for annotation in chunk.term_annotations
        ],
        "colloquial": chunk.colloquial.model_dump(mode="json")
        if chunk.colloquial is not None
        else None,
        "chunker_version": chunk.chunker_version,
    }


def _placement_row(release_id: uuid.UUID, placement: ReleaseChunk) -> dict[str, Any]:
    return {
        "release_id": release_id,
        "chunk_version_id": placement.chunk_version_id,
        "section_id": placement.section_id,
        "section_revision_id": placement.section_revision_id,
        "ordinal": placement.ordinal,
        "hydrate_strategy": placement.hydrate_strategy.value,
    }


async def _insert(
    session: AsyncSession,
    table: type[Base],
    rows: Sequence[dict[str, Any]],
    *,
    conflict: Sequence[str],
    update_columns: Sequence[str] = (),
) -> None:
    for batch in itertools.batched(rows, ROWS_PER_STATEMENT):
        statement = insert(table).values(list(batch))
        if update_columns:
            statement = statement.on_conflict_do_update(
                index_elements=list(conflict),
                set_={name: statement.excluded[name] for name in update_columns},
            )
        else:
            statement = statement.on_conflict_do_nothing(index_elements=list(conflict))
        await session.execute(statement)


async def _delete_unreferenced(
    session: AsyncSession,
    table: type[ChunkVersionTable] | type[SectionRevisionTable],
    ids: Sequence[uuid.UUID],
) -> tuple[int, int]:
    """Delete rows, keeping those another table still references (FK RESTRICT).

    Each batch runs in a savepoint; a batch blocked by a foreign key is retried row by row so
    only the referenced rows survive. Returns (deleted, kept).
    """
    deleted = kept = 0
    for batch in itertools.batched(ids, ROWS_PER_STATEMENT):
        try:
            async with session.begin_nested():
                await session.execute(delete(table).where(table.id.in_(batch)))
            deleted += len(batch)
        except IntegrityError:
            for row_id in batch:
                try:
                    async with session.begin_nested():
                        await session.execute(delete(table).where(table.id == row_id))
                    deleted += 1
                except IntegrityError:
                    kept += 1
    return deleted, kept


class PostgresCorpusRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def get_collection(self, key: str) -> Collection | None:
        async with self._sessions() as session:
            row = (
                await session.execute(
                    select(CollectionTable).where(CollectionTable.key == key)
                )
            ).scalar_one_or_none()
        return _collection(row) if row is not None else None

    async def get_release(self, release_id: uuid.UUID) -> Release | None:
        async with self._sessions() as session:
            row = await session.get(ReleaseTable, release_id)
        return _release(row) if row is not None else None

    async def find_release(
        self,
        collection_id: uuid.UUID,
        *,
        bundle_digest: str,
        chunker_version: str,
        embedding_model: str,
    ) -> Release | None:
        query = (
            select(ReleaseTable)
            .where(
                ReleaseTable.collection_id == collection_id,
                ReleaseTable.bundle_digest == bundle_digest,
                ReleaseTable.chunker_version == chunker_version,
                ReleaseTable.embedding_model == embedding_model,
                ReleaseTable.status != ReleaseStatus.RETIRED.value,
            )
            .order_by(ReleaseTable.number.desc())
            .limit(1)
        )
        async with self._sessions() as session:
            row = (await session.execute(query)).scalar_one_or_none()
        return _release(row) if row is not None else None

    async def list_releases(self, collection_key: str | None) -> list[ReleaseSummary]:
        chunk_counts = (
            select(
                ReleaseChunkTable.release_id,
                func.count().label("chunks"),
            )
            .group_by(ReleaseChunkTable.release_id)
            .subquery()
        )
        query = (
            select(
                ReleaseTable,
                CollectionTable.key,
                CollectionTable.current_release_id,
                func.coalesce(chunk_counts.c.chunks, 0),
            )
            .join(CollectionTable, CollectionTable.id == ReleaseTable.collection_id)
            .outerjoin(chunk_counts, chunk_counts.c.release_id == ReleaseTable.id)
            .order_by(CollectionTable.key, ReleaseTable.number.desc())
        )
        if collection_key is not None:
            query = query.where(CollectionTable.key == collection_key)
        async with self._sessions() as session:
            rows = (await session.execute(query)).all()
        return [
            ReleaseSummary(
                collection_key=key,
                release=_release(release),
                chunk_count=int(chunks),
                current=current_release_id == release.id,
            )
            for release, key, current_release_id, chunks in rows
        ]

    async def stage_release(
        self,
        snapshot: CorpusSnapshot,
        *,
        release_id: uuid.UUID,
        embedding_model: str,
        at: datetime,
    ) -> Release:
        collection = snapshot.collection
        async with self._sessions.begin() as session:
            statement = insert(CollectionTable).values(
                id=collection.id,
                key=collection.key,
                title=collection.title,
                owner_user_id=collection.owner_user_id,
                visibility=collection.visibility.value,
            )
            await session.execute(
                statement.on_conflict_do_update(
                    index_elements=["key"],
                    set_={"title": statement.excluded.title, "updated_at": func.now()},
                )
            )
            await _insert(
                session,
                DocumentTable,
                [_document_row(document) for document in snapshot.documents],
                conflict=["collection_id", "key"],
                update_columns=["kind", "title", "source_title", "source_url", "attributes"],
            )
            await _insert(
                session,
                SectionTable,
                [_section_row(section) for section in snapshot.sections],
                conflict=["document_id", "key"],
                update_columns=["heading", "context_path", "ordinal", "retrieval_mode"],
            )
            await _insert(
                session,
                SectionRevisionTable,
                [_revision_row(revision) for revision in snapshot.revisions],
                conflict=["id"],
            )
            await _insert(
                session,
                ChunkVersionTable,
                [_chunk_row(chunk) for chunk in snapshot.chunks],
                conflict=["id"],
            )
            row = await session.get(ReleaseTable, release_id)
            if row is None:
                last = (
                    await session.execute(
                        select(func.coalesce(func.max(ReleaseTable.number), 0)).where(
                            ReleaseTable.collection_id == collection.id
                        )
                    )
                ).scalar_one()
                row = ReleaseTable(
                    id=release_id,
                    collection_id=collection.id,
                    number=int(last) + 1,
                    status=ReleaseStatus.BUILDING.value,
                    bundle_digest=snapshot.bundle_digest,
                    chunker_version=snapshot.chunker_version,
                    embedding_model=embedding_model,
                    created_at=at,
                )
                session.add(row)
                await session.flush()
                await _insert(
                    session,
                    ReleaseChunkTable,
                    [
                        _placement_row(release_id, placement)
                        for placement in snapshot.release_chunks
                    ],
                    conflict=["release_id", "chunk_version_id"],
                )
                session.add_all(
                    [
                        GlossaryEntryTable(
                            release_id=release_id,
                            term=entry.term,
                            data=entry.model_dump(mode="json"),
                        )
                        for entry in snapshot.glossary
                    ]
                )
                session.add_all(
                    [
                        ColloquialMappingTable(
                            release_id=release_id,
                            position=position,
                            key=mapping.key,
                            data=mapping.model_dump(mode="json"),
                        )
                        for position, mapping in enumerate(snapshot.colloquial_mappings)
                    ]
                )
            return _release(row)

    async def index_items(self, release_ids: Sequence[uuid.UUID]) -> list[IndexItem]:
        if not release_ids:
            return []
        chunk_ids = select(ReleaseChunkTable.chunk_version_id).where(
            ReleaseChunkTable.release_id.in_(list(release_ids))
        )
        items_query = (
            select(
                ChunkVersionTable.id,
                DocumentTable.collection_id,
                SectionTable.document_id,
                SectionRevisionTable.section_id,
                ChunkVersionTable.section_revision_id,
                ChunkVersionTable.kind,
                ChunkVersionTable.embedding_text,
                ChunkVersionTable.embedding_text_sha256,
            )
            .join(
                SectionRevisionTable,
                SectionRevisionTable.id == ChunkVersionTable.section_revision_id,
            )
            .join(SectionTable, SectionTable.id == SectionRevisionTable.section_id)
            .join(DocumentTable, DocumentTable.id == SectionTable.document_id)
            .where(ChunkVersionTable.id.in_(chunk_ids))
            .order_by(ChunkVersionTable.id)
        )
        memberships_query = (
            select(
                ReleaseChunkTable.chunk_version_id,
                func.array_agg(ReleaseChunkTable.release_id),
            )
            .join(ReleaseTable, ReleaseTable.id == ReleaseChunkTable.release_id)
            .where(
                ReleaseTable.status != ReleaseStatus.RETIRED.value,
                ReleaseChunkTable.chunk_version_id.in_(chunk_ids),
            )
            .group_by(ReleaseChunkTable.chunk_version_id)
        )
        async with self._sessions() as session:
            rows = (await session.execute(items_query)).all()
            memberships: dict[uuid.UUID, list[uuid.UUID]] = {
                chunk_id: sorted(releases)
                for chunk_id, releases in (await session.execute(memberships_query)).all()
            }
        return [
            IndexItem(
                chunk_version_id=chunk_id,
                collection_id=collection_id,
                document_id=document_id,
                section_id=section_id,
                section_revision_id=section_revision_id,
                kind=BlockKind(kind),
                embedding_text=embedding_text,
                embedding_text_sha256=embedding_text_sha256,
                release_ids=memberships.get(chunk_id, []),
            )
            for (
                chunk_id,
                collection_id,
                document_id,
                section_id,
                section_revision_id,
                kind,
                embedding_text,
                embedding_text_sha256,
            ) in rows
        ]

    async def mark_ready(
        self, release_id: uuid.UUID, stats: ReleaseStats, at: datetime
    ) -> None:
        async with self._sessions.begin() as session:
            await session.execute(
                update(ReleaseTable)
                .where(ReleaseTable.id == release_id)
                .values(
                    status=ReleaseStatus.READY.value,
                    stats=stats.model_dump(mode="json"),
                    ready_at=at,
                )
            )

    async def publish(self, release_id: uuid.UUID, at: datetime) -> None:
        async with self._sessions.begin() as session:
            row = await session.get(ReleaseTable, release_id)
            if row is None:
                raise ReleaseNotFound(str(release_id))
            await session.execute(
                update(CollectionTable)
                .where(CollectionTable.id == row.collection_id)
                .values(current_release_id=release_id, updated_at=func.now())
            )
            await session.execute(
                update(ReleaseTable)
                .where(ReleaseTable.id == release_id)
                .values(published_at=func.coalesce(ReleaseTable.published_at, at))
            )

    async def retire(self, release_ids: Sequence[uuid.UUID], at: datetime) -> None:
        if not release_ids:
            return
        async with self._sessions.begin() as session:
            await session.execute(
                update(ReleaseTable)
                .where(
                    ReleaseTable.id.in_(list(release_ids)),
                    ReleaseTable.status != ReleaseStatus.RETIRED.value,
                )
                .values(status=ReleaseStatus.RETIRED.value, retired_at=at)
            )

    async def purge_retired(self, collection_id: uuid.UUID) -> PurgeResult:
        retired = select(ReleaseTable.id).where(
            ReleaseTable.collection_id == collection_id,
            ReleaseTable.status == ReleaseStatus.RETIRED.value,
        )
        collection_revisions = (
            select(SectionRevisionTable.id)
            .join(SectionTable, SectionTable.id == SectionRevisionTable.section_id)
            .join(DocumentTable, DocumentTable.id == SectionTable.document_id)
            .where(DocumentTable.collection_id == collection_id)
        )
        async with self._sessions.begin() as session:
            placements = (
                await session.execute(
                    delete(ReleaseChunkTable)
                    .where(ReleaseChunkTable.release_id.in_(retired))
                    .returning(ReleaseChunkTable.chunk_version_id)
                )
            ).all()
            await session.execute(
                delete(GlossaryEntryTable).where(GlossaryEntryTable.release_id.in_(retired))
            )
            await session.execute(
                delete(ColloquialMappingTable).where(
                    ColloquialMappingTable.release_id.in_(retired)
                )
            )
            orphan_chunks = (
                (
                    await session.execute(
                        select(ChunkVersionTable.id)
                        .where(
                            ChunkVersionTable.section_revision_id.in_(collection_revisions),
                            ~exists().where(
                                ReleaseChunkTable.chunk_version_id == ChunkVersionTable.id
                            ),
                        )
                        .order_by(ChunkVersionTable.id)
                    )
                )
                .scalars()
                .all()
            )
            chunks_deleted, chunks_kept = await _delete_unreferenced(
                session, ChunkVersionTable, orphan_chunks
            )
            orphan_revisions = (
                (
                    await session.execute(
                        select(SectionRevisionTable.id)
                        .where(
                            SectionRevisionTable.id.in_(collection_revisions),
                            ~exists().where(
                                ReleaseChunkTable.section_revision_id
                                == SectionRevisionTable.id
                            ),
                            ~exists().where(
                                ChunkVersionTable.section_revision_id
                                == SectionRevisionTable.id
                            ),
                        )
                        .order_by(SectionRevisionTable.id)
                    )
                )
                .scalars()
                .all()
            )
            revisions_deleted, _ = await _delete_unreferenced(
                session, SectionRevisionTable, orphan_revisions
            )
        return PurgeResult(
            release_chunks_deleted=len(placements),
            chunk_versions_deleted=chunks_deleted,
            chunk_versions_kept=chunks_kept,
            section_revisions_deleted=revisions_deleted,
        )
```

- [ ] **Step 5: Run the integration tests**

Run: `uv run pytest -q -m integration tests/infrastructure/test_corpus_repository.py`
Expected: PASS (5 passed).

- [ ] **Step 6: Run the full check**

Run: `uv run ruff check src tests && uv run ruff format --check src tests && uv run pyrefly check --min-severity warn && uv run pytest -q && uv run pytest -q -m integration`
Expected: all green. `ruff format` may rewrap long lines in the code above; apply `uv run ruff format src tests` before the check rather than changing logic.

- [ ] **Step 7: Commit**

```bash
git add src/pharma_agent/infrastructure/persistence/postgres/corpus_repository.py \
  src/pharma_agent/infrastructure/persistence/postgres/embedding_cache.py \
  tests/infrastructure/test_corpus_repository.py
git commit -m "feat(corpus): add Postgres corpus repository and embedding cache"
```

The commit message ends with the session attribution trailer.

---

### Task 5: `QdrantVectorIndex` and the shared Qdrant test container

**Files:**
- Create: `backend/src/pharma_agent/infrastructure/retrieval/qdrant_index.py`
- Create: `backend/tests/qdrant.py`
- Modify: `backend/tests/conftest.py:1`
- Test: `backend/tests/infrastructure/test_qdrant_index.py`

**Interfaces:**
- Consumes: `DENSE_VECTOR_NAME`, `BM25_SPARSE_VECTOR_NAME`, `BM25_MODEL_NAME` (`infrastructure/retrieval/qdrant_adapter.py`); `model_slug` (P1 `bundle`); `IndexItem`, `IndexMismatch` (Task 3); `sha256_hex` (P1 `identity`); `FAKE_EMBEDDING_MODEL`, `fake_vector` (Task 2); qdrant-client 1.19 `AsyncQdrantClient.collection_exists/get_collection/create_collection(metadata=...)/create_payload_index/get_aliases/update_collection_aliases/retrieve/upsert/batch_update_points/delete/count`.
- Produces: `CURRENT_ALIAS = "chunks_current"`, `IDS_PER_REQUEST = 500`, `POINTS_PER_UPSERT = 256`, `KEYWORD_PAYLOAD_FIELDS`, `physical_collection_name(model: str) -> str`, `point_payload(item: IndexItem) -> dict[str, object]`, `QdrantVectorIndex(client: AsyncQdrantClient, *, model: str, dimension: int, alias: str = CURRENT_ALIAS)` implementing `VectorIndex` with attributes `collection_name: str` and `alias: str` (the alias created, checked and never moved by `ensure_collection`); test fixtures `qdrant_url` (session) and `qdrant_client` (function, all collections deleted first), image `qdrant/qdrant:v1.19.1`.

- [ ] **Step 1: Add the Qdrant test plugin**

`backend/tests/qdrant.py`:

```python
"""Qdrant for integration tests: one container per session, collections wiped per test."""

from collections.abc import AsyncGenerator, Iterator

import pytest
from qdrant_client import AsyncQdrantClient

# Same server version as docker-compose.yml; BM25 inference from models.Document needs >= 1.15.3.
QDRANT_IMAGE = "qdrant/qdrant:v1.19.1"


@pytest.fixture(scope="session")
def qdrant_url() -> Iterator[str]:
    from testcontainers.community.qdrant import QdrantContainer

    with QdrantContainer(QDRANT_IMAGE) as container:
        host = container.get_container_host_ip()
        yield f"http://{host}:{container.get_exposed_port(6333)}"


@pytest.fixture
async def qdrant_client(qdrant_url: str) -> AsyncGenerator[AsyncQdrantClient]:
    client = AsyncQdrantClient(url=qdrant_url)
    for collection in (await client.get_collections()).collections:
        await client.delete_collection(collection.name)
    yield client
    await client.close()
```

Replace `backend/tests/conftest.py` with:

```python
pytest_plugins = ["tests.postgres", "tests.qdrant"]
```

- [ ] **Step 2: Write the failing integration tests**

`backend/tests/infrastructure/test_qdrant_index.py`:

```python
import uuid

import pytest
from qdrant_client import AsyncQdrantClient, models

from pharma_agent.domain.corpus.bundle import BlockKind
from pharma_agent.domain.corpus.identity import sha256_hex
from pharma_agent.domain.corpus.models import IndexItem, IndexMismatch
from pharma_agent.domain.corpus.ports import VectorIndex
from pharma_agent.infrastructure.retrieval.qdrant_adapter import (
    BM25_MODEL_NAME,
    BM25_SPARSE_VECTOR_NAME,
    DENSE_VECTOR_NAME,
)
from pharma_agent.infrastructure.retrieval.qdrant_index import (
    CURRENT_ALIAS,
    QdrantVectorIndex,
)
from tests.fakes import FAKE_EMBEDDING_DIMENSION, FAKE_EMBEDDING_MODEL, fake_vector

pytestmark = pytest.mark.integration

COLLECTION = "chunks_fake_embedding_4d"
COLLECTION_ID = uuid.uuid4()


def item(text: str, release_ids: list[uuid.UUID]) -> IndexItem:
    return IndexItem(
        chunk_version_id=uuid.uuid4(),
        collection_id=COLLECTION_ID,
        document_id=uuid.uuid4(),
        section_id=uuid.uuid4(),
        section_revision_id=uuid.uuid4(),
        kind=BlockKind.PROSE,
        embedding_text=text,
        embedding_text_sha256=sha256_hex(text),
        release_ids=release_ids,
    )


def fake_index(client: AsyncQdrantClient, dimension: int = 4) -> QdrantVectorIndex:
    return QdrantVectorIndex(client, model=FAKE_EMBEDDING_MODEL, dimension=dimension)


async def test_ensure_collection_creates_the_layout_once(
    qdrant_client: AsyncQdrantClient,
) -> None:
    index: VectorIndex = fake_index(qdrant_client)
    assert await index.ensure_collection() == COLLECTION
    assert await index.ensure_collection() == COLLECTION

    info = await qdrant_client.get_collection(COLLECTION)
    assert info.config.metadata == {
        "embedding_model": FAKE_EMBEDDING_MODEL,
        "dims": FAKE_EMBEDDING_DIMENSION,
    }
    vectors = info.config.params.vectors
    assert isinstance(vectors, dict)
    assert vectors[DENSE_VECTOR_NAME].size == FAKE_EMBEDDING_DIMENSION
    assert vectors[DENSE_VECTOR_NAME].distance is models.Distance.COSINE
    sparse = info.config.params.sparse_vectors
    assert sparse is not None
    assert sparse[BM25_SPARSE_VECTOR_NAME].modifier is models.Modifier.IDF
    assert set(info.payload_schema) == {
        "collection_id",
        "release_ids",
        "document_id",
        "section_id",
        "section_revision_id",
        "kind",
    }
    tenant = info.payload_schema["collection_id"].params
    assert isinstance(tenant, models.KeywordIndexParams) and tenant.is_tenant is True
    aliases = (await qdrant_client.get_aliases()).aliases
    assert [(a.alias_name, a.collection_name) for a in aliases] == [
        (CURRENT_ALIAS, COLLECTION)
    ]


async def test_ensure_collection_rejects_mismatched_metadata_and_keeps_alias(
    qdrant_client: AsyncQdrantClient,
) -> None:
    await fake_index(qdrant_client).ensure_collection()
    with pytest.raises(IndexMismatch, match="fake-embedding-4d"):
        await fake_index(qdrant_client, dimension=8).ensure_collection()

    other = QdrantVectorIndex(qdrant_client, model="Other-Embedding:v2", dimension=4)
    assert await other.ensure_collection() == "chunks_other_embedding_v2"
    aliases = (await qdrant_client.get_aliases()).aliases
    assert [(a.alias_name, a.collection_name) for a in aliases] == [
        (CURRENT_ALIAS, COLLECTION)
    ]


async def test_non_default_alias_is_created_and_used(
    qdrant_client: AsyncQdrantClient,
) -> None:
    index = QdrantVectorIndex(
        qdrant_client,
        model=FAKE_EMBEDDING_MODEL,
        dimension=FAKE_EMBEDDING_DIMENSION,
        alias="e2e_chunks_current",
    )
    assert index.alias == "e2e_chunks_current"
    assert await index.ensure_collection() == COLLECTION
    assert await index.ensure_collection() == COLLECTION
    aliases = (await qdrant_client.get_aliases()).aliases
    assert [(a.alias_name, a.collection_name) for a in aliases] == [
        ("e2e_chunks_current", COLLECTION)
    ]

    release_id = uuid.uuid4()
    point = item("paracetamol hạ sốt", [release_id])
    await index.upsert(
        [point], {point.embedding_text_sha256: fake_vector(point.embedding_text)}
    )
    found = await qdrant_client.query_points(
        "e2e_chunks_current",
        query=fake_vector(point.embedding_text),
        using=DENSE_VECTOR_NAME,
        limit=1,
    )
    assert [str(p.id) for p in found.points] == [str(point.chunk_version_id)]

    await fake_index(qdrant_client).ensure_collection()
    names = {a.alias_name for a in (await qdrant_client.get_aliases()).aliases}
    assert names == {"e2e_chunks_current", CURRENT_ALIAS}


async def test_upsert_payload_release_ids_count_and_delete(
    qdrant_client: AsyncQdrantClient,
) -> None:
    index = fake_index(qdrant_client)
    await index.ensure_collection()
    r1, r2 = uuid.uuid4(), uuid.uuid4()
    fever = item("paracetamol hạ sốt cho người lớn", [r1])
    warfarin = item("warfarin tương tác chống đông máu", [r1])
    vectors = {x.embedding_text_sha256: fake_vector(x.embedding_text) for x in (fever, warfarin)}

    await index.upsert([fever, warfarin], vectors)

    unknown = uuid.uuid4()
    assert await index.existing_ids(
        [fever.chunk_version_id, warfarin.chunk_version_id, unknown]
    ) == {fever.chunk_version_id, warfarin.chunk_version_id}
    (record,) = await qdrant_client.retrieve(
        COLLECTION, ids=[str(fever.chunk_version_id)], with_payload=True
    )
    assert record.payload == {
        "collection_id": str(COLLECTION_ID),
        "release_ids": [str(r1)],
        "document_id": str(fever.document_id),
        "section_id": str(fever.section_id),
        "section_revision_id": str(fever.section_revision_id),
        "kind": "prose",
    }
    bm25 = await qdrant_client.query_points(
        COLLECTION,
        query=models.Document(text="chống đông", model=BM25_MODEL_NAME),
        using=BM25_SPARSE_VECTOR_NAME,
        limit=1,
    )
    assert bm25.points[0].id == str(warfarin.chunk_version_id)
    assert await index.count_release(r1) == 2

    await index.set_release_ids(
        [
            fever.model_copy(update={"release_ids": sorted([r1, r2])}),
            warfarin.model_copy(update={"release_ids": [r2]}),
        ]
    )
    assert (await index.count_release(r1), await index.count_release(r2)) == (1, 2)

    await index.delete([warfarin.chunk_version_id])
    assert await index.existing_ids([warfarin.chunk_version_id]) == set()
    assert await index.count_release(r2) == 1

    with pytest.raises(ValueError, match="no vector"):
        await index.upsert([item("không có vector", [r1])], vectors)
```

- [ ] **Step 3: Run the tests and watch them fail**

Run: `uv run pytest -q -m integration tests/infrastructure/test_qdrant_index.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'pharma_agent.infrastructure.retrieval.qdrant_index'`.

- [ ] **Step 4: Write the adapter**

`backend/src/pharma_agent/infrastructure/retrieval/qdrant_index.py`:

```python
"""Qdrant writes for the corpus index (spec C §8.3).

One physical collection per embedding model (`chunks_<model_slug>`), partitioned by the
`collection_id` payload (tenant index), with `release_ids` listing the non-retired releases that
contain each chunk version. Payloads carry no text; BM25 vectors are inferred by the Qdrant
server from `models.Document`.
"""

import itertools
import uuid
from collections.abc import Mapping, Sequence

from qdrant_client import AsyncQdrantClient, models

from pharma_agent.domain.corpus.bundle import model_slug
from pharma_agent.domain.corpus.models import IndexItem, IndexMismatch
from pharma_agent.infrastructure.retrieval.qdrant_adapter import (
    BM25_MODEL_NAME,
    BM25_SPARSE_VECTOR_NAME,
    DENSE_VECTOR_NAME,
)

CURRENT_ALIAS = "chunks_current"
IDS_PER_REQUEST = 500
POINTS_PER_UPSERT = 256
OPERATIONS_PER_BATCH = 100
TENANT_FIELD = "collection_id"
KEYWORD_PAYLOAD_FIELDS = (
    "release_ids",
    "document_id",
    "section_id",
    "section_revision_id",
    "kind",
)


def physical_collection_name(model: str) -> str:
    return f"chunks_{model_slug(model)}"


def point_payload(item: IndexItem) -> dict[str, object]:
    return {
        "collection_id": str(item.collection_id),
        "release_ids": [str(release_id) for release_id in item.release_ids],
        "document_id": str(item.document_id),
        "section_id": str(item.section_id),
        "section_revision_id": str(item.section_revision_id),
        "kind": item.kind.value,
    }


class QdrantVectorIndex:
    def __init__(
        self,
        client: AsyncQdrantClient,
        *,
        model: str,
        dimension: int,
        alias: str = CURRENT_ALIAS,
    ) -> None:
        self._client = client
        self._model = model
        self._dimension = dimension
        self.collection_name = physical_collection_name(model)
        # Retrieval reads this alias (P3: settings.retrieval.qdrant_collection); the E2E
        # server uses its own alias so it never touches the dev index.
        self.alias = alias

    async def ensure_collection(self) -> str:
        if await self._client.collection_exists(self.collection_name):
            await self._verify_metadata()
        else:
            await self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    DENSE_VECTOR_NAME: models.VectorParams(
                        size=self._dimension, distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    BM25_SPARSE_VECTOR_NAME: models.SparseVectorParams(
                        modifier=models.Modifier.IDF
                    )
                },
                metadata={"embedding_model": self._model, "dims": self._dimension},
            )
        await self._ensure_payload_indexes()
        await self._ensure_alias()
        return self.collection_name

    async def existing_ids(self, ids: Sequence[uuid.UUID]) -> set[uuid.UUID]:
        found: set[uuid.UUID] = set()
        for batch in itertools.batched(ids, IDS_PER_REQUEST):
            records = await self._client.retrieve(
                collection_name=self.collection_name,
                ids=[str(point_id) for point_id in batch],
                with_payload=False,
                with_vectors=False,
            )
            found.update(uuid.UUID(str(record.id)) for record in records)
        return found

    async def upsert(
        self, items: Sequence[IndexItem], vectors: Mapping[str, Sequence[float]]
    ) -> None:
        for batch in itertools.batched(items, POINTS_PER_UPSERT):
            points: list[models.PointStruct] = []
            for item in batch:
                vector = vectors.get(item.embedding_text_sha256)
                if vector is None:
                    raise ValueError(
                        f"no vector for embedding_text_sha256 {item.embedding_text_sha256}"
                    )
                if len(vector) != self._dimension:
                    raise IndexMismatch(
                        f"vector dimension {len(vector)} != {self._dimension} "
                        f"for {self.collection_name}"
                    )
                points.append(
                    models.PointStruct(
                        id=str(item.chunk_version_id),
                        vector={
                            DENSE_VECTOR_NAME: list(vector),
                            BM25_SPARSE_VECTOR_NAME: models.Document(
                                text=item.embedding_text, model=BM25_MODEL_NAME
                            ),
                        },
                        payload=point_payload(item),
                    )
                )
            await self._client.upsert(
                collection_name=self.collection_name, points=points, wait=True
            )

    async def set_release_ids(self, items: Sequence[IndexItem]) -> None:
        groups: dict[tuple[str, ...], list[models.ExtendedPointId]] = {}
        for item in items:
            key = tuple(str(release_id) for release_id in item.release_ids)
            groups.setdefault(key, []).append(str(item.chunk_version_id))
        operations: list[models.SetPayloadOperation] = [
            models.SetPayloadOperation(
                set_payload=models.SetPayload(
                    payload={"release_ids": list(release_ids)}, points=list(batch)
                )
            )
            for release_ids, point_ids in groups.items()
            for batch in itertools.batched(point_ids, IDS_PER_REQUEST)
        ]
        for batch in itertools.batched(operations, OPERATIONS_PER_BATCH):
            await self._client.batch_update_points(
                collection_name=self.collection_name,
                update_operations=list(batch),
                wait=True,
            )

    async def delete(self, ids: Sequence[uuid.UUID]) -> None:
        for batch in itertools.batched(ids, IDS_PER_REQUEST):
            points: list[models.ExtendedPointId] = [str(point_id) for point_id in batch]
            await self._client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=points),
                wait=True,
            )

    async def count_release(self, release_id: uuid.UUID) -> int:
        result = await self._client.count(
            collection_name=self.collection_name,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="release_ids", match=models.MatchValue(value=str(release_id))
                    )
                ]
            ),
            exact=True,
        )
        return result.count

    async def _verify_metadata(self) -> None:
        info = await self._client.get_collection(self.collection_name)
        metadata = info.config.metadata or {}
        if (
            metadata.get("embedding_model") != self._model
            or metadata.get("dims") != self._dimension
        ):
            raise IndexMismatch(
                f"collection {self.collection_name} has metadata {metadata}, expected "
                f"embedding_model={self._model} dims={self._dimension}"
            )

    async def _ensure_payload_indexes(self) -> None:
        info = await self._client.get_collection(self.collection_name)
        existing = set(info.payload_schema)
        if TENANT_FIELD not in existing:
            await self._client.create_payload_index(
                collection_name=self.collection_name,
                field_name=TENANT_FIELD,
                field_schema=models.KeywordIndexParams(
                    type=models.KeywordIndexType.KEYWORD, is_tenant=True
                ),
                wait=True,
            )
        for field in KEYWORD_PAYLOAD_FIELDS:
            if field not in existing:
                await self._client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                    wait=True,
                )

    async def _ensure_alias(self) -> None:
        aliases = (await self._client.get_aliases()).aliases
        if any(alias.alias_name == self.alias for alias in aliases):
            return
        await self._client.update_collection_aliases(
            change_aliases_operations=[
                models.CreateAliasOperation(
                    create_alias=models.CreateAlias(
                        collection_name=self.collection_name, alias_name=self.alias
                    )
                )
            ]
        )
```

- [ ] **Step 5: Run the integration tests**

Run: `uv run pytest -q -m integration tests/infrastructure/test_qdrant_index.py tests/infrastructure/test_qdrant_integration.py`
Expected: PASS (4 passed in the new file; the existing Qdrant test still passes with its own container).

- [ ] **Step 6: Run the full check**

Run: `uv run ruff check src tests && uv run ruff format --check src tests && uv run pyrefly check --min-severity warn && uv run pytest -q && uv run pytest -q -m integration`
Expected: all green. The unit run (`-m "not integration"`) never starts a container because `qdrant_url` is only requested by integration tests.

- [ ] **Step 7: Commit**

```bash
git add src/pharma_agent/infrastructure/retrieval/qdrant_index.py \
  tests/qdrant.py tests/conftest.py \
  tests/infrastructure/test_qdrant_index.py
git commit -m "feat(corpus): add Qdrant vector index with tenant payload and release ids"
```

The commit message ends with the session attribution trailer.

---

### Task 6: `ImportKnowledgeBundle` with embedding resolution and index writing

**Files:**
- Create: `backend/src/pharma_agent/application/corpus/__init__.py` (empty)
- Create: `backend/src/pharma_agent/application/corpus/indexing.py`
- Create: `backend/src/pharma_agent/application/corpus/import_bundle.py`
- Create: `backend/tests/corpus_memory.py`
- Test: `backend/tests/application/test_import_bundle.py`

**Interfaces:**
- Consumes: domain models and ports (Task 3); `Clock`, `FixedClock` (`domain/shared/clock.py`); `KnowledgeBundle`, `model_slug` (P1); `RetrievalError` (`domain/retrieval/ports.py`, raised by embedders); test support from Task 2.
- Produces: `EmbeddingCounts`, `EmbeddingResolver`, `IndexCounts`, `IndexWriter`, `CACHE_WRITE_BATCH = 500`, `INDEX_BATCH = 256` (`indexing.py`); `ImportOutcome`, `ImportReport`, `bundle_vectors(bundle: KnowledgeBundle, model: str, dimension: int) -> Mapping[str, list[float]]`, `ImportKnowledgeBundle` (`import_bundle.py`), all as listed in "Interfaces pinned by this plan"; test doubles `InMemoryCorpusRepository` (with `stage_calls: int`, `protected_chunk_ids: set[UUID]`), `InMemoryEmbeddingCache` (`vectors: dict[tuple[str, str], list[float]]`), `InMemoryVectorIndex(*, drop_ids: set[UUID] | None = None)` (`points`, `ensure_calls`), `CorpusAdapters` dataclass, `build_importer(adapters, embedder, *, batch_size=2, max_concurrent=1) -> ImportKnowledgeBundle`.

Flow implemented by `ImportKnowledgeBundle.__call__` (spec C §8.2):

| Step | Code |
| --- | --- |
| 1 Validate | `build_snapshot(bundle)` (document references) and `bundle_vectors(...)` (vector lengths) raise `CorpusImportError` before any write; file-level §5.3 checks already ran in `read_bundle` |
| 2 No change | current release of the collection has the same `bundle_digest`, `CHUNKER_VERSION` and embedding model → `ImportOutcome.NO_CHANGE` |
| 3–6 Stage | `repository.stage_release(...)`; a matching `building` release is resumed with its id, a matching `ready` one is reused (`REUSED`) |
| 7 Embedding | `EmbeddingResolver.ensure`: cache → bundle vectors (model and dims declared in the manifest) → embedder in batches of `embed_batch_size`, at most `embed_max_concurrent` at once, cache written after each batch |
| 8 Index | `VectorIndex.ensure_collection()` runs before staging; `IndexWriter.write` upserts missing points with vectors read from the cache per batch and rewrites `release_ids` on existing points |
| 9 Verify | `count_release == len(snapshot.release_chunks)` → `mark_ready(stats)`, otherwise `CorpusImportError` and the release stays `building` |
| 10 Publish | `repository.publish(release.id, now)` when `publish=True` |

- [ ] **Step 1: Write the in-memory adapters**

`backend/tests/corpus_memory.py`:

```python
"""In-memory corpus adapters that behave like the Postgres and Qdrant ones."""

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime

from pharma_agent.application.corpus.import_bundle import ImportKnowledgeBundle
from pharma_agent.domain.corpus.bundle import ColloquialMappingRecord, GlossaryEntry
from pharma_agent.domain.corpus.models import (
    ChunkVersion,
    Collection,
    CorpusSnapshot,
    Document,
    IndexItem,
    PurgeResult,
    Release,
    ReleaseChunk,
    ReleaseNotFound,
    ReleaseStats,
    ReleaseStatus,
    ReleaseSummary,
    Section,
    SectionRevision,
)
from pharma_agent.domain.corpus.ports import Embedder
from pharma_agent.domain.shared.clock import FixedClock
from tests.fakes import NOW


class InMemoryCorpusRepository:
    def __init__(self) -> None:
        self.collections: dict[str, Collection] = {}
        self.documents: dict[uuid.UUID, Document] = {}
        self.sections: dict[uuid.UUID, Section] = {}
        self.revisions: dict[uuid.UUID, SectionRevision] = {}
        self.chunks: dict[uuid.UUID, ChunkVersion] = {}
        self.releases: dict[uuid.UUID, Release] = {}
        self.placements: dict[uuid.UUID, list[ReleaseChunk]] = {}
        self.glossary: dict[uuid.UUID, list[GlossaryEntry]] = {}
        self.mappings: dict[uuid.UUID, list[ColloquialMappingRecord]] = {}
        self.protected_chunk_ids: set[uuid.UUID] = set()
        self.stage_calls = 0

    async def get_collection(self, key: str) -> Collection | None:
        return self.collections.get(key)

    async def get_release(self, release_id: uuid.UUID) -> Release | None:
        return self.releases.get(release_id)

    async def find_release(
        self,
        collection_id: uuid.UUID,
        *,
        bundle_digest: str,
        chunker_version: str,
        embedding_model: str,
    ) -> Release | None:
        matches = [
            release
            for release in self.releases.values()
            if release.collection_id == collection_id
            and release.status is not ReleaseStatus.RETIRED
            and (release.bundle_digest, release.chunker_version, release.embedding_model)
            == (bundle_digest, chunker_version, embedding_model)
        ]
        return max(matches, key=lambda release: release.number, default=None)

    async def list_releases(self, collection_key: str | None) -> list[ReleaseSummary]:
        by_id = {collection.id: collection for collection in self.collections.values()}
        summaries = [
            ReleaseSummary(
                collection_key=by_id[release.collection_id].key,
                release=release,
                chunk_count=len(self.placements.get(release.id, [])),
                current=by_id[release.collection_id].current_release_id == release.id,
            )
            for release in self.releases.values()
            if collection_key is None or by_id[release.collection_id].key == collection_key
        ]
        return sorted(
            summaries, key=lambda summary: (summary.collection_key, -summary.release.number)
        )

    async def stage_release(
        self,
        snapshot: CorpusSnapshot,
        *,
        release_id: uuid.UUID,
        embedding_model: str,
        at: datetime,
    ) -> Release:
        self.stage_calls += 1
        collection = snapshot.collection
        stored = self.collections.get(collection.key)
        self.collections[collection.key] = (
            collection
            if stored is None
            else stored.model_copy(update={"title": collection.title})
        )
        self.documents.update({document.id: document for document in snapshot.documents})
        self.sections.update({section.id: section for section in snapshot.sections})
        for revision in snapshot.revisions:
            self.revisions.setdefault(revision.id, revision)
        for chunk in snapshot.chunks:
            self.chunks.setdefault(chunk.id, chunk)
        if release_id not in self.releases:
            number = 1 + max(
                (
                    release.number
                    for release in self.releases.values()
                    if release.collection_id == collection.id
                ),
                default=0,
            )
            self.releases[release_id] = Release(
                id=release_id,
                collection_id=collection.id,
                number=number,
                status=ReleaseStatus.BUILDING,
                bundle_digest=snapshot.bundle_digest,
                chunker_version=snapshot.chunker_version,
                embedding_model=embedding_model,
                created_at=at,
            )
            self.placements[release_id] = list(snapshot.release_chunks)
            self.glossary[release_id] = list(snapshot.glossary)
            self.mappings[release_id] = list(snapshot.colloquial_mappings)
        return self.releases[release_id]

    async def index_items(self, release_ids: Sequence[uuid.UUID]) -> list[IndexItem]:
        wanted = {
            placement.chunk_version_id
            for release_id in release_ids
            for placement in self.placements.get(release_id, [])
        }
        return [self._item(chunk_id) for chunk_id in sorted(wanted)]

    async def mark_ready(
        self, release_id: uuid.UUID, stats: ReleaseStats, at: datetime
    ) -> None:
        self.releases[release_id] = self.releases[release_id].model_copy(
            update={"status": ReleaseStatus.READY, "stats": stats, "ready_at": at}
        )

    async def publish(self, release_id: uuid.UUID, at: datetime) -> None:
        release = self.releases.get(release_id)
        if release is None:
            raise ReleaseNotFound(str(release_id))
        collection = next(
            c for c in self.collections.values() if c.id == release.collection_id
        )
        self.collections[collection.key] = collection.model_copy(
            update={"current_release_id": release_id}
        )
        if release.published_at is None:
            self.releases[release_id] = release.model_copy(update={"published_at": at})

    async def retire(self, release_ids: Sequence[uuid.UUID], at: datetime) -> None:
        for release_id in release_ids:
            release = self.releases[release_id]
            if release.status is not ReleaseStatus.RETIRED:
                self.releases[release_id] = release.model_copy(
                    update={"status": ReleaseStatus.RETIRED, "retired_at": at}
                )

    async def purge_retired(self, collection_id: uuid.UUID) -> PurgeResult:
        retired = [
            release.id
            for release in self.releases.values()
            if release.collection_id == collection_id
            and release.status is ReleaseStatus.RETIRED
        ]
        placements_deleted = sum(
            len(self.placements.pop(release_id, [])) for release_id in retired
        )
        for release_id in retired:
            self.glossary.pop(release_id, None)
            self.mappings.pop(release_id, None)
        referenced = {
            placement.chunk_version_id
            for placements in self.placements.values()
            for placement in placements
        }
        orphans = [
            chunk_id
            for chunk_id in self.chunks
            if self._collection_of_revision(self.chunks[chunk_id].section_revision_id)
            == collection_id
            and chunk_id not in referenced
        ]
        kept = [chunk_id for chunk_id in orphans if chunk_id in self.protected_chunk_ids]
        for chunk_id in orphans:
            if chunk_id not in self.protected_chunk_ids:
                del self.chunks[chunk_id]
        live_revisions = {
            placement.section_revision_id
            for placements in self.placements.values()
            for placement in placements
        } | {chunk.section_revision_id for chunk in self.chunks.values()}
        orphan_revisions = [
            revision_id
            for revision_id in self.revisions
            if self._collection_of_revision(revision_id) == collection_id
            and revision_id not in live_revisions
        ]
        for revision_id in orphan_revisions:
            del self.revisions[revision_id]
        return PurgeResult(
            release_chunks_deleted=placements_deleted,
            chunk_versions_deleted=len(orphans) - len(kept),
            chunk_versions_kept=len(kept),
            section_revisions_deleted=len(orphan_revisions),
        )

    def _collection_of_revision(self, revision_id: uuid.UUID) -> uuid.UUID:
        section = self.sections[self.revisions[revision_id].section_id]
        return self.documents[section.document_id].collection_id

    def _item(self, chunk_id: uuid.UUID) -> IndexItem:
        chunk = self.chunks[chunk_id]
        section = self.sections[self.revisions[chunk.section_revision_id].section_id]
        return IndexItem(
            chunk_version_id=chunk.id,
            collection_id=self.documents[section.document_id].collection_id,
            document_id=section.document_id,
            section_id=section.id,
            section_revision_id=chunk.section_revision_id,
            kind=chunk.kind,
            embedding_text=chunk.embedding_text,
            embedding_text_sha256=chunk.embedding_text_sha256,
            release_ids=sorted(
                release_id
                for release_id, placements in self.placements.items()
                if self.releases[release_id].status is not ReleaseStatus.RETIRED
                and any(p.chunk_version_id == chunk_id for p in placements)
            ),
        )


class InMemoryEmbeddingCache:
    def __init__(self) -> None:
        self.vectors: dict[tuple[str, str], list[float]] = {}

    async def missing(self, model: str, hashes: Sequence[str]) -> set[str]:
        return {sha for sha in hashes if (model, sha) not in self.vectors}

    async def get_many(self, model: str, hashes: Sequence[str]) -> dict[str, list[float]]:
        return {
            sha: list(self.vectors[(model, sha)])
            for sha in hashes
            if (model, sha) in self.vectors
        }

    async def put_many(
        self, model: str, dimension: int, vectors: Mapping[str, Sequence[float]]
    ) -> None:
        for sha, vector in vectors.items():
            if len(vector) != dimension:
                raise ValueError(f"dimension {len(vector)} != {dimension}")
            self.vectors.setdefault((model, sha), list(vector))


class InMemoryVectorIndex:
    def __init__(self, *, drop_ids: set[uuid.UUID] | None = None) -> None:
        self.points: dict[uuid.UUID, tuple[IndexItem, list[float]]] = {}
        self.ensure_calls = 0
        self._drop_ids = drop_ids or set()

    async def ensure_collection(self) -> str:
        self.ensure_calls += 1
        return "chunks_fake_embedding_4d"

    async def existing_ids(self, ids: Sequence[uuid.UUID]) -> set[uuid.UUID]:
        return {point_id for point_id in ids if point_id in self.points}

    async def upsert(
        self, items: Sequence[IndexItem], vectors: Mapping[str, Sequence[float]]
    ) -> None:
        for item in items:
            vector = vectors.get(item.embedding_text_sha256)
            if vector is None:
                raise ValueError(f"no vector for {item.embedding_text_sha256}")
            if item.chunk_version_id not in self._drop_ids:
                self.points[item.chunk_version_id] = (item, list(vector))

    async def set_release_ids(self, items: Sequence[IndexItem]) -> None:
        for item in items:
            if item.chunk_version_id in self.points:
                stored, vector = self.points[item.chunk_version_id]
                self.points[item.chunk_version_id] = (
                    stored.model_copy(update={"release_ids": list(item.release_ids)}),
                    vector,
                )

    async def delete(self, ids: Sequence[uuid.UUID]) -> None:
        for point_id in ids:
            self.points.pop(point_id, None)

    async def count_release(self, release_id: uuid.UUID) -> int:
        return sum(release_id in item.release_ids for item, _ in self.points.values())

    def release_ids_of(self, chunk_id: uuid.UUID) -> list[uuid.UUID]:
        return self.points[chunk_id][0].release_ids


@dataclass
class CorpusAdapters:
    repository: InMemoryCorpusRepository = field(default_factory=InMemoryCorpusRepository)
    cache: InMemoryEmbeddingCache = field(default_factory=InMemoryEmbeddingCache)
    index: InMemoryVectorIndex = field(default_factory=InMemoryVectorIndex)


def build_importer(
    adapters: CorpusAdapters,
    embedder: Embedder,
    *,
    batch_size: int = 2,
    max_concurrent: int = 1,
) -> ImportKnowledgeBundle:
    return ImportKnowledgeBundle(
        adapters.repository,
        adapters.cache,
        adapters.index,
        embedder,
        FixedClock(NOW),
        embed_batch_size=batch_size,
        embed_max_concurrent=max_concurrent,
    )
```

- [ ] **Step 2: Write the failing tests**

`backend/tests/application/test_import_bundle.py`:

```python
import asyncio
from collections.abc import Sequence

import pytest

from pharma_agent.application.corpus.import_bundle import ImportOutcome
from pharma_agent.application.corpus.indexing import EmbeddingCounts, EmbeddingResolver
from pharma_agent.domain.corpus.bundle import KnowledgeBundle
from pharma_agent.domain.corpus.models import (
    CorpusImportError,
    ReleaseStatus,
    build_snapshot,
)
from pharma_agent.domain.retrieval.ports import RetrievalError
from tests.corpus_fixtures import DOSAGE_SECTION, small_bundle, with_section_text
from tests.corpus_memory import (
    CorpusAdapters,
    InMemoryEmbeddingCache,
    InMemoryVectorIndex,
    build_importer,
)
from tests.fakes import FAKE_EMBEDDING_MODEL, NOW, FakeEmbedder, fake_vector


def without_vectors(bundle: KnowledgeBundle) -> KnowledgeBundle:
    return bundle.model_copy(update={"embeddings": {}})


async def test_import_takes_vectors_from_cache_then_bundle_then_embedder() -> None:
    adapters = CorpusAdapters()
    bundle = small_bundle()
    snapshot = build_snapshot(bundle)
    texts = snapshot.embedding_texts()
    hashes = sorted(texts)
    cached, bundled = hashes[0], hashes[1:3]
    await adapters.cache.put_many(
        FAKE_EMBEDDING_MODEL, 4, {cached: fake_vector(texts[cached])}
    )
    partial = bundle.model_copy(
        update={
            "embeddings": {
                FAKE_EMBEDDING_MODEL: {
                    sha: bundle.embeddings[FAKE_EMBEDDING_MODEL][sha] for sha in bundled
                }
            }
        }
    )
    embedder = FakeEmbedder()

    report = await build_importer(adapters, embedder, batch_size=2)(partial, publish=False)

    assert (report.outcome, report.published, report.collection_key) == (
        ImportOutcome.IMPORTED,
        False,
        "formulary",
    )
    release = report.release
    assert (release.number, release.status, release.ready_at, release.published_at) == (
        1,
        ReleaseStatus.READY,
        NOW,
        None,
    )
    assert release.stats is not None
    assert (
        release.stats.embeddings_cached,
        release.stats.embeddings_from_bundle,
        release.stats.embeddings_computed,
    ) == (1, 2, len(hashes) - 3)
    assert (release.stats.points_upserted, release.stats.points_updated) == (
        len(snapshot.release_chunks),
        0,
    )
    assert release.stats.chunks == len(snapshot.release_chunks)
    assert sorted(text for batch in embedder.batches for text in batch) == sorted(
        texts[sha] for sha in hashes[3:]
    )
    assert all(len(batch) <= 2 for batch in embedder.batches)
    assert set(adapters.cache.vectors) == {(FAKE_EMBEDDING_MODEL, sha) for sha in hashes}
    assert len(adapters.index.points) == len(snapshot.release_chunks)
    assert all(item.release_ids == [release.id] for item, _ in adapters.index.points.values())
    collection = await adapters.repository.get_collection("formulary")
    assert collection is not None and collection.current_release_id is None


async def test_publish_then_same_bundle_is_no_change() -> None:
    adapters = CorpusAdapters()
    embedder = FakeEmbedder()
    first = await build_importer(adapters, embedder)(small_bundle(), publish=True)
    assert first.outcome is ImportOutcome.IMPORTED and first.published
    assert first.release.published_at == NOW
    assert embedder.batches == []  # the fixture ships every vector

    again_embedder = FakeEmbedder()
    again = await build_importer(adapters, again_embedder)(small_bundle(), publish=True)

    assert again.outcome is ImportOutcome.NO_CHANGE
    assert again.release == first.release
    assert adapters.repository.stage_calls == 1
    assert adapters.index.ensure_calls == 1
    assert again_embedder.batches == []
    collection = await adapters.repository.get_collection("formulary")
    assert collection is not None and collection.current_release_id == first.release.id


async def test_ready_unpublished_release_is_reused() -> None:
    adapters = CorpusAdapters()
    first = await build_importer(adapters, FakeEmbedder())(small_bundle(), publish=False)
    second = await build_importer(adapters, FakeEmbedder())(small_bundle(), publish=True)

    assert second.outcome is ImportOutcome.REUSED
    assert second.release.id == first.release.id and second.published
    assert second.release.published_at == NOW
    assert adapters.repository.stage_calls == 1
    assert len(await adapters.repository.list_releases("formulary")) == 1


async def test_failed_embedding_leaves_building_release_and_rerun_resumes() -> None:
    adapters = CorpusAdapters()
    bundle = without_vectors(small_bundle())
    texts = build_snapshot(bundle).embedding_texts()

    with pytest.raises(RetrievalError):
        await build_importer(adapters, FakeEmbedder(fail_on_batch=2), batch_size=2)(
            bundle, publish=True
        )

    (summary,) = await adapters.repository.list_releases("formulary")
    assert summary.release.status is ReleaseStatus.BUILDING and not summary.current
    assert len(adapters.cache.vectors) == 2

    retry = FakeEmbedder()
    report = await build_importer(adapters, retry, batch_size=2)(bundle, publish=True)

    assert report.outcome is ImportOutcome.IMPORTED
    assert report.release.id == summary.release.id
    assert report.release.status is ReleaseStatus.READY and report.published
    assert sum(len(batch) for batch in retry.batches) == len(texts) - 2
    assert report.release.stats is not None
    assert report.release.stats.embeddings_cached == 2
    assert len(await adapters.repository.list_releases("formulary")) == 1


async def test_changed_bundle_adds_release_and_rewrites_release_ids() -> None:
    adapters = CorpusAdapters()
    original = small_bundle()
    first = await build_importer(adapters, FakeEmbedder())(original, publish=True)
    edited = with_section_text(original, DOSAGE_SECTION, "Ghi chú liều mới.")
    embedder = FakeEmbedder()

    second = await build_importer(adapters, embedder)(edited, publish=True)

    old = build_snapshot(original)
    new = build_snapshot(edited)
    old_ids, new_ids = {c.id for c in old.chunks}, {c.id for c in new.chunks}
    assert second.outcome is ImportOutcome.IMPORTED and second.release.number == 2
    assert second.release.stats is not None
    assert second.release.stats.points_upserted == len(new_ids - old_ids)
    assert second.release.stats.points_updated == len(new_ids & old_ids)
    ids = sorted([first.release.id, second.release.id])
    for chunk_id in new_ids & old_ids:
        assert adapters.index.release_ids_of(chunk_id) == ids
    for chunk_id in new_ids - old_ids:
        assert adapters.index.release_ids_of(chunk_id) == [second.release.id]
    for chunk_id in old_ids - new_ids:
        assert adapters.index.release_ids_of(chunk_id) == [first.release.id]
    new_hashes = set(new.embedding_texts()) - set(old.embedding_texts())
    assert sum(len(batch) for batch in embedder.batches) == len(new_hashes)
    collection = await adapters.repository.get_collection("formulary")
    assert collection is not None and collection.current_release_id == second.release.id


async def test_bundle_vectors_declared_with_other_dims_are_ignored() -> None:
    adapters = CorpusAdapters()
    bundle = small_bundle()
    manifest = bundle.manifest.model_copy(
        update={
            "embeddings": [
                entry.model_copy(update={"dims": 8}) for entry in bundle.manifest.embeddings
            ]
        }
    )
    report = await build_importer(adapters, FakeEmbedder())(
        bundle.model_copy(update={"manifest": manifest}), publish=False
    )
    assert report.release.stats is not None
    assert report.release.stats.embeddings_from_bundle == 0
    assert report.release.stats.embeddings_computed == len(
        build_snapshot(bundle).embedding_texts()
    )


async def test_wrong_vector_length_is_rejected_before_any_write() -> None:
    adapters = CorpusAdapters()
    bundle = small_bundle()
    vectors = dict(bundle.embeddings[FAKE_EMBEDDING_MODEL])
    vectors[next(iter(vectors))] = [0.5]
    broken = bundle.model_copy(update={"embeddings": {FAKE_EMBEDDING_MODEL: vectors}})

    with pytest.raises(CorpusImportError, match="dims"):
        await build_importer(adapters, FakeEmbedder())(broken, publish=True)

    assert adapters.repository.stage_calls == 0
    assert adapters.index.ensure_calls == 0


async def test_missing_points_keep_the_release_building() -> None:
    snapshot = build_snapshot(small_bundle())
    adapters = CorpusAdapters(index=InMemoryVectorIndex(drop_ids={snapshot.chunks[0].id}))

    with pytest.raises(CorpusImportError, match="expected"):
        await build_importer(adapters, FakeEmbedder())(small_bundle(), publish=True)

    (summary,) = await adapters.repository.list_releases("formulary")
    assert summary.release.status is ReleaseStatus.BUILDING and not summary.current


class TrackingEmbedder(FakeEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self.active = 0
        self.peak = 0

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        return await super().embed(texts)


async def test_embedding_resolver_respects_the_concurrency_limit() -> None:
    embedder = TrackingEmbedder()
    resolver = EmbeddingResolver(
        InMemoryEmbeddingCache(), embedder, batch_size=1, max_concurrent=2
    )
    counts = await resolver.ensure({f"h{i}": f"text {i}" for i in range(6)}, {})
    assert counts == EmbeddingCounts(cached=0, from_bundle=0, computed=6)
    assert embedder.peak == 2 and len(embedder.batches) == 6
    with pytest.raises(ValueError, match="batch_size"):
        EmbeddingResolver(InMemoryEmbeddingCache(), embedder, batch_size=0, max_concurrent=1)
```

- [ ] **Step 3: Run the tests and watch them fail**

Run: `uv run pytest -q tests/application/test_import_bundle.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'pharma_agent.application.corpus'`.

- [ ] **Step 4: Write `indexing.py`**

`backend/src/pharma_agent/application/corpus/indexing.py`:

```python
"""Embedding and index writing shared by import and reindex (spec C §8.2 steps 7-8).

Vectors never pile up in memory: they are written to the embedding cache batch by batch and
read back per upsert batch, so a rerun resumes from whatever the cache already holds.
"""

import asyncio
import itertools
from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict

from pharma_agent.domain.corpus.models import CorpusImportError, IndexItem
from pharma_agent.domain.corpus.ports import Embedder, EmbeddingCache, VectorIndex

CACHE_WRITE_BATCH = 500
INDEX_BATCH = 256


class EmbeddingCounts(BaseModel):
    model_config = ConfigDict(frozen=True)

    cached: int
    from_bundle: int
    computed: int


class IndexCounts(BaseModel):
    model_config = ConfigDict(frozen=True)

    upserted: int
    updated: int


class EmbeddingResolver:
    def __init__(
        self,
        cache: EmbeddingCache,
        embedder: Embedder,
        *,
        batch_size: int,
        max_concurrent: int,
    ) -> None:
        if batch_size < 1 or max_concurrent < 1:
            raise ValueError("batch_size and max_concurrent must be at least 1")
        self._cache = cache
        self._embedder = embedder
        self._batch_size = batch_size
        self._max_concurrent = max_concurrent

    async def ensure(
        self, texts: Mapping[str, str], bundled: Mapping[str, Sequence[float]]
    ) -> EmbeddingCounts:
        """Make every `texts` hash (sha -> embedding text) present in the cache."""
        model, dimension = self._embedder.model, self._embedder.dimension
        missing = await self._cache.missing(model, list(texts))
        from_bundle = {sha: bundled[sha] for sha in sorted(missing) if sha in bundled}
        for batch in itertools.batched(from_bundle.items(), CACHE_WRITE_BATCH):
            await self._cache.put_many(model, dimension, dict(batch))
        remaining = sorted(missing - from_bundle.keys())
        semaphore = asyncio.Semaphore(self._max_concurrent)

        async def embed_batch(hashes: tuple[str, ...]) -> None:
            async with semaphore:
                vectors = await self._embedder.embed([texts[sha] for sha in hashes])
                if len(vectors) != len(hashes) or any(
                    len(vector) != dimension for vector in vectors
                ):
                    raise CorpusImportError(
                        f"embedder {model} returned vectors that do not match "
                        f"{len(hashes)} inputs of {dimension} dims"
                    )
                await self._cache.put_many(
                    model, dimension, dict(zip(hashes, vectors, strict=True))
                )

        try:
            async with asyncio.TaskGroup() as group:
                for batch in itertools.batched(remaining, self._batch_size):
                    group.create_task(embed_batch(batch))
        except ExceptionGroup as errors:
            raise errors.exceptions[0] from errors
        return EmbeddingCounts(
            cached=len(texts) - len(missing),
            from_bundle=len(from_bundle),
            computed=len(remaining),
        )


class IndexWriter:
    def __init__(
        self,
        index: VectorIndex,
        cache: EmbeddingCache,
        *,
        model: str,
        batch_size: int = INDEX_BATCH,
    ) -> None:
        self._index = index
        self._cache = cache
        self._model = model
        self._batch_size = batch_size

    async def write(
        self, items: Sequence[IndexItem], *, overwrite: bool = False
    ) -> IndexCounts:
        """Upsert points that are missing (or all with `overwrite`), rewrite release_ids of
        the rest. Vectors must already be in the cache."""
        if overwrite:
            new, existing = list(items), []
        else:
            present = await self._index.existing_ids(
                [item.chunk_version_id for item in items]
            )
            new = [item for item in items if item.chunk_version_id not in present]
            existing = [item for item in items if item.chunk_version_id in present]
        for batch in itertools.batched(new, self._batch_size):
            hashes = sorted({item.embedding_text_sha256 for item in batch})
            vectors = await self._cache.get_many(self._model, hashes)
            absent = set(hashes) - vectors.keys()
            if absent:
                raise CorpusImportError(
                    f"{len(absent)} embeddings for model {self._model} are not cached, "
                    f"first {min(absent)}"
                )
            await self._index.upsert(batch, vectors)
        if existing:
            await self._index.set_release_ids(existing)
        return IndexCounts(upserted=len(new), updated=len(existing))
```

- [ ] **Step 5: Write `import_bundle.py`**

`backend/src/pharma_agent/application/corpus/import_bundle.py`:

```python
"""Import a knowledge bundle as a new corpus release (spec C §8.2).

The only ingest path: the CLI calls it today and a future upload API will call the same
service. Every step is idempotent on hashed ids, so rerunning after a failure resumes the
`building` release from the embedding cache and the points already indexed.
"""

import uuid
from collections.abc import Mapping
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from pharma_agent.application.corpus.indexing import EmbeddingResolver, IndexWriter
from pharma_agent.domain.corpus.bundle import KnowledgeBundle, model_slug
from pharma_agent.domain.corpus.models import (
    CorpusImportError,
    CorpusSnapshot,
    Release,
    ReleaseNotFound,
    ReleaseStatus,
    build_snapshot,
)
from pharma_agent.domain.corpus.ports import (
    CorpusRepository,
    Embedder,
    EmbeddingCache,
    VectorIndex,
)
from pharma_agent.domain.shared.clock import Clock


class ImportOutcome(StrEnum):
    IMPORTED = "imported"
    REUSED = "reused"
    NO_CHANGE = "no_change"


class ImportReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    outcome: ImportOutcome
    collection_key: str
    release: Release
    published: bool


def bundle_vectors(
    bundle: KnowledgeBundle, model: str, dimension: int
) -> Mapping[str, list[float]]:
    """Precomputed vectors usable for `model`, keyed by embedding_text_sha256.

    Only vectors declared in the manifest with the same model and dims are used. Raises
    CorpusImportError when a declared vector has another length.
    """
    declared = any(
        entry.model == model and entry.dims == dimension
        for entry in bundle.manifest.embeddings
    )
    if not declared:
        return {}
    vectors = bundle.embeddings.get(model, {})
    wrong = sorted(sha for sha, vector in vectors.items() if len(vector) != dimension)
    if wrong:
        raise CorpusImportError(
            f"embeddings/{model_slug(model)}.jsonl: {len(wrong)} vectors do not have "
            f"{dimension} dims, first {wrong[0]}"
        )
    return vectors


class ImportKnowledgeBundle:
    def __init__(
        self,
        repository: CorpusRepository,
        cache: EmbeddingCache,
        index: VectorIndex,
        embedder: Embedder,
        clock: Clock,
        *,
        embed_batch_size: int,
        embed_max_concurrent: int,
    ) -> None:
        self._repository = repository
        self._index = index
        self._embedder = embedder
        self._clock = clock
        self._resolver = EmbeddingResolver(
            cache,
            embedder,
            batch_size=embed_batch_size,
            max_concurrent=embed_max_concurrent,
        )
        self._writer = IndexWriter(index, cache, model=embedder.model)

    async def __call__(self, bundle: KnowledgeBundle, *, publish: bool) -> ImportReport:
        model = self._embedder.model
        snapshot = build_snapshot(bundle)
        vectors = bundle_vectors(bundle, model, self._embedder.dimension)
        key = snapshot.collection.key

        current = await self._current_release(key)
        if current is not None and (
            current.bundle_digest,
            current.chunker_version,
            current.embedding_model,
        ) == (snapshot.bundle_digest, snapshot.chunker_version, model):
            return ImportReport(
                outcome=ImportOutcome.NO_CHANGE,
                collection_key=key,
                release=current,
                published=True,
            )

        await self._index.ensure_collection()
        existing = await self._repository.find_release(
            snapshot.collection.id,
            bundle_digest=snapshot.bundle_digest,
            chunker_version=snapshot.chunker_version,
            embedding_model=model,
        )
        if existing is not None and existing.status is ReleaseStatus.READY:
            release, outcome = existing, ImportOutcome.REUSED
        else:
            release = await self._repository.stage_release(
                snapshot,
                release_id=existing.id if existing is not None else uuid.uuid4(),
                embedding_model=model,
                at=self._clock.now(),
            )
            await self._build(snapshot, vectors, release)
            outcome = ImportOutcome.IMPORTED

        if publish:
            await self._repository.publish(release.id, self._clock.now())
        stored = await self._repository.get_release(release.id)
        if stored is None:
            raise ReleaseNotFound(str(release.id))
        return ImportReport(
            outcome=outcome, collection_key=key, release=stored, published=publish
        )

    async def _current_release(self, collection_key: str) -> Release | None:
        collection = await self._repository.get_collection(collection_key)
        if collection is None or collection.current_release_id is None:
            return None
        return await self._repository.get_release(collection.current_release_id)

    async def _build(
        self,
        snapshot: CorpusSnapshot,
        vectors: Mapping[str, list[float]],
        release: Release,
    ) -> None:
        embeddings = await self._resolver.ensure(snapshot.embedding_texts(), vectors)
        items = await self._repository.index_items([release.id])
        written = await self._writer.write(items)
        indexed = await self._index.count_release(release.id)
        expected = len(snapshot.release_chunks)
        if indexed != expected:
            raise CorpusImportError(
                f"release {release.number}: {indexed} points carry the release id, "
                f"expected {expected}; the release stays building"
            )
        stats = snapshot.stats().model_copy(
            update={
                "embeddings_cached": embeddings.cached,
                "embeddings_from_bundle": embeddings.from_bundle,
                "embeddings_computed": embeddings.computed,
                "points_upserted": written.upserted,
                "points_updated": written.updated,
            }
        )
        await self._repository.mark_ready(release.id, stats, self._clock.now())
```

- [ ] **Step 6: Run the tests**

Run: `uv run pytest -q tests/application/test_import_bundle.py tests/architecture/test_layering.py`
Expected: PASS (9 passed in the new file).

- [ ] **Step 7: Run the full check**

Run: `uv run ruff check src tests && uv run ruff format --check src tests && uv run pyrefly check --min-severity warn && uv run pytest -q`
Expected: all green.

- [ ] **Step 8: Commit**

```bash
git add src/pharma_agent/application/corpus \
  tests/corpus_memory.py \
  tests/application/test_import_bundle.py
git commit -m "feat(corpus): import knowledge bundles into resumable releases"
```

The commit message ends with the session attribution trailer.

---

### Task 7: `ReleaseService`: list, publish, rollback, gc, reindex

**Files:**
- Create: `backend/src/pharma_agent/application/corpus/releases.py`
- Modify: `backend/tests/corpus_memory.py` (imports; `CorpusAdapters` block to end of file)
- Test: `backend/tests/application/test_release_service.py`

**Interfaces:**
- Consumes: `EmbeddingResolver`, `IndexWriter` (Task 6); `releases_to_retire`, `Release`, `ReleaseStatus`, `ReleaseSummary`, `PurgeResult`, `CollectionNotFound`, `ReleaseNotFound`, `ReleaseNotPublishable`, `NoEarlierRelease` (Task 3); ports (Task 3); `Clock` (`domain/shared/clock.py`).
- Produces: `GcReport`, `ReindexReport`, `ReleaseService` as listed in "Interfaces pinned by this plan"; test helpers `SteppingClock(start: datetime = NOW, step: timedelta = timedelta(minutes=1))`, `build_importer(adapters, embedder, *, clock: Clock | None = None, batch_size: int = 2, max_concurrent: int = 1)`, `build_release_service(adapters, embedder, *, clock: Clock | None = None, batch_size: int = 2, max_concurrent: int = 1) -> ReleaseService`.

Rules implemented (spec C §8.1, §8.5):

- `publish(release_id)`: only a `ready` release; `repository.publish` sets `current_release_id` and the first `published_at` in one transaction.
- `rollback(collection_key)`: the `ready` release with the greatest `published_at` earlier than the current release's `published_at`; none → `NoEarlierRelease`.
- `gc(collection_key, keep)`: (1) retire `releases_to_retire(...)`; (2) for every retired release that still has `release_chunks` (this makes a crashed gc resumable) recompute `release_ids`, rewrite them on existing points and delete points whose list is empty; (3–4) `repository.purge_retired`; the embedding cache is never touched.
- `reindex(collection_key)`: `ensure_collection`, load items of every non-retired release built with the configured embedding model (others are reported as `skipped` and removed from payload `release_ids`), fill missing cache entries through the embedder, upsert every point, count points per release.

- [ ] **Step 1: Extend the in-memory helpers**

In `backend/tests/corpus_memory.py` make three import changes:

- replace `from datetime import datetime` with `from datetime import datetime, timedelta`;
- replace `from pharma_agent.domain.shared.clock import FixedClock` with `from pharma_agent.domain.shared.clock import Clock, FixedClock`;
- add `from pharma_agent.application.corpus.releases import ReleaseService` directly below `from pharma_agent.application.corpus.import_bundle import ImportKnowledgeBundle`.

Then replace everything from `@dataclass` above `class CorpusAdapters` to the end of the file with:

```python
@dataclass
class CorpusAdapters:
    repository: InMemoryCorpusRepository = field(default_factory=InMemoryCorpusRepository)
    cache: InMemoryEmbeddingCache = field(default_factory=InMemoryEmbeddingCache)
    index: InMemoryVectorIndex = field(default_factory=InMemoryVectorIndex)


class SteppingClock:
    """Each call returns a later time, so published_at values are ordered."""

    def __init__(
        self, start: datetime = NOW, step: timedelta = timedelta(minutes=1)
    ) -> None:
        self._next = start
        self._step = step

    def now(self) -> datetime:
        current = self._next
        self._next = current + self._step
        return current


def build_importer(
    adapters: CorpusAdapters,
    embedder: Embedder,
    *,
    clock: Clock | None = None,
    batch_size: int = 2,
    max_concurrent: int = 1,
) -> ImportKnowledgeBundle:
    return ImportKnowledgeBundle(
        adapters.repository,
        adapters.cache,
        adapters.index,
        embedder,
        clock if clock is not None else FixedClock(NOW),
        embed_batch_size=batch_size,
        embed_max_concurrent=max_concurrent,
    )


def build_release_service(
    adapters: CorpusAdapters,
    embedder: Embedder,
    *,
    clock: Clock | None = None,
    batch_size: int = 2,
    max_concurrent: int = 1,
) -> ReleaseService:
    return ReleaseService(
        adapters.repository,
        adapters.cache,
        adapters.index,
        embedder,
        clock if clock is not None else FixedClock(NOW),
        embed_batch_size=batch_size,
        embed_max_concurrent=max_concurrent,
    )
```

- [ ] **Step 2: Write the failing tests**

`backend/tests/application/test_release_service.py`:

```python
import uuid
from dataclasses import dataclass

import pytest

from pharma_agent.application.corpus.import_bundle import ImportKnowledgeBundle
from pharma_agent.application.corpus.releases import ReleaseService
from pharma_agent.domain.corpus.bundle import KnowledgeBundle
from pharma_agent.domain.corpus.models import (
    CollectionNotFound,
    NoEarlierRelease,
    Release,
    ReleaseNotFound,
    ReleaseNotPublishable,
    ReleaseStatus,
    build_snapshot,
)
from pharma_agent.domain.retrieval.ports import RetrievalError
from tests.corpus_fixtures import (
    DOSAGE_SECTION,
    LEAFLET_SECTION,
    small_bundle,
    with_section_text,
)
from tests.corpus_memory import (
    CorpusAdapters,
    SteppingClock,
    build_importer,
    build_release_service,
)
from tests.fakes import FakeEmbedder


@dataclass
class Services:
    adapters: CorpusAdapters
    embedder: FakeEmbedder
    clock: SteppingClock
    importer: ImportKnowledgeBundle
    releases: ReleaseService

    async def publish_bundle(self, bundle: KnowledgeBundle, *, publish: bool = True) -> Release:
        return (await self.importer(bundle, publish=publish)).release


def services() -> Services:
    adapters, embedder, clock = CorpusAdapters(), FakeEmbedder(), SteppingClock()
    return Services(
        adapters=adapters,
        embedder=embedder,
        clock=clock,
        importer=build_importer(adapters, embedder, clock=clock),
        releases=build_release_service(adapters, embedder, clock=clock),
    )


def v2() -> KnowledgeBundle:
    return with_section_text(small_bundle(), DOSAGE_SECTION, "Ghi chú liều v2.")


def v3() -> KnowledgeBundle:
    return with_section_text(v2(), LEAFLET_SECTION, "Ghi chú tờ HDSD v3.")


def chunk_ids(bundle: KnowledgeBundle) -> set[uuid.UUID]:
    return {chunk.id for chunk in build_snapshot(bundle).chunks}


async def test_list_and_publish_follow_release_status() -> None:
    s = services()
    r1 = await s.publish_bundle(small_bundle())
    r2 = await s.publish_bundle(v2(), publish=False)

    summaries = await s.releases.list_releases("formulary")
    assert [(x.release.number, x.current) for x in summaries] == [(2, False), (1, True)]

    published = await s.releases.publish(r2.id)
    assert published.id == r2.id and published.published_at is not None
    collection = await s.adapters.repository.get_collection("formulary")
    assert collection is not None and collection.current_release_id == r2.id

    with pytest.raises(RetrievalError):
        await build_importer(s.adapters, FakeEmbedder(fail_on_batch=1), clock=s.clock)(
            v3(), publish=False
        )
    building = (await s.releases.list_releases("formulary"))[0].release
    assert building.status is ReleaseStatus.BUILDING
    with pytest.raises(ReleaseNotPublishable, match="building"):
        await s.releases.publish(building.id)
    with pytest.raises(ReleaseNotFound):
        await s.releases.publish(uuid.uuid4())
    assert len(await s.releases.list_releases(None)) == 3
    assert r1.number == 1


async def test_rollback_points_back_to_the_previously_published_release() -> None:
    s = services()
    r1 = await s.publish_bundle(small_bundle())
    r2 = await s.publish_bundle(v2())

    rolled = await s.releases.rollback("formulary")
    assert rolled.id == r1.id
    collection = await s.adapters.repository.get_collection("formulary")
    assert collection is not None and collection.current_release_id == r1.id
    with pytest.raises(NoEarlierRelease):
        await s.releases.rollback("formulary")

    await s.releases.publish(r2.id)
    assert (await s.releases.rollback("formulary")).id == r1.id
    with pytest.raises(CollectionNotFound):
        await s.releases.rollback("missing")


async def test_gc_retires_old_releases_cleans_points_and_keeps_protected_chunks() -> None:
    s = services()
    r1 = await s.publish_bundle(small_bundle())
    r2 = await s.publish_bundle(v2())
    r3 = await s.publish_bundle(v3())
    ids1, ids2, ids3 = chunk_ids(small_bundle()), chunk_ids(v2()), chunk_ids(v3())
    gone = (ids1 | ids2) - ids3
    assert len(gone) >= 2
    protected = sorted(ids1 - ids2 - ids3)[0]
    s.adapters.repository.protected_chunk_ids.add(protected)

    report = await s.releases.gc("formulary", keep=1)

    assert report.retired == [r2.id, r1.id]
    assert set(s.adapters.index.points) == ids3
    assert all(
        item.release_ids == [r3.id] for item, _ in s.adapters.index.points.values()
    )
    assert report.points_deleted == len(gone)
    assert report.points_updated == len((ids1 | ids2) & ids3)
    assert report.purge.chunk_versions_kept == 1
    assert report.purge.chunk_versions_deleted == len(gone) - 1
    assert protected in s.adapters.repository.chunks
    statuses = {
        x.release.id: (x.release.status, x.chunk_count, x.current)
        for x in await s.releases.list_releases("formulary")
    }
    assert statuses[r3.id][0] is ReleaseStatus.READY and statuses[r3.id][2]
    assert statuses[r1.id][:2] == (ReleaseStatus.RETIRED, 0)
    assert statuses[r2.id][:2] == (ReleaseStatus.RETIRED, 0)

    again = await s.releases.gc("formulary", keep=1)
    assert (again.retired, again.points_updated, again.points_deleted) == ([], 0, 0)
    assert again.purge.chunk_versions_kept == 1


async def test_reindex_rebuilds_points_from_repository_and_cache() -> None:
    s = services()
    r1 = await s.publish_bundle(small_bundle())
    r2 = await s.publish_bundle(v2())
    ids1, ids2 = chunk_ids(small_bundle()), chunk_ids(v2())
    snapshot1, snapshot2 = build_snapshot(small_bundle()), build_snapshot(v2())

    foreign = r1.model_copy(
        update={"id": uuid.uuid4(), "number": 9, "embedding_model": "other-model"}
    )
    s.adapters.repository.releases[foreign.id] = foreign
    s.adapters.repository.placements[foreign.id] = list(snapshot1.release_chunks)

    s.adapters.index.points.clear()
    dropped = next(iter(snapshot2.embedding_texts()))
    del s.adapters.cache.vectors[(s.embedder.model, dropped)]
    s.embedder.batches.clear()

    report = await s.releases.reindex("formulary")

    assert report.points_upserted == len(ids1 | ids2)
    assert report.release_points == {
        r1.id: len(snapshot1.release_chunks),
        r2.id: len(snapshot2.release_chunks),
    }
    assert report.skipped == [foreign.id]
    shared = next(iter(ids1 & ids2))
    assert s.adapters.index.release_ids_of(shared) == sorted([r1.id, r2.id])
    assert s.embedder.batches == [[snapshot2.embedding_texts()[dropped]]]
```

- [ ] **Step 3: Run the tests and watch them fail**

Run: `uv run pytest -q tests/application/test_release_service.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'pharma_agent.application.corpus.releases'` (raised while importing `tests/corpus_memory.py`).

- [ ] **Step 4: Write `releases.py`**

`backend/src/pharma_agent/application/corpus/releases.py`:

```python
"""Release commands (spec C §8.1, §8.5): list, publish, rollback, gc, reindex."""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from pharma_agent.application.corpus.indexing import EmbeddingResolver, IndexWriter
from pharma_agent.domain.corpus.models import (
    Collection,
    CollectionNotFound,
    NoEarlierRelease,
    PurgeResult,
    Release,
    ReleaseNotFound,
    ReleaseNotPublishable,
    ReleaseStatus,
    ReleaseSummary,
    releases_to_retire,
)
from pharma_agent.domain.corpus.ports import (
    CorpusRepository,
    Embedder,
    EmbeddingCache,
    VectorIndex,
)
from pharma_agent.domain.shared.clock import Clock


class GcReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    retired: list[uuid.UUID]
    points_updated: int
    points_deleted: int
    purge: PurgeResult


class ReindexReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    points_upserted: int
    release_points: dict[uuid.UUID, int]
    skipped: list[uuid.UUID]


class ReleaseService:
    def __init__(
        self,
        repository: CorpusRepository,
        cache: EmbeddingCache,
        index: VectorIndex,
        embedder: Embedder,
        clock: Clock,
        *,
        embed_batch_size: int,
        embed_max_concurrent: int,
    ) -> None:
        self._repository = repository
        self._index = index
        self._embedder = embedder
        self._clock = clock
        self._resolver = EmbeddingResolver(
            cache,
            embedder,
            batch_size=embed_batch_size,
            max_concurrent=embed_max_concurrent,
        )
        self._writer = IndexWriter(index, cache, model=embedder.model)

    async def list_releases(self, collection_key: str | None = None) -> list[ReleaseSummary]:
        return await self._repository.list_releases(collection_key)

    async def publish(self, release_id: uuid.UUID) -> Release:
        release = await self._release(release_id)
        if release.status is not ReleaseStatus.READY:
            raise ReleaseNotPublishable(
                f"release {release.number} ({release.id}) is {release.status.value}"
            )
        await self._repository.publish(release_id, self._clock.now())
        return await self._release(release_id)

    async def rollback(self, collection_key: str) -> Release:
        collection = await self._collection(collection_key)
        if collection.current_release_id is None:
            raise NoEarlierRelease(f"collection {collection_key} has no current release")
        current = await self._release(collection.current_release_id)
        current_published_at = current.published_at
        if current_published_at is None:
            raise NoEarlierRelease(f"current release {current.number} was never published")
        candidates: list[tuple[datetime, int, Release]] = []
        for summary in await self._repository.list_releases(collection_key):
            published_at = summary.release.published_at
            if (
                summary.release.status is ReleaseStatus.READY
                and published_at is not None
                and published_at < current_published_at
            ):
                candidates.append((published_at, summary.release.number, summary.release))
        if not candidates:
            raise NoEarlierRelease(
                f"collection {collection_key} has no release published before "
                f"release {current.number}"
            )
        _, _, target = max(candidates, key=lambda candidate: candidate[:2])
        await self._repository.publish(target.id, self._clock.now())
        return await self._release(target.id)

    async def gc(self, collection_key: str, keep: int) -> GcReport:
        collection = await self._collection(collection_key)
        summaries = await self._repository.list_releases(collection_key)
        doomed = releases_to_retire(
            [summary.release for summary in summaries],
            collection.current_release_id,
            keep,
        )
        await self._repository.retire([release.id for release in doomed], self._clock.now())

        pending = [
            summary.release.id
            for summary in await self._repository.list_releases(collection_key)
            if summary.release.status is ReleaseStatus.RETIRED and summary.chunk_count > 0
        ]
        points_updated = points_deleted = 0
        if pending:
            await self._index.ensure_collection()
            items = await self._repository.index_items(pending)
            present = await self._index.existing_ids(
                [item.chunk_version_id for item in items]
            )
            live = [
                item
                for item in items
                if item.release_ids and item.chunk_version_id in present
            ]
            dead = [
                item.chunk_version_id
                for item in items
                if not item.release_ids and item.chunk_version_id in present
            ]
            if live:
                await self._index.set_release_ids(live)
            if dead:
                await self._index.delete(dead)
            points_updated, points_deleted = len(live), len(dead)

        purge = await self._repository.purge_retired(collection.id)
        return GcReport(
            retired=[release.id for release in doomed],
            points_updated=points_updated,
            points_deleted=points_deleted,
            purge=purge,
        )

    async def reindex(self, collection_key: str) -> ReindexReport:
        await self._collection(collection_key)
        live = [
            summary.release
            for summary in await self._repository.list_releases(collection_key)
            if summary.release.status is not ReleaseStatus.RETIRED
        ]
        usable = [r for r in live if r.embedding_model == self._embedder.model]
        usable_ids = {release.id for release in usable}
        skipped = [release.id for release in live if release.id not in usable_ids]

        await self._index.ensure_collection()
        items = [
            item.model_copy(
                update={
                    "release_ids": [r for r in item.release_ids if r in usable_ids]
                }
            )
            for item in await self._repository.index_items(
                [release.id for release in usable]
            )
        ]
        await self._resolver.ensure(
            {item.embedding_text_sha256: item.embedding_text for item in items}, {}
        )
        written = await self._writer.write(items, overwrite=True)
        release_points = {
            release.id: await self._index.count_release(release.id) for release in usable
        }
        return ReindexReport(
            points_upserted=written.upserted,
            release_points=release_points,
            skipped=skipped,
        )

    async def _collection(self, key: str) -> Collection:
        collection = await self._repository.get_collection(key)
        if collection is None:
            raise CollectionNotFound(f"collection {key} does not exist")
        return collection

    async def _release(self, release_id: uuid.UUID) -> Release:
        release = await self._repository.get_release(release_id)
        if release is None:
            raise ReleaseNotFound(f"release {release_id} does not exist")
        return release
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest -q tests/application/test_release_service.py tests/application/test_import_bundle.py`
Expected: PASS (4 passed in the new file; the Task 6 tests still pass with the new `build_importer` signature).

- [ ] **Step 6: Run the full check**

Run: `uv run ruff check src tests && uv run ruff format --check src tests && uv run pyrefly check --min-severity warn && uv run pytest -q`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add src/pharma_agent/application/corpus/releases.py \
  tests/corpus_memory.py \
  tests/application/test_release_service.py
git commit -m "feat(corpus): add release publish, rollback, gc and reindex service"
```

The commit message ends with the session attribution trailer.

---

### Task 8: `CorpusSettings`, `open_corpus_services` and the `pharma-agent corpus` commands

**Files:**
- Modify: `backend/src/pharma_agent/infrastructure/settings.py:185-208` (add `CorpusSettings` after `CheckpointSettings`, field `corpus` on `Settings`)
- Create: `backend/src/pharma_agent/infrastructure/corpus_factory.py`
- Modify: `backend/src/pharma_agent/cli.py` (imports at lines 3-14; corpus sub-app appended at the end of the file)
- Modify: `backend/tests/infrastructure/test_settings.py` (append one test)
- Test: `backend/tests/infrastructure/test_corpus_factory.py`, `backend/tests/test_cli_corpus.py`

**Interfaces:**
- Consumes: `ImportKnowledgeBundle`, `ImportReport` (Task 6); `ReleaseService` (Task 7); `PostgresCorpusRepository`, `PostgresEmbeddingCache` (Task 4); `QdrantVectorIndex` (Task 5); `OpenAiEmbedder` (with `model`/`dimension`, Task 2); `Database`; `build_async_openai` (`infrastructure/openai_client.py`); `SystemClock`; `read_bundle`, `BundleValidationError` (P1); `DomainError`; `ReleaseSummary` (Task 3); test helpers from Tasks 2, 6, 7.
- Produces: `CorpusSettings(embed_batch_size: int = 32, embed_max_concurrent: int = 1, gc_keep: int = 2)` and `Settings.corpus` (env `PHARMA_CORPUS__EMBED_BATCH_SIZE`, `PHARMA_CORPUS__EMBED_MAX_CONCURRENT`, `PHARMA_CORPUS__GC_KEEP`); `CorpusServices(importer, releases, index, embedder)`; `open_corpus_services(settings, *, embedder: Embedder | None = None, alias: str = CURRENT_ALIAS)`; typer sub-app `corpus_app` mounted as `pharma-agent corpus` with:

| Command | Signature (spec C §8.1) | Exit codes |
| --- | --- | --- |
| `import` | `pharma-agent corpus import <bundle_dir> --collection <key> [--publish]` | 0; 2 invalid bundle or collection mismatch (no connection opened); 1 corpus/domain error |
| `releases` | `pharma-agent corpus releases [--collection <key>]` | 0; 1 |
| `publish` | `pharma-agent corpus publish <release_id>` | 0; 1 |
| `rollback` | `pharma-agent corpus rollback --collection <key>` | 0; 1 |
| `gc` | `pharma-agent corpus gc --collection <key> [--keep N]` (default `Settings.corpus.gc_keep`) | 0; 1 |
| `reindex` | `pharma-agent corpus reindex --collection <key>` | 0; 1 |

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/infrastructure/test_settings.py`:

```python
def test_corpus_settings_defaults_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    corpus = Settings(_env_file=None).corpus
    assert (corpus.embed_batch_size, corpus.embed_max_concurrent, corpus.gc_keep) == (
        32,
        1,
        2,
    )
    monkeypatch.setenv("PHARMA_CORPUS__EMBED_MAX_CONCURRENT", "4")
    monkeypatch.setenv("PHARMA_CORPUS__GC_KEEP", "0")
    configured = Settings(_env_file=None).corpus
    assert (configured.embed_max_concurrent, configured.gc_keep) == (4, 0)
    monkeypatch.setenv("PHARMA_CORPUS__EMBED_BATCH_SIZE", "0")
    with pytest.raises(ValueError, match="embed_batch_size"):
        Settings(_env_file=None)
```

`backend/tests/infrastructure/test_corpus_factory.py`:

```python
import pytest

from pharma_agent.application.corpus.import_bundle import ImportKnowledgeBundle
from pharma_agent.application.corpus.releases import ReleaseService
from pharma_agent.infrastructure.corpus_factory import open_corpus_services
from pharma_agent.infrastructure.retrieval.qdrant_adapter import OpenAiEmbedder
from pharma_agent.infrastructure.retrieval.qdrant_index import QdrantVectorIndex
from pharma_agent.infrastructure.settings import Settings
from tests.fakes import FakeEmbedder


async def test_open_corpus_services_wires_real_adapters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHARMA_QDRANT__CHECK_COMPATIBILITY", "false")
    monkeypatch.setenv("PHARMA_RETRIEVAL__EMBEDDING__MODEL", "qwen3-embedding:0.6b")
    monkeypatch.setenv("PHARMA_RETRIEVAL__EMBEDDING__DIMENSION", "1024")
    settings = Settings(_env_file=None)

    async with open_corpus_services(settings) as services:
        assert isinstance(services.importer, ImportKnowledgeBundle)
        assert isinstance(services.releases, ReleaseService)
        assert isinstance(services.index, QdrantVectorIndex)
        assert services.index.collection_name == "chunks_qwen3_embedding_0_6b"
        assert isinstance(services.embedder, OpenAiEmbedder)
        assert (services.embedder.model, services.embedder.dimension) == (
            "qwen3-embedding:0.6b",
            1024,
        )


async def test_open_corpus_services_uses_an_injected_embedder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHARMA_QDRANT__CHECK_COMPATIBILITY", "false")
    embedder = FakeEmbedder()

    async with open_corpus_services(
        Settings(_env_file=None), embedder=embedder
    ) as services:
        assert services.embedder is embedder
        assert isinstance(services.index, QdrantVectorIndex)
        assert services.index.collection_name == "chunks_fake_embedding_4d"
        assert services.index.alias == "chunks_current"

    async with open_corpus_services(
        Settings(_env_file=None), embedder=embedder, alias="e2e_chunks_current"
    ) as services:
        assert isinstance(services.index, QdrantVectorIndex)
        assert services.index.alias == "e2e_chunks_current"
```

`backend/tests/test_cli_corpus.py`:

```python
import asyncio
import shutil
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from click.testing import Result
from typer.testing import CliRunner

from pharma_agent import cli
from pharma_agent.domain.corpus.models import build_snapshot
from pharma_agent.infrastructure.corpus_factory import CorpusServices
from pharma_agent.infrastructure.settings import Settings
from tests.corpus_fixtures import (
    DOSAGE_SECTION,
    FIXTURE_DIR,
    small_bundle,
    with_section_text,
)
from tests.corpus_memory import (
    CorpusAdapters,
    SteppingClock,
    build_importer,
    build_release_service,
)
from tests.fakes import FakeEmbedder


@dataclass
class FakeCorpus:
    adapters: CorpusAdapters = field(default_factory=CorpusAdapters)
    embedder: FakeEmbedder = field(default_factory=FakeEmbedder)
    clock: SteppingClock = field(default_factory=SteppingClock)
    opened: int = 0

    @asynccontextmanager
    async def open(self, settings: Settings) -> AsyncGenerator[CorpusServices]:
        self.opened += 1
        yield CorpusServices(
            importer=build_importer(self.adapters, self.embedder, clock=self.clock),
            releases=build_release_service(self.adapters, self.embedder, clock=self.clock),
            index=self.adapters.index,
            embedder=self.embedder,
        )


@pytest.fixture
def corpus(monkeypatch: pytest.MonkeyPatch) -> FakeCorpus:
    fake = FakeCorpus()
    monkeypatch.setattr(cli, "open_corpus_services", fake.open)
    return fake


def invoke(*args: str) -> Result:
    return CliRunner().invoke(cli.app, ["corpus", *args])


def test_import_publishes_then_reports_no_change(corpus: FakeCorpus) -> None:
    args = ("import", str(FIXTURE_DIR), "--collection", "formulary", "--publish")

    first = invoke(*args)
    assert first.exit_code == 0, first.output
    assert "imported: formulary release 1 " in first.output
    assert "[ready] published" in first.output
    assert "chunks " in first.output and "points new " in first.output

    again = invoke(*args)
    assert again.exit_code == 0, again.output
    assert "no_change: formulary release 1 " in again.output
    assert corpus.opened == 2


def test_import_rejects_wrong_collection_and_invalid_bundle(
    corpus: FakeCorpus, tmp_path: Path
) -> None:
    mismatch = invoke("import", str(FIXTURE_DIR), "--collection", "other")
    assert mismatch.exit_code == 2
    assert "bundle collection is formulary, not other" in mismatch.output

    broken = tmp_path / "bundle"
    shutil.copytree(FIXTURE_DIR, broken)
    with (broken / "documents.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    invalid = invoke("import", str(broken), "--collection", "formulary")
    assert invalid.exit_code == 2
    assert "BUNDLE_INVALID" in invalid.output
    assert corpus.opened == 0


def test_release_commands(corpus: FakeCorpus, monkeypatch: pytest.MonkeyPatch) -> None:
    imported = invoke("import", str(FIXTURE_DIR), "--collection", "formulary", "--publish")
    assert imported.exit_code == 0, imported.output
    edited = with_section_text(small_bundle(), DOSAGE_SECTION, "Ghi chú từ CLI.")
    second = asyncio.run(
        build_importer(corpus.adapters, corpus.embedder, clock=corpus.clock)(
            edited, publish=True
        )
    ).release

    listing = invoke("releases")
    assert listing.exit_code == 0, listing.output
    lines = listing.output.splitlines()
    assert lines[0].startswith(f"* formulary #2 {second.id} ready chunks=")
    assert lines[1].startswith("  formulary #1 ")
    assert "no releases" in invoke("releases", "--collection", "missing").output

    rolled = invoke("rollback", "--collection", "formulary")
    assert rolled.exit_code == 0, rolled.output
    assert "rolled back formulary to release 1 " in rolled.output
    republished = invoke("publish", str(second.id))
    assert republished.exit_code == 0, republished.output
    assert f"published release 2 {second.id}" in republished.output
    missing = invoke("publish", str(uuid.uuid4()))
    assert missing.exit_code == 1 and "RELEASE_NOT_FOUND" in missing.output

    monkeypatch.setenv("PHARMA_CORPUS__GC_KEEP", "5")
    kept = invoke("gc", "--collection", "formulary")
    assert kept.exit_code == 0, kept.output
    assert "retired 0 releases (keep 5)" in kept.output
    collected = invoke("gc", "--collection", "formulary", "--keep", "1")
    assert collected.exit_code == 0, collected.output
    assert "retired 1 releases (keep 1)" in collected.output

    corpus.adapters.index.points.clear()
    reindexed = invoke("reindex", "--collection", "formulary")
    assert reindexed.exit_code == 0, reindexed.output
    assert f"points upserted {len(build_snapshot(edited).chunks)}" in reindexed.output
    assert f"release {second.id}: " in reindexed.output

    unknown = invoke("rollback", "--collection", "missing")
    assert unknown.exit_code == 1 and "COLLECTION_NOT_FOUND" in unknown.output
```

- [ ] **Step 2: Run the tests and watch them fail**

Run: `uv run pytest -q tests/infrastructure/test_settings.py tests/infrastructure/test_corpus_factory.py tests/test_cli_corpus.py`
Expected: FAIL: `AttributeError: 'Settings' object has no attribute 'corpus'` and `ModuleNotFoundError: No module named 'pharma_agent.infrastructure.corpus_factory'`.

- [ ] **Step 3: Add `CorpusSettings`**

In `backend/src/pharma_agent/infrastructure/settings.py`, directly after `class CheckpointSettings` add:

```python
class CorpusSettings(BaseModel):
    """Corpus import and release commands (spec C §8.2, §8.5)."""

    embed_batch_size: int = Field(default=32, ge=1)
    embed_max_concurrent: int = Field(default=1, ge=1)
    gc_keep: int = Field(default=2, ge=0)
```

and in `class Settings`, after `checkpoints: CheckpointSettings = Field(default_factory=CheckpointSettings)`:

```python
    corpus: CorpusSettings = Field(default_factory=CorpusSettings)
```

- [ ] **Step 4: Write the factory**

`backend/src/pharma_agent/infrastructure/corpus_factory.py`:

```python
"""Builds the corpus import and release services with resources they own.

Used by `pharma-agent corpus ...`, which runs outside the HTTP container: it opens its own
database engine, Qdrant client and embedding client from Settings and closes them on exit.
"""

from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient

from pharma_agent.application.corpus.import_bundle import ImportKnowledgeBundle
from pharma_agent.application.corpus.releases import ReleaseService
from pharma_agent.domain.corpus.ports import Embedder, VectorIndex
from pharma_agent.domain.shared.clock import SystemClock
from pharma_agent.infrastructure.openai_client import build_async_openai
from pharma_agent.infrastructure.persistence.postgres.corpus_repository import (
    PostgresCorpusRepository,
)
from pharma_agent.infrastructure.persistence.postgres.database import Database
from pharma_agent.infrastructure.persistence.postgres.embedding_cache import (
    PostgresEmbeddingCache,
)
from pharma_agent.infrastructure.retrieval.qdrant_adapter import OpenAiEmbedder
from pharma_agent.infrastructure.retrieval.qdrant_index import (
    CURRENT_ALIAS,
    QdrantVectorIndex,
)
from pharma_agent.infrastructure.settings import Settings

# A CLI run needs one connection at a time plus one for gc savepoint retries.
CORPUS_POOL_SIZE = 2


@dataclass
class CorpusServices:
    importer: ImportKnowledgeBundle
    releases: ReleaseService
    index: VectorIndex
    embedder: Embedder


@asynccontextmanager
async def open_corpus_services(
    settings: Settings,
    *,
    embedder: Embedder | None = None,
    alias: str = CURRENT_ALIAS,
) -> AsyncGenerator[CorpusServices]:
    """Open resources and yield the services; everything is closed on exit.

    `embedder` replaces the OpenAI-compatible embedder built from settings (tests and the
    E2E server pass `FakeEmbedder`); the Qdrant collection follows its model and dimension.
    `alias` is the Qdrant alias retrieval reads (P3 switches the default to
    `settings.retrieval.qdrant_collection`; the E2E server passes `e2e_chunks_current`).
    """
    database = Database(
        settings.postgres.dsn, pool_size=CORPUS_POOL_SIZE, echo=settings.postgres.echo
    )
    qdrant = AsyncQdrantClient(
        url=settings.qdrant.url,
        api_key=settings.qdrant.api_key,
        timeout=int(settings.qdrant.timeout_seconds),
        check_compatibility=settings.qdrant.check_compatibility,
    )
    async with AsyncExitStack() as stack:
        stack.push_async_callback(database.dispose)
        stack.push_async_callback(qdrant.close)
        if embedder is None:
            embedding = settings.retrieval.embedding
            # Bulk corpus embeddings are not chat turns, so they are not sent to Langfuse.
            embed_client = build_async_openai(
                api_key=embedding.api_key,
                base_url=embedding.base_url,
                timeout=embedding.timeout_seconds,
                max_retries=embedding.max_retries,
                traced=False,
            )
            stack.push_async_callback(embed_client.close)
            embedder = OpenAiEmbedder(
                embed_client, model=embedding.model, dimension=embedding.dimension
            )
        repository = PostgresCorpusRepository(database.sessions)
        cache = PostgresEmbeddingCache(database.sessions)
        index = QdrantVectorIndex(
            qdrant, model=embedder.model, dimension=embedder.dimension, alias=alias
        )
        clock = SystemClock()
        corpus = settings.corpus
        yield CorpusServices(
            importer=ImportKnowledgeBundle(
                repository,
                cache,
                index,
                embedder,
                clock,
                embed_batch_size=corpus.embed_batch_size,
                embed_max_concurrent=corpus.embed_max_concurrent,
            ),
            releases=ReleaseService(
                repository,
                cache,
                index,
                embedder,
                clock,
                embed_batch_size=corpus.embed_batch_size,
                embed_max_concurrent=corpus.embed_max_concurrent,
            ),
            index=index,
            embedder=embedder,
        )
```

- [ ] **Step 5: Add the CLI group**

In `backend/src/pharma_agent/cli.py` replace the import block (lines 3–14) with:

```python
import asyncio
import json
import sys
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import typer

from pharma_agent.application.corpus.import_bundle import ImportReport
from pharma_agent.application.progress import EventType, ProgressEvent
from pharma_agent.domain.corpus.bundle import BundleValidationError, read_bundle
from pharma_agent.domain.corpus.models import ReleaseSummary
from pharma_agent.domain.retrieval.models import Hit, HydrateStrategy
from pharma_agent.domain.shared.errors import DomainError
from pharma_agent.infrastructure.composition import build_application
from pharma_agent.infrastructure.corpus_factory import (
    CorpusServices,
    open_corpus_services,
)
from pharma_agent.infrastructure.langgraph.cleanup import delete_expired_checkpoints
from pharma_agent.infrastructure.settings import Settings
```

and append at the end of the file:

```python
corpus_app = typer.Typer(
    help="Corpus: import knowledge bundle, quản lý release", no_args_is_help=True
)
app.add_typer(corpus_app, name="corpus")

CorpusAction = Callable[[CorpusServices], Awaitable[None]]


def _run_corpus(action: CorpusAction, settings: Settings | None = None) -> None:
    try:
        asyncio.run(_with_corpus_services(settings or Settings(), action))
    except DomainError as exc:
        typer.echo(f"!! {exc.code}: {exc}", err=True)
        raise typer.Exit(code=1) from exc


async def _with_corpus_services(settings: Settings, action: CorpusAction) -> None:
    async with open_corpus_services(settings) as services:
        await action(services)


def _echo_import(report: ImportReport) -> None:
    release = report.release
    published = " published" if report.published else ""
    typer.echo(
        f"{report.outcome.value}: {report.collection_key} release {release.number} "
        f"{release.id} [{release.status.value}]{published}"
    )
    stats = release.stats
    if stats is not None:
        typer.echo(
            f"chunks {stats.chunks} | embeddings cached {stats.embeddings_cached} "
            f"bundle {stats.embeddings_from_bundle} computed {stats.embeddings_computed} "
            f"| points new {stats.points_upserted} updated {stats.points_updated}"
        )


def _release_line(summary: ReleaseSummary) -> str:
    release = summary.release
    marker = "*" if summary.current else " "
    published = release.published_at.isoformat() if release.published_at else "-"
    return (
        f"{marker} {summary.collection_key} #{release.number} {release.id} "
        f"{release.status.value} chunks={summary.chunk_count} published={published}"
    )


@corpus_app.command("import")
def corpus_import(
    bundle_dir: Path = typer.Argument(
        ..., exists=True, file_okay=False, dir_okay=True, help="Thư mục knowledge bundle"
    ),
    collection: str = typer.Option(
        ..., "--collection", help="Key của collection, phải trùng manifest"
    ),
    publish: bool = typer.Option(
        False, "--publish", help="Đặt release mới làm release hiện hành"
    ),
) -> None:
    """Import knowledge bundle thành một release mới."""
    try:
        bundle = read_bundle(bundle_dir)
    except BundleValidationError as exc:
        typer.echo(f"!! {exc.code}: {bundle_dir}", err=True)
        for problem in exc.problems:
            typer.echo(f"   - {problem}", err=True)
        raise typer.Exit(code=2) from exc
    if bundle.manifest.collection.key != collection:
        typer.echo(
            f"!! bundle collection is {bundle.manifest.collection.key}, not {collection}",
            err=True,
        )
        raise typer.Exit(code=2)

    async def action(services: CorpusServices) -> None:
        _echo_import(await services.importer(bundle, publish=publish))

    _run_corpus(action)


@corpus_app.command("releases")
def corpus_releases(
    collection: str | None = typer.Option(None, "--collection", help="Lọc theo collection"),
) -> None:
    """Liệt kê release, trạng thái, số chunk và release hiện hành (*)."""

    async def action(services: CorpusServices) -> None:
        summaries = await services.releases.list_releases(collection)
        if not summaries:
            typer.echo("no releases")
        for summary in summaries:
            typer.echo(_release_line(summary))

    _run_corpus(action)


@corpus_app.command("publish")
def corpus_publish(
    release_id: uuid.UUID = typer.Argument(..., help="Id của release ở trạng thái ready"),
) -> None:
    """Trỏ release hiện hành của collection tới release này."""

    async def action(services: CorpusServices) -> None:
        release = await services.releases.publish(release_id)
        typer.echo(f"published release {release.number} {release.id}")

    _run_corpus(action)


@corpus_app.command("rollback")
def corpus_rollback(
    collection: str = typer.Option(..., "--collection", help="Key của collection"),
) -> None:
    """Trỏ về release được publish liền trước."""

    async def action(services: CorpusServices) -> None:
        release = await services.releases.rollback(collection)
        typer.echo(f"rolled back {collection} to release {release.number} {release.id}")

    _run_corpus(action)


@corpus_app.command("gc")
def corpus_gc(
    collection: str = typer.Option(..., "--collection", help="Key của collection"),
    keep: int | None = typer.Option(
        None, "--keep", min=0, help="Số release gần nhất được giữ (mặc định từ settings)"
    ),
) -> None:
    """Retire release cũ, dọn point Qdrant và chunk version không còn dùng."""
    settings = Settings()
    retained = keep if keep is not None else settings.corpus.gc_keep

    async def action(services: CorpusServices) -> None:
        report = await services.releases.gc(collection, retained)
        typer.echo(
            f"retired {len(report.retired)} releases (keep {retained}) | points updated "
            f"{report.points_updated} deleted {report.points_deleted} | chunk versions "
            f"deleted {report.purge.chunk_versions_deleted} kept "
            f"{report.purge.chunk_versions_kept} | section revisions deleted "
            f"{report.purge.section_revisions_deleted}"
        )

    _run_corpus(action, settings)


@corpus_app.command("reindex")
def corpus_reindex(
    collection: str = typer.Option(..., "--collection", help="Key của collection"),
) -> None:
    """Dựng lại point Qdrant của các release chưa retire từ Postgres và embedding cache."""

    async def action(services: CorpusServices) -> None:
        report = await services.releases.reindex(collection)
        typer.echo(f"points upserted {report.points_upserted}")
        for release_id, count in report.release_points.items():
            typer.echo(f"release {release_id}: {count} points")
        for release_id in report.skipped:
            typer.echo(f"skipped release {release_id} (other embedding model)")

    _run_corpus(action)
```

- [ ] **Step 6: Run the tests**

Run: `uv run pytest -q tests/infrastructure/test_settings.py tests/infrastructure/test_corpus_factory.py tests/test_cli_corpus.py tests/test_cli.py`
Expected: PASS (existing `test_cli.py` tests, including `test_migrate_upgrades_to_head`, still pass).

- [ ] **Step 7: Smoke-test the command tree**

Run: `uv run pharma-agent corpus --help && uv run pharma-agent corpus import --help`
Expected: the six commands `import`, `releases`, `publish`, `rollback`, `gc`, `reindex` are listed; `import` shows `BUNDLE_DIR`, `--collection` and `--publish`.

- [ ] **Step 8: Run the full check**

Run: `uv run ruff check src tests && uv run ruff format --check src tests && uv run pyrefly check --min-severity warn && uv run pytest -q`
Expected: all green.

- [ ] **Step 9: Commit**

```bash
git add src/pharma_agent/infrastructure/settings.py \
  src/pharma_agent/infrastructure/corpus_factory.py \
  src/pharma_agent/cli.py \
  tests/infrastructure/test_settings.py \
  tests/infrastructure/test_corpus_factory.py \
  tests/test_cli_corpus.py
git commit -m "feat(cli): add pharma-agent corpus import and release commands"
```

The commit message ends with the session attribution trailer.

---

### Task 9: End-to-end integration: import, publish/rollback, gc with RESTRICT, reindex

**Files:**
- Test: `backend/tests/infrastructure/test_corpus_integration.py`

**Interfaces:**
- Consumes: `open_corpus_services(settings, *, embedder=FakeEmbedder())` (Task 8), so the test runs the production wiring with only the embedder swapped; `PostgresCorpusRepository` for reading collections; fixtures `migrated_dsn` (`tests/postgres.py`), `qdrant_url` and `qdrant_client` (`tests/qdrant.py`); fixture bundle helpers (Task 2).
- Produces: no production code. This task proves the spec C §12 integration row for P2 against real services: import → index on the alias → publish/rollback switches the current release → `gc` keeps a chunk version referenced through FK `RESTRICT` and deletes orphan points → `reindex` rebuilds a deleted Qdrant collection from Postgres and the embedding cache. Hybrid retrieval and hydrate on this data are P3's integration tests.

Why the probe table: `message_citations` (the real `RESTRICT` reference) arrives with migration `0008` in P6. The test creates `public.gc_probe_citations` with the same kind of foreign key, inserts one reference, and drops the table in `finally`, so it proves that Postgres blocks the delete and that gc keeps going without knowing which table holds the reference.

- [ ] **Step 1: Write the integration tests**

`backend/tests/infrastructure/test_corpus_integration.py`:

```python
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass

import pytest
from qdrant_client import AsyncQdrantClient, models
from sqlalchemy import text

from pharma_agent.application.corpus.import_bundle import (
    ImportKnowledgeBundle,
    ImportOutcome,
)
from pharma_agent.application.corpus.releases import ReleaseService
from pharma_agent.domain.corpus.bundle import KnowledgeBundle
from pharma_agent.domain.corpus.models import ReleaseStatus, build_snapshot
from pharma_agent.domain.corpus.ports import VectorIndex
from pharma_agent.infrastructure.corpus_factory import open_corpus_services
from pharma_agent.infrastructure.persistence.postgres.corpus_repository import (
    PostgresCorpusRepository,
)
from pharma_agent.infrastructure.persistence.postgres.database import Database
from pharma_agent.infrastructure.retrieval.qdrant_adapter import DENSE_VECTOR_NAME
from pharma_agent.infrastructure.retrieval.qdrant_index import CURRENT_ALIAS
from pharma_agent.infrastructure.settings import Settings
from tests.corpus_fixtures import (
    DOSAGE_SECTION,
    LEAFLET_SECTION,
    small_bundle,
    with_section_text,
)
from tests.fakes import FakeEmbedder, fake_vector

pytestmark = pytest.mark.integration

PHYSICAL = "chunks_fake_embedding_4d"
PAYLOAD_KEYS = {
    "collection_id",
    "release_ids",
    "document_id",
    "section_id",
    "section_revision_id",
    "kind",
}


@dataclass
class Stack:
    database: Database
    client: AsyncQdrantClient
    embedder: FakeEmbedder
    repository: PostgresCorpusRepository
    index: VectorIndex
    importer: ImportKnowledgeBundle
    releases: ReleaseService


@pytest.fixture
async def stack(
    migrated_dsn: str,
    qdrant_url: str,
    qdrant_client: AsyncQdrantClient,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncGenerator[Stack]:
    database = Database(migrated_dsn, pool_size=2)
    async with database.engine.begin() as connection:
        await connection.execute(
            text("TRUNCATE corpus.collections, corpus.embedding_cache CASCADE")
        )
    monkeypatch.setenv("PHARMA_POSTGRES__DSN", migrated_dsn)
    monkeypatch.setenv("PHARMA_QDRANT__URL", qdrant_url)
    monkeypatch.setenv("PHARMA_CORPUS__EMBED_BATCH_SIZE", "4")
    monkeypatch.setenv("PHARMA_CORPUS__EMBED_MAX_CONCURRENT", "2")
    embedder = FakeEmbedder()
    async with open_corpus_services(
        Settings(_env_file=None), embedder=embedder
    ) as services:
        yield Stack(
            database=database,
            client=qdrant_client,
            embedder=embedder,
            repository=PostgresCorpusRepository(database.sessions),
            index=services.index,
            importer=services.importer,
            releases=services.releases,
        )
    await database.dispose()


def edited() -> KnowledgeBundle:
    bundle = with_section_text(small_bundle(), DOSAGE_SECTION, "Ghi chú liều mới.")
    return with_section_text(bundle, LEAFLET_SECTION, "Ghi chú tờ HDSD mới.")


def chunk_ids(bundle: KnowledgeBundle) -> set[uuid.UUID]:
    return {chunk.id for chunk in build_snapshot(bundle).chunks}


async def alias_target(client: AsyncQdrantClient) -> str | None:
    aliases = (await client.get_aliases()).aliases
    return next((a.collection_name for a in aliases if a.alias_name == CURRENT_ALIAS), None)


async def test_import_indexes_points_without_text_and_is_idempotent(stack: Stack) -> None:
    bundle = small_bundle()
    snapshot = build_snapshot(bundle)

    report = await stack.importer(bundle, publish=True)

    release = report.release
    assert report.outcome is ImportOutcome.IMPORTED
    assert release.status is ReleaseStatus.READY and release.published_at is not None
    assert release.stats is not None
    assert release.stats.embeddings_from_bundle == len(snapshot.embedding_texts())
    assert release.stats.embeddings_computed == 0
    assert await stack.index.count_release(release.id) == len(snapshot.release_chunks)
    assert await alias_target(stack.client) == PHYSICAL

    chunk = snapshot.chunks[0]
    (record,) = await stack.client.retrieve(PHYSICAL, ids=[str(chunk.id)], with_payload=True)
    assert record.payload is not None and set(record.payload) == PAYLOAD_KEYS
    assert record.payload["release_ids"] == [str(release.id)]
    assert record.payload["collection_id"] == str(snapshot.collection.id)

    found = await stack.client.query_points(
        CURRENT_ALIAS,
        query=fake_vector(chunk.embedding_text),
        using=DENSE_VECTOR_NAME,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="collection_id",
                    match=models.MatchValue(value=str(snapshot.collection.id)),
                ),
                models.FieldCondition(
                    key="release_ids", match=models.MatchValue(value=str(release.id))
                ),
            ]
        ),
        limit=3,
    )
    assert str(chunk.id) in [str(point.id) for point in found.points]

    again = await stack.importer(bundle, publish=True)
    assert again.outcome is ImportOutcome.NO_CHANGE and again.release == release
    assert await stack.index.count_release(release.id) == len(snapshot.release_chunks)
    assert stack.embedder.batches == []


async def test_publish_and_rollback_switch_the_current_release(stack: Stack) -> None:
    r1 = (await stack.importer(small_bundle(), publish=True)).release
    r2 = (await stack.importer(edited(), publish=True)).release
    ids1, ids2 = chunk_ids(small_bundle()), chunk_ids(edited())

    collection = await stack.repository.get_collection("formulary")
    assert collection is not None and collection.current_release_id == r2.id
    assert (await stack.releases.rollback("formulary")).id == r1.id
    current = [s.release.id for s in await stack.releases.list_releases("formulary") if s.current]
    assert current == [r1.id]
    assert (await stack.releases.publish(r2.id)).id == r2.id
    collection = await stack.repository.get_collection("formulary")
    assert collection is not None and collection.current_release_id == r2.id

    assert await stack.index.count_release(r1.id) == len(ids1)
    assert await stack.index.count_release(r2.id) == len(ids2)
    shared = sorted(ids1 & ids2)[0]
    (record,) = await stack.client.retrieve(PHYSICAL, ids=[str(shared)], with_payload=True)
    assert record.payload is not None
    assert record.payload["release_ids"] == sorted([str(r1.id), str(r2.id)])


async def test_gc_keeps_cited_chunk_versions_and_deletes_orphan_points(
    stack: Stack,
) -> None:
    r1 = (await stack.importer(small_bundle(), publish=True)).release
    r2 = (await stack.importer(edited(), publish=True)).release
    ids1, ids2 = chunk_ids(small_bundle()), chunk_ids(edited())
    old_only = ids1 - ids2
    assert len(old_only) >= 2
    protected = sorted(old_only)[0]

    async with stack.database.engine.begin() as connection:
        await connection.execute(
            text(
                "CREATE TABLE public.gc_probe_citations (chunk_version_id uuid NOT NULL "
                "REFERENCES corpus.chunk_versions(id) ON DELETE RESTRICT)"
            )
        )
        await connection.execute(
            text("INSERT INTO public.gc_probe_citations VALUES (:id)"), {"id": protected}
        )
    try:
        report = await stack.releases.gc("formulary", keep=1)
        async with stack.database.engine.connect() as connection:
            kept_rows = (
                await connection.execute(
                    text("SELECT count(*) FROM corpus.chunk_versions WHERE id = :id"),
                    {"id": protected},
                )
            ).scalar_one()
    finally:
        async with stack.database.engine.begin() as connection:
            await connection.execute(text("DROP TABLE public.gc_probe_citations"))

    assert report.retired == [r1.id]
    assert report.points_deleted == len(old_only)
    assert report.points_updated == len(ids1 & ids2)
    assert report.purge.chunk_versions_kept == 1
    assert report.purge.chunk_versions_deleted == len(old_only) - 1
    assert kept_rows == 1
    assert await stack.index.existing_ids(sorted(ids1 | ids2)) == ids2
    assert await stack.index.count_release(r1.id) == 0
    assert await stack.index.count_release(r2.id) == len(ids2)
    statuses = {
        s.release.id: s.release.status for s in await stack.releases.list_releases("formulary")
    }
    assert statuses == {r1.id: ReleaseStatus.RETIRED, r2.id: ReleaseStatus.READY}


async def test_reindex_rebuilds_a_deleted_qdrant_collection(stack: Stack) -> None:
    r1 = (await stack.importer(small_bundle(), publish=True)).release
    r2 = (await stack.importer(edited(), publish=True)).release
    ids1, ids2 = chunk_ids(small_bundle()), chunk_ids(edited())

    await stack.client.delete_collection(PHYSICAL)
    stack.embedder.batches.clear()

    report = await stack.releases.reindex("formulary")

    assert report.points_upserted == len(ids1 | ids2)
    assert report.release_points == {r1.id: len(ids1), r2.id: len(ids2)}
    assert report.skipped == []
    assert stack.embedder.batches == []
    assert await alias_target(stack.client) == PHYSICAL
    info = await stack.client.get_collection(PHYSICAL)
    assert info.config.metadata == {"embedding_model": "fake-embedding-4d", "dims": 4}
```

- [ ] **Step 2: Run the integration tests**

Run: `uv run pytest -q -m integration tests/infrastructure/test_corpus_integration.py`
Expected: PASS (4 passed). These tests only use code from Tasks 1–8, so a failure here is a real defect in one of them: fix it in that task's module (and its unit test), never by loosening an assertion.

- [ ] **Step 3: Run the full check**

Run: `uv run ruff check src tests && uv run ruff format --check src tests && uv run pyrefly check --min-severity warn && uv run pytest -q && uv run pytest -q -m integration`
Expected: all green. Run `uv run ruff format tests` first if the formatter rewraps the constructor calls in the `stack` fixture.

- [ ] **Step 4: Manual smoke test against the local Docker stack (optional, dev machine)**

Run from `backend/` with Postgres and Qdrant from `docker-compose.yml` running and `PHARMA_RETRIEVAL__EMBEDDING__*` pointing at a reachable embedding server:

```bash
uv run pharma-agent migrate
uv run pharma-agent corpus import tests/fixtures/knowledge_bundle_small --collection formulary --publish
uv run pharma-agent corpus releases
```

Expected: `imported: formulary release 1 ... [ready] published`, then one `*` line. The fixture's vectors are for `fake-embedding-4d`, so the configured model embeds every chunk through the endpoint; that is the resume path of spec C §8.2 step 7.3.

- [ ] **Step 5: Commit**

```bash
git add tests/infrastructure/test_corpus_integration.py
git commit -m "test(corpus): cover import, release switch, gc and reindex on Postgres and Qdrant"
```

The commit message ends with the session attribution trailer.

---

## Self-Review

### Spec coverage

| Spec requirement | Task |
| --- | --- |
| C §4 `domain/corpus/models.py` (Collection, Document, Section, SectionRevision, ChunkVersion, Release) and `ports.py` (CorpusRepository, EmbeddingCache, VectorIndex) | Task 3 (plus `Embedder`, `IndexItem`, `ReleaseStats`, `ReleaseSummary`, `CorpusSnapshot`, `PurgeResult`); `CorpusReader` is P3 (overview §3.2) |
| C §4 `application/corpus/import_bundle.py`, `releases.py` | Tasks 6, 7 |
| C §4 `infrastructure/persistence/postgres/corpus_tables.py`, `corpus_repository.py`, `embedding_cache.py` | Tasks 1, 4 |
| C §4 `retrieval/qdrant_index.py` | Task 5 (`qdrant_adapter.py` retrieval changes and `postgres_corpus.py` are P3) |
| C §4 `cli.py` group `pharma-agent corpus ...` | Task 8 (resources built by `infrastructure/corpus_factory.py`, not in `cli.py`) |
| C §6.2 migration creates schema `corpus`; `env.py` `include_schemas=True` and `include_name` accepts the schema | Task 1 (`metadata.py`, `env.py`, migration `0005`, migration diff test with `include_schemas`) |
| C §6.2 `collections`: `key` unique, `visibility` check, `owner_user_id` FK `user` CASCADE, `current_release_id` FK deferrable | Task 1 (`use_alter`, `DEFERRABLE INITIALLY DEFERRED`; metadata test and Alembic diff) |
| C §6.2 `documents` unique `(collection_id, key)`; `sections` unique `(document_id, key)`, `context_path text[]` | Task 1; upsert by key in Task 4 |
| C §6.2 `section_revisions` immutable, id from §6.1 | Tasks 1, 3 (`section_revision_id`), 4 (insert on conflict do nothing) |
| C §6.2 `chunk_versions` FK `RESTRICT`, index `embedding_text_sha256` | Tasks 1, 4 |
| C §6.2 `releases` unique `(collection_id, number)`, status check, stats and timestamps | Tasks 1, 4 |
| C §6.2 `release_chunks` PK, index `(release_id, section_revision_id, ordinal)`, FK chunk `RESTRICT`, `hydrate_strategy` per release | Tasks 1, 3, 4 |
| C §6.2 `glossary_entries`, `colloquial_mappings` FK release CASCADE; `embedding_cache` PK `(model, sha)` | Tasks 1, 4 |
| C §6.2 note: `published_at` written the first time a release becomes current | Task 4 (`coalesce(published_at, at)`), Task 7 rollback |
| C §8.1 six commands with their signatures | Task 8 |
| C §8.2 step 1 validate | `read_bundle` in the CLI (Task 8) plus reference and vector checks before writes (Tasks 3, 6) |
| C §8.2 step 2 no change | Task 6 (`NO_CHANGE`; unit test and Task 9) |
| C §8.2 steps 3–6 identity upsert, revisions, chunks, release rows | Task 4 `stage_release`, Task 3 `build_snapshot` |
| C §8.2 step 7 cache → bundle (model and dims match) → embedder in batches with `embed_batch_size` / `embed_max_concurrent`, cache after each batch, resume | Task 6 (`EmbeddingResolver`; tests for each source, resume, concurrency limit) |
| C §8.2 step 8 upsert missing points, add release id on existing ones | Tasks 5, 6 (`IndexWriter`) |
| C §8.2 step 9 point count equals `release_chunks` → `ready` + stats | Task 6 (test keeps `building` on mismatch) |
| C §8.2 step 10 publish in one transaction | Task 4 `publish`, Task 6 |
| C §8.2 idempotent, failure leaves `building`, rerun resumes | Task 6 tests, Task 9 no-change test |
| C §8.3 collection per model, dense cosine, sparse IDF, alias `chunks_current`, point id, tenant payload, no text, BM25 `models.Document`, metadata `{embedding_model, dims}` | Task 5 (startup check and query filter are P3) |
| C §8.4 capacity (25k × 2560 dims) | Tasks 4–6: vectors streamed through the cache in batches, never all in memory |
| C §8.5 gc steps 1–5 including RESTRICT-protected chunk versions and revisions, embedding cache untouched | Tasks 3 (`releases_to_retire`), 4 (`purge_retired`, probe table test), 7 (`gc`), 9 |
| C §8.1 `reindex` from Postgres and `embedding_cache` after Qdrant loss | Tasks 7, 9 |
| C §12 unit application row (cache, bundle, embedder, re-import, no change, status) | Tasks 6, 7 |
| C §12 integration row for P2 (import → publish/rollback → gc keeps cited chunk, deletes points → reindex) | Task 9 (hybrid search and hydrate on the imported data belong to P3) |
| C §12 contract row: shared fixture bundle | Task 2 commits `tests/fixtures/knowledge_bundle_small/`; the seed-pipeline side is P4 |
| Overview §3.3 `pharma-agent migrate` keeps working | Task 1 step 7 |

### Names checked against the overview

- Revision id `0005`; schema `corpus`; the ten table names and class names; `corpus_tables.py` on `Base.metadata`: Task 1.
- `ImportKnowledgeBundle.__call__(self, bundle: KnowledgeBundle, *, publish: bool) -> ImportReport` in `pharma_agent.application.corpus.import_bundle`: Task 6.
- `ReleaseService` with `list_releases(collection_key)`, `publish(release_id)`, `rollback(collection_key)`, `gc(collection_key, keep)`, `reindex(collection_key)` in `pharma_agent.application.corpus.releases`: Task 7.
- `CorpusSettings(embed_batch_size=32, embed_max_concurrent=1, gc_keep=2)` as `Settings.corpus`: Task 8.
- `QdrantVectorIndex` in `pharma_agent.infrastructure.retrieval.qdrant_index`; `chunks_<model_slug>`, alias `chunks_current`, `dense_vector`, `bm25_sparse_vector` (imported constants), payload keys and `str(chunk_version_id)` point ids: Task 5.
- Fixture directory, `tests/corpus_fixtures.py::small_bundle()`, `tests/fakes.py::FakeEmbedder`, model `fake-embedding-4d`, 2 documents, prose, table, `index_only` with `index_entries`, leaflet with colloquial mapping, 2 glossary entries: Task 2.
- P1 names are used only as pinned in overview §3.2 (`chunk_section(document, section, glossary, mappings)`, not the older spec §7.1 `SectionInput` form).
- Corpus ids are `uuid.UUID` in Python and `str(uuid)` in Qdrant payloads and CLI output.

### Placeholder and consistency scan

- Every step that changes code shows the code. Later tasks use the signatures from "Interfaces pinned by this plan" (`stage_release(snapshot, *, release_id, embedding_model, at)`, `index_items(release_ids)`, `set_release_ids(items)`, `EmbeddingResolver.ensure(texts, bundled)`, `IndexWriter.write(items, *, overwrite)`, `build_importer(..., clock=...)`).
- `CorpusServices` exposes `index` and `embedder` in addition to the two services, so the factory tests can check the real adapters. With `embedder=` injected, the factory builds no OpenAI client, and the Qdrant collection is named from the injected model.
- Commit steps say only that the message ends with the session attribution trailer; no session URL is written into the plan.

### Conflicts found and resolutions

1. **`--collection` in spec C §8.1 vs the pinned `__call__` without a collection parameter.** The manifest's `collection.key` is the target, and the CLI exits 2 when `--collection` differs.
2. **Spec C §6.2 gives `glossary_entries` and `colloquial_mappings` no key.** Glossary terms are unique per bundle (P1 validation), so `glossary_entries` uses PK `(release_id, term)`. P1 allows empty and repeated `ColloquialMappingRecord.key`, so `colloquial_mappings` gets `position` and PK `(release_id, position)`.
3. **Spec C §8.2 step 1 inside the service vs an in-memory `KnowledgeBundle`.** The file checks (§5.3) run in `read_bundle` at the boundary. The service re-checks document references and vector lengths before any write.
4. **`chunk_version_id` (§6.1) does not include the section revision, while `chunk_versions` has a `section_revision_id` column.** That column stores the revision where the version first appeared, and the Qdrant payload uses it. Per-release placement lives in `release_chunks`. gc keeps a revision while any chunk version still references it.
5. **Spec §8.2 step 2 only covers the current release.** An identical non-current release is resumed (`building`) or reused (`ready`) instead of being duplicated, which satisfies §12 "import lại không sinh dữ liệu".
6. **Deleting chunk versions protected by `RESTRICT` (P6 table not yet present).** Deletes run in batches inside savepoints, falling back to one row at a time. The tests use a temporary `public.gc_probe_citations` FK table.
7. **`tables.include_name` moved to the new `metadata.py`.** `env.py` and `test_migrations.py` import it from there. The existing `tests/infrastructure/test_qdrant_integration.py` still uses `qdrant/qdrant:latest` and is left to P3, which rewrites that adapter. New tests pin `v1.19.1`.
8. **Verified against the P1 plan (`2026-09-13-corpus-domain.md`).**
   - `BundleManifest` has no validators, so the generator's placeholders (`files={}`) construct fine, and `write_bundle` rebuilds `files`, both counts and the `embeddings` list from `bundle.embeddings`.
   - Bundle models are `strict=True`. The generator passes enum members and exact types.
   - `write_bundle` validates content: unique keys, unique ordinals per document, non-empty blocks, pages, case-insensitive unique glossary terms, and existing mapping `section_keys`. The fixture satisfies all of these.
   - `build_context_header(title, context_path)` prefixes the title itself, so the fixture's `context_path` no longer repeats the document title.
   - `BundleValidationError(problems)` exposes `.problems`, as the CLI uses it.
9. **Added in the cross-review.**
   - `chunk_versions.context_header` is covered by a domain test and a repository test.
   - `open_corpus_services(settings, *, embedder=None)` is covered by a factory test. Task 9 now runs the real factory with `FakeEmbedder` injected instead of wiring the services by hand.
10. **Alias is configurable (reconciling P3 and P7).** `QdrantVectorIndex(..., alias: str = CURRENT_ALIAS)` creates, checks and names only that alias. `open_corpus_services(..., alias=CURRENT_ALIAS)` passes it through. P3 switches the default to `settings.retrieval.qdrant_collection`, and P7's E2E server passes `e2e_chunks_current`. Covered by `test_non_default_alias_is_created_and_used` (Task 5) and the factory test (Task 8).
