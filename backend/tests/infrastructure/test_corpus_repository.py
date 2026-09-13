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
    return build_snapshot(
        with_section_text(bundle, LEAFLET_SECTION, "Ghi chú tờ HDSD.")
    )


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
    assert (first.number, first.status, first.stats) == (
        1,
        ReleaseStatus.BUILDING,
        None,
    )
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
    assert stored_headers == {
        chunk.id: chunk.context_header for chunk in snapshot.chunks
    }
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


async def test_second_release_shares_unchanged_chunk_versions(
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

    old_ids = {chunk.id for chunk in old.chunks}
    new_ids = {chunk.id for chunk in new.chunks}
    assert r2.number == 2
    assert await count(database, "chunk_versions") == len(old_ids | new_ids)
    assert await count(database, "release_chunks") == len(old.release_chunks) + len(
        new.release_chunks
    )

    items = {
        item.chunk_version_id: item for item in await repository.index_items([r2.id])
    }
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
    orphans = [
        item for item in await repository.index_items([r1.id]) if not item.release_ids
    ]
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
