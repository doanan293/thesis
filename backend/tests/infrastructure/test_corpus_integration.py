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
    return next(
        (a.collection_name for a in aliases if a.alias_name == CURRENT_ALIAS), None
    )


async def test_import_indexes_points_without_text_and_is_idempotent(
    stack: Stack,
) -> None:
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
    (record,) = await stack.client.retrieve(
        PHYSICAL, ids=[str(chunk.id)], with_payload=True
    )
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
    current = [
        s.release.id
        for s in await stack.releases.list_releases("formulary")
        if s.current
    ]
    assert current == [r1.id]
    assert (await stack.releases.publish(r2.id)).id == r2.id
    collection = await stack.repository.get_collection("formulary")
    assert collection is not None and collection.current_release_id == r2.id

    assert await stack.index.count_release(r1.id) == len(ids1)
    assert await stack.index.count_release(r2.id) == len(ids2)
    shared = sorted(ids1 & ids2)[0]
    (record,) = await stack.client.retrieve(
        PHYSICAL, ids=[str(shared)], with_payload=True
    )
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
            text("INSERT INTO public.gc_probe_citations VALUES (:id)"),
            {"id": protected},
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
        s.release.id: s.release.status
        for s in await stack.releases.list_releases("formulary")
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
