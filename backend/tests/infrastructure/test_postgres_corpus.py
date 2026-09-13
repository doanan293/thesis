import uuid
from collections.abc import AsyncIterator

import pytest

from pharma_agent.domain.corpus.hydrate import hydrate_strategy_for
from pharma_agent.domain.retrieval.models import HydrateStrategy
from pharma_agent.infrastructure.persistence.postgres.database import Database
from pharma_agent.infrastructure.retrieval.postgres_corpus import (
    PostgresCorpusReader,
    PostgresHydrator,
)
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


async def test_current_releases_maps_only_published_collections(
    database: Database,
) -> None:
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
    assert (record.release_id, record.collection_id) == (
        seeded.release_id,
        seeded.collection_id,
    )
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
