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
from pharma_agent.domain.corpus.chunking import (
    CHUNKER_VERSION,
    ChunkDraft,
    chunk_section,
)
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
    labels = (
        list(markers)
        if markers is not None
        else [f"kxdoan{i}" for i in range(1, blocks + 1)]
    )
    return SectionRecord(
        key=key,
        document_key="drug:paracetamol",
        heading="Liều lượng và cách dùng",
        context_path=["Liều lượng và cách dùng"],
        ordinal=ordinal,
        start_page=812,
        end_page=813,
        blocks=[
            BlockRecord(
                kind=BlockKind.PROSE,
                markdown=long_paragraph(label),
                start_page=812,
                end_page=813,
            )
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
        blocks=[
            BlockRecord(
                kind=BlockKind.PROSE, markdown="Panadol chứa paracetamol 500 mg."
            )
        ],
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
                    term_annotations=[
                        a.model_dump(mode="json") for a in draft.term_annotations
                    ],
                    colloquial=draft.colloquial.model_dump(mode="json")
                    if draft.colloquial is not None
                    else None,
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
    return SeededRelease(
        collection_id=collection_id,
        release_id=release_id,
        drafts=drafts,
        revisions=revisions,
    )
