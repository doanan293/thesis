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
        return [
            _record(link, chunk, section, document)
            for link, chunk, section, document in rows
        ]
