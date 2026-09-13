"""`message_citations` rows and the corpus joins that turn them back into `Citation` values."""

import uuid
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from pharma_agent.domain.agent.citations import SNIPPET_CHARS
from pharma_agent.domain.conversation.models import Citation, Message
from pharma_agent.domain.retrieval.models import HydrateStrategy
from pharma_agent.domain.shared.text import make_snippet
from pharma_agent.infrastructure.persistence.postgres.corpus_tables import (
    ChunkVersionTable,
    DocumentTable,
    SectionRevisionTable,
    SectionTable,
)
from pharma_agent.infrastructure.persistence.postgres.tables import (
    MessageCitationTable,
)


def citation_link_rows(message: Message) -> list[MessageCitationTable]:
    message_id = uuid.UUID(hex=message.message_id)
    return [
        MessageCitationTable(
            message_id=message_id,
            index=citation.index,
            chunk_version_id=citation.chunk_version_id,
            release_id=citation.release_id,
            strategy=citation.strategy.value,
            block_chunk_version_ids=list(citation.block_chunk_version_ids),
        )
        for citation in message.citations
    ]


def citation_from_row(
    link: MessageCitationTable,
    chunk: ChunkVersionTable,
    *,
    heading: str,
    document_title: str,
    source_title: str,
) -> Citation:
    """The same display fields and snippet `citations_from` builds from a live `Hit`."""
    return Citation(
        index=link.index,
        chunk_version_id=link.chunk_version_id,
        release_id=link.release_id,
        strategy=HydrateStrategy(link.strategy),
        block_chunk_version_ids=list(link.block_chunk_version_ids),
        source=source_title,
        title=document_title,
        section=heading,
        start_page=chunk.start_page,
        end_page=chunk.end_page,
        snippet=make_snippet(chunk.chunk_text, SNIPPET_CHARS),
    )


async def load_citations(
    session: AsyncSession, message_ids: Sequence[uuid.UUID]
) -> dict[uuid.UUID, list[Citation]]:
    """Citations of the given messages in one query, each list ordered by index."""
    if not message_ids:
        return {}
    query = (
        select(
            MessageCitationTable,
            ChunkVersionTable,
            SectionTable.heading,
            DocumentTable.title,
            DocumentTable.source_title,
        )
        .join(
            ChunkVersionTable,
            ChunkVersionTable.id == MessageCitationTable.chunk_version_id,
        )
        .join(
            SectionRevisionTable,
            SectionRevisionTable.id == ChunkVersionTable.section_revision_id,
        )
        .join(SectionTable, SectionTable.id == SectionRevisionTable.section_id)
        .join(DocumentTable, DocumentTable.id == SectionTable.document_id)
        .where(MessageCitationTable.message_id.in_(message_ids))
        .order_by(MessageCitationTable.message_id, MessageCitationTable.index)
    )
    grouped: dict[uuid.UUID, list[Citation]] = {}
    result = await session.execute(query)
    for link, chunk, heading, document_title, source_title in result.tuples():
        grouped.setdefault(link.message_id, []).append(
            citation_from_row(
                link,
                chunk,
                heading=heading,
                document_title=document_title,
                source_title=source_title,
            )
        )
    return grouped
