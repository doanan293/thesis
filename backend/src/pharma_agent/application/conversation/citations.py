"""Full text behind a citation (spec A §5.2); REST models keep snake_case."""

import uuid

from pydantic import BaseModel

from pharma_agent.domain.conversation.models import CitationBlock
from pharma_agent.domain.retrieval.models import HydrateStrategy


class CitationChunk(BaseModel):
    id: uuid.UUID
    text: str
    matched: bool
    start_page: int | None
    end_page: int | None


class CitationDetail(BaseModel):
    index: int
    source: str
    document_title: str
    section: str
    start_page: int | None
    end_page: int | None
    strategy: HydrateStrategy
    is_current: bool
    chunks: list[CitationChunk]

    @classmethod
    def of(cls, block: CitationBlock) -> "CitationDetail":
        citation = block.citation
        return cls(
            index=citation.index,
            source=citation.source,
            document_title=citation.title,
            section=citation.section,
            start_page=citation.start_page,
            end_page=citation.end_page,
            strategy=citation.strategy,
            is_current=block.is_current,
            chunks=[
                CitationChunk(
                    id=chunk.chunk_version_id,
                    text=chunk.text,
                    matched=chunk.chunk_version_id == citation.chunk_version_id,
                    start_page=chunk.start_page,
                    end_page=chunk.end_page,
                )
                for chunk in block.chunks
            ],
        )
