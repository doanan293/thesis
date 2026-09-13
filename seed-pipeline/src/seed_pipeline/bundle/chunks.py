"""Run the backend chunker over every section of a knowledge bundle."""

from collections.abc import Iterator
from dataclasses import dataclass

from pharma_agent.domain.corpus.bundle import (
    DocumentRecord,
    KnowledgeBundle,
    SectionRecord,
)
from pharma_agent.domain.corpus.chunking import (
    MAX_CHUNK_CHARS,
    ChunkDraft,
    chunk_section,
)


@dataclass(frozen=True)
class SectionChunks:
    document: DocumentRecord
    section: SectionRecord
    drafts: list[ChunkDraft]


def gold_chunk_label(section_key: str, ordinal: int) -> str:
    """Position label used by evaluation gold sets; equals the pre-migration chunk_id."""
    return f"{section_key}:chunk-{ordinal:03d}"


def iter_section_chunks(
    bundle: KnowledgeBundle, *, max_chars: int = MAX_CHUNK_CHARS
) -> Iterator[SectionChunks]:
    documents = {document.key: document for document in bundle.documents}
    for section in bundle.sections:
        document = documents[section.document_key]
        yield SectionChunks(
            document=document,
            section=section,
            drafts=chunk_section(
                document,
                section,
                bundle.glossary,
                bundle.colloquial_mappings,
                max_chars=max_chars,
            ),
        )
