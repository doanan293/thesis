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
    attributes: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


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
        return {
            chunk.embedding_text_sha256: chunk.embedding_text for chunk in self.chunks
        }

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
        (
            release
            for release in releases
            if release.status is not ReleaseStatus.RETIRED
        ),
        key=lambda release: release.number,
        reverse=True,
    )
    kept = {release.id for release in live[:keep]}
    return [
        release
        for release in live
        if release.id not in kept and release.id != current_release_id
    ]
