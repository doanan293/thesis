"""In-memory corpus adapters that behave like the Postgres and Qdrant ones."""

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from pharma_agent.application.corpus.import_bundle import ImportKnowledgeBundle
from pharma_agent.application.corpus.releases import ReleaseService
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
from pharma_agent.domain.shared.clock import Clock, FixedClock
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
            and (
                release.bundle_digest,
                release.chunker_version,
                release.embedding_model,
            )
            == (bundle_digest, chunker_version, embedding_model)
        ]
        if not matches:
            return None
        return max(matches, key=lambda release: release.number)

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
            if collection_key is None
            or by_id[release.collection_id].key == collection_key
        ]
        return sorted(
            summaries,
            key=lambda summary: (summary.collection_key, -summary.release.number),
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
        self.documents.update(
            {document.id: document for document in snapshot.documents}
        )
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
        kept = [
            chunk_id for chunk_id in orphans if chunk_id in self.protected_chunk_ids
        ]
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

    async def get_many(
        self, model: str, hashes: Sequence[str]
    ) -> dict[str, list[float]]:
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
    repository: InMemoryCorpusRepository = field(
        default_factory=InMemoryCorpusRepository
    )
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
