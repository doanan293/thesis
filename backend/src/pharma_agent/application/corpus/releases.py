"""Release commands (spec C §8.1, §8.5): list, publish, rollback, gc, reindex."""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from pharma_agent.application.corpus.indexing import EmbeddingResolver, IndexWriter
from pharma_agent.domain.corpus.models import (
    Collection,
    CollectionNotFound,
    NoEarlierRelease,
    PurgeResult,
    Release,
    ReleaseNotFound,
    ReleaseNotPublishable,
    ReleaseStatus,
    ReleaseSummary,
    releases_to_retire,
)
from pharma_agent.domain.corpus.ports import (
    CorpusRepository,
    Embedder,
    EmbeddingCache,
    VectorIndex,
)
from pharma_agent.domain.shared.clock import Clock


class GcReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    retired: list[uuid.UUID]
    points_updated: int
    points_deleted: int
    purge: PurgeResult


class ReindexReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    points_upserted: int
    release_points: dict[uuid.UUID, int]
    skipped: list[uuid.UUID]


class ReleaseService:
    def __init__(
        self,
        repository: CorpusRepository,
        cache: EmbeddingCache,
        index: VectorIndex,
        embedder: Embedder,
        clock: Clock,
        *,
        embed_batch_size: int,
        embed_max_concurrent: int,
    ) -> None:
        self._repository = repository
        self._index = index
        self._embedder = embedder
        self._clock = clock
        self._resolver = EmbeddingResolver(
            cache,
            embedder,
            batch_size=embed_batch_size,
            max_concurrent=embed_max_concurrent,
        )
        self._writer = IndexWriter(index, cache, model=embedder.model)

    async def list_releases(
        self, collection_key: str | None = None
    ) -> list[ReleaseSummary]:
        return await self._repository.list_releases(collection_key)

    async def publish(self, release_id: uuid.UUID) -> Release:
        release = await self._release(release_id)
        if release.status is not ReleaseStatus.READY:
            raise ReleaseNotPublishable(
                f"release {release.number} ({release.id}) is {release.status.value}"
            )
        await self._repository.publish(release_id, self._clock.now())
        return await self._release(release_id)

    async def rollback(self, collection_key: str) -> Release:
        collection = await self._collection(collection_key)
        if collection.current_release_id is None:
            raise NoEarlierRelease(
                f"collection {collection_key} has no current release"
            )
        current = await self._release(collection.current_release_id)
        current_published_at = current.published_at
        if current_published_at is None:
            raise NoEarlierRelease(
                f"current release {current.number} was never published"
            )
        candidates: list[tuple[datetime, int, Release]] = []
        for summary in await self._repository.list_releases(collection_key):
            published_at = summary.release.published_at
            if (
                summary.release.status is ReleaseStatus.READY
                and published_at is not None
                and published_at < current_published_at
            ):
                candidates.append(
                    (published_at, summary.release.number, summary.release)
                )
        if not candidates:
            raise NoEarlierRelease(
                f"collection {collection_key} has no release published before "
                f"release {current.number}"
            )
        _, _, target = max(candidates, key=lambda candidate: candidate[:2])
        await self._repository.publish(target.id, self._clock.now())
        return await self._release(target.id)

    async def gc(self, collection_key: str, keep: int) -> GcReport:
        collection = await self._collection(collection_key)
        summaries = await self._repository.list_releases(collection_key)
        doomed = releases_to_retire(
            [summary.release for summary in summaries],
            collection.current_release_id,
            keep,
        )
        await self._repository.retire(
            [release.id for release in doomed], self._clock.now()
        )

        pending = [
            summary.release.id
            for summary in await self._repository.list_releases(collection_key)
            if summary.release.status is ReleaseStatus.RETIRED
            and summary.chunk_count > 0
        ]
        points_updated = points_deleted = 0
        if pending:
            await self._index.ensure_collection()
            items = await self._repository.index_items(pending)
            present = await self._index.existing_ids(
                [item.chunk_version_id for item in items]
            )
            live = [
                item
                for item in items
                if item.release_ids and item.chunk_version_id in present
            ]
            dead = [
                item.chunk_version_id
                for item in items
                if not item.release_ids and item.chunk_version_id in present
            ]
            if live:
                await self._index.set_release_ids(live)
            if dead:
                await self._index.delete(dead)
            points_updated, points_deleted = len(live), len(dead)

        purge = await self._repository.purge_retired(collection.id)
        return GcReport(
            retired=[release.id for release in doomed],
            points_updated=points_updated,
            points_deleted=points_deleted,
            purge=purge,
        )

    async def reindex(self, collection_key: str) -> ReindexReport:
        await self._collection(collection_key)
        live = [
            summary.release
            for summary in await self._repository.list_releases(collection_key)
            if summary.release.status is not ReleaseStatus.RETIRED
        ]
        usable = [r for r in live if r.embedding_model == self._embedder.model]
        usable_ids = {release.id for release in usable}
        skipped = [release.id for release in live if release.id not in usable_ids]

        await self._index.ensure_collection()
        items = [
            item.model_copy(
                update={"release_ids": [r for r in item.release_ids if r in usable_ids]}
            )
            for item in await self._repository.index_items(
                [release.id for release in usable]
            )
        ]
        await self._resolver.ensure(
            {item.embedding_text_sha256: item.embedding_text for item in items}, {}
        )
        written = await self._writer.write(items, overwrite=True)
        release_points = {
            release.id: await self._index.count_release(release.id)
            for release in usable
        }
        return ReindexReport(
            points_upserted=written.upserted,
            release_points=release_points,
            skipped=skipped,
        )

    async def _collection(self, key: str) -> Collection:
        collection = await self._repository.get_collection(key)
        if collection is None:
            raise CollectionNotFound(f"collection {key} does not exist")
        return collection

    async def _release(self, release_id: uuid.UUID) -> Release:
        release = await self._repository.get_release(release_id)
        if release is None:
            raise ReleaseNotFound(f"release {release_id} does not exist")
        return release
