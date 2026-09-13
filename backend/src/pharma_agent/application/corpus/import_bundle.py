"""Import a knowledge bundle as a new corpus release (spec C §8.2).

The only ingest path: the CLI calls it today and a future upload API will call the same
service. Every step is idempotent on hashed ids, so rerunning after a failure resumes the
`building` release from the embedding cache and the points already indexed.
"""

import uuid
from collections.abc import Mapping
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from pharma_agent.application.corpus.indexing import EmbeddingResolver, IndexWriter
from pharma_agent.domain.corpus.bundle import KnowledgeBundle, model_slug
from pharma_agent.domain.corpus.models import (
    CorpusImportError,
    CorpusSnapshot,
    Release,
    ReleaseNotFound,
    ReleaseStatus,
    build_snapshot,
)
from pharma_agent.domain.corpus.ports import (
    CorpusRepository,
    Embedder,
    EmbeddingCache,
    VectorIndex,
)
from pharma_agent.domain.shared.clock import Clock


class ImportOutcome(StrEnum):
    IMPORTED = "imported"
    REUSED = "reused"
    NO_CHANGE = "no_change"


class ImportReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    outcome: ImportOutcome
    collection_key: str
    release: Release
    published: bool


def bundle_vectors(
    bundle: KnowledgeBundle, model: str, dimension: int
) -> Mapping[str, list[float]]:
    """Precomputed vectors usable for `model`, keyed by embedding_text_sha256.

    Only vectors declared in the manifest with the same model and dims are used. Raises
    CorpusImportError when a declared vector has another length.
    """
    declared = any(
        entry.model == model and entry.dims == dimension
        for entry in bundle.manifest.embeddings
    )
    if not declared:
        return {}
    vectors = bundle.embeddings.get(model, {})
    wrong = sorted(sha for sha, vector in vectors.items() if len(vector) != dimension)
    if wrong:
        raise CorpusImportError(
            f"embeddings/{model_slug(model)}.jsonl: {len(wrong)} vectors do not have "
            f"{dimension} dims, first {wrong[0]}"
        )
    return vectors


class ImportKnowledgeBundle:
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

    async def __call__(self, bundle: KnowledgeBundle, *, publish: bool) -> ImportReport:
        model = self._embedder.model
        snapshot = build_snapshot(bundle)
        vectors = bundle_vectors(bundle, model, self._embedder.dimension)
        key = snapshot.collection.key

        current = await self._current_release(key)
        if current is not None and (
            current.bundle_digest,
            current.chunker_version,
            current.embedding_model,
        ) == (snapshot.bundle_digest, snapshot.chunker_version, model):
            return ImportReport(
                outcome=ImportOutcome.NO_CHANGE,
                collection_key=key,
                release=current,
                published=True,
            )

        await self._index.ensure_collection()
        existing = await self._repository.find_release(
            snapshot.collection.id,
            bundle_digest=snapshot.bundle_digest,
            chunker_version=snapshot.chunker_version,
            embedding_model=model,
        )
        if existing is not None and existing.status is ReleaseStatus.READY:
            release, outcome = existing, ImportOutcome.REUSED
        else:
            release = await self._repository.stage_release(
                snapshot,
                release_id=existing.id if existing is not None else uuid.uuid4(),
                embedding_model=model,
                at=self._clock.now(),
            )
            await self._build(snapshot, vectors, release)
            outcome = ImportOutcome.IMPORTED

        if publish:
            await self._repository.publish(release.id, self._clock.now())
        stored = await self._repository.get_release(release.id)
        if stored is None:
            raise ReleaseNotFound(str(release.id))
        return ImportReport(
            outcome=outcome, collection_key=key, release=stored, published=publish
        )

    async def _current_release(self, collection_key: str) -> Release | None:
        collection = await self._repository.get_collection(collection_key)
        if collection is None or collection.current_release_id is None:
            return None
        return await self._repository.get_release(collection.current_release_id)

    async def _build(
        self,
        snapshot: CorpusSnapshot,
        vectors: Mapping[str, list[float]],
        release: Release,
    ) -> None:
        embeddings = await self._resolver.ensure(snapshot.embedding_texts(), vectors)
        items = await self._repository.index_items([release.id])
        written = await self._writer.write(items)
        indexed = await self._index.count_release(release.id)
        expected = len(snapshot.release_chunks)
        if indexed != expected:
            raise CorpusImportError(
                f"release {release.number}: {indexed} points carry the release id, "
                f"expected {expected}; the release stays building"
            )
        stats = snapshot.stats().model_copy(
            update={
                "embeddings_cached": embeddings.cached,
                "embeddings_from_bundle": embeddings.from_bundle,
                "embeddings_computed": embeddings.computed,
                "points_upserted": written.upserted,
                "points_updated": written.updated,
            }
        )
        await self._repository.mark_ready(release.id, stats, self._clock.now())
