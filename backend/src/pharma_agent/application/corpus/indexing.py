"""Embedding and index writing shared by import and reindex (spec C §8.2 steps 7-8).

Vectors never pile up in memory: they are written to the embedding cache batch by batch and
read back per upsert batch, so a rerun resumes from whatever the cache already holds.
"""

import asyncio
import itertools
from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict

from pharma_agent.domain.corpus.models import CorpusImportError, IndexItem
from pharma_agent.domain.corpus.ports import Embedder, EmbeddingCache, VectorIndex

CACHE_WRITE_BATCH = 500
INDEX_BATCH = 256


class EmbeddingCounts(BaseModel):
    model_config = ConfigDict(frozen=True)

    cached: int
    from_bundle: int
    computed: int


class IndexCounts(BaseModel):
    model_config = ConfigDict(frozen=True)

    upserted: int
    updated: int


class EmbeddingResolver:
    def __init__(
        self,
        cache: EmbeddingCache,
        embedder: Embedder,
        *,
        batch_size: int,
        max_concurrent: int,
    ) -> None:
        if batch_size < 1 or max_concurrent < 1:
            raise ValueError("batch_size and max_concurrent must be at least 1")
        self._cache = cache
        self._embedder = embedder
        self._batch_size = batch_size
        self._max_concurrent = max_concurrent

    async def ensure(
        self, texts: Mapping[str, str], bundled: Mapping[str, Sequence[float]]
    ) -> EmbeddingCounts:
        """Make every `texts` hash (sha -> embedding text) present in the cache."""
        model, dimension = self._embedder.model, self._embedder.dimension
        missing = await self._cache.missing(model, list(texts))
        from_bundle = {sha: bundled[sha] for sha in sorted(missing) if sha in bundled}
        for batch in itertools.batched(from_bundle.items(), CACHE_WRITE_BATCH):
            await self._cache.put_many(model, dimension, dict(batch))
        remaining = sorted(missing - from_bundle.keys())
        # Workers pull from one shared iterator: at most `max_concurrent` batches are in
        # flight, and a worker stops at its first failed batch instead of starting more.
        batches = itertools.batched(remaining, self._batch_size)

        async def worker() -> None:
            for hashes in batches:
                vectors = await self._embedder.embed([texts[sha] for sha in hashes])
                if len(vectors) != len(hashes) or any(
                    len(vector) != dimension for vector in vectors
                ):
                    raise CorpusImportError(
                        f"embedder {model} returned vectors that do not match "
                        f"{len(hashes)} inputs of {dimension} dims"
                    )
                await self._cache.put_many(
                    model, dimension, dict(zip(hashes, vectors, strict=True))
                )

        try:
            async with asyncio.TaskGroup() as group:
                for _ in range(self._max_concurrent):
                    group.create_task(worker())
        except ExceptionGroup as errors:
            raise errors.exceptions[0] from errors
        return EmbeddingCounts(
            cached=len(texts) - len(missing),
            from_bundle=len(from_bundle),
            computed=len(remaining),
        )


class IndexWriter:
    def __init__(
        self,
        index: VectorIndex,
        cache: EmbeddingCache,
        *,
        model: str,
        batch_size: int = INDEX_BATCH,
    ) -> None:
        self._index = index
        self._cache = cache
        self._model = model
        self._batch_size = batch_size

    async def write(
        self, items: Sequence[IndexItem], *, overwrite: bool = False
    ) -> IndexCounts:
        """Upsert points that are missing (or all with `overwrite`), rewrite release_ids of
        the rest. Vectors must already be in the cache."""
        if overwrite:
            new, existing = list(items), []
        else:
            present = await self._index.existing_ids(
                [item.chunk_version_id for item in items]
            )
            new = [item for item in items if item.chunk_version_id not in present]
            existing = [item for item in items if item.chunk_version_id in present]
        for batch in itertools.batched(new, self._batch_size):
            hashes = sorted({item.embedding_text_sha256 for item in batch})
            vectors = await self._cache.get_many(self._model, hashes)
            absent = set(hashes) - vectors.keys()
            if absent:
                raise CorpusImportError(
                    f"{len(absent)} embeddings for model {self._model} are not cached, "
                    f"first {min(absent)}"
                )
            await self._index.upsert(batch, vectors)
        if existing:
            await self._index.set_release_ids(existing)
        return IndexCounts(upserted=len(new), updated=len(existing))
