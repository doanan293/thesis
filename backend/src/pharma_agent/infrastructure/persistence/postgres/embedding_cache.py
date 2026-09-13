"""Postgres `corpus.embedding_cache`: vectors keyed by (model, sha256(embedding_text))."""

import itertools
import struct
from collections.abc import Mapping, Sequence
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pharma_agent.infrastructure.persistence.postgres.corpus_tables import (
    EmbeddingCacheTable,
)

HASHES_PER_QUERY = 1000
ROWS_PER_STATEMENT = 500


def pack_vector(values: Sequence[float]) -> bytes:
    """float32 little-endian, the same layout as knowledge-bundle vectors."""
    return struct.pack(f"<{len(values)}f", *values)


def unpack_vector(data: bytes, dimension: int) -> list[float]:
    return list(struct.unpack(f"<{dimension}f", data))


class PostgresEmbeddingCache:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def missing(self, model: str, hashes: Sequence[str]) -> set[str]:
        wanted = set(hashes)
        found: set[str] = set()
        async with self._sessions() as session:
            for batch in itertools.batched(sorted(wanted), HASHES_PER_QUERY):
                result = await session.execute(
                    select(EmbeddingCacheTable.embedding_text_sha256).where(
                        EmbeddingCacheTable.model == model,
                        EmbeddingCacheTable.embedding_text_sha256.in_(batch),
                    )
                )
                found.update(result.scalars().all())
        return wanted - found

    async def get_many(
        self, model: str, hashes: Sequence[str]
    ) -> dict[str, list[float]]:
        vectors: dict[str, list[float]] = {}
        async with self._sessions() as session:
            for batch in itertools.batched(sorted(set(hashes)), HASHES_PER_QUERY):
                result = await session.execute(
                    select(
                        EmbeddingCacheTable.embedding_text_sha256,
                        EmbeddingCacheTable.dims,
                        EmbeddingCacheTable.vector,
                    ).where(
                        EmbeddingCacheTable.model == model,
                        EmbeddingCacheTable.embedding_text_sha256.in_(batch),
                    )
                )
                for sha, dims, vector in result.all():
                    vectors[sha] = unpack_vector(vector, dims)
        return vectors

    async def put_many(
        self, model: str, dimension: int, vectors: Mapping[str, Sequence[float]]
    ) -> None:
        rows: list[dict[str, Any]] = []
        for sha, vector in vectors.items():
            if len(vector) != dimension:
                raise ValueError(
                    f"vector for {sha} has dimension {len(vector)}, expected {dimension}"
                )
            rows.append(
                {
                    "model": model,
                    "embedding_text_sha256": sha,
                    "dims": dimension,
                    "vector": pack_vector(vector),
                }
            )
        if not rows:
            return
        async with self._sessions.begin() as session:
            for batch in itertools.batched(rows, ROWS_PER_STATEMENT):
                await session.execute(
                    insert(EmbeddingCacheTable)
                    .values(list(batch))
                    .on_conflict_do_nothing(
                        index_elements=["model", "embedding_text_sha256"]
                    )
                )
