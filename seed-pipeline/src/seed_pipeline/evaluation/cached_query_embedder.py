"""Query embedder for evaluation, backed by the `seed embed queries` cache.

It never calls a model: evaluation queries were embedded once (on Kaggle or locally) and
retrieval must use exactly those vectors.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from seed_pipeline.evaluation.query_embedding_cache import (
    QueryEmbeddingCache,
    QueryEmbeddingCacheError,
)
from seed_pipeline.evaluation.query_hash import query_hash


class QueryEmbeddingMissing(QueryEmbeddingCacheError):
    """An evaluation query has no cached vector for the backend embedding model."""


class CachedQueryEmbedder:
    def __init__(
        self,
        vectors: Mapping[str, Sequence[float]],
        *,
        model: str,
        dimension: int,
        source: Path,
    ) -> None:
        for digest, vector in vectors.items():
            if len(vector) != dimension:
                raise QueryEmbeddingCacheError(
                    f"Query vector {digest} in {source} has dimension {len(vector)}, "
                    f"expected {dimension}"
                )
        self._vectors = {digest: list(vector) for digest, vector in vectors.items()}
        self._model = model
        self._dimension = dimension
        self._source = source

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        missing = [text for text in texts if query_hash(text) not in self._vectors]
        if missing:
            raise QueryEmbeddingMissing(
                f"Query embedding cache {self._source} has no {self._model} vector for "
                f"{len(missing)} query text(s), first {missing[0][:80]!r}; run "
                f"`uv run seed embed queries --model {self._model}` for this evaluation"
            )
        return [list(self._vectors[query_hash(text)]) for text in texts]


def cached_query_embedder(
    cache: QueryEmbeddingCache,
    rows: Sequence[Mapping[str, Any]],
    *,
    model: str,
    dimension: int,
) -> tuple[CachedQueryEmbedder, str]:
    """Check that every row has a cached vector and return the embedder and subset sha256."""
    row_list = [dict(row) for row in rows]
    subset = cache.validate_subset(row_list, model)
    if not subset.is_complete or subset.sha256 is None:
        raise QueryEmbeddingMissing(
            f"Query embedding cache {cache.path} is missing {subset.missing} of "
            f"{subset.total} queries for {model}; run "
            f"`uv run seed embed queries --model {model}` for this evaluation"
        )
    vectors: dict[str, list[float]] = {}
    for row in row_list:
        text = str(row["query"])
        vector = cache.get(model, str(row["query_id"]), text)
        if vector is None:
            raise QueryEmbeddingMissing(
                f"Query embedding cache {cache.path} lost {row['query_id']}"
            )
        vectors[query_hash(text)] = vector
    return (
        CachedQueryEmbedder(
            vectors, model=model, dimension=dimension, source=cache.path
        ),
        subset.sha256,
    )
