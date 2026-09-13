"""Qdrant hybrid search over the corpus index (spec C §8.3, §9).

Qdrant holds vectors and routing payload only (`collection_id`, `release_ids`). A search reads
the current release of every scoped collection from Postgres, asks Qdrant for chunk version ids
and fused scores inside those releases, then loads the chunks from Postgres in one query.
"""

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any, Literal
from uuid import UUID

from qdrant_client import models

from pharma_agent.domain.corpus.ports import Embedder
from pharma_agent.domain.retrieval.models import Hit, Query
from pharma_agent.domain.retrieval.ports import ChunkKey, CorpusReader, RetrievalError

DENSE_VECTOR_NAME = "dense_vector"
BM25_SPARSE_VECTOR_NAME = "bm25_sparse_vector"
BM25_MODEL_NAME = "Qdrant/bm25"
COLLECTION_ID_KEY = "collection_id"
RELEASE_IDS_KEY = "release_ids"


class OpenAiEmbedder:
    """Query embeddings through an OpenAI-compatible /v1/embeddings endpoint (llama.cpp here)."""

    def __init__(self, client: Any, *, model: str, dimension: int) -> None:
        self._client = client
        self._model = model
        self._dimension = dimension

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = await self._client.embeddings.create(
                model=self._model, input=list(texts)
            )
        except Exception as exc:
            raise RetrievalError(f"embedding request failed: {exc}") from exc
        ordered = sorted(response.data, key=lambda item: item.index)
        vectors = [list(item.embedding) for item in ordered]
        if len(vectors) != len(texts):
            raise RetrievalError(
                f"embedding returned {len(vectors)} vectors for {len(texts)} inputs"
            )
        for vector in vectors:
            if len(vector) != self._dimension:
                raise RetrievalError(
                    f"embedding dimension {len(vector)} != configured {self._dimension}"
                )
        return vectors


def release_scope_filter(releases: Mapping[UUID, UUID]) -> models.Filter:
    """Points of each scoped collection that belong to that collection's current release."""
    branches: list[models.Condition] = [
        models.Filter(
            must=[
                models.FieldCondition(
                    key=COLLECTION_ID_KEY,
                    match=models.MatchValue(value=str(collection_id)),
                ),
                models.FieldCondition(
                    key=RELEASE_IDS_KEY, match=models.MatchValue(value=str(release_id))
                ),
            ]
        )
        for collection_id, release_id in releases.items()
    ]
    return models.Filter(should=branches)


def _scored_point(point: Any) -> tuple[UUID, UUID, float]:
    """(chunk_version_id, collection_id, fused score) of one Qdrant point."""
    payload = point.payload or {}
    return (
        UUID(str(point.id)),
        UUID(str(payload[COLLECTION_ID_KEY])),
        float(point.score or 0.0),
    )


class QdrantHybridRetriever:
    def __init__(
        self,
        client: Any,
        embedder: Embedder,
        reader: CorpusReader,
        *,
        collection: str,
        scope: Sequence[str],
        mode: Literal["hybrid", "dense", "bm25"] = "hybrid",
        prefetch_k: int = 50,
        rrf_k: int = 2,
        max_concurrent: int = 3,
    ) -> None:
        self._client = client
        self._embedder = embedder
        self._reader = reader
        self._collection = collection
        self._scope = list(scope)
        self._mode = mode
        self._prefetch_k = prefetch_k
        self._rrf_k = rrf_k
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def search_many(
        self, queries: Sequence[Query], top_k: int
    ) -> list[list[Hit]]:
        if not queries:
            return []
        releases = await self._reader.current_releases(self._scope)
        if not releases:
            raise RetrievalError(
                f"no current release for collections {', '.join(self._scope)}"
            )
        # bm25 needs no query vector, so the embedding endpoint is not called at all.
        vectors: list[list[float] | None] = [None] * len(queries)
        if self._mode != "bm25":
            embedded = await self._embedder.embed([q.text for q in queries])
            vectors = [*embedded]
        scope = release_scope_filter(releases)
        try:
            ranked = await asyncio.gather(
                *(
                    self._search_one(q, v, top_k, scope)
                    for q, v in zip(queries, vectors, strict=True)
                )
            )
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(f"qdrant query failed: {exc}") from exc

        per_query: list[list[tuple[ChunkKey, float]]] = [
            [
                ((releases[collection_id], chunk_id), score)
                for chunk_id, collection_id, score in points
                if collection_id in releases
            ]
            for points in ranked
        ]
        keys = list(dict.fromkeys(key for scored in per_query for key, _ in scored))
        records = {
            (record.release_id, record.chunk_version_id): record
            for record in await self._reader.load_chunks(keys)
        }
        return [
            [
                records[key].to_hit(fusion_score=score, query_text=query.text)
                for key, score in scored
                if key in records
            ]
            for query, scored in zip(queries, per_query, strict=True)
        ]

    async def _search_one(
        self,
        query: Query,
        vector: list[float] | None,
        top_k: int,
        scope: models.Filter,
    ) -> list[tuple[UUID, UUID, float]]:
        async with self._semaphore:
            if self._mode == "bm25":
                response = await self._client.query_points(
                    collection_name=self._collection,
                    query=models.Document(text=query.text, model=BM25_MODEL_NAME),
                    using=BM25_SPARSE_VECTOR_NAME,
                    query_filter=scope,
                    limit=top_k,
                    with_payload=[COLLECTION_ID_KEY],
                )
            elif vector is None:
                raise RetrievalError(f"{self._mode} search needs a query vector")
            elif self._mode == "hybrid":
                response = await self._client.query_points(
                    collection_name=self._collection,
                    prefetch=[
                        models.Prefetch(
                            query=vector,
                            using=DENSE_VECTOR_NAME,
                            filter=scope,
                            limit=self._prefetch_k,
                        ),
                        models.Prefetch(
                            query=models.Document(
                                text=query.text, model=BM25_MODEL_NAME
                            ),
                            using=BM25_SPARSE_VECTOR_NAME,
                            filter=scope,
                            limit=self._prefetch_k,
                        ),
                    ],
                    query=models.RrfQuery(rrf=models.Rrf(k=self._rrf_k)),
                    query_filter=scope,
                    limit=top_k,
                    with_payload=[COLLECTION_ID_KEY],
                )
            else:
                response = await self._client.query_points(
                    collection_name=self._collection,
                    query=vector,
                    using=DENSE_VECTOR_NAME,
                    query_filter=scope,
                    limit=top_k,
                    with_payload=[COLLECTION_ID_KEY],
                )
        return [_scored_point(point) for point in response.points]

    async def verify_collection(self, *, embedding_model: str, dimension: int) -> None:
        """Raise RetrievalError unless the collection was built for this embedding model."""
        try:
            info = await self._client.get_collection(collection_name=self._collection)
        except Exception as exc:
            raise RetrievalError(
                f"cannot read collection {self._collection}: {exc}"
            ) from exc
        vectors = info.config.params.vectors
        params = vectors[DENSE_VECTOR_NAME] if isinstance(vectors, dict) else vectors
        size = int(getattr(params, "size", 0))
        if size != dimension:
            raise RetrievalError(
                f"collection {self._collection} dense vector dimension {size} != configured {dimension}"
            )
        metadata = info.config.metadata or {}
        if (
            metadata.get("embedding_model") != embedding_model
            or metadata.get("dims") != dimension
        ):
            raise RetrievalError(
                f"collection {self._collection} metadata {metadata!r} does not match "
                f"embedding model {embedding_model} with {dimension} dims"
            )

    async def verify_corpus(self, *, embedding_model: str, dimension: int) -> None:
        """Raise RetrievalError unless the collection matches the embedding settings and every
        scoped corpus collection has a current release (spec C §9, health `corpus`)."""
        await self.verify_collection(
            embedding_model=embedding_model, dimension=dimension
        )
        scope = sorted(set(self._scope))
        releases = await self._reader.current_releases(scope)
        if len(releases) < len(scope):
            raise RetrievalError(
                f"{len(releases)} of {len(scope)} collections in {', '.join(scope)} "
                "have a current release"
            )
