"""Qdrant adapters matching the corpus-pipeline collection layout (dense + BM25 sparse, RRF)."""

import asyncio
from collections.abc import Sequence
from typing import Any, Literal, Protocol

from qdrant_client import models

from pharma_agent.domain.retrieval.models import (
    Chunk,
    ColloquialMapping,
    Hit,
    HydrateStrategy,
    Query,
    TermAnnotation,
)
from pharma_agent.domain.retrieval.ports import RetrievalError

DENSE_VECTOR_NAME = "dense_vector"
BM25_SPARSE_VECTOR_NAME = "bm25_sparse_vector"
BM25_MODEL_NAME = "Qdrant/bm25"
_SCROLL_LIMIT = 256


class Embedder(Protocol):
    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class OpenAiEmbedder:
    """Query embeddings through an OpenAI-compatible /v1/embeddings endpoint (llama.cpp here)."""

    def __init__(self, client: Any, *, model: str, dimension: int) -> None:
        self._client = client
        self._model = model
        self._dimension = dimension

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


def hit_from_point(payload: dict[str, Any], score: float, query_text: str) -> Hit:
    mapping = payload.get("colloquial_mapping")
    annotations = payload.get("term_annotations") or []
    return Hit(
        chunk_id=str(payload["chunk_id"]),
        section_id=str(payload["section_id"]),
        chunk_index=int(payload["chunk_index"]),
        hydrate_strategy=HydrateStrategy(payload["hydrate_strategy"]),
        source=str(payload.get("source", "")),
        title=str(payload.get("title", "")),
        section=str(payload.get("section", "")),
        start_page=int(payload.get("start_page", 0)),
        end_page=int(payload.get("end_page", 0)),
        context_header=str(payload.get("context_header", "")),
        chunk_text=str(payload.get("chunk_text", "")),
        embedding_text=str(payload.get("embedding_text", "")),
        content_type=str(payload.get("content_type", "") or ""),
        table_id=str(payload.get("table_id", "") or ""),
        colloquial_mapping=ColloquialMapping.model_validate(mapping)
        if isinstance(mapping, dict)
        else None,
        term_annotations=[
            TermAnnotation.model_validate(a) for a in annotations if isinstance(a, dict)
        ],
        fusion_score=float(score),
        matched_queries=[query_text],
    )


class QdrantHybridRetriever:
    def __init__(
        self,
        client: Any,
        embedder: Embedder,
        collection: str,
        *,
        mode: Literal["hybrid", "dense"] = "hybrid",
        prefetch_k: int = 50,
        rrf_k: int = 2,
        max_concurrent: int = 3,
    ) -> None:
        self._client = client
        self._embedder = embedder
        self._collection = collection
        self._mode = mode
        self._prefetch_k = prefetch_k
        self._rrf_k = rrf_k
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def search_many(
        self, queries: Sequence[Query], top_k: int
    ) -> list[list[Hit]]:
        if not queries:
            return []
        vectors = await self._embedder.embed([q.text for q in queries])
        try:
            return list(
                await asyncio.gather(
                    *(
                        self._search_one(q, v, top_k)
                        for q, v in zip(queries, vectors, strict=True)
                    )
                )
            )
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(f"qdrant query failed: {exc}") from exc

    async def _search_one(
        self, query: Query, vector: list[float], top_k: int
    ) -> list[Hit]:
        async with self._semaphore:
            if self._mode == "hybrid":
                response = await self._client.query_points(
                    collection_name=self._collection,
                    prefetch=[
                        models.Prefetch(
                            query=vector,
                            using=DENSE_VECTOR_NAME,
                            limit=self._prefetch_k,
                        ),
                        models.Prefetch(
                            query=models.Document(
                                text=query.text, model=BM25_MODEL_NAME
                            ),
                            using=BM25_SPARSE_VECTOR_NAME,
                            limit=self._prefetch_k,
                        ),
                    ],
                    query=models.RrfQuery(rrf=models.Rrf(k=self._rrf_k)),
                    limit=top_k,
                    with_payload=True,
                )
            else:
                response = await self._client.query_points(
                    collection_name=self._collection,
                    query=vector,
                    using=DENSE_VECTOR_NAME,
                    limit=top_k,
                    with_payload=True,
                )
        return [
            hit_from_point(point.payload or {}, float(point.score or 0.0), query.text)
            for point in response.points
        ]

    async def verify_collection(self, expected_dimension: int) -> None:
        try:
            info = await self._client.get_collection(collection_name=self._collection)
        except Exception as exc:
            raise RetrievalError(
                f"cannot read collection {self._collection}: {exc}"
            ) from exc
        vectors = info.config.params.vectors
        params = vectors[DENSE_VECTOR_NAME] if isinstance(vectors, dict) else vectors
        size = int(getattr(params, "size", 0))
        if size != expected_dimension:
            raise RetrievalError(
                f"collection {self._collection} dense vector dimension {size} != configured {expected_dimension}"
            )


class QdrantHydrator:
    def __init__(self, client: Any, collection: str, *, window: int = 1) -> None:
        self._client = client
        self._collection = collection
        self._window = window

    async def hydrate(self, hit: Hit, strategy: HydrateStrategy) -> list[Chunk]:
        if strategy is HydrateStrategy.SEARCH_ONLY:
            return []
        must: list[models.Condition] = [
            models.FieldCondition(
                key="section_id", match=models.MatchValue(value=hit.section_id)
            )
        ]
        if strategy is HydrateStrategy.CHUNK_WINDOW:
            must.append(
                models.FieldCondition(
                    key="chunk_index",
                    range=models.Range(
                        gte=hit.chunk_index - self._window,
                        lte=hit.chunk_index + self._window,
                    ),
                )
            )
        try:
            points, _ = await self._client.scroll(
                collection_name=self._collection,
                scroll_filter=models.Filter(must=must),
                limit=_SCROLL_LIMIT,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            raise RetrievalError(
                f"hydrate failed for section {hit.section_id}: {exc}"
            ) from exc
        chunks = [
            Chunk(
                chunk_id=str(p.payload["chunk_id"]),
                section_id=str(p.payload["section_id"]),
                chunk_index=int(p.payload["chunk_index"]),
                text=str(p.payload.get("chunk_text", "")),
                content_type=str(p.payload.get("content_type", "") or ""),
                table_id=str(p.payload.get("table_id", "") or ""),
            )
            for p in points
            if p.payload
        ]
        return sorted(chunks, key=lambda c: c.chunk_index)
