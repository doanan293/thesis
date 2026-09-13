import time
from collections import defaultdict
from collections.abc import Iterable

from qdrant_client import QdrantClient, models

from seed_pipeline.corpus.metadata.payload_layers import (
    compact_colloquial_mapping,
    format_colloquial_mapping,
)
from seed_pipeline.evaluation.retrieval_types import RetrievalCandidate
from seed_pipeline.vector_store.qdrant_client_helper import (
    BM25_MODEL_NAME,
    BM25_SPARSE_VECTOR_NAME,
    DENSE_VECTOR_NAME,
)


def candidate_document_text(payload: dict) -> str:
    embedding_text = str(payload.get("embedding_text") or "").strip()
    if embedding_text:
        return embedding_text
    context_header = str(payload.get("context_header") or "").strip()
    text = str(payload.get("chunk_text") or "").strip()
    visible_text = (
        f"{context_header}\n\n{text}"
        if context_header and text
        else context_header or text
    )
    colloquial_text = format_colloquial_mapping(
        compact_colloquial_mapping(payload), visible_text
    )
    parts = []
    if context_header:
        parts.append(context_header)
    if colloquial_text:
        parts.append(colloquial_text)
    if text:
        parts.append(text)
    return "\n\n".join(parts).strip()


def reciprocal_rank_fusion(
    rankings: Iterable[list[RetrievalCandidate]],
    limit: int,
    rrf_k: int,
) -> list[RetrievalCandidate]:
    scores: defaultdict[str, float] = defaultdict(float)
    payloads: dict[str, dict] = {}
    best_original: dict[str, RetrievalCandidate] = {}

    for ranking in rankings:
        for candidate in ranking:
            chunk_id = candidate.resolved_chunk_id
            if not chunk_id:
                continue
            scores[chunk_id] += 1 / (rrf_k + candidate.rank)
            payloads.setdefault(chunk_id, candidate.payload)
            if (
                chunk_id not in best_original
                or candidate.rank < best_original[chunk_id].rank
            ):
                best_original[chunk_id] = candidate

    ordered_ids = sorted(
        scores,
        key=lambda chunk_id: (
            -scores[chunk_id],
            best_original[chunk_id].rank,
            chunk_id,
        ),
    )
    fused: list[RetrievalCandidate] = []
    for rank, chunk_id in enumerate(ordered_ids[:limit], start=1):
        fused.append(
            RetrievalCandidate(
                chunk_id=chunk_id,
                score=scores[chunk_id],
                rank=rank,
                source="hybrid",
                payload=payloads[chunk_id],
            )
        )
    return fused


class DenseQdrantRetriever:
    accepts_query_row = True

    def __init__(
        self,
        qdrant_client: QdrantClient,
        collection_name: str,
        embed_query,
        max_retries: int = 5,
        retry_sleep=time.sleep,
        filter_builder=None,
    ):
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        self.embed_query = embed_query
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep
        self.filter_builder = filter_builder

    def search(self, query, limit: int) -> list[RetrievalCandidate]:
        vector = self.embed_query(query)
        for attempt in range(1, self.max_retries + 1):
            try:
                query_kwargs = {
                    "collection_name": self.collection_name,
                    "query": vector,
                    "using": DENSE_VECTOR_NAME,
                    "limit": limit,
                    "with_payload": True,
                }
                if self.filter_builder is not None:
                    query_filter = self.filter_builder(query)
                    if query_filter is not None:
                        query_kwargs["query_filter"] = query_filter
                response = self.qdrant_client.query_points(**query_kwargs)
                break
            except Exception:
                if attempt >= self.max_retries:
                    raise
                self.retry_sleep(min(0.5 * (2 ** (attempt - 1)), 5.0))
        else:
            raise RuntimeError(
                "Qdrant search was not attempted: max_retries must be >= 1"
            )
        results = response.points
        candidates: list[RetrievalCandidate] = []
        for rank, result in enumerate(results, start=1):
            payload = result.payload or {}
            candidates.append(
                RetrievalCandidate(
                    chunk_id=str(
                        payload.get("chunk_id") or payload.get("chunk_key") or ""
                    ),
                    score=float(getattr(result, "score", 0.0) or 0.0),
                    rank=rank,
                    source="dense",
                    payload=payload,
                )
            )
        return candidates

    def search_batch(
        self, queries: list[object], limit: int
    ) -> list[list[RetrievalCandidate]]:
        requests = []
        for query in queries:
            kwargs = {
                "query": self.embed_query(query),
                "using": DENSE_VECTOR_NAME,
                "limit": limit,
                "with_payload": True,
            }
            if self.filter_builder is not None:
                kwargs["filter"] = self.filter_builder(query)
            requests.append(models.QueryRequest(**kwargs))
        responses = self.qdrant_client.query_batch_points(
            collection_name=self.collection_name, requests=requests
        )
        return [
            [
                RetrievalCandidate(
                    chunk_id=str(
                        (point.payload or {}).get("chunk_id")
                        or (point.payload or {}).get("chunk_key")
                        or ""
                    ),
                    score=float(getattr(point, "score", 0.0) or 0.0),
                    rank=rank,
                    source="dense",
                    payload=point.payload or {},
                )
                for rank, point in enumerate(response.points, start=1)
            ]
            for response in responses
        ]


class QdrantBm25Retriever:
    accepts_query_row = True

    def __init__(
        self,
        qdrant_client: QdrantClient,
        collection_name: str,
        max_retries: int = 5,
        retry_sleep=time.sleep,
        filter_builder=None,
    ):
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep
        self.filter_builder = filter_builder

    def search(self, query, limit: int) -> list[RetrievalCandidate]:
        query_text = query.get("query", "") if isinstance(query, dict) else str(query)
        for attempt in range(1, self.max_retries + 1):
            try:
                query_kwargs = {
                    "collection_name": self.collection_name,
                    "query": models.Document(text=query_text, model=BM25_MODEL_NAME),
                    "using": BM25_SPARSE_VECTOR_NAME,
                    "limit": limit,
                    "with_payload": True,
                }
                if self.filter_builder is not None:
                    query_filter = self.filter_builder(query)
                    if query_filter is not None:
                        query_kwargs["query_filter"] = query_filter
                response = self.qdrant_client.query_points(**query_kwargs)
                break
            except Exception:
                if attempt >= self.max_retries:
                    raise
                self.retry_sleep(min(0.5 * (2 ** (attempt - 1)), 5.0))
        else:
            raise RuntimeError(
                "Qdrant search was not attempted: max_retries must be >= 1"
            )

        candidates: list[RetrievalCandidate] = []
        for rank, result in enumerate(response.points, start=1):
            payload = result.payload or {}
            candidates.append(
                RetrievalCandidate(
                    chunk_id=str(
                        payload.get("chunk_id") or payload.get("chunk_key") or ""
                    ),
                    score=float(getattr(result, "score", 0.0) or 0.0),
                    rank=rank,
                    source="bm25",
                    payload=payload,
                )
            )
        return candidates

    def search_batch(
        self, queries: list[object], limit: int
    ) -> list[list[RetrievalCandidate]]:
        requests = [
            models.QueryRequest(
                query=models.Document(
                    text=str(
                        query.get("query", "") if isinstance(query, dict) else query
                    ),
                    model=BM25_MODEL_NAME,
                ),
                using=BM25_SPARSE_VECTOR_NAME,
                limit=limit,
                with_payload=True,
            )
            for query in queries
        ]
        responses = self.qdrant_client.query_batch_points(
            collection_name=self.collection_name, requests=requests
        )
        return [
            [
                RetrievalCandidate(
                    chunk_id=str(
                        (point.payload or {}).get("chunk_id")
                        or (point.payload or {}).get("chunk_key")
                        or ""
                    ),
                    score=float(getattr(point, "score", 0.0) or 0.0),
                    rank=rank,
                    source="bm25",
                    payload=point.payload or {},
                )
                for rank, point in enumerate(response.points, start=1)
            ]
            for response in responses
        ]


class QdrantHybridRetriever(DenseQdrantRetriever):
    def __init__(
        self,
        qdrant_client: QdrantClient,
        collection_name: str,
        embed_query,
        max_retries: int = 5,
        retry_sleep=time.sleep,
        filter_builder=None,
        *,
        rrf_k: int,
        prefetch_k: int | None = None,
    ):
        super().__init__(
            qdrant_client,
            collection_name,
            embed_query,
            max_retries=max_retries,
            retry_sleep=retry_sleep,
            filter_builder=filter_builder,
        )
        self.rrf_k = rrf_k
        self.prefetch_k = prefetch_k

    def search(self, query, limit: int) -> list[RetrievalCandidate]:
        query_text = query.get("query", "") if isinstance(query, dict) else str(query)
        vector = self.embed_query(query)
        prefetch_k = self.prefetch_k or limit
        query_filter = (
            self.filter_builder(query) if self.filter_builder is not None else None
        )
        for attempt in range(1, self.max_retries + 1):
            try:
                prefetch = [
                    models.Prefetch(
                        query=vector,
                        using=DENSE_VECTOR_NAME,
                        limit=prefetch_k,
                        filter=query_filter,
                    ),
                    models.Prefetch(
                        query=models.Document(text=query_text, model=BM25_MODEL_NAME),
                        using=BM25_SPARSE_VECTOR_NAME,
                        limit=prefetch_k,
                        filter=query_filter,
                    ),
                ]
                response = self.qdrant_client.query_points(
                    collection_name=self.collection_name,
                    prefetch=prefetch,
                    query=models.RrfQuery(rrf=models.Rrf(k=self.rrf_k)),
                    limit=limit,
                    with_payload=True,
                )
                break
            except Exception:
                if attempt >= self.max_retries:
                    raise
                self.retry_sleep(min(0.5 * (2 ** (attempt - 1)), 5.0))
        else:
            raise RuntimeError(
                "Qdrant search was not attempted: max_retries must be >= 1"
            )
        candidates: list[RetrievalCandidate] = []
        for rank, result in enumerate(response.points, start=1):
            payload = result.payload or {}
            candidates.append(
                RetrievalCandidate(
                    chunk_id=str(
                        payload.get("chunk_id") or payload.get("chunk_key") or ""
                    ),
                    score=float(getattr(result, "score", 0.0) or 0.0),
                    rank=rank,
                    source="hybrid",
                    payload=payload,
                )
            )
        return candidates

    def search_batch(
        self, queries: list[object], limit: int
    ) -> list[list[RetrievalCandidate]]:
        prefetch_k = self.prefetch_k or limit
        requests = []
        for query in queries:
            query_text = (
                query.get("query", "") if isinstance(query, dict) else str(query)
            )
            query_filter = (
                self.filter_builder(query) if self.filter_builder is not None else None
            )
            requests.append(
                models.QueryRequest(
                    prefetch=[
                        models.Prefetch(
                            query=self.embed_query(query),
                            using=DENSE_VECTOR_NAME,
                            limit=prefetch_k,
                            filter=query_filter,
                        ),
                        models.Prefetch(
                            query=models.Document(
                                text=query_text, model=BM25_MODEL_NAME
                            ),
                            using=BM25_SPARSE_VECTOR_NAME,
                            limit=prefetch_k,
                            filter=query_filter,
                        ),
                    ],
                    query=models.RrfQuery(rrf=models.Rrf(k=self.rrf_k)),
                    limit=limit,
                    with_payload=True,
                )
            )
        responses = self.qdrant_client.query_batch_points(
            collection_name=self.collection_name, requests=requests
        )
        return [
            [
                RetrievalCandidate(
                    chunk_id=str(
                        (point.payload or {}).get("chunk_id")
                        or (point.payload or {}).get("chunk_key")
                        or ""
                    ),
                    score=float(getattr(point, "score", 0.0) or 0.0),
                    rank=rank,
                    source="hybrid",
                    payload=point.payload or {},
                )
                for rank, point in enumerate(response.points, start=1)
            ]
            for response in responses
        ]
