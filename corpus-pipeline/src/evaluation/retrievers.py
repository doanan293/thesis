import json
import time
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from qdrant_client import QdrantClient, models

from evaluation.retrieval_types import RetrievalCandidate
from rag_metadata.payload_layers import (
    compact_colloquial_mapping,
    format_colloquial_mapping,
)
from vector_store.qdrant_client_helper import (
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
        last_error = None
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
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise
                self.retry_sleep(min(0.5 * (2 ** (attempt - 1)), 5.0))
        else:
            raise last_error
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
        last_error = None
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
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise
                self.retry_sleep(min(0.5 * (2 ** (attempt - 1)), 5.0))
        else:
            raise last_error

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

    def search(self, query, limit: int) -> list[RetrievalCandidate]:
        query_text = query.get("query", "") if isinstance(query, dict) else str(query)
        vector = self.embed_query(query)
        query_filter = (
            self.filter_builder(query) if self.filter_builder is not None else None
        )
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                prefetch = [
                    models.Prefetch(
                        query=vector,
                        using=DENSE_VECTOR_NAME,
                        limit=limit,
                        filter=query_filter,
                    ),
                    models.Prefetch(
                        query=models.Document(text=query_text, model=BM25_MODEL_NAME),
                        using=BM25_SPARSE_VECTOR_NAME,
                        limit=limit,
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
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise
                self.retry_sleep(min(0.5 * (2 ** (attempt - 1)), 5.0))
        else:
            raise last_error
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


class PrecomputedJsonRetriever:
    accepts_query_row = True

    def __init__(self, dump_json_path: str | Path):
        self.dump_json_path = Path(dump_json_path)
        self.by_id: dict[str, list[RetrievalCandidate]] = {}
        self.by_text: dict[str, list[RetrievalCandidate]] = {}
        self._load_dump()

    def _load_dump(self):
        with open(self.dump_json_path, encoding="utf-8") as f:
            data = json.load(f)

        queries_map = data.get("queries", {})
        for q_key, q_data in queries_map.items():
            candidates_raw = q_data.get("candidates", [])
            candidates = [
                RetrievalCandidate(
                    chunk_id=str(c.get("chunk_id") or ""),
                    score=float(c.get("score") or 0.0),
                    rank=int(c.get("rank") or idx + 1),
                    source=str(c.get("source") or "precomputed"),
                    payload=c.get("payload") or {},
                )
                for idx, c in enumerate(candidates_raw)
            ]
            q_id = str(q_data.get("query_id") or q_key).strip()
            q_text = str(q_data.get("query") or "").strip()

            if q_id:
                self.by_id[q_id] = candidates
            if q_text:
                self.by_text[q_text] = candidates

    def search_query_row(self, row: dict, limit: int) -> list[RetrievalCandidate]:
        q_id = str(row.get("query_id") or "").strip()
        q_text = str(row.get("query") or "").strip()

        candidates = self.by_id.get(q_id) or self.by_text.get(q_text) or []
        return candidates[:limit]
