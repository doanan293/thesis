import time
from typing import Protocol

import requests

from pharma_lab.evaluation.candidate_text import candidate_document_text
from pharma_lab.evaluation.rerank_score_cache import RerankScoreCache
from pharma_lab.evaluation.retrieval_types import RetrievalCandidate
from pharma_lab.runtime.catalog import ModelSpec
from pharma_lab.runtime.client import LlamaCppClient

DEFAULT_RERANK_MAX_RETRIES = 3
DEFAULT_RERANK_RETRY_SLEEP_SECONDS = 5.0
RERANK_TRANSIENT_HTTP_STATUS_CODES = {408, 429, 500, 502, 503, 504}


class Reranker(Protocol):
    def rerank(
        self, query: str, candidates: list[RetrievalCandidate]
    ) -> list[RetrievalCandidate]: ...


class CachedReranker:
    accepts_query_row = True

    def __init__(self, cache: RerankScoreCache, reranker: str):
        self.cache = cache
        self.reranker = reranker

    def rerank(
        self, query_row: dict, candidates: list[RetrievalCandidate]
    ) -> list[RetrievalCandidate]:
        scored = []
        for candidate in candidates:
            score = self.cache.require_score(self.reranker, query_row, candidate)
            scored.append(
                (score, candidate.rank, candidate.resolved_chunk_id, candidate)
            )
        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        return [
            item[3].with_rerank_score(item[0], rank)
            for rank, item in enumerate(scored, start=1)
        ]


def _exception_chain(exc: BaseException):
    seen = set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _is_transient(exc: BaseException) -> bool:
    for current in _exception_chain(exc):
        if isinstance(current, requests.exceptions.HTTPError):
            response = current.response
            return (
                response is None
                or response.status_code in RERANK_TRANSIENT_HTTP_STATUS_CODES
            )
        if isinstance(
            current,
            ConnectionError | TimeoutError | requests.exceptions.RequestException,
        ):
            return True
    return False


class LlamaCppReranker:
    """Scores every candidate of a query in one llama.cpp /v1/rerank request."""

    def __init__(
        self,
        spec: ModelSpec,
        client: LlamaCppClient,
        max_retries: int = DEFAULT_RERANK_MAX_RETRIES,
        retry_sleep_seconds: float = DEFAULT_RERANK_RETRY_SLEEP_SECONDS,
        retry_sleep=time.sleep,
    ) -> None:
        self.spec = spec
        self.client = client
        self.max_retries = max_retries
        self.retry_sleep_seconds = float(retry_sleep_seconds)
        self.retry_sleep = retry_sleep

    def _call(self, operation):
        for attempt in range(1, self.max_retries + 1):
            try:
                return operation()
            except Exception as exc:
                if attempt >= self.max_retries or not _is_transient(exc):
                    raise
                self.retry_sleep(self.retry_sleep_seconds)
        raise AssertionError("unreachable")

    def rerank(
        self, query: str, candidates: list[RetrievalCandidate]
    ) -> list[RetrievalCandidate]:
        documents = [
            candidate.document_text or candidate_document_text(candidate.payload)
            for candidate in candidates
        ]
        scores = self._call(
            lambda: self.client.rerank_native(query, documents, self.spec.name)
        )
        scored = [
            (score, original_index, candidate)
            for original_index, (score, candidate) in enumerate(
                zip(scores, candidates, strict=True)
            )
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            candidate.with_rerank_score(rerank_score=score, rank=rank)
            for rank, (score, _original_index, candidate) in enumerate(scored, start=1)
        ]
