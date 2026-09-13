import time
from typing import Protocol

import requests

from seed_pipeline.evaluation.rerank_score_cache import RerankScoreCache
from seed_pipeline.evaluation.retrieval_types import RetrievalCandidate
from seed_pipeline.evaluation.retrievers import candidate_document_text
from seed_pipeline.runtime.catalog import ModelSpec
from seed_pipeline.runtime.client import LlamaCppClient
from seed_pipeline.runtime.model_profiles import (
    DEFAULT_RERANK_INSTRUCTION,
    QWEN3_SYSTEM_PROMPT,
    build_qwen3_yes_no_prompt,
)

DEFAULT_RERANK_MAX_RETRIES = 3
DEFAULT_RERANK_RETRY_SLEEP_SECONDS = 5.0
RERANK_TRANSIENT_HTTP_STATUS_CODES = {408, 429, 500, 502, 503, 504}
QWEN_RERANK_SYSTEM_PROMPT = QWEN3_SYSTEM_PROMPT


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


def build_qwen_rerank_prompt(
    query: str,
    document: str,
    instruction: str = DEFAULT_RERANK_INSTRUCTION,
) -> str:
    return build_qwen3_yes_no_prompt(query, document, instruction)


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
            candidate_document_text(candidate.payload) for candidate in candidates
        ]
        if self.spec.reranker_protocol == "native_rerank":
            scores = self._call(
                lambda: self.client.rerank_native(query, documents, self.spec.name)
            )
        elif self.spec.reranker_protocol == "completion_logprobs":
            contract = self.spec.rerank_contract
            if contract is None:
                raise ValueError(f"Reranker {self.spec.name} has no scoring contract")
            prompts = [contract.build_prompt(query, document) for document in documents]
            if hasattr(self.client, "rerank_completions_async"):
                import asyncio

                scores = self._call(
                    lambda: asyncio.run(
                        self.client.rerank_completions_async(
                            prompts, self.spec.name, contract=contract
                        )
                    )
                )
            else:
                scores = [
                    self._call(
                        lambda prompt=prompt: self.client.rerank_completion(
                            prompt, self.spec.name, contract=contract
                        )
                    )
                    for prompt in prompts
                ]
        else:
            raise ValueError(
                f"Unsupported reranker protocol for {self.spec.name}: {self.spec.reranker_protocol}"
            )
        scored = [
            (score, original_index, candidate)
            for original_index, (score, candidate) in enumerate(
                zip(scores, candidates, strict=False)
            )
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            candidate.with_rerank_score(rerank_score=score, rank=rank)
            for rank, (score, _original_index, candidate) in enumerate(scored, start=1)
        ]
