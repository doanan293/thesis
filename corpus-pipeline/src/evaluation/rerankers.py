import time
from typing import Protocol

import requests

from evaluation.retrieval_types import RetrievalCandidate
from evaluation.retrievers import candidate_document_text
from model_runtime.catalog import ModelSpec
from model_runtime.client import LlamaCppClient

DEFAULT_RERANK_MAX_RETRIES = 3
DEFAULT_RERANK_RETRY_SLEEP_SECONDS = 5.0
RERANK_TRANSIENT_HTTP_STATUS_CODES = {408, 429, 500, 502, 503, 504}
QWEN_RERANK_SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
    'Note that the answer can only be "yes" or "no".'
)
DEFAULT_RERANK_INSTRUCTION = "Given a Vietnamese medical retrieval query, retrieve relevant passages that answer the query"


class Reranker(Protocol):
    def rerank(
        self, query: str, candidates: list[RetrievalCandidate]
    ) -> list[RetrievalCandidate]: ...


class NoopReranker:
    def rerank(
        self, query: str, candidates: list[RetrievalCandidate]
    ) -> list[RetrievalCandidate]:
        return [
            candidate.with_rank(rank)
            for rank, candidate in enumerate(candidates, start=1)
        ]


def build_qwen_rerank_prompt(
    query: str,
    document: str,
    instruction: str = DEFAULT_RERANK_INSTRUCTION,
) -> str:
    return (
        f"<|im_start|>system\n{QWEN_RERANK_SYSTEM_PROMPT}<|im_end|>\n"
        "<|im_start|>user\n"
        f"<Instruct>: {instruction}\n"
        f"<Query>: {query}\n"
        f"<Document>: {document}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


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
        self.max_retries = int(max_retries)
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
            scores = [
                self._call(
                    lambda document=document: self.client.rerank_completion(
                        build_qwen_rerank_prompt(query, document), self.spec.name
                    )
                )
                for document in documents
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
