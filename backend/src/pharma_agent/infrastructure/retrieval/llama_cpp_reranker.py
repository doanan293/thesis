"""Rerankers over llama.cpp server. The qwen3 prompt and scoring are copied from corpus-pipeline so
runtime scores match the evaluation (corpus_pipeline.runtime.model_profiles.qwen3_rerank_contract)."""

import asyncio
import math
from collections.abc import Sequence
from typing import Any

import httpx

from pharma_agent.domain.retrieval.models import Hit
from pharma_agent.domain.retrieval.ports import Reranker, RetrievalError
from pharma_agent.infrastructure.settings import RerankSettings

QWEN3_SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
    'Note that the answer can only be "yes" or "no".'
)
DEFAULT_RERANK_INSTRUCTION = "Given a Vietnamese medical retrieval query, retrieve relevant passages that answer the query"
_POSITIVE_TOKEN = "yes"
_NEGATIVE_TOKEN = "no"
_LOGIT_BIAS = 100.0


def build_qwen3_yes_no_prompt(
    query: str, document: str, instruction: str = DEFAULT_RERANK_INSTRUCTION
) -> str:
    return (
        f"<|im_start|>system\n{QWEN3_SYSTEM_PROMPT}<|im_end|>\n"
        "<|im_start|>user\n"
        f"<Instruct>: {instruction}\n"
        f"<Query>: {query}\n"
        f"<Document>: {document}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def _document_text(hit: Hit) -> str:
    return hit.embedding_text or f"{hit.context_header}\n\n{hit.chunk_text}".strip()


def _sorted_top(
    hits: Sequence[Hit], scores: Sequence[float | None], top_n: int
) -> list[Hit]:
    scored = [
        h.model_copy(update={"rerank_score": s})
        for h, s in zip(hits, scores, strict=True)
    ]
    scored.sort(key=lambda h: h.score, reverse=True)
    return scored[:top_n]


class LlamaCppCompletionReranker:
    """Yes/no logprob scoring through llama.cpp `/completion` (protocol `completion_logprobs`)."""

    def __init__(
        self,
        http: httpx.AsyncClient,
        *,
        model: str,
        max_concurrent: int = 2,
        instruction: str = DEFAULT_RERANK_INSTRUCTION,
    ) -> None:
        self._http = http
        self._model = model
        self._instruction = instruction
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._token_ids: dict[str, int] = {}

    async def aclose(self) -> None:
        await self._http.aclose()

    async def rerank(self, query: str, hits: Sequence[Hit], top_n: int) -> list[Hit]:
        if not hits:
            return []
        yes_id, no_id = (
            await self._token_id(_POSITIVE_TOKEN),
            await self._token_id(_NEGATIVE_TOKEN),
        )
        if yes_id == no_id:
            raise RetrievalError("rerank candidate tokens must differ")
        scores = await asyncio.gather(
            *(
                self._score(
                    build_qwen3_yes_no_prompt(
                        query, _document_text(h), self._instruction
                    ),
                    yes_id,
                    no_id,
                )
                for h in hits
            )
        )
        return _sorted_top(hits, list(scores), top_n)

    async def _token_id(self, text: str) -> int:
        cached = self._token_ids.get(text)
        if cached is not None:
            return cached
        payload = await self._post(
            "/tokenize", {"content": text, "add_special": False, "parse_special": False}
        )
        tokens = payload.get("tokens")
        if (
            not isinstance(tokens, list)
            or len(tokens) != 1
            or not isinstance(tokens[0], int)
        ):
            raise RetrievalError(
                f"rerank token {text!r} must encode to exactly one token, got {tokens!r}"
            )
        self._token_ids[text] = tokens[0]
        return tokens[0]

    async def _score(self, prompt: str, yes_id: int, no_id: int) -> float:
        body = {
            "model": self._model,
            "prompt": prompt,
            "cache_prompt": True,
            "n_predict": 1,
            "temperature": 1.0,
            "samplers": ["temperature"],
            "n_probs": 2,
            "min_keep": 2,
            "post_sampling_probs": True,
            "logit_bias": [[yes_id, _LOGIT_BIAS], [no_id, _LOGIT_BIAS]],
        }
        async with self._semaphore:
            payload = await self._post("/completion", body)
        probabilities = payload.get("completion_probabilities")
        if not isinstance(probabilities, list) or not probabilities:
            raise RetrievalError(
                "completion rerank response is missing token probabilities"
            )
        first = probabilities[0]
        candidates = first.get("top_probs") if isinstance(first, dict) else None
        if not isinstance(candidates, list):
            raise RetrievalError(
                "completion rerank response is missing top probabilities"
            )
        found: dict[int, float] = {}
        for item in candidates:
            token_id, probability = item.get("id"), item.get("prob")
            if token_id in (yes_id, no_id) and isinstance(probability, int | float):
                if not math.isfinite(probability) or probability < 0:
                    raise RetrievalError(
                        "completion rerank probability must be finite and non-negative"
                    )
                found[int(token_id)] = float(probability)
        if set(found) != {yes_id, no_id}:
            raise RetrievalError(
                "completion rerank response requires yes and no probabilities"
            )
        total = found[yes_id] + found[no_id]
        if total <= 0:
            raise RetrievalError("completion rerank probability total must be positive")
        return found[yes_id] / total

    async def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await self._http.post(path, json=body)
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise RetrievalError(f"llama.cpp {path} failed: {exc}") from exc
        if not isinstance(payload, dict):
            raise RetrievalError(f"llama.cpp {path} returned a non-object response")
        return payload


class NativeReranker:
    """`POST /v1/rerank` (llama.cpp, TEI, vLLM; protocol `native_rerank`)."""

    def __init__(
        self, http: httpx.AsyncClient, *, model: str, max_concurrent: int = 2
    ) -> None:
        self._http = http
        self._model = model
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def rerank(self, query: str, hits: Sequence[Hit], top_n: int) -> list[Hit]:
        if not hits:
            return []
        body = {
            "model": self._model,
            "query": query,
            "documents": [_document_text(h) for h in hits],
            "top_n": len(hits),
        }
        try:
            async with self._semaphore:
                response = await self._http.post("/v1/rerank", json=body)
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise RetrievalError(f"llama.cpp /v1/rerank failed: {exc}") from exc
        results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(results, list):
            raise RetrievalError("rerank response has no results")
        scores: list[float | None] = [None] * len(hits)
        for item in results:
            index, score = item.get("index"), item.get("relevance_score")
            if (
                isinstance(index, int)
                and 0 <= index < len(hits)
                and isinstance(score, int | float)
            ):
                scores[index] = float(score)
        return _sorted_top(hits, scores, top_n)


class NoopReranker:
    async def rerank(self, query: str, hits: Sequence[Hit], top_n: int) -> list[Hit]:
        return sorted(hits, key=lambda h: h.fusion_score, reverse=True)[:top_n]


def build_reranker(settings: RerankSettings) -> Reranker:
    if settings.protocol == "none":
        return NoopReranker()
    headers = (
        {"Authorization": f"Bearer {settings.api_key}"} if settings.api_key else None
    )
    http = httpx.AsyncClient(
        base_url=settings.base_url, timeout=settings.timeout_seconds, headers=headers
    )
    if settings.protocol == "native_rerank":
        return NativeReranker(
            http, model=settings.model, max_concurrent=settings.max_concurrent
        )
    return LlamaCppCompletionReranker(
        http, model=settings.model, max_concurrent=settings.max_concurrent
    )
