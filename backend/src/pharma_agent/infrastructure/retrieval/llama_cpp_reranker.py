"""Rerankers over an OpenAI-style `/v1/rerank` endpoint (llama.cpp, TEI, vLLM).

The rerank template ships inside the GGUF, so the agent and the seed-pipeline evaluation
score candidates with the same computation without copying prompts."""

import asyncio
from collections.abc import Sequence

import httpx

from pharma_agent.domain.retrieval.models import Hit
from pharma_agent.domain.retrieval.ports import Reranker, RetrievalError
from pharma_agent.infrastructure.settings import RerankSettings


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


class NativeReranker:
    """`POST /v1/rerank` with every candidate of a search round in one request."""

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
    return NativeReranker(
        http, model=settings.model, max_concurrent=settings.max_concurrent
    )
