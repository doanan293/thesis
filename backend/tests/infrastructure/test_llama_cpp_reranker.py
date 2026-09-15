import json

import httpx
import pytest
import respx

from pharma_agent.domain.retrieval.ports import RetrievalError
from pharma_agent.infrastructure.retrieval.llama_cpp_reranker import (
    NativeReranker,
    NoopReranker,
    build_reranker,
)
from pharma_agent.infrastructure.settings import RerankSettings
from tests.domain.factories import chunk_uuid, make_hit

BASE = "http://rerank"


@respx.mock(base_url=BASE)
async def test_native_reranker_scores_every_candidate_in_one_request(
    respx_mock: respx.MockRouter,
) -> None:
    route = respx_mock.post("/v1/rerank").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"index": 1, "relevance_score": 0.7},
                    {"index": 0, "relevance_score": 0.3},
                ]
            },
        )
    )
    async with httpx.AsyncClient(base_url=BASE) as http:
        hits = await NativeReranker(http, model="qwen3-reranker:4b-fp16").rerank(
            "Q", [make_hit("a"), make_hit("b")], top_n=1
        )

    assert [h.chunk_version_id for h in hits] == [chunk_uuid("b")]
    assert hits[0].rerank_score == 0.7
    assert len(route.calls) == 1
    body = json.loads(route.calls[0].request.content)
    assert (body["model"], body["query"], body["top_n"]) == (
        "qwen3-reranker:4b-fp16",
        "Q",
        2,
    )
    assert len(body["documents"]) == 2
    # Scores embedding_text, like the evaluation.
    assert body["documents"][0].startswith("Paracetamol > Liều dùng")


@respx.mock(base_url=BASE)
async def test_native_reranker_maps_http_errors(respx_mock: respx.MockRouter) -> None:
    respx_mock.post("/v1/rerank").mock(return_value=httpx.Response(500))
    async with httpx.AsyncClient(base_url=BASE) as http:
        reranker = NativeReranker(http, model="m")
        with pytest.raises(RetrievalError):
            await reranker.rerank("Q", [make_hit("a")], top_n=1)


async def test_noop_reranker_keeps_fusion_order() -> None:
    hits = await NoopReranker().rerank(
        "Q", [make_hit("a", fusion=0.1), make_hit("b", fusion=0.9)], top_n=1
    )
    assert [h.chunk_version_id for h in hits] == [chunk_uuid("b")]
    assert hits[0].rerank_score is None


def test_build_reranker_switches_on_protocol() -> None:
    assert isinstance(build_reranker(RerankSettings(protocol="none")), NoopReranker)
    assert isinstance(
        build_reranker(RerankSettings(protocol="native_rerank")), NativeReranker
    )


@respx.mock(base_url=BASE)
async def test_build_reranker_sends_the_api_key_only_when_configured(
    respx_mock: respx.MockRouter,
) -> None:
    route = respx_mock.post("/v1/rerank").mock(
        return_value=httpx.Response(
            200, json={"results": [{"index": 0, "relevance_score": 0.5}]}
        )
    )
    for api_key in ("rerank-secret", None):
        reranker = build_reranker(
            RerankSettings(protocol="native_rerank", base_url=BASE, api_key=api_key)
        )
        assert isinstance(reranker, NativeReranker)
        await reranker.rerank("Q", [make_hit("a")], top_n=1)
        await reranker.aclose()
    assert route.calls[0].request.headers["Authorization"] == "Bearer rerank-secret"
    assert "Authorization" not in route.calls[1].request.headers
