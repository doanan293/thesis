import json

import httpx
import pytest
import respx

from pharma_agent.domain.retrieval.ports import RetrievalError
from pharma_agent.infrastructure.retrieval.llama_cpp_reranker import (
    LlamaCppCompletionReranker,
    NativeReranker,
    NoopReranker,
    build_qwen3_yes_no_prompt,
    build_reranker,
)
from pharma_agent.infrastructure.settings import RerankSettings
from tests.domain.factories import make_hit

BASE = "http://rerank"

# Copied verbatim from corpus_pipeline.runtime.model_profiles.build_qwen3_yes_no_prompt: the agent must score
# candidates with the exact prompt the evaluation used.
EXPECTED_PROMPT = (
    "<|im_start|>system\n"
    'Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
    "<Instruct>: Given a Vietnamese medical retrieval query, retrieve relevant passages that answer the query\n"
    "<Query>: Q\n"
    "<Document>: D<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)


def test_prompt_matches_pipeline_contract() -> None:
    assert build_qwen3_yes_no_prompt("Q", "D") == EXPECTED_PROMPT


def completion_response(p_yes: float, p_no: float) -> dict:
    return {
        "content": "yes",
        "completion_probabilities": [
            {
                "id": 9693,
                "token": "yes",
                "prob": p_yes,
                "top_probs": [
                    {"id": 9693, "token": "yes", "prob": p_yes},
                    {"id": 2152, "token": "no", "prob": p_no},
                ],
            }
        ],
    }


@respx.mock(base_url=BASE)
async def test_completion_reranker_scores_and_sorts(
    respx_mock: respx.MockRouter,
) -> None:
    respx_mock.post("/tokenize").mock(
        side_effect=lambda request: httpx.Response(
            200, json={"tokens": [9693 if b'"yes"' in request.content else 2152]}
        )
    )
    scores = iter([completion_response(0.2, 0.8), completion_response(0.9, 0.1)])
    completion = respx_mock.post("/completion").mock(
        side_effect=lambda request: httpx.Response(200, json=next(scores))
    )

    async with httpx.AsyncClient(base_url=BASE) as http:
        reranker = LlamaCppCompletionReranker(
            http, model="qwen3-reranker:4b-fp16", max_concurrent=1
        )
        hits = await reranker.rerank(
            "Q", [make_hit("a", fusion=0.9), make_hit("b", fusion=0.1)], top_n=2
        )

    assert [h.chunk_id for h in hits] == ["b", "a"]
    assert hits[0].rerank_score == pytest.approx(0.9) and hits[
        1
    ].rerank_score == pytest.approx(0.2)
    body = json.loads(completion.calls[0].request.content)
    assert (body["n_predict"], body["n_probs"], body["post_sampling_probs"]) == (
        1,
        2,
        True,
    )
    assert body["logit_bias"] == [[9693, 100.0], [2152, 100.0]]
    assert (
        "<Document>: Paracetamol > Liều dùng" in body["prompt"]
    )  # scores embedding_text, like the evaluation


@respx.mock(base_url=BASE)
async def test_completion_reranker_maps_http_errors(
    respx_mock: respx.MockRouter,
) -> None:
    respx_mock.post("/tokenize").mock(return_value=httpx.Response(500))
    async with httpx.AsyncClient(base_url=BASE) as http:
        reranker = LlamaCppCompletionReranker(http, model="m")
        with pytest.raises(RetrievalError):
            await reranker.rerank("Q", [make_hit("a")], top_n=1)


@respx.mock(base_url=BASE)
async def test_native_reranker_uses_v1_rerank(respx_mock: respx.MockRouter) -> None:
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
        hits = await NativeReranker(http, model="bge-reranker-v2-m3:f16").rerank(
            "Q", [make_hit("a"), make_hit("b")], top_n=1
        )
    assert [h.chunk_id for h in hits] == ["b"] and hits[0].rerank_score == 0.7
    assert b'"documents"' in route.calls[0].request.content


async def test_noop_reranker_keeps_fusion_order() -> None:
    hits = await NoopReranker().rerank(
        "Q", [make_hit("a", fusion=0.1), make_hit("b", fusion=0.9)], top_n=1
    )
    assert [h.chunk_id for h in hits] == ["b"] and hits[0].rerank_score is None


def test_build_reranker_switches_on_protocol() -> None:
    assert isinstance(build_reranker(RerankSettings(protocol="none")), NoopReranker)
    assert isinstance(
        build_reranker(RerankSettings(protocol="native_rerank")), NativeReranker
    )
    assert isinstance(
        build_reranker(RerankSettings(protocol="completion_logprobs")),
        LlamaCppCompletionReranker,
    )
