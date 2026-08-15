import pytest

from corpus_pipeline.runtime.catalog import require_model
from corpus_pipeline.runtime.client import LlamaCppClient, LlamaCppRequestError


def test_completion_payload_can_include_model_name():
    contract = require_model("qwen3-reranker:0.6b-fp16").rerank_contract
    assert contract is not None

    payload = LlamaCppClient.completion_payload(
        "prompt",
        contract,
        model="qwen3-reranker:0.6b-fp16",
        yes_id=10,
        no_id=11,
    )

    assert payload["model"] == "qwen3-reranker:0.6b-fp16"


@pytest.mark.asyncio
async def test_async_completion_retries_retryable_http_status(monkeypatch):
    import httpx

    attempts = 0

    class Response:
        text = "busy"

        def __init__(self, status_code):
            self.status_code = status_code

        def json(self):
            return {
                "completion_probabilities": [
                    {"top_probs": [{"id": 10, "prob": 0.8}, {"id": 11, "prob": 0.2}]}
                ]
            }

    class AsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, *_args, **_kwargs):
            nonlocal attempts
            attempts += 1
            return Response(503 if attempts == 1 else 200)

    monkeypatch.setattr(httpx, "AsyncClient", AsyncClient)
    client = LlamaCppClient("http://server", max_attempts=2, retry_delay_seconds=0)
    monkeypatch.setattr(client, "_single_token_id", lambda text, _model: 10 if text == "yes" else 11)
    contract = require_model("qwen3-reranker:0.6b-fp16").rerank_contract

    result = await client.rerank_completions_async(["prompt"], "model", contract=contract)

    assert result == [pytest.approx(0.8)]
    assert attempts == 2


@pytest.mark.asyncio
async def test_async_completion_does_not_retry_non_retryable_status(monkeypatch):
    import httpx

    attempts = 0

    class Response:
        status_code = 400
        text = "bad request"

    class AsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, *_args, **_kwargs):
            nonlocal attempts
            attempts += 1
            return Response()

    monkeypatch.setattr(httpx, "AsyncClient", AsyncClient)
    client = LlamaCppClient("http://server", max_attempts=3, retry_delay_seconds=0)
    monkeypatch.setattr(client, "_single_token_id", lambda text, _model: 10 if text == "yes" else 11)
    contract = require_model("qwen3-reranker:0.6b-fp16").rerank_contract

    with pytest.raises(LlamaCppRequestError):
        await client.rerank_completions_async(["prompt"], "model", contract=contract)

    assert attempts == 1


def test_sync_request_retries_retryable_http_status():
    attempts = 0

    class Response:
        def __init__(self, status_code):
            self.status_code = status_code
            self.text = "busy"

        def json(self):
            return {"ok": True}

    def post(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        return Response(503 if attempts == 1 else 200)

    client = LlamaCppClient(
        "http://server", post=post, max_attempts=2, retry_delay_seconds=0
    )

    assert client._request("/health", {}) == {"ok": True}
    assert attempts == 2
