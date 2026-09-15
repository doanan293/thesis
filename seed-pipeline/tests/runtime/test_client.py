import pytest

from seed_pipeline.runtime.client import LlamaCppClient, LlamaCppResponseError


class JsonResponse:
    def __init__(self, payload: object, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code
        self.text = ""

    def json(self) -> object:
        return self.payload


def test_rerank_native_returns_scores_in_document_order():
    seen = []

    def post(url, json, timeout):
        seen.append((url, json))
        return JsonResponse(
            {
                "results": [
                    {"index": 1, "relevance_score": 0.2},
                    {"index": 0, "relevance_score": 0.9},
                ]
            }
        )

    client = LlamaCppClient("http://server", post=post)

    assert client.rerank_native("q", ["a", "b"], "model") == [0.9, 0.2]
    assert seen == [
        (
            "http://server/v1/rerank",
            {"model": "model", "query": "q", "documents": ["a", "b"]},
        )
    ]


def test_rerank_native_rejects_duplicate_indices():
    client = LlamaCppClient(
        "http://server",
        post=lambda *_args, **_kwargs: JsonResponse(
            {
                "results": [
                    {"index": 0, "relevance_score": 0.5},
                    {"index": 0, "relevance_score": 0.4},
                ]
            }
        ),
    )

    with pytest.raises(LlamaCppResponseError, match="duplicate"):
        client.rerank_native("q", ["a", "b"], "model")


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
