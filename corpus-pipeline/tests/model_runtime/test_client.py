from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from corpus_pipeline.runtime.client import LlamaCppClient


@pytest.mark.asyncio
async def test_rerank_completions_async():
    client = LlamaCppClient(base_url="http://localhost:8080")
    client._single_token_id = MagicMock(
        side_effect=lambda token, model: 500 if token == "yes" else 501
    )

    mock_resp1 = MagicMock()
    mock_resp1.status_code = 200
    mock_resp1.json.return_value = {
        "completion_probabilities": [
            {"top_probs": [{"id": 500, "prob": 0.8}, {"id": 501, "prob": 0.2}]}
        ]
    }

    mock_resp2 = MagicMock()
    mock_resp2.status_code = 200
    mock_resp2.json.return_value = {
        "completion_probabilities": [
            {"top_probs": [{"id": 500, "prob": 0.3}, {"id": 501, "prob": 0.7}]}
        ]
    }

    mock_async_client = AsyncMock()
    mock_async_client.post.side_effect = [mock_resp1, mock_resp2]

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client_cls.return_value.__aenter__.return_value = mock_async_client

        scores = await client.rerank_completions_async(
            ["prompt1", "prompt2"], model="qwen-reranker", concurrency=2
        )

    assert len(scores) == 2
    assert scores[0] == pytest.approx(0.8)
    assert scores[1] == pytest.approx(0.3)
