from unittest.mock import AsyncMock, MagicMock

import pytest

from corpus_pipeline.evaluation.rerankers import LlamaCppReranker
from corpus_pipeline.evaluation.retrieval_types import RetrievalCandidate


class DummySpec:
    name = "qwen3-reranker:0.6b-fp16"
    reranker_protocol = "completion_logprobs"


def test_llama_cpp_reranker_uses_async_completions():
    mock_client = MagicMock()
    mock_client.rerank_completions_async = AsyncMock(return_value=[0.85])

    reranker = LlamaCppReranker(spec=DummySpec(), client=mock_client)

    candidates = [
        RetrievalCandidate(
            chunk_id="chunk1",
            score=0.5,
            rank=1,
            source="dense",
            payload={"text": "doc 1"},
            document_text="doc 1 text",
        )
    ]
    result = reranker.rerank("query text", candidates)
    assert len(result) == 1
    assert result[0].rerank_score == pytest.approx(0.85)
    mock_client.rerank_completions_async.assert_called_once()
