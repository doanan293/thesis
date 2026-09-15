from unittest.mock import create_autospec

from seed_pipeline.evaluation.rerankers import LlamaCppReranker
from seed_pipeline.evaluation.retrieval_types import RetrievalCandidate
from seed_pipeline.runtime.catalog import require_model
from seed_pipeline.runtime.client import LlamaCppClient


def _candidate(chunk_id: str, text: str, rank: int) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=chunk_id,
        score=0.5,
        rank=rank,
        source="test",
        payload={"chunk_text": text},
    )


def test_local_reranker_scores_every_candidate_in_one_native_request():
    spec = require_model("qwen3-reranker:4b-fp16")
    client = create_autospec(LlamaCppClient, instance=True)
    client.rerank_native.return_value = [0.2, 0.9]
    reranker = LlamaCppReranker(spec, client)

    result = reranker.rerank(
        "thuốc gì",
        [_candidate("a", "tài liệu a", 1), _candidate("b", "tài liệu b", 2)],
    )

    client.rerank_native.assert_called_once_with(
        "thuốc gì", ["tài liệu a", "tài liệu b"], spec.name
    )
    assert [(item.chunk_id, item.rerank_score, item.rank) for item in result] == [
        ("b", 0.9, 1),
        ("a", 0.2, 2),
    ]
