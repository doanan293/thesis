from unittest.mock import create_autospec

from corpus_pipeline.evaluation.rerankers import LlamaCppReranker
from corpus_pipeline.evaluation.retrieval_types import RetrievalCandidate
from corpus_pipeline.runtime.catalog import require_model
from corpus_pipeline.runtime.client import LlamaCppClient


def test_local_completion_reranker_executes_catalog_contract():
    spec = require_model("bge-reranker-v2-gemma:f16")
    client = create_autospec(LlamaCppClient, instance=True)
    client.rerank_completions_async.return_value = [0.8]
    reranker = LlamaCppReranker(spec, client)
    candidate = RetrievalCandidate(
        chunk_id="chunk-1",
        score=0.5,
        rank=1,
        source="test",
        payload={"chunk_text": "tài liệu"},
    )

    result = reranker.rerank("thuốc gì", [candidate])

    call = client.rerank_completions_async.call_args
    prompts = call.args[0] if call.args else call.kwargs["prompts"]
    assert prompts == [
        "<bos>A: thuốc gì\n"
        "B: tài liệu\n"
        "Given a query A and a passage B, determine whether the passage contains "
        "an answer to the query by providing a prediction of either 'Yes' or 'No'."
    ]
    assert call.kwargs["contract"] is spec.rerank_contract
    assert result[0].rerank_score == 0.8
