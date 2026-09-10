from corpus_pipeline.evaluation.rerankers import LlamaCppReranker
from corpus_pipeline.evaluation.retrieval_types import RetrievalCandidate
from corpus_pipeline.runtime.catalog import require_model


class ContractCapturingClient:
    def __init__(self):
        self.prompts = None
        self.contract = None

    async def rerank_completions_async(
        self, prompts, _model, concurrency=16, contract=None, cache_prompt=True
    ):
        del concurrency, cache_prompt
        self.prompts = prompts
        self.contract = contract
        return [0.8]


def test_local_completion_reranker_executes_catalog_contract():
    spec = require_model("bge-reranker-v2-gemma:f16")
    client = ContractCapturingClient()
    reranker = LlamaCppReranker(spec, client)
    candidate = RetrievalCandidate(
        chunk_id="chunk-1",
        score=0.5,
        rank=1,
        source="test",
        payload={"chunk_text": "tài liệu"},
    )

    result = reranker.rerank("thuốc gì", [candidate])

    assert client.prompts == [
        "<bos>A: thuốc gì\n"
        "B: tài liệu\n"
        "Given a query A and a passage B, determine whether the passage contains "
        "an answer to the query by providing a prediction of either 'Yes' or 'No'."
    ]
    assert client.contract is spec.rerank_contract
    assert result[0].rerank_score == 0.8
