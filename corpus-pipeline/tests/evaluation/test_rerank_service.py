from corpus_pipeline.evaluation.rerank_service import (
    RerankRequest,
    rerank_checkpoint_path,
)


def test_rerank_request_supports_run_resolution(tmp_path):
    request = RerankRequest(tmp_path / "run", None, None, "model", False, True, 10, 1.0)
    assert request.candidates_dir is None


def test_rerank_checkpoint_is_scoped_to_candidate_identity(tmp_path):
    output = tmp_path / "rerank"
    first = rerank_checkpoint_path(output, "a" * 64, "qwen3-reranker:0.6b-fp16")
    assert first != rerank_checkpoint_path(output, "b" * 64, "qwen3-reranker:0.6b-fp16")
    assert first != rerank_checkpoint_path(output, "a" * 64, "bge-reranker-v2-m3:f16")
