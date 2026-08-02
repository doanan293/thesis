import pytest

from corpus_pipeline.evaluation.artifact_contracts import ArtifactContractError
from corpus_pipeline.evaluation.metrics_service import (
    MetricsRequest,
    load_and_validate_metric_inputs,
)
from corpus_pipeline.evaluation.retrieval_candidate_artifact import (
    build_candidate_artifact,
)
from corpus_pipeline.evaluation.retrieval_types import RetrievalCandidate
from corpus_pipeline.evaluation.run_workspace import RunIdentity, RunWorkspace


def test_metrics_request_is_offline(tmp_path):
    request = MetricsRequest(tmp_path / "run")
    assert request.top_k == 10


def test_metrics_rejects_evaluation_changed_after_retrieval(tmp_path):
    evaluation = tmp_path / "evaluation.jsonl"
    evaluation.write_text('{"query_id":"q1","query":"query"}\n', encoding="utf-8")
    identity = RunIdentity(
        str(evaluation),
        "0" * 64,
        "collection",
        "qwen3-embedding:0.6b-fp16",
        None,
        "bm25",
        1,
        60,
        None,
    )
    workspace = RunWorkspace.open_or_create(tmp_path / "run", identity)
    candidate = RetrievalCandidate(
        chunk_id="c1",
        score=1.0,
        rank=1,
        source="bm25",
        payload={"chunk_id": "c1", "chunk_text": "document"},
    )
    artifact = build_candidate_artifact(
        rows=[{"query_id": "q1", "query": "query"}],
        retriever=type(
            "Retriever", (), {"search": lambda self, _query, limit: [candidate]}
        )(),
        output_path=workspace.candidates_dir / "candidates.jsonl",
        identity={},
        candidate_k=1,
    )
    workspace.record_candidates(artifact)

    with pytest.raises(ArtifactContractError, match=r"evaluation.*changed"):
        load_and_validate_metric_inputs(MetricsRequest(workspace.root))
