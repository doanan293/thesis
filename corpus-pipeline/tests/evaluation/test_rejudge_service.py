import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from corpus_pipeline.evaluation.artifact_contracts import (
    ArtifactContractError,
    ArtifactManifest,
    sha256_file,
    write_json,
)
from corpus_pipeline.evaluation.metrics_artifacts import MetricsArtifactResult
from corpus_pipeline.evaluation.metrics_service import MetricsRequest, MetricsResult
from corpus_pipeline.evaluation.query_hash import query_hash
from corpus_pipeline.evaluation.rejudge_service import (
    RejudgeRequest,
    run_rejudging,
)
from corpus_pipeline.evaluation.rerank_contract import document_hash
from corpus_pipeline.evaluation.run_workspace import (
    RunIdentity,
    RunWorkspace,
    load_run_record,
)


def _write_candidate_bundle(root: Path, evaluation_sha: str) -> str:
    candidate_root = root / "candidates"
    candidate_root.mkdir(parents=True, exist_ok=True)
    record = {
        "query_id": "q1",
        "query": "cho tôi thông tin về X",
        "query_hash": query_hash("cho tôi thông tin về X"),
        "candidates": [
            {
                "chunk_id": "drug:x:thong-tin-chung:chunk-001",
                "retrieval_score": 0.9,
                "retrieval_rank": 1,
                "source": "test",
                "document_text": "X thông tin chung",
                "document_hash": document_hash("X thông tin chung"),
                "payload": {
                    "chunk_id": "drug:x:thong-tin-chung:chunk-001",
                    "section_id": "drug:x:thong-tin-chung",
                    "chunk_index": 1,
                },
            }
        ],
    }
    data_path = candidate_root / "candidates.jsonl"
    data_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    manifest = ArtifactManifest.create(
        artifact_type="retrieval_candidates",
        data_path=data_path,
        record_count=1,
        identity={
            "evaluation_sha256": evaluation_sha,
            "pair_count": 1,
            "candidate_k": 1,
        },
    )
    payload = manifest.to_dict()
    payload.update(total=1, complete=1, missing=0)
    write_json(candidate_root / "manifest.json", payload)
    return manifest.data_sha256


def _write_run(
    metadata_root: Path,
    artifact_root: Path,
    evaluation_path: Path,
    evaluation_sha: str,
    *,
    retriever: str,
) -> None:
    identity = RunIdentity(
        evaluation_path=str(evaluation_path),
        evaluation_sha256=evaluation_sha,
        collection_name="collection",
        embedding_model="embedding",
        query_embeddings_sha256="query-cache",
        retriever=retriever,
        candidate_k=1,
        rrf_k=2,
        limit=None,
        prefetch_k=1 if retriever == "hybrid" else None,
    )
    metadata_root.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    workspace = RunWorkspace.open_or_create(
        metadata_root, identity, artifact_root=artifact_root
    )
    data_sha = _write_candidate_bundle(artifact_root, evaluation_sha)
    shutil.copy2(
        artifact_root / "candidates" / "manifest.json",
        metadata_root / "candidates-manifest.json",
    )
    record = load_run_record(metadata_root / "run.json")
    workspace.write_record(
        replace(record, candidates_dir="candidates", status="retrieved")
    )
    assert data_sha
    reports = artifact_root / "reports"
    reports.mkdir()
    (reports / "old-report.md").write_text("old", encoding="utf-8")


@pytest.fixture
def rejudge_fixture(tmp_path: Path):
    evaluation_path = tmp_path / "evaluation.jsonl"
    evaluation_path.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "query": "cho tôi thông tin về X",
                "eval_group": "formulary",
                "eval_tags": ["drug_fact"],
                "expected_section_id": "drug:x:tuong-ky",
                "expected_section_ids": ["drug:x:tuong-ky"],
                "expected_chunk_id": "drug:x:tuong-ky:chunk-001",
                "retrieval_granularity": "section",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    evaluation_sha = sha256_file(evaluation_path)
    dense_metadata = tmp_path / "dense-meta"
    dense_artifacts = tmp_path / "dense-artifacts"
    hybrid_metadata = tmp_path / "hybrid-meta"
    hybrid_artifacts = tmp_path / "hybrid-artifacts"
    _write_run(
        dense_metadata,
        dense_artifacts,
        evaluation_path,
        evaluation_sha,
        retriever="dense",
    )
    _write_run(
        hybrid_metadata,
        hybrid_artifacts,
        evaluation_path,
        evaluation_sha,
        retriever="hybrid",
    )
    return RejudgeRequest(
        evaluation_path=evaluation_path,
        dense_run_root=dense_metadata,
        dense_artifact_root=dense_artifacts,
        hybrid_run_root=hybrid_metadata,
        hybrid_artifact_root=hybrid_artifacts,
        apply=False,
    )


def _write_fake_metrics(request: MetricsRequest) -> MetricsResult:
    artifact_root = request.artifact_root
    assert artifact_root is not None
    generated_report = artifact_root / "reports" / "generated" / "report.md"
    generated_report.parent.mkdir(parents=True, exist_ok=True)
    generated_report.write_text("new report", encoding="utf-8")
    return MetricsResult(
        baseline=MetricsArtifactResult(
            artifact_dir=artifact_root / "reports" / "baseline",
            report_path=generated_report,
            results_path=artifact_root / "reports" / "metrics.jsonl",
            metrics_sha256="metrics",
        ),
        reranked=(),
    )


def test_rejudge_dry_run_does_not_modify_or_run_metrics(
    rejudge_fixture: RejudgeRequest, monkeypatch: pytest.MonkeyPatch
):
    before = rejudge_fixture.evaluation_path.read_bytes()
    calls = []

    def fail_if_called(request):
        calls.append(request)
        raise AssertionError("metrics must not run during dry-run")

    monkeypatch.setattr(
        "corpus_pipeline.evaluation.rejudge_service.RAG_FINAL_SECTIONS_PATH",
        rejudge_fixture.evaluation_path,
    )
    result = run_rejudging(
        rejudge_fixture,
        metrics_runner=fail_if_called,
        known_section_ids={"drug:x:thong-tin-chung"},
    )

    assert result.applied is False
    assert result.summary["drug_fact_changed"] == 1
    assert calls == []
    assert rejudge_fixture.evaluation_path.read_bytes() == before


def test_rejudge_apply_replaces_reports_and_updates_hashes(
    rejudge_fixture: RejudgeRequest, monkeypatch: pytest.MonkeyPatch
):
    requests = []

    def fake_metrics(request: MetricsRequest) -> MetricsResult:
        requests.append(request)
        return _write_fake_metrics(request)

    monkeypatch.setattr(
        "corpus_pipeline.evaluation.rejudge_service.RAG_FINAL_SECTIONS_PATH",
        rejudge_fixture.evaluation_path,
    )
    result = run_rejudging(
        replace(rejudge_fixture, apply=True),
        metrics_runner=fake_metrics,
        known_section_ids={"drug:x:thong-tin-chung"},
    )

    assert result.applied is True
    assert len(requests) == 2
    assert all(request.top_k == 30 for request in requests)
    assert not (
        rejudge_fixture.dense_artifact_root / "reports" / "old-report.md"
    ).exists()
    assert not (
        rejudge_fixture.hybrid_artifact_root / "reports" / "old-report.md"
    ).exists()
    assert (rejudge_fixture.dense_run_root / "reports" / "baseline.md").read_text(
        encoding="utf-8"
    ) == "new report"
    assert (rejudge_fixture.hybrid_run_root / "reports" / "baseline.md").read_text(
        encoding="utf-8"
    ) == "new report"
    assert sha256_file(rejudge_fixture.evaluation_path) == result.new_evaluation_sha256
    assert (
        load_run_record(
            rejudge_fixture.dense_run_root / "run.json"
        ).identity.evaluation_sha256
        == result.new_evaluation_sha256
    )


def test_rejudge_rolls_back_when_metrics_fails(
    rejudge_fixture: RejudgeRequest, monkeypatch: pytest.MonkeyPatch
):
    original_eval = rejudge_fixture.evaluation_path.read_bytes()
    original_dense = (
        rejudge_fixture.dense_artifact_root / "reports" / "old-report.md"
    ).read_bytes()

    def fail_on_hybrid(request: MetricsRequest) -> MetricsResult:
        if request.run_root == rejudge_fixture.hybrid_run_root:
            raise RuntimeError("metrics failure")
        return _write_fake_metrics(request)

    monkeypatch.setattr(
        "corpus_pipeline.evaluation.rejudge_service.RAG_FINAL_SECTIONS_PATH",
        rejudge_fixture.evaluation_path,
    )
    with pytest.raises(RuntimeError, match="metrics failure"):
        run_rejudging(
            replace(rejudge_fixture, apply=True),
            metrics_runner=fail_on_hybrid,
            known_section_ids={"drug:x:thong-tin-chung"},
        )

    assert rejudge_fixture.evaluation_path.read_bytes() == original_eval
    assert (
        rejudge_fixture.dense_artifact_root / "reports" / "old-report.md"
    ).read_bytes() == original_dense
    assert load_run_record(
        rejudge_fixture.dense_run_root / "run.json"
    ).identity.evaluation_sha256 == sha256_file(rejudge_fixture.evaluation_path)


def test_rejudge_rejects_stale_run_before_apply(
    rejudge_fixture: RejudgeRequest,
):
    run_path = rejudge_fixture.dense_run_root / "run.json"
    payload = json.loads(run_path.read_text(encoding="utf-8"))
    payload["identity"]["evaluation_sha256"] = "f" * 64
    run_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ArtifactContractError, match="dense run evaluation hash"):
        run_rejudging(rejudge_fixture)
