import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import pytest

from seed_pipeline.artifacts.manifest import Completion
from seed_pipeline.cache.jsonl_records import append_record
from seed_pipeline.evaluation import rerank_service
from seed_pipeline.evaluation.rerank_score_cache import RerankScoreCache
from seed_pipeline.evaluation.rerank_service import (
    KaggleRerankBackend,
    LocalRerankBackend,
    RerankRequest,
    _cleanup_completed_stage_artifact,
)
from seed_pipeline.evaluation.run_workspace import load_run_record
from seed_pipeline.integrations.kaggle import auto_profile
from seed_pipeline.integrations.kaggle import service as kaggle_service
from seed_pipeline.integrations.kaggle.models import ActionVerb, ReconcileAction
from seed_pipeline.runtime.catalog import require_model


class FakeReranker:
    def rerank(self, query, candidates):
        del query
        return [
            candidate.with_rerank_score(float(index), index)
            for index, candidate in enumerate(candidates, start=1)
        ]


def request(run_root, model, *, force=False, dry_run=False):
    return RerankRequest(
        run_root=run_root,
        candidates_dir=None,
        model=model,
        force=force,
        dry_run=dry_run,
        budget_seconds=60,
        request_timeout_seconds=5.0,
    )


def fake_local_backend():
    return LocalRerankBackend(reranker_factory=lambda _spec, _timeout: FakeReranker())


def _score_record(*, query_id: str, score: float) -> dict[str, object]:
    return {
        "reranker": "qwen3-reranker:0.6b-fp16",
        "model_sha256": "model-sha",
        "request_contract_sha256": "contract-sha",
        "query_id": query_id,
        "query_hash": f"query-hash-{query_id}",
        "chunk_id": "chunk-1",
        "document_hash": "document-hash-1",
        "score": score,
    }


def test_remote_rerank_reconciliation_replaces_conflicts_and_keeps_local_only(
    tmp_path,
):
    """A missing replacement would keep conflicting local scores and fail merge."""
    local_path = tmp_path / "local.jsonl"
    remote_path = tmp_path / "remote.jsonl"
    append_record(
        local_path,
        _score_record(query_id="shared", score=0.1),
        schema="rerank-score-v2",
    )
    append_record(
        local_path,
        _score_record(query_id="local-only", score=0.2),
        schema="rerank-score-v2",
    )
    append_record(
        remote_path,
        _score_record(query_id="shared", score=0.9),
        schema="rerank-score-v2",
    )

    remote = RerankScoreCache(
        remote_path,
        model_sha256="model-sha",
        request_contract_sha256="contract-sha",
    )
    rerank_service._merge_remote_rerank_scores(
        local_path,
        remote,
        model_sha256="model-sha",
        request_contract_sha256="contract-sha",
    )

    reconciled = RerankScoreCache(
        local_path,
        model_sha256="model-sha",
        request_contract_sha256="contract-sha",
    )
    scores_by_query = {key.query_id: score for key, score in reconciled.records.items()}
    assert scores_by_query == {"shared": 0.9, "local-only": 0.2}


def test_two_models_register_two_variants_without_mutating_candidates(complete_run):
    candidates_path = complete_run / "candidates" / "candidates.jsonl"
    before = candidates_path.read_bytes()
    backend = fake_local_backend()

    first = backend.run(request(complete_run, "qwen3-reranker:0.6b-fp16"))
    second = backend.run(request(complete_run, "bge-reranker-v2-m3:f16"))

    record = load_run_record(complete_run / "run.json")
    assert len(record.rerank_variants) == 2
    assert first.artifact_dir is not None
    assert second.artifact_dir is not None
    assert first.artifact_dir != second.artifact_dir
    assert candidates_path.read_bytes() == before


def test_local_rerun_reuses_the_same_variant(complete_run):
    backend = fake_local_backend()
    first = backend.run(request(complete_run, "qwen3-reranker:0.6b-fp16"))
    assert first.artifact_dir is not None
    score_bytes = (first.artifact_dir / "rerank_scores.jsonl").read_bytes()

    second = backend.run(request(complete_run, "qwen3-reranker:0.6b-fp16"))

    assert second.artifact_dir is not None
    assert second.variant_sha256 == first.variant_sha256
    assert second.artifact_dir == first.artifact_dir
    assert (second.artifact_dir / "rerank_scores.jsonl").read_bytes() == score_bytes


def fake_kaggle_dry_run(**_kwargs):
    return SimpleNamespace(
        artifact_path=None,
        completion=Completion(total=1, complete=0, missing=1),
        actions=(
            ReconcileAction(
                "model", "owner/model", ActionVerb.CREATE, "dataset is missing"
            ),
            ReconcileAction(
                "input", "owner/input", ActionVerb.CREATE, "dataset is missing"
            ),
        ),
    )


def test_kaggle_dry_run_formats_resource_identity_and_does_not_register(
    complete_run, monkeypatch, tmp_path
):
    monkeypatch.setattr(kaggle_service, "run_kaggle_stage", fake_kaggle_dry_run)
    original_ensure_profile = auto_profile.ensure_runtime_profile

    def isolated_profile(**kwargs):
        return original_ensure_profile(
            **kwargs,
            profile_root=tmp_path / "profiles",
        )

    monkeypatch.setattr(auto_profile, "ensure_runtime_profile", isolated_profile)

    result = KaggleRerankBackend().run(
        request(complete_run, "qwen3-reranker:0.6b-fp16", dry_run=True)
    )

    assert result.incomplete
    assert "profile=benchmark-required" in result.actions
    assert load_run_record(complete_run / "run.json").rerank_variants == {}


def test_kaggle_rerank_propagates_selected_account(complete_run, monkeypatch):
    seen = []
    spec = require_model("qwen3-reranker:0.6b-fp16")
    assert spec.rerank_search_space is not None
    selected = spec.rerank_search_space.candidates[0]
    monkeypatch.setattr(
        auto_profile,
        "ensure_runtime_profile",
        lambda **kwargs: (
            seen.append(("profile", kwargs["kaggle_account"]))
            or SimpleNamespace(
                profile=SimpleNamespace(selected=selected), action="reuse"
            )
        ),
    )
    monkeypatch.setattr(
        kaggle_service,
        "run_kaggle_stage",
        lambda **kwargs: (
            seen.append(("stage", kwargs["kaggle_account"]))
            or SimpleNamespace(
                artifact_path=None,
                completion=Completion(1, 0, 1),
                actions=(),
            )
        ),
    )

    result = KaggleRerankBackend().run(
        replace(
            request(complete_run, "qwen3-reranker:0.6b-fp16"),
            kaggle_account="acc2",
        )
    )

    assert result.incomplete
    assert seen == [("profile", "acc2"), ("stage", "acc2")]


def test_kaggle_rerank_allows_different_models_in_the_same_run(
    complete_run, monkeypatch, tmp_path
):
    from seed_pipeline.evaluation import rerank_service

    monkeypatch.setattr(rerank_service, "WORK_DIR", tmp_path / "work")

    first_entered = threading.Event()
    release_first = threading.Event()

    def run_unlocked(rerank_request):
        if rerank_request.model == "qwen3-reranker:0.6b-fp16":
            first_entered.set()
            assert release_first.wait(2)
        return rerank_request.model

    monkeypatch.setattr(
        KaggleRerankBackend,
        "_run_kaggle_unlocked",
        staticmethod(run_unlocked),
    )
    backend = KaggleRerankBackend()
    first_request = request(complete_run, "qwen3-reranker:0.6b-fp16")
    second_request = request(complete_run, "bge-reranker-v2-m3:f16")

    with ThreadPoolExecutor(max_workers=1) as executor:
        first = executor.submit(backend.run, first_request)
        assert first_entered.wait(2)
        try:
            second = backend.run(second_request)
        finally:
            release_first.set()

        assert first.result(timeout=2) == first_request.model
    assert second == second_request.model


def test_kaggle_rerank_serializes_same_model_across_runs(tmp_path, monkeypatch):
    from seed_pipeline.evaluation import rerank_service

    monkeypatch.setattr(rerank_service, "WORK_DIR", tmp_path / "work")

    first_entered = threading.Event()
    release_first = threading.Event()

    def run_unlocked(_request):
        first_entered.set()
        assert release_first.wait(2)
        return "first"

    monkeypatch.setattr(
        KaggleRerankBackend,
        "_run_kaggle_unlocked",
        staticmethod(run_unlocked),
    )
    backend = KaggleRerankBackend()
    first_request = request(tmp_path / "run-a", "qwen3-reranker:0.6b-fp16")
    second_request = request(tmp_path / "run-b", "qwen3-reranker:0.6b-fp16")

    with ThreadPoolExecutor(max_workers=1) as executor:
        first = executor.submit(backend.run, first_request)
        assert first_entered.wait(2)
        try:
            with pytest.raises(RuntimeError, match="already has an active Kaggle job"):
                backend.run(second_request)
        finally:
            release_first.set()
        assert first.result(timeout=2) == "first"


def test_kaggle_migrates_matching_legacy_variant_before_submit(
    complete_run, complete_rerank_cache, monkeypatch
):
    run_path = complete_run / "run.json"
    current = load_run_record(run_path)
    run_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "identity": current.identity.__dict__,
                "status": "reranked",
                "candidates_dir": "candidates",
                "rerank_scores_dir": str(complete_rerank_cache.path),
                "reranker": "qwen3-reranker:0.6b-fp16",
            }
        ),
        encoding="utf-8",
    )

    def fail_submit(**_kwargs):
        raise AssertionError("matching legacy variant should be reused")

    monkeypatch.setattr(kaggle_service, "run_kaggle_stage", fail_submit)
    result = KaggleRerankBackend().run(
        request(complete_run, "qwen3-reranker:0.6b-fp16")
    )

    assert result.artifact_dir is not None
    assert load_run_record(run_path).schema_version == 2


def test_completed_stage_cleanup_removes_only_job_directory(tmp_path):
    root = tmp_path / "kaggle-rerank-scores" / "model"
    job = root / "rerank" / "model" / "job-a"
    sibling = root / "rerank" / "model" / "job-b"
    artifact = job / "rerank_scores.jsonl"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("score\n", encoding="utf-8")
    sibling.mkdir(parents=True)
    (sibling / "keep.jsonl").write_text("keep\n", encoding="utf-8")

    _cleanup_completed_stage_artifact(artifact, root)

    assert not job.exists()
    assert (sibling / "keep.jsonl").is_file()
    assert root.is_dir()


def test_completed_stage_cleanup_rejects_artifact_outside_staging_root(tmp_path):
    root = tmp_path / "expected"
    artifact = tmp_path / "outside" / "rerank_scores.jsonl"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("score\n", encoding="utf-8")

    with pytest.raises(ValueError, match="outside Kaggle rerank staging root"):
        _cleanup_completed_stage_artifact(artifact, root)

    assert artifact.is_file()
