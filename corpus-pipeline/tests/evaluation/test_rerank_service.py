import json
from types import SimpleNamespace

from corpus_pipeline.artifacts.manifest import Completion
from corpus_pipeline.evaluation.rerank_service import (
    KaggleRerankBackend,
    LocalRerankBackend,
    RerankRequest,
)
from corpus_pipeline.evaluation.run_workspace import load_run_record
from corpus_pipeline.integrations.kaggle import service as kaggle_service
from corpus_pipeline.integrations.kaggle.models import ActionVerb, ReconcileAction


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
    complete_run, monkeypatch
):
    monkeypatch.setattr(kaggle_service, "run_kaggle_stage", fake_kaggle_dry_run)

    result = KaggleRerankBackend().run(
        request(complete_run, "qwen3-reranker:0.6b-fp16", dry_run=True)
    )

    assert result.incomplete
    assert any(
        "create model owner/model: dataset is missing" in item
        for item in result.actions
    )
    assert any(
        "create input owner/input: dataset is missing" in item
        for item in result.actions
    )
    assert load_run_record(complete_run / "run.json").rerank_variants == {}


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
