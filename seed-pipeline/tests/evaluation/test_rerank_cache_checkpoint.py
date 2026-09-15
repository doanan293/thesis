from pathlib import Path

from tests.integrations.kaggle.factories import rerank_runtime_profile, stage_request

from seed_pipeline.artifacts.manifest import Completion
from seed_pipeline.evaluation.rerank_cache_checkpoint import RerankCacheCheckpoint
from seed_pipeline.integrations.kaggle.artifacts import load_cloud_artifact
from seed_pipeline.integrations.kaggle.models import StageJob, StageName
from seed_pipeline.integrations.kaggle.stages import RerankStage
from seed_pipeline.integrations.kaggle.workers.rerank import run_rerank_worker

MODEL = "qwen3-reranker:0.6b-fp16"


def _job(tmp_path: Path, candidate_bundle) -> StageJob:
    request = stage_request(
        StageName.RERANK,
        MODEL,
        candidate_bundle.data_path,
        output_dir=tmp_path / "remote",
        runtime_profile=rerank_runtime_profile(MODEL),
    )
    return RerankStage().build_job(request)


def _no_scoring(_query: str, _document: str, _model: str) -> float:
    raise AssertionError("pairs from the local cache must not be scored again")


def test_empty_cache_offers_no_checkpoint(tmp_path, candidate_bundle):
    job = _job(tmp_path, candidate_bundle)

    state = RerankCacheCheckpoint(tmp_path / "empty.jsonl").inspect(
        job, download_root=tmp_path / "local"
    )

    assert state.reference is None
    assert state.artifact is None
    assert state.completion == Completion(1, 0, 1)


def test_cached_scores_become_a_checkpoint_the_worker_resumes(
    tmp_path, candidate_bundle, complete_rerank_cache
):
    job = _job(tmp_path, candidate_bundle)

    state = RerankCacheCheckpoint(complete_rerank_cache.path).inspect(
        job, download_root=tmp_path / "local"
    )

    assert state.reference is None
    assert state.completion == Completion(1, 1, 0)
    assert state.artifact is not None
    assert state.artifact.checkpoint_path is not None
    loaded = load_cloud_artifact(
        state.artifact.data_path,
        state.artifact.manifest_path,
        job.identity,
        allow_partial=True,
        expected_artifact_type="rerank_scores",
    )
    assert loaded.strict_identity_match
    config = dict(job.worker_config) | {
        "output_dir": str(tmp_path / "worker"),
        "identity": job.identity.payload,
        "job_sha256": job.identity.sha256,
        "input_files": {
            item.key: {
                "path": str(item.source_path),
                "filename": item.filename,
                "sha256": item.sha256,
            }
            for item in job.input_bundle.files
        },
        "checkpoint_filename": str(state.artifact.checkpoint_path),
    }

    resumed = run_rerank_worker(
        config, score_pair=_no_scoring, emit=lambda _message: None, clock=lambda: 0.0
    )

    assert resumed.completion == Completion(1, 1, 0)
