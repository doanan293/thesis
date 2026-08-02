import json
from pathlib import Path

from corpus_pipeline.integrations.kaggle.api import KaggleCommandRunner
from corpus_pipeline.integrations.kaggle.kernel_service import KernelService
from corpus_pipeline.integrations.kaggle.kernels import PipelineKernelService
from corpus_pipeline.integrations.kaggle.models import JobIdentity, StageJob, StageName


def make_job(tmp_path: Path) -> StageJob:
    identity = JobIdentity.create(
        stage=StageName.QUERY_EMBED,
        contract_version=1,
        model="fake",
        model_sha256="a" * 64,
        input_sha256="b" * 64,
        runtime_parameters={},
    )
    return StageJob(
        StageName.QUERY_EMBED,
        1,
        "fake",
        identity,
        tmp_path / "input.jsonl",
        tmp_path / "output",
        tmp_path / "output" / "query_embeddings.jsonl",
        "query_embeddings.jsonl",
        1,
        "corpus_pipeline.integrations.kaggle.workers.query_embed",
        {
            "model": "fake",
            "input_path": str(tmp_path / "input.jsonl"),
            "identity": identity.payload,
            "job_sha256": identity.sha256,
        },
    )


def test_bundle_uses_generated_runner_and_embedded_source(tmp_path: Path):
    service = PipelineKernelService(
        KernelService(KaggleCommandRunner(dry_run=True), "alice"),
        owner="alice",
        source_root=Path("src"),
    )
    bundle = service.prepare_bundle(
        make_job(tmp_path),
        root=tmp_path / "bundle",
        dataset_references=["alice/input"],
        checkpoint_reference="alice/checkpoint",
        total_budget_seconds=60,
    )
    metadata = json.loads((bundle / "kernel-metadata.json").read_text())
    assert metadata["code_file"] == "runner.py"
    assert metadata["dataset_sources"] == ["alice/input", "alice/checkpoint"]
    assert (
        "corpus_pipeline.integrations.kaggle.workers.query_embed"
        in (bundle / "runner.py").read_text()
    )
    assert (bundle / "source_bundle.zip").is_file()
    config = json.loads((bundle / "stage_config.json").read_text())
    assert config["input_path"] == (
        f"/kaggle/input/pipeline-input-{make_job(tmp_path).identity.sha256[:16]}"
        "/input.jsonl"
    )


def test_rerank_bundle_rewrites_candidate_paths_to_dataset_mount(tmp_path: Path):
    job = make_job(tmp_path)
    job = StageJob(
        StageName.RERANK,
        job.contract_version,
        job.model,
        job.identity,
        tmp_path / "candidates.jsonl",
        job.output_dir,
        job.local_cache_path,
        "rerank_scores.jsonl",
        1,
        "corpus_pipeline.integrations.kaggle.workers.rerank",
        {
            "model": "fake",
            "candidate_path": str(tmp_path / "candidates.jsonl"),
            "candidate_manifest_path": str(tmp_path / "manifest.json"),
        },
    )
    service = PipelineKernelService(
        KernelService(KaggleCommandRunner(dry_run=True), "alice"),
        owner="alice",
        source_root=Path("src"),
    )
    bundle = service.prepare_bundle(
        job,
        root=tmp_path / "bundle",
        dataset_references=["alice/input"],
        checkpoint_reference=None,
        total_budget_seconds=60,
    )
    config = json.loads((bundle / "stage_config.json").read_text())
    mount = f"/kaggle/input/pipeline-input-{job.identity.sha256[:16]}"
    assert config["candidate_path"] == f"{mount}/candidates.jsonl"
    assert config["candidate_manifest_path"] == f"{mount}/manifest.json"
