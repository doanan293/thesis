"""Real Kaggle pipeline data objects for tests (no loosely typed stand-ins)."""

from __future__ import annotations

from pathlib import Path

from seed_pipeline.artifacts.manifest import Completion
from seed_pipeline.integrations.kaggle.config import OwnerConfiguration
from seed_pipeline.integrations.kaggle.models import (
    CloudArtifact,
    InputBundle,
    InputFile,
    JobIdentity,
    KernelRemoteState,
    KernelStatus,
    StageJob,
    StageName,
    StageRequest,
)
from seed_pipeline.runtime.catalog import require_model
from seed_pipeline.runtime.runtime_profiles import RuntimeCandidate

RERANK_MODEL = "qwen3-reranker:0.6b-fp16"


def owners(
    execution: str = "owner",
    *,
    runtime: str | None = None,
    corpus: str | None = None,
    checkpoint: str | None = None,
) -> OwnerConfiguration:
    return OwnerConfiguration(
        execution=execution,
        runtime=runtime or execution,
        corpus=corpus or execution,
        checkpoint=checkpoint or execution,
    )


def rerank_runtime_profile(
    model: str = RERANK_MODEL, *, index: int = 0
) -> RuntimeCandidate:
    search_space = require_model(model).rerank_search_space
    assert search_space is not None
    return search_space.candidates[index]


def stage_request(
    stage: StageName,
    model: str,
    input_path: Path,
    *,
    output_dir: Path | None = None,
    gguf_root: Path | None = None,
    owner_configuration: OwnerConfiguration | None = None,
    runtime_profile: RuntimeCandidate | None = None,
    max_runs: int = 10,
    total_budget_seconds: int = 21_600,
) -> StageRequest:
    return StageRequest(
        stage=stage,
        model=model,
        input_path=input_path,
        output_dir=output_dir or input_path.parent / "output",
        gguf_root=gguf_root or input_path.parent / "gguf",
        owners=owner_configuration or owners(),
        max_runs=max_runs,
        total_budget_seconds=total_budget_seconds,
        runtime_profile=runtime_profile,
    )


def job_identity(
    stage: StageName = StageName.RERANK,
    model: str = RERANK_MODEL,
    *,
    input_sha256: str = "b" * 64,
) -> JobIdentity:
    return JobIdentity.create(
        stage=stage,
        contract_version=1,
        model=model,
        model_sha256="a" * 64,
        input_sha256=input_sha256,
        runtime_parameters={},
    )


def stage_job(
    root: Path,
    *,
    stage: StageName = StageName.RERANK,
    model: str = RERANK_MODEL,
    identity: JobIdentity | None = None,
    data_filename: str = "rerank_scores.jsonl",
    expected_total: int = 1,
) -> StageJob:
    source = root / "job-input.jsonl"
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("{}\n", encoding="utf-8")
    bundle = InputBundle.create((InputFile.create("input", source),))
    output_dir = root / "output"
    return StageJob(
        stage=stage,
        contract_version=1,
        model=model,
        identity=identity or job_identity(stage, model, input_sha256=bundle.sha256),
        input_bundle=bundle,
        output_dir=output_dir,
        local_cache_path=output_dir / data_filename,
        data_filename=data_filename,
        expected_total=expected_total,
        worker_module="seed_pipeline.integrations.kaggle.workers.rerank",
        worker_config={},
    )


def cloud_artifact(
    completion: Completion, *, root: Path = Path("artifact")
) -> CloudArtifact:
    return CloudArtifact(
        data_path=root / "data.jsonl",
        manifest_path=root / "manifest.json",
        identity=job_identity(),
        completion=completion,
    )


class UnusedKernelService:
    """Kernel port for tests that must never call Kaggle."""

    def inspect_state(self, reference: str) -> KernelRemoteState:
        raise AssertionError(f"unexpected kernel inspection: {reference}")

    def confirm_missing(self, reference: str) -> bool | None:
        raise AssertionError(f"unexpected kernel listing: {reference}")

    def push(self, bundle: Path, *, timeout_seconds: int) -> None:
        raise AssertionError(f"unexpected kernel push: {bundle}")

    def wait_for_terminal(
        self, reference: str, *, timeout_seconds: float
    ) -> KernelStatus:
        raise AssertionError(f"unexpected kernel wait: {reference}")

    def download_output(self, reference: str, destination: Path) -> None:
        raise AssertionError(f"unexpected kernel download: {reference}")

    def poll(self, reference: str, *, timeout_seconds: float) -> None:
        raise AssertionError(f"unexpected kernel poll: {reference}")

    def log_tail(self, reference: str) -> str:
        raise AssertionError(f"unexpected kernel log fetch: {reference}")
