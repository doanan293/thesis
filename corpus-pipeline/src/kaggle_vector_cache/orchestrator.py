from __future__ import annotations

import dataclasses
import subprocess
import time
from pathlib import Path

from kaggle_vector_cache.canonical_gguf import CanonicalModelArtifact
from kaggle_vector_cache.checkpoint_service import CloudCheckpointService
from kaggle_vector_cache.dataset_service import (
    CORPUS_DATASET_SLUG,
    DatasetService,
)
from kaggle_vector_cache.kaggle_api import (
    KaggleCommandRunner,
    kernel_status_command,
    quota_command,
)
from kaggle_vector_cache.kernel_service import (
    LLAMA_CPP_DATASET_SLUG,
    KernelService,
)
from kaggle_vector_cache.lock import ModelRunLock, RunResult
from kaggle_vector_cache.manifests import load_json, validate_checkpoint_manifest
from kaggle_vector_cache.models import require_model
from kaggle_vector_cache.parsers import (
    checkpoint_dataset_slug,
    format_compact_elapsed,
    format_elapsed,
    format_timed_log_lines,
    kernel_slug,
    parse_dataset_status,
    parse_gpu_quota_hours,
    parse_kernel_log_entries,
    parse_kernel_status,
)
from kaggle_vector_cache.runtime_service import RuntimeService
from kaggle_vector_cache.temp_workspace import TemporaryWorkspace
from vector_store.ingest_vectors import model_slug

__all__ = [
    "KaggleVectorCacheOrchestrator",
    "ModelRunLock",
    "RunResult",
    "checkpoint_dataset_slug",
    "format_compact_elapsed",
    "format_elapsed",
    "format_timed_log_lines",
    "kernel_slug",
    "parse_dataset_status",
    "parse_gpu_quota_hours",
    "parse_kernel_log_entries",
    "parse_kernel_status",
    "time",
]


class KaggleVectorCacheOrchestrator:
    def __init__(
        self,
        owner: str = "owner",
        project_root: Path | str = ".",
        temp_root: Path | str = "/tmp",
        output_dir: Path | str | None = None,
        runtime_owner: str | None = None,
        corpus_owner: str | None = None,
        checkpoint_owner: str | None = None,
        command_runner: KaggleCommandRunner | None = None,
        poll_interval_seconds: int = 20,
    ):
        self.owner = owner
        self.runtime_owner = runtime_owner or owner
        self.corpus_owner = corpus_owner or self.runtime_owner
        self.checkpoint_owner = checkpoint_owner or owner
        self.project_root = Path(project_root)
        self.temp_root = Path(temp_root)
        self.output_dir = (
            Path(output_dir)
            if output_dir is not None
            else self.project_root / "data/cache/vector_embeddings"
        )
        self.runner = command_runner or KaggleCommandRunner()
        self.command_runner = self.runner
        self.poll_interval_seconds = max(1, min(int(poll_interval_seconds), 60))
        self.dataset_service = DatasetService(self.runner, self.owner)
        self.runtime_datasets = DatasetService(self.runner, self.runtime_owner)
        self.checkpoint_datasets = DatasetService(
            self.runner,
            self.checkpoint_owner,
        )
        self.kernel_service = KernelService(
            runner=self.runner,
            owner=self.owner,
            runtime_owner=self.runtime_owner,
            corpus_owner=self.corpus_owner,
            checkpoint_owner=self.checkpoint_owner,
        )
        self.checkpoints = CloudCheckpointService(
            self.runner,
            self.checkpoint_datasets,
            self.checkpoint_owner,
        )
        self.runtime_service = RuntimeService(
            self.runner,
            self.runtime_datasets,
            self.kernel_service,
        )

    @property
    def corpus_path(self) -> Path:
        return self.project_root / "data/processed/rag-final/chunks.jsonl"

    def check_quota(self, required_hours: float) -> None:
        output = self.runner.run(quota_command(), capture_output=True)
        remaining = parse_gpu_quota_hours(output) if output else 30.0
        if remaining < required_hours:
            raise RuntimeError(
                f"Insufficient GPU quota: remaining {remaining}h < required {required_hours}h"
            )

    def require_cloud_dependencies(self, model: str) -> None:
        spec = require_model(model)
        checks = [
            (
                f"{self.corpus_owner}/{CORPUS_DATASET_SLUG}",
                "uv run python -m cli.kaggle_vector_cache corpus publish",
            ),
            (
                f"{self.runtime_owner}/{spec.gguf_dataset_slug}",
                "uv run python -m cli.kaggle_vector_cache models publish",
            ),
            (
                f"{self.runtime_owner}/{LLAMA_CPP_DATASET_SLUG}",
                "uv run python -m cli.kaggle_vector_cache llama-cpp-runtime build",
            ),
        ]
        for reference, command in checks:
            self.dataset_service.require_ready(reference, command)

    def _corpus_manifest(self, destination: Path) -> dict:
        return self.dataset_service.fetch_json(
            f"{self.corpus_owner}/{CORPUS_DATASET_SLUG}",
            "manifest.json",
            destination,
        )

    def _kernel_status_optional(self, model: str) -> str | None:
        reference = f"{self.owner}/{kernel_slug(model)}"
        try:
            output = self.runner.run(
                kernel_status_command(reference),
                capture_output=True,
            )
        except subprocess.CalledProcessError as exc:
            detail = str(exc.stdout or exc.output or "")
            if "not found" in detail.casefold() or "404" in detail:
                return None
            raise RuntimeError(
                f"Kaggle kernel status failed for {reference}: {detail}"
            ) from exc
        return parse_kernel_status(output)

    def inspect_cloud(self, model: str) -> RunResult:
        with TemporaryWorkspace(self.temp_root, command="status") as workspace:
            self.require_cloud_dependencies(model)
            corpus_manifest = self._corpus_manifest(workspace.incoming_dir / "corpus")
            checkpoint = self.checkpoints.inspect(
                model,
                workspace.incoming_dir / "checkpoint-status",
                corpus_sha256=str(corpus_manifest["sha256"]),
            )
            return dataclasses.replace(
                checkpoint,
                kernel_status=self._kernel_status_optional(model),
            )

    def recover_latest_kernel_output(
        self,
        model: str,
        workspace: TemporaryWorkspace,
        current: RunResult,
        corpus_sha256: str,
    ) -> RunResult | None:
        if self._kernel_status_optional(model) != "COMPLETE":
            return None
        reference = f"{self.owner}/{kernel_slug(model)}"
        manifest_dir = workspace.incoming_dir / "recovery-manifest"
        manifest_dir.mkdir()
        self.kernel_service.download_manifest(reference, manifest_dir)
        candidates = list(manifest_dir.rglob("checkpoint_manifest.json"))
        if len(candidates) != 1:
            return None
        candidate = load_json(candidates[0])
        spec = require_model(model)
        validate_checkpoint_manifest(
            candidate,
            model=model,
            corpus_sha256=corpus_sha256,
            vector_dimension=int(spec.vector_dimension),
        )
        if int(candidate["complete"]) <= current.complete:
            return None
        full_output = workspace.incoming_dir / "recovery-output"
        full_output.mkdir()
        self.kernel_service.download_checkpoint(reference, full_output)
        artifact = self.checkpoints.materialize_kernel_output(
            full_output,
            workspace.checkpoint_dir / "recovered",
            model=model,
            corpus_sha256=corpus_sha256,
        )
        previous = None if current.total == 0 else {"complete": current.complete}
        return self.checkpoints.publish(
            artifact,
            model=model,
            previous_manifest=previous,
        )

    def execute_one_run(
        self,
        model: str,
        workspace: TemporaryWorkspace,
        previous: RunResult,
        corpus_sha256: str,
        *,
        autotune_parallel: int,
        total_budget_seconds: int,
        export_reserve_seconds: int,
        run_index: int,
    ) -> RunResult:
        checkpoint_reference = self.checkpoints.reference(model)
        checkpoint_status = self.checkpoint_datasets.optional_status(
            checkpoint_reference
        )
        if checkpoint_status not in {None, "READY"}:
            raise RuntimeError(
                f"Checkpoint dataset is not READY: "
                f"{checkpoint_reference} ({checkpoint_status})"
            )
        checkpoint_dataset = (
            checkpoint_dataset_slug(model) if checkpoint_status == "READY" else None
        )
        bundle = self.kernel_service.prepare_kernel_bundle(
            workspace.kernel_dir,
            model=model,
            checkpoint_dataset=checkpoint_dataset,
            autotune_parallel=autotune_parallel,
            total_budget_seconds=total_budget_seconds,
            export_reserve_seconds=export_reserve_seconds,
        )
        self.kernel_service.push_kernel(bundle)
        kernel_reference = f"{self.owner}/{kernel_slug(model)}"
        self.kernel_service.poll_kernel(kernel_reference)
        incoming = workspace.incoming_dir / f"run-{run_index}"
        incoming.mkdir()
        self.kernel_service.download_checkpoint(kernel_reference, incoming)
        artifact = self.checkpoints.materialize_kernel_output(
            incoming,
            workspace.checkpoint_dir / f"run-{run_index}",
            model=model,
            corpus_sha256=corpus_sha256,
        )
        previous_manifest = (
            None if previous.total == 0 else {"complete": previous.complete}
        )
        return self.checkpoints.publish(
            artifact,
            model=model,
            previous_manifest=previous_manifest,
        )

    def run_until_complete(
        self,
        model: str,
        max_runs: int = 1,
        total_budget_seconds: int = 21_600,
        export_reserve_seconds: int = 900,
        retune: bool = False,
        **_kwargs,
    ) -> RunResult:
        del retune
        require_model(model)
        lock_path = (
            self.temp_root
            / "corpus-pipeline-kaggle-locks"
            / f"{model_slug(model)}.lock"
        )
        with (
            ModelRunLock(lock_path, model),
            TemporaryWorkspace(self.temp_root, command=f"run:{model}") as workspace,
        ):
            self.require_cloud_dependencies(model)
            corpus_manifest = self._corpus_manifest(workspace.incoming_dir / "corpus")
            corpus_sha256 = str(corpus_manifest["sha256"])
            current = self.checkpoints.inspect(
                model,
                workspace.incoming_dir / "checkpoint-status",
                corpus_sha256=corpus_sha256,
            )
            if current.is_complete:
                return current
            recovered = self.recover_latest_kernel_output(
                model,
                workspace,
                current,
                corpus_sha256,
            )
            if recovered is not None:
                current = recovered
                if current.is_complete:
                    return current
            self.check_quota(required_hours=1.0)
            for run_index in range(max_runs):
                result = self.execute_one_run(
                    model=model,
                    workspace=workspace,
                    previous=current,
                    corpus_sha256=corpus_sha256,
                    autotune_parallel=1,
                    total_budget_seconds=total_budget_seconds,
                    export_reserve_seconds=export_reserve_seconds,
                    run_index=run_index,
                )
                if result.complete <= current.complete:
                    raise RuntimeError(
                        f"Vector cache ingest for {model} made no progress: "
                        f"complete={result.complete}"
                    )
                current = result
                if current.is_complete:
                    return current
            return current

    def publish_corpus(self, corpus_path: Path | None = None) -> str:
        source = Path(corpus_path) if corpus_path is not None else self.corpus_path
        if not source.is_file():
            raise ValueError(f"Corpus file does not exist: {source}")
        with TemporaryWorkspace(self.temp_root, command="corpus-publish") as workspace:
            return self.dataset_service.publish_corpus(
                source,
                workspace.checkpoint_dir,
            )

    def publish_canonical_gguf(self, artifact: CanonicalModelArtifact) -> str:
        with TemporaryWorkspace(
            self.temp_root,
            command=f"model-publish:{artifact.model}",
        ) as workspace:
            return self.runtime_datasets.publish_canonical_gguf(
                artifact,
                workspace.checkpoint_dir,
                public=True,
            )

    def build_llama_cpp_runtime(self, *, force: bool = False) -> str:
        with TemporaryWorkspace(self.temp_root, command="runtime-build") as workspace:
            return self.runtime_service.build_llama_cpp_runtime(
                workspace.kernel_dir,
                workspace.incoming_dir,
                force=force,
            )

    def benchmark(self, model: str, total_budget_seconds: int = 600) -> RunResult:
        reserve = min(120, max(30, total_budget_seconds // 10))
        return self.run_until_complete(
            model,
            max_runs=1,
            total_budget_seconds=total_budget_seconds,
            export_reserve_seconds=reserve,
        )

    def download_checkpoint(
        self,
        model: str,
        output_dir: Path | None = None,
    ) -> tuple[Path, Path]:
        destination = Path(output_dir) if output_dir is not None else self.output_dir
        with TemporaryWorkspace(
            self.temp_root,
            command=f"download:{model}",
        ) as workspace:
            self.require_cloud_dependencies(model)
            corpus_manifest = self._corpus_manifest(workspace.incoming_dir / "corpus")
            return self.checkpoints.download(
                model,
                workspace.incoming_dir / "checkpoint",
                destination,
                corpus_sha256=str(corpus_manifest["sha256"]),
            )

    def poll_kernel(self, reference: str) -> None:
        self.kernel_service.poll_kernel(reference)

    def forward_kernel_logs(
        self,
        reference: str,
        cursor: int = 0,
        elapsed: float = 0.0,
    ) -> tuple[int, bool]:
        return self.kernel_service.forward_kernel_logs(
            reference,
            cursor=cursor,
            elapsed=elapsed,
        )
