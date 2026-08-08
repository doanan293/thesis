from __future__ import annotations

import tempfile
from pathlib import Path

from corpus_pipeline.integrations.kaggle.artifacts import (
    ArtifactContractError,
    load_cloud_artifact,
    promote_complete_artifact,
)
from corpus_pipeline.integrations.kaggle.checkpoints import CheckpointState
from corpus_pipeline.integrations.kaggle.models import (
    ActionVerb,
    PipelineResult,
    ReconcileAction,
    StageRequest,
)
from corpus_pipeline.integrations.kaggle.stages import get_stage_adapter

RUNTIME_DATASET_SLUG = "vector-cache-llama-cpp-cuda-t4"


def runtime_dataset_reference(owners) -> str:
    return f"{owners.runtime}/{RUNTIME_DATASET_SLUG}"


class KagglePipelineOrchestrator:
    def __init__(
        self,
        stages,
        dependencies,
        checkpoints,
        kernels,
        *,
        temp_root: Path = Path("/tmp"),
    ):
        self.stages = stages
        self.dependencies = dependencies
        self.checkpoints = checkpoints
        self.kernels = kernels
        self.temp_root = Path(temp_root)

    def _adapter(self, stage):
        if hasattr(self.stages, "build_job"):
            return self.stages
        if hasattr(self.stages, "get"):
            return self.stages.get(stage)
        return get_stage_adapter(stage)

    def run(self, request: StageRequest) -> PipelineResult:
        if request.max_runs < 1:
            raise ValueError("max_runs must be at least 1")
        job = self._adapter(request.stage).build_job(request)
        self.dependencies.require_ready(runtime_dataset_reference(request.owners))
        local = self._inspect_local(job)
        if local is not None and local.completion.is_complete:
            return PipelineResult(job, local.completion, (), local.data_path, 0)
        with tempfile.TemporaryDirectory(
            prefix="kaggle-pipeline-", dir=str(self.temp_root)
        ) as raw:
            workspace = Path(raw)
            actions = tuple(
                self.dependencies.reconcile(
                    job,
                    request.owners,
                    workspace,
                    force=request.force,
                    check_only=request.check_only,
                )
            )
            checkpoint = self.checkpoints.inspect(job)
            if request.check_only:
                return PipelineResult(job, checkpoint.completion, actions, None, 0)
            if (
                checkpoint.completion.is_complete
                and checkpoint.artifact is not None
                and checkpoint.artifact.strict_identity_match
            ):
                destination = promote_complete_artifact(
                    checkpoint.artifact, job.local_cache_path
                )
                return PipelineResult(
                    job,
                    checkpoint.completion,
                    (
                        *actions,
                        ReconcileAction(
                            "artifact",
                            checkpoint.reference or "",
                            ActionVerb.SYNC,
                            "complete checkpoint",
                        ),
                    ),
                    destination,
                    0,
                )
            return self._run_until_complete(
                job, request, workspace, actions, checkpoint
            )

    def _run_until_complete(
        self, job, request, workspace, actions, checkpoint: CheckpointState
    ) -> PipelineResult:
        state = checkpoint
        for attempt in range(1, request.max_runs + 1):
            checkpoint_reference = state.reference
            dataset_references = [
                runtime_dataset_reference(request.owners),
                *(
                    action.reference
                    for action in actions
                    if action.resource_kind != "artifact"
                ),
            ]
            bundle = self.kernels.prepare_bundle(
                job,
                root=workspace / str(attempt),
                dataset_references=dataset_references,
                checkpoint_reference=checkpoint_reference,
                total_budget_seconds=request.total_budget_seconds,
            )
            output_root = self.kernels.run(
                job, bundle, timeout_seconds=request.total_budget_seconds
            )
            artifact = self._load_downloaded_artifact(output_root, job)
            if artifact.completion.is_complete:
                destination = promote_complete_artifact(artifact, job.local_cache_path)
                return PipelineResult(
                    job,
                    artifact.completion,
                    (
                        *actions,
                        ReconcileAction(
                            "kernel",
                            self.kernels.reference(job),
                            ActionVerb.SUBMIT,
                            f"attempt {attempt}",
                        ),
                        ReconcileAction(
                            "artifact",
                            str(destination),
                            ActionVerb.SYNC,
                            "complete output",
                        ),
                    ),
                    destination,
                    attempt,
                )
            self.checkpoints.publish(job, artifact)
            state = CheckpointState(
                self.checkpoints.reference(job), artifact, artifact.completion
            )
        return PipelineResult(job, state.completion, actions, None, request.max_runs)

    @staticmethod
    def _inspect_local(job):
        if not job.local_cache_path.is_file():
            return None
        manifest = job.local_cache_path.with_name("manifest.json")
        if not manifest.is_file():
            return None
        try:
            return load_cloud_artifact(job.local_cache_path, manifest, job.identity)
        except ArtifactContractError:
            return None

    @staticmethod
    def _load_downloaded_artifact(root: Path, job):
        root = Path(root)
        manifests = list(root.rglob("manifest.json"))
        if len(manifests) != 1:
            raise ArtifactContractError(
                f"Expected one downloaded artifact manifest, found {len(manifests)}"
            )
        manifest = manifests[0]
        data = root / job.data_filename
        if not data.is_file():
            matches = list(root.rglob(job.data_filename))
            if len(matches) != 1:
                raise ArtifactContractError(
                    f"Expected one downloaded artifact data file, found {len(matches)}"
                )
            data = matches[0]
        return load_cloud_artifact(data, manifest, job.identity, allow_partial=True)
