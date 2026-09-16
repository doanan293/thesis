from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from pharma_lab.integrations.kaggle.artifacts import (
    ArtifactContractError,
    load_cloud_artifact,
    promote_complete_artifact,
)
from pharma_lab.integrations.kaggle.checkpoint_inheritance import (
    CheckpointInheritanceResult,
)
from pharma_lab.integrations.kaggle.checkpoints import CheckpointState
from pharma_lab.integrations.kaggle.kernel_reconciler import (
    KernelReconciler,
    KernelResolution,
)
from pharma_lab.integrations.kaggle.models import (
    ActionVerb,
    CloudArtifact,
    KernelPresence,
    KernelRemoteState,
    KernelStatus,
    PipelineResult,
    ReconcileAction,
    StageJob,
    StageRequest,
)
from pharma_lab.integrations.kaggle.stages import get_stage_adapter
from pharma_lab.integrations.kaggle.workspace import managed_staging_directory

RUNTIME_DATASET_SLUG = "vector-cache-llama-cpp-cuda-t4"
ArtifactSink = Callable[[StageJob, CloudArtifact], None]


def runtime_dataset_reference(owners) -> str:
    return f"{owners.runtime}/{RUNTIME_DATASET_SLUG}"


class KernelReconciliation(Protocol):
    """Kernel lifecycle operations the orchestrator depends on."""

    def reference(self, job: StageJob, /) -> str: ...

    def log_tail(self, reference: str, /) -> str: ...

    def inspect(self, job: StageJob, /) -> KernelRemoteState: ...

    def attach_or_recover(
        self,
        job: StageJob,
        remote: KernelRemoteState,
        staging_root: Path,
        /,
        *,
        timeout_seconds: int,
    ) -> KernelResolution: ...

    def submit(
        self,
        job: StageJob,
        bundle: Path,
        staging_root: Path,
        /,
        *,
        timeout_seconds: int,
    ) -> KernelResolution: ...


class CheckpointInheritanceResolver(Protocol):
    """Resolves the best checkpoint across the local source and account profiles."""

    def resolve(
        self,
        job: StageJob,
        target_state: CheckpointState,
        /,
        *,
        check_only: bool,
        include_profiles: bool = True,
    ) -> CheckpointInheritanceResult: ...


class KagglePipelineOrchestrator:
    def __init__(
        self,
        stages,
        dependencies,
        checkpoints,
        kernels,
        *,
        temp_root: Path = Path("/tmp"),
        reconciler: KernelReconciliation | None = None,
        checkpoint_inheritance: CheckpointInheritanceResolver | None = None,
        artifact_sink: ArtifactSink | None = None,
    ):
        self.stages = stages
        self.dependencies = dependencies
        self.checkpoints = checkpoints
        self.kernels = kernels
        self.reconciler: KernelReconciliation = reconciler or KernelReconciler(kernels)
        self.temp_root = Path(temp_root)
        self.checkpoint_inheritance = checkpoint_inheritance
        self.artifact_sink = artifact_sink

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
        benchmark = job.stage.value.endswith("-benchmark")
        local = self._inspect_local(job)
        if not request.force and local is not None and local.completion.is_complete:
            return PipelineResult(job, local.completion, (), local.data_path, 0)

        ignore_remote = request.force or not request.resume_remote
        checkpoint_actions: tuple[ReconcileAction, ...] = ()
        if benchmark or ignore_remote:
            checkpoint = self.checkpoints.empty(job)
        else:
            checkpoint = self.checkpoints.inspect(job)
        if not benchmark and self.checkpoint_inheritance is not None:
            inheritance = self.checkpoint_inheritance.resolve(
                job,
                checkpoint,
                check_only=request.check_only,
                include_profiles=not ignore_remote,
            )
            checkpoint = inheritance.state
            checkpoint_actions = inheritance.actions
        remote = (
            KernelRemoteState(self.reconciler.reference(job), KernelPresence.ABSENT)
            if ignore_remote
            else self.reconciler.inspect(job)
        )

        if request.check_only:
            return PipelineResult(
                job,
                checkpoint.completion,
                (*checkpoint_actions, *self._check_actions(job, remote, checkpoint)),
                None,
                0,
            )

        with managed_staging_directory(
            self.temp_root, prefix="kaggle-pipeline-"
        ) as workspace:
            actions: tuple[ReconcileAction, ...] = checkpoint_actions
            # A queued or running kernel is a session already in progress.
            attached_runs = int(
                remote.presence is KernelPresence.EXISTS
                and remote.status in {KernelStatus.QUEUED, KernelStatus.RUNNING}
            )

            if remote.presence is KernelPresence.EXISTS:
                resolution = self.reconciler.attach_or_recover(
                    job,
                    remote,
                    workspace,
                    timeout_seconds=request.total_budget_seconds,
                )
                actions += resolution.actions
                if resolution.output_root is not None:
                    artifact = None
                    try:
                        artifact = self._load_downloaded_artifact(
                            resolution.output_root, job
                        )
                    except ArtifactContractError as error:
                        if (
                            resolution.remote.status is not KernelStatus.ERROR
                            or resolution.submitted
                        ):
                            raise ArtifactContractError(
                                f"{error}; kernel={resolution.remote.reference}; "
                                f"log_tail={self.reconciler.log_tail(resolution.remote.reference)}"
                            ) from error
                    if artifact is not None:
                        if not benchmark:
                            self._deliver(job, artifact)
                        if artifact.completion.is_complete:
                            return self._complete_result(
                                job, artifact, actions, attempts=0
                            )
                        if not benchmark:
                            # A re-attached kernel may hold only pairs the checkpoint
                            # already has; that is not an error.
                            published = self.checkpoints.publish_if_better(
                                job, artifact, checkpoint
                            )
                            if (
                                published.completion.complete
                                > checkpoint.completion.complete
                            ):
                                checkpoint = published
                                actions += (self._checkpoint_action(checkpoint),)

            if (
                remote.presence is KernelPresence.ABSENT
                and checkpoint.completion.is_complete
                and checkpoint.artifact is not None
                and checkpoint.artifact.strict_identity_match
            ):
                if not benchmark:
                    self._deliver(job, checkpoint.artifact)
                destination = promote_complete_artifact(
                    checkpoint.artifact, job.local_cache_path
                )
                return PipelineResult(
                    job,
                    checkpoint.completion,
                    (*actions, self._finalize_action(destination, "checkpoint")),
                    destination,
                    0,
                )

            runs = request.max_runs - attached_runs
            if runs < 1:
                return PipelineResult(
                    job, checkpoint.completion, actions, None, attached_runs
                )
            return self._run_until_complete(
                job, request, workspace, actions, checkpoint, runs=runs
            )

    def _run_until_complete(
        self,
        job,
        request,
        workspace,
        actions: tuple[ReconcileAction, ...],
        checkpoint: CheckpointState,
        *,
        runs: int,
    ) -> PipelineResult:
        benchmark = job.stage.value.endswith("-benchmark")
        state = checkpoint
        for attempt in range(1, runs + 1):
            self.dependencies.require_ready(runtime_dataset_reference(request.owners))
            dependency_actions = tuple(
                self.dependencies.reconcile(
                    job,
                    request.owners,
                    workspace,
                    force=request.force,
                    check_only=request.check_only,
                )
            )
            dataset_references = [
                runtime_dataset_reference(request.owners),
                *(
                    action.reference
                    for action in dependency_actions
                    if action.resource_kind != "artifact"
                ),
            ]
            bundle = self.kernels.prepare_bundle(
                job,
                root=workspace / str(attempt),
                dataset_references=dataset_references,
                checkpoint_reference=None if benchmark else state.reference,
                total_budget_seconds=request.total_budget_seconds,
            )
            resolution = self.reconciler.submit(
                job,
                bundle,
                workspace / str(attempt),
                timeout_seconds=request.total_budget_seconds,
            )
            if resolution.output_root is None:
                return PipelineResult(
                    job,
                    state.completion,
                    (*actions, *dependency_actions, *resolution.actions),
                    None,
                    attempt,
                )
            try:
                artifact = self._load_downloaded_artifact(resolution.output_root, job)
            except ArtifactContractError as error:
                raise ArtifactContractError(
                    f"{error}; kernel={resolution.remote.reference}; "
                    f"log_tail={self.reconciler.log_tail(resolution.remote.reference)}"
                ) from error
            combined_actions = (*actions, *dependency_actions, *resolution.actions)
            if not benchmark:
                self._deliver(job, artifact)
            if artifact.completion.is_complete:
                return self._complete_result(
                    job, artifact, combined_actions, attempts=attempt
                )
            if benchmark:
                actions = combined_actions
            else:
                state = self._publish_checkpoint(job, artifact, state)
                actions = (*combined_actions, self._checkpoint_action(state))
        return PipelineResult(job, state.completion, actions, None, runs)

    def _deliver(self, job: StageJob, artifact: CloudArtifact) -> None:
        """Hand scores to the local sink before any checkpoint is published."""
        if self.artifact_sink is not None:
            self.artifact_sink(job, artifact)

    def _publish_checkpoint(
        self, job, artifact, current: CheckpointState
    ) -> CheckpointState:
        checkpoint = self.checkpoints.publish_if_better(job, artifact, current)
        if checkpoint.completion.complete <= current.completion.complete:
            raise RuntimeError(
                "checkpoint publication made no progress: "
                f"complete={current.completion.complete}/"
                f"{current.completion.total}"
            )
        return checkpoint

    @staticmethod
    def _complete_result(job, artifact, actions, *, attempts: int):
        destination = promote_complete_artifact(artifact, job.local_cache_path)
        return PipelineResult(
            job,
            artifact.completion,
            (
                *actions,
                KagglePipelineOrchestrator._finalize_action(destination, "output"),
            ),
            destination,
            attempts,
        )

    @staticmethod
    def _checkpoint_action(checkpoint: CheckpointState) -> ReconcileAction:
        return ReconcileAction(
            "checkpoint",
            checkpoint.reference or "",
            ActionVerb.CHECKPOINT,
            f"{checkpoint.completion.complete}/{checkpoint.completion.total}",
        )

    @staticmethod
    def _finalize_action(destination: Path, reason: str) -> ReconcileAction:
        return ReconcileAction(
            "artifact", str(destination), ActionVerb.FINALIZE, reason
        )

    @staticmethod
    def _check_actions(job, remote, checkpoint):
        if remote.presence is KernelPresence.EXISTS:
            return (
                ReconcileAction(
                    "kernel",
                    remote.reference,
                    ActionVerb.ATTACH,
                    str(remote.status),
                ),
            )
        if checkpoint.reference is not None:
            return (
                ReconcileAction(
                    "checkpoint",
                    checkpoint.reference,
                    ActionVerb.REUSE,
                    "checkpoint available",
                ),
            )
        return (
            ReconcileAction(
                "kernel",
                job.identity.sha256[:16],
                ActionVerb.SUBMIT,
                "kernel absent",
            ),
        )

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
    def _load_downloaded_artifact(root: Path | None, job):
        if root is None:
            raise ArtifactContractError("Kernel completed without downloadable output")
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
