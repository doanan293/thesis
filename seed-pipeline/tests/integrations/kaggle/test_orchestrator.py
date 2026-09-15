import json
from dataclasses import replace
from pathlib import Path

import pytest
from tests.integrations.kaggle.factories import (
    cloud_artifact,
    owners,
    rerank_runtime_profile,
    stage_job,
    stage_request,
)

from seed_pipeline.artifacts.manifest import Completion
from seed_pipeline.integrations.kaggle.checkpoint_inheritance import (
    CheckpointInheritanceResult,
)
from seed_pipeline.integrations.kaggle.checkpoints import CheckpointState
from seed_pipeline.integrations.kaggle.kernel_reconciler import KernelResolution
from seed_pipeline.integrations.kaggle.models import (
    ActionVerb,
    KernelPresence,
    KernelRemoteState,
    KernelStatus,
    ReconcileAction,
    StageJob,
    StageName,
)
from seed_pipeline.integrations.kaggle.orchestrator import (
    KagglePipelineOrchestrator,
)
from seed_pipeline.integrations.kaggle.stages import (
    RerankStage,
    get_stage_adapter,
)
from seed_pipeline.integrations.kaggle.workers.runtime import artifact_from_output


class FakeDependencies:
    def __init__(self):
        self.required = 0
        self.reconciles = 0
        self.forced: list[bool] = []

    def require_ready(self, _reference):
        self.required += 1

    def reconcile(self, *_args, **kwargs):
        self.reconciles += 1
        self.forced.append(kwargs["force"])
        return []


class FakeCheckpoints:
    def __init__(self, total, state=None):
        self.empty_state = CheckpointState(None, None, Completion(total, 0, total))
        self.current = state or self.empty_state
        self.inspected = []
        self.emptied = []
        self.published = []

    def empty(self, job):
        self.emptied.append(job)
        return self.empty_state

    def inspect(self, job):
        self.inspected.append(job)
        return self.current

    def publish_if_better(self, _job, artifact, current):
        if artifact.completion.complete <= current.completion.complete:
            return current
        self.published.append(artifact.completion)
        return CheckpointState("owner/checkpoint", artifact, artifact.completion)

    def reference(self, _job):
        return "owner/checkpoint"


class FakeKernels:
    def __init__(self):
        self.prepared = []
        self.checkpoint_references = []

    def prepare_bundle(self, _job, *, root, **_kwargs):
        bundle = Path(root) / "bundle"
        bundle.mkdir(parents=True, exist_ok=True)
        self.prepared.append(bundle)
        self.checkpoint_references.append(_kwargs.get("checkpoint_reference"))
        return bundle


class FakeInheritance:
    def __init__(self, state, actions=()):
        self.state = state
        self.actions = tuple(actions)
        self.calls = []

    def resolve(
        self,
        job: StageJob,
        state: CheckpointState,
        *,
        check_only: bool,
        include_profiles: bool = True,
    ) -> CheckpointInheritanceResult:
        self.calls.append((job, state, check_only, include_profiles))
        return CheckpointInheritanceResult(self.state, self.actions)


class FakeReconciler:
    def __init__(self, resolution, submission_resolution=None):
        self.remote = resolution.remote
        self.resolution = resolution
        self.submission_resolution = submission_resolution or resolution
        self.attached = []
        self.submissions = []

    def reference(self, job: StageJob) -> str:
        return self.remote.reference

    def log_tail(self, reference: str) -> str:
        return f"log tail for {reference}"

    def inspect(self, job: StageJob) -> KernelRemoteState:
        return self.remote

    def attach_or_recover(
        self,
        job: StageJob,
        remote: KernelRemoteState,
        staging_root: Path,
        *,
        timeout_seconds: int,
    ) -> KernelResolution:
        self.attached.append(remote)
        return self.resolution

    def submit(
        self,
        job: StageJob,
        bundle: Path,
        staging_root: Path,
        *,
        timeout_seconds: int,
    ) -> KernelResolution:
        self.submissions.append((job, bundle, staging_root, timeout_seconds))
        return self.submission_resolution


def _request(tmp_path, *, candidates: int = 1):
    candidates_path = tmp_path / "candidates.jsonl"
    candidates_path.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "query": "query",
                "candidates": [
                    {"chunk_id": f"c{index}", "document_text": "text"}
                    for index in range(1, candidates + 1)
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "manifest.json").write_text("{}\n", encoding="utf-8")
    return stage_request(
        StageName.RERANK,
        "qwen3-reranker:0.6b-fp16",
        candidates_path,
        output_dir=tmp_path / "remote",
        owner_configuration=owners(),
        max_runs=1,
        total_budget_seconds=60,
        runtime_profile=rerank_runtime_profile(),
    )


def test_running_remote_is_attached_before_dependency_reconciliation(
    tmp_path, monkeypatch
):
    request = _request(tmp_path)
    job = RerankStage().build_job(request)
    data = tmp_path / "remote-output" / "rerank_scores.jsonl"
    data.parent.mkdir()
    data.write_text(json.dumps({"pair": "score"}) + "\n", encoding="utf-8")
    artifact = artifact_from_output(
        data,
        artifact_type="rerank_scores",
        identity=job.identity,
        total=1,
        complete=1,
    )
    remote = KernelRemoteState(
        "owner/rerank-legacy", KernelPresence.EXISTS, KernelStatus.RUNNING
    )
    resolution = KernelResolution(
        remote,
        data.parent,
        (ReconcileAction("kernel", remote.reference, ActionVerb.ATTACH, "running"),),
    )
    dependencies = FakeDependencies()
    reconciler = FakeReconciler(resolution)
    orchestrator = KagglePipelineOrchestrator(
        RerankStage(),
        dependencies,
        FakeCheckpoints(1),
        FakeKernels(),
        reconciler=reconciler,
    )
    monkeypatch.setattr(
        orchestrator, "_load_downloaded_artifact", lambda root, job: artifact
    )

    result = orchestrator.run(request)

    assert reconciler.attached == [remote]
    assert reconciler.submissions == []
    assert dependencies.reconciles == 0
    assert result.completion.is_complete


def test_failed_remote_with_invalid_output_starts_fresh_attempt(tmp_path):
    request = _request(tmp_path)
    failed_output = tmp_path / "failed-output"
    failed_output.mkdir()
    failed_remote = KernelRemoteState(
        "secondary-owner/rerank-failed",
        KernelPresence.EXISTS,
        KernelStatus.ERROR,
    )
    failed_resolution = KernelResolution(failed_remote, failed_output, ())
    queued_remote = KernelRemoteState(
        "secondary-owner/rerank-retry",
        KernelPresence.EXISTS,
        KernelStatus.QUEUED,
    )
    detached_retry = KernelResolution(queued_remote, None, (), submitted=True)
    orchestrator = KagglePipelineOrchestrator(
        RerankStage(),
        FakeDependencies(),
        FakeCheckpoints(1),
        FakeKernels(),
        reconciler=FakeReconciler(failed_resolution, detached_retry),
    )

    result = orchestrator.run(request)

    assert result.run_count == 1
    assert result.artifact_path is None


def test_benchmark_does_not_inspect_or_publish_checkpoint(tmp_path):
    request = replace(
        _request(tmp_path),
        stage=StageName.RERANK_BENCHMARK,
        check_only=True,
    )
    remote = KernelRemoteState(
        "owner/rerank-benchmark-12345678",
        KernelPresence.ABSENT,
    )
    resolution = KernelResolution(remote, None, ())
    checkpoints = FakeCheckpoints(1)
    orchestrator = KagglePipelineOrchestrator(
        {StageName.RERANK_BENCHMARK: get_stage_adapter(StageName.RERANK_BENCHMARK)},
        FakeDependencies(),
        checkpoints,
        FakeKernels(),
        reconciler=FakeReconciler(resolution),
    )

    orchestrator.run(request)

    assert len(checkpoints.emptied) == 1
    assert checkpoints.inspected == []
    assert checkpoints.published == []


def test_checkpoint_publication_requires_strict_progress(tmp_path):
    checkpoints = FakeCheckpoints(2)
    orchestrator = KagglePipelineOrchestrator(
        RerankStage(), FakeDependencies(), checkpoints, FakeKernels()
    )
    current = CheckpointState(None, None, Completion(2, 1, 1))
    artifact = cloud_artifact(Completion(2, 1, 1))

    with pytest.raises(RuntimeError, match="made no progress"):
        orchestrator._publish_checkpoint(stage_job(tmp_path), artifact, current)


def test_production_resolves_inheritance_before_kernel_reconciliation(
    tmp_path, monkeypatch
):
    request = _request(tmp_path)
    remote = KernelRemoteState("owner/kernel", KernelPresence.ABSENT)
    resolution = KernelResolution(remote, None, ())
    events = []
    checkpoints = FakeCheckpoints(1)
    inherited = CheckpointState("owner/checkpoint", None, Completion(1, 0, 1))
    inheritance = FakeInheritance(
        inherited,
        (
            ReconcileAction(
                "checkpoint", "owner/checkpoint", ActionVerb.SYNC, "inherited"
            ),
        ),
    )
    reconciler = FakeReconciler(resolution)
    original_inspect = reconciler.inspect
    monkeypatch.setattr(
        reconciler,
        "inspect",
        lambda job: events.append("inspect-kernel") or original_inspect(job),
    )
    original_submit = reconciler.submit
    monkeypatch.setattr(
        reconciler,
        "submit",
        lambda *args, **kwargs: (
            events.append("submit") or original_submit(*args, **kwargs)
        ),
    )
    original_resolve = inheritance.resolve
    monkeypatch.setattr(
        inheritance,
        "resolve",
        lambda *args, **kwargs: (
            events.append("inherit") or original_resolve(*args, **kwargs)
        ),
    )
    orchestrator = KagglePipelineOrchestrator(
        RerankStage(),
        FakeDependencies(),
        checkpoints,
        FakeKernels(),
        reconciler=reconciler,
        checkpoint_inheritance=inheritance,
    )

    result = orchestrator.run(request)

    assert events == ["inherit", "inspect-kernel", "submit"]
    assert result.actions[0].verb is ActionVerb.SYNC


def test_check_only_includes_planned_inheritance_action_without_submit(tmp_path):
    request = replace(_request(tmp_path), check_only=True)
    remote = KernelRemoteState("owner/kernel", KernelPresence.ABSENT)
    resolution = KernelResolution(remote, None, ())
    action = ReconcileAction(
        "checkpoint", "owner/checkpoint", ActionVerb.SYNC, "would inherit acc1 -> acc2"
    )
    inheritance = FakeInheritance(
        CheckpointState("owner/checkpoint", None, Completion(1, 0, 1)), (action,)
    )
    reconciler = FakeReconciler(resolution)
    orchestrator = KagglePipelineOrchestrator(
        RerankStage(),
        FakeDependencies(),
        FakeCheckpoints(1),
        FakeKernels(),
        reconciler=reconciler,
        checkpoint_inheritance=inheritance,
    )

    result = orchestrator.run(request)

    assert result.actions[0] is action
    assert reconciler.submissions == []
    assert inheritance.calls[0][2] is True


def test_inheritance_failure_prevents_kernel_submission(tmp_path):
    request = _request(tmp_path)

    class BrokenInheritance:
        def resolve(
            self,
            job: StageJob,
            target_state: CheckpointState,
            *,
            check_only: bool,
            include_profiles: bool = True,
        ) -> CheckpointInheritanceResult:
            raise RuntimeError("inheritance failed")

    remote = KernelRemoteState("owner/kernel", KernelPresence.ABSENT)
    resolution = KernelResolution(remote, None, ())
    reconciler = FakeReconciler(resolution)
    orchestrator = KagglePipelineOrchestrator(
        RerankStage(),
        FakeDependencies(),
        FakeCheckpoints(1),
        FakeKernels(),
        reconciler=reconciler,
        checkpoint_inheritance=BrokenInheritance(),
    )

    with pytest.raises(RuntimeError, match="inheritance failed"):
        orchestrator.run(request)
    assert reconciler.submissions == []


def test_inherited_target_checkpoint_is_mounted_before_submit(tmp_path):
    request = _request(tmp_path)
    remote = KernelRemoteState("secondary-user/kernel", KernelPresence.ABSENT)
    resolution = KernelResolution(remote, None, ())
    kernels = FakeKernels()
    target_state = CheckpointState(
        "secondary-user/checkpoint", None, Completion(1, 0, 1)
    )
    inheritance = FakeInheritance(target_state)
    reconciler = FakeReconciler(resolution)
    orchestrator = KagglePipelineOrchestrator(
        RerankStage(),
        FakeDependencies(),
        FakeCheckpoints(1),
        kernels,
        reconciler=reconciler,
        checkpoint_inheritance=inheritance,
    )

    orchestrator.run(request)

    assert kernels.checkpoint_references == ["secondary-user/checkpoint"]


def _artifact(tmp_path, job, *, complete, total, name="output"):
    data = tmp_path / name / "rerank_scores.jsonl"
    data.parent.mkdir(parents=True)
    data.write_text(
        "".join(json.dumps({"pair": index}) + "\n" for index in range(complete)),
        encoding="utf-8",
    )
    return artifact_from_output(
        data,
        artifact_type="rerank_scores",
        identity=job.identity,
        total=total,
        complete=complete,
    )


def _detached_submission():
    queued = KernelRemoteState(
        "owner/kernel", KernelPresence.EXISTS, KernelStatus.QUEUED
    )
    return KernelResolution(queued, None, (), submitted=True)


def test_reattached_output_without_new_pairs_does_not_fail(tmp_path, monkeypatch):
    request = _request(tmp_path, candidates=2)
    job = RerankStage().build_job(request)
    artifact = _artifact(tmp_path, job, complete=1, total=2)
    checkpoints = FakeCheckpoints(
        2, state=CheckpointState("owner/checkpoint", artifact, Completion(2, 1, 1))
    )
    finished = KernelRemoteState(
        "owner/kernel", KernelPresence.EXISTS, KernelStatus.COMPLETE
    )
    reconciler = FakeReconciler(
        KernelResolution(finished, artifact.data_path.parent, ()),
        _detached_submission(),
    )
    delivered = []
    orchestrator = KagglePipelineOrchestrator(
        RerankStage(),
        FakeDependencies(),
        checkpoints,
        FakeKernels(),
        reconciler=reconciler,
        artifact_sink=lambda _job, item: delivered.append(item.completion),
    )
    monkeypatch.setattr(
        orchestrator, "_load_downloaded_artifact", lambda root, job: artifact
    )

    result = orchestrator.run(request)

    assert checkpoints.published == []
    assert delivered == [Completion(2, 1, 1)]
    assert len(reconciler.submissions) == 1
    assert result.completion == Completion(2, 1, 1)


def test_session_output_reaches_the_sink_before_checkpoint_publication(
    tmp_path, monkeypatch
):
    request = _request(tmp_path, candidates=2)
    job = RerankStage().build_job(request)
    artifact = _artifact(tmp_path, job, complete=1, total=2)
    events: list[str] = []

    class OrderedCheckpoints(FakeCheckpoints):
        def publish_if_better(self, _job, artifact, current):
            events.append("publish")
            return super().publish_if_better(_job, artifact, current)

    absent = KernelRemoteState("owner/kernel", KernelPresence.ABSENT)
    done = KernelRemoteState(
        "owner/kernel", KernelPresence.EXISTS, KernelStatus.COMPLETE
    )
    reconciler = FakeReconciler(
        KernelResolution(absent, None, ()),
        KernelResolution(done, artifact.data_path.parent, (), submitted=True),
    )
    orchestrator = KagglePipelineOrchestrator(
        RerankStage(),
        FakeDependencies(),
        OrderedCheckpoints(2),
        FakeKernels(),
        reconciler=reconciler,
        artifact_sink=lambda _job, _artifact: events.append("sink"),
    )
    monkeypatch.setattr(
        orchestrator, "_load_downloaded_artifact", lambda root, job: artifact
    )

    result = orchestrator.run(request)

    assert events == ["sink", "publish"]
    assert result.completion == Completion(2, 1, 1)


def test_attaching_a_running_kernel_uses_up_a_one_run_request(tmp_path, monkeypatch):
    request = _request(tmp_path, candidates=2)
    job = RerankStage().build_job(request)
    artifact = _artifact(tmp_path, job, complete=1, total=2)
    running = KernelRemoteState(
        "owner/kernel", KernelPresence.EXISTS, KernelStatus.RUNNING
    )
    reconciler = FakeReconciler(
        KernelResolution(running, artifact.data_path.parent, ()),
        _detached_submission(),
    )
    checkpoints = FakeCheckpoints(2)
    dependencies = FakeDependencies()
    orchestrator = KagglePipelineOrchestrator(
        RerankStage(), dependencies, checkpoints, FakeKernels(), reconciler=reconciler
    )
    monkeypatch.setattr(
        orchestrator, "_load_downloaded_artifact", lambda root, job: artifact
    )

    result = orchestrator.run(request)

    assert reconciler.submissions == []
    assert dependencies.reconciles == 0
    assert checkpoints.published == [Completion(2, 1, 1)]
    assert result.run_count == 1
    assert result.completion == Completion(2, 1, 1)


def test_run_without_remote_resume_ignores_account_state(tmp_path):
    request = replace(_request(tmp_path), resume_remote=False)
    checkpoints = FakeCheckpoints(1)
    inheritance = FakeInheritance(CheckpointState(None, None, Completion(1, 0, 1)))
    finished = KernelRemoteState(
        "owner/kernel", KernelPresence.EXISTS, KernelStatus.COMPLETE
    )
    reconciler = FakeReconciler(
        KernelResolution(finished, None, ()), _detached_submission()
    )
    dependencies = FakeDependencies()
    orchestrator = KagglePipelineOrchestrator(
        RerankStage(),
        dependencies,
        checkpoints,
        FakeKernels(),
        reconciler=reconciler,
        checkpoint_inheritance=inheritance,
    )

    orchestrator.run(request)

    assert checkpoints.inspected == []
    assert inheritance.calls[0][3] is False
    assert reconciler.attached == []
    assert len(reconciler.submissions) == 1
    assert dependencies.forced == [False]
