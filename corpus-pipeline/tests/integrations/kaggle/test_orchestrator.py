import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from corpus_pipeline.artifacts.manifest import Completion
from corpus_pipeline.integrations.kaggle.checkpoints import CheckpointState
from corpus_pipeline.integrations.kaggle.kernel_reconciler import KernelResolution
from corpus_pipeline.integrations.kaggle.models import (
    ActionVerb,
    KernelPresence,
    KernelRemoteState,
    KernelStatus,
    StageName,
    StageRequest,
)
from corpus_pipeline.integrations.kaggle.orchestrator import (
    KagglePipelineOrchestrator,
)
from corpus_pipeline.integrations.kaggle.stages import (
    RerankStage,
    get_stage_adapter,
)
from corpus_pipeline.integrations.kaggle.workers.runtime import artifact_from_output
from corpus_pipeline.runtime.catalog import require_model


class FakeDependencies:
    def __init__(self):
        self.required = 0
        self.reconciles = 0

    def require_ready(self, _reference):
        self.required += 1

    def reconcile(self, *_args, **_kwargs):
        self.reconciles += 1
        return []


class FakeCheckpoints:
    def __init__(self, total):
        self.empty_state = CheckpointState(None, None, Completion(total, 0, total))
        self.inspected = []
        self.emptied = []
        self.published = []

    def empty(self, job):
        self.emptied.append(job)
        return self.empty_state

    def inspect(self, job):
        self.inspected.append(job)
        return self.empty_state

    def publish_if_better(self, _job, artifact, current):
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

    def resolve(self, job, state, *, check_only):
        self.calls.append((job, state, check_only))
        return SimpleNamespace(state=self.state, actions=self.actions)


class FakeReconciler:
    def __init__(self, resolution, submission_resolution=None):
        self.remote = resolution.remote
        self.resolution = resolution
        self.submission_resolution = submission_resolution or resolution
        self.attached = []
        self.submissions = []

    def inspect(self, _job):
        return self.remote

    def attach_or_recover(self, job, remote, staging_root, *, timeout_seconds):
        self.attached.append(remote)
        return self.resolution

    def submit(self, job, bundle, staging_root, *, timeout_seconds):
        self.submissions.append((job, bundle, staging_root, timeout_seconds))
        return self.submission_resolution


def _request(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "query": "query",
                "candidates": [{"chunk_id": "c1", "document_text": "text"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "manifest.json").write_text("{}\n", encoding="utf-8")
    return StageRequest(
        stage=StageName.RERANK,
        model="qwen3-reranker:0.6b-fp16",
        input_path=candidates,
        output_dir=tmp_path / "remote",
        gguf_root=tmp_path / "gguf",
        owners=SimpleNamespace(
            execution="owner", runtime="owner", corpus="owner", checkpoint="owner"
        ),
        max_runs=1,
        total_budget_seconds=60,
        runtime_profile=require_model(
            "qwen3-reranker:0.6b-fp16"
        ).rerank_search_space.candidates[0],
    )


def test_running_remote_is_attached_before_dependency_reconciliation(tmp_path):
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
        (
            SimpleNamespace(
                resource_kind="kernel",
                reference=remote.reference,
                verb=ActionVerb.ATTACH,
                reason="running",
            ),
        ),
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
    orchestrator._load_downloaded_artifact = lambda _root, _job: artifact

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
    artifact = SimpleNamespace(completion=Completion(2, 1, 1))

    with pytest.raises(RuntimeError, match="made no progress"):
        orchestrator._publish_checkpoint(None, artifact, current)


def test_production_resolves_inheritance_before_kernel_reconciliation(tmp_path):
    request = _request(tmp_path)
    remote = KernelRemoteState("owner/kernel", KernelPresence.ABSENT)
    resolution = KernelResolution(remote, None, ())
    events = []
    checkpoints = FakeCheckpoints(1)
    inherited = CheckpointState("owner/checkpoint", None, Completion(1, 0, 1))
    inheritance = FakeInheritance(
        inherited,
        (
            SimpleNamespace(
                resource_kind="checkpoint",
                reference="owner/checkpoint",
                verb=ActionVerb.SYNC,
                reason="inherited",
            ),
        ),
    )
    reconciler = FakeReconciler(resolution)
    original_inspect = reconciler.inspect
    reconciler.inspect = lambda job: (
        events.append("inspect-kernel") or original_inspect(job)
    )
    original_submit = reconciler.submit
    reconciler.submit = lambda *args, **kwargs: (
        events.append("submit") or original_submit(*args, **kwargs)
    )
    original_resolve = inheritance.resolve
    inheritance.resolve = lambda *args, **kwargs: (
        events.append("inherit") or original_resolve(*args, **kwargs)
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
    action = SimpleNamespace(
        resource_kind="checkpoint",
        reference="owner/checkpoint",
        verb=ActionVerb.SYNC,
        reason="would inherit acc1 -> acc2",
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
        def resolve(self, *_args, **_kwargs):
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
