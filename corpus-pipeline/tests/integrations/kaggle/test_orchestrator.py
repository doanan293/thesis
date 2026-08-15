import json
from pathlib import Path
from types import SimpleNamespace

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
from corpus_pipeline.integrations.kaggle.stages import RerankStage
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
        self.empty = CheckpointState(None, None, Completion(total, 0, total))
        self.published = []

    def inspect(self, _job):
        return self.empty

    def publish_if_better(self, _job, artifact, current):
        self.published.append(artifact.completion)
        return CheckpointState("owner/checkpoint", artifact, artifact.completion)

    def reference(self, _job):
        return "owner/checkpoint"


class FakeKernels:
    def __init__(self):
        self.prepared = []

    def prepare_bundle(self, _job, *, root, **_kwargs):
        bundle = Path(root) / "bundle"
        bundle.mkdir(parents=True, exist_ok=True)
        self.prepared.append(bundle)
        return bundle


class FakeReconciler:
    def __init__(self, resolution):
        self.remote = resolution.remote
        self.resolution = resolution
        self.attached = []
        self.submissions = []

    def inspect(self, _job):
        return self.remote

    def attach_or_recover(self, job, remote, staging_root, *, timeout_seconds):
        self.attached.append(remote)
        return self.resolution

    def submit(self, job, bundle, staging_root, *, timeout_seconds):
        self.submissions.append((job, bundle, staging_root, timeout_seconds))
        return self.resolution


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
