from pathlib import Path

from corpus_pipeline.integrations.kaggle.checkpoints import CheckpointState
from corpus_pipeline.integrations.kaggle.config import OwnerConfiguration
from corpus_pipeline.integrations.kaggle.models import (
    Completion,
    JobIdentity,
    StageJob,
    StageName,
    StageRequest,
)
from corpus_pipeline.integrations.kaggle.orchestrator import KagglePipelineOrchestrator
from corpus_pipeline.integrations.kaggle.workers.runtime import write_artifact_manifest

OWNERS = OwnerConfiguration("owner", "runtime", "corpus", "checkpoint")


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
        tmp_path / "input",
        tmp_path / "output",
        tmp_path / "cache.jsonl",
        "query_embeddings.jsonl",
        1,
        "worker",
        {"identity": identity.payload, "job_sha256": identity.sha256},
    )


class FakeStage:
    def __init__(self, job):
        self.job = job

    def build_job(self, request):
        del request
        return self.job


class FakeDependencies:
    def __init__(self, actions):
        self.actions = actions
        self.calls = 0

    def reconcile(self, *args, **kwargs):
        del args, kwargs
        self.calls += 1
        return self.actions


class FakeCheckpoints:
    def __init__(self, state):
        self.state = state
        self.published = 0

    def inspect(self, job):
        del job
        return self.state

    def publish(self, job, artifact):
        del job, artifact
        self.published += 1
        return "owner/checkpoint"


class FakeKernels:
    def __init__(self):
        self.runs = 0

    def prepare_bundle(self, *args, **kwargs):
        del args, kwargs
        return Path("/tmp/bundle")

    def run(self, *args, **kwargs):
        del args, kwargs
        self.runs += 1
        raise AssertionError("kernel should not run in this test")


def request(tmp_path: Path, *, check_only: bool = False) -> StageRequest:
    return StageRequest(
        StageName.QUERY_EMBED,
        "fake",
        tmp_path / "input",
        tmp_path / "output",
        tmp_path / "gguf",
        OWNERS,
        check_only=check_only,
    )


def test_check_only_reports_plan_without_mutation(tmp_path: Path):
    job = make_job(tmp_path)
    dependencies = FakeDependencies([])
    kernels = FakeKernels()
    checkpoints = FakeCheckpoints(CheckpointState(None, None, Completion(1, 0, 1)))
    orchestrator = KagglePipelineOrchestrator(
        FakeStage(job), dependencies, checkpoints, kernels
    )
    result = orchestrator.run(request(tmp_path, check_only=True))
    assert result.completion.missing == 1
    assert dependencies.calls == 1
    assert kernels.runs == 0


def test_complete_local_artifact_skips_dependencies_and_kernel(tmp_path: Path):
    job = make_job(tmp_path)
    job.local_cache_path.parent.mkdir(parents=True, exist_ok=True)
    job.local_cache_path.write_text(
        '{"query_id":"q1","embedding":[0.1]}\n', encoding="utf-8"
    )
    write_artifact_manifest(
        job.local_cache_path,
        artifact_type="query_embedding_cache",
        identity=job.identity,
        completion=Completion(1, 1, 0),
    )
    dependencies = FakeDependencies([])
    kernels = FakeKernels()
    checkpoints = FakeCheckpoints(CheckpointState(None, None, Completion(1, 0, 1)))
    result = KagglePipelineOrchestrator(
        FakeStage(job), dependencies, checkpoints, kernels
    ).run(request(tmp_path))
    assert result.completion.is_complete
    assert result.run_count == 0
    assert dependencies.calls == 0
    assert kernels.runs == 0
