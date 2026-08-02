import hashlib
from pathlib import Path

from corpus_pipeline.integrations.kaggle.checkpoints import CheckpointService
from corpus_pipeline.integrations.kaggle.dataset_service import (
    DatasetPresence,
    DatasetRemoteState,
)
from corpus_pipeline.integrations.kaggle.models import JobIdentity, StageJob, StageName


class FakeDatasets:
    owner = "checkpoint"

    def __init__(self, state):
        self.state = state

    def inspect_state(self, reference, *, active_owner=None):
        del reference, active_owner
        return self.state

    def fetch_json(self, reference, filename):
        del reference, filename
        data = b'{"id":"q1"}\n{"id":"q2"}\n'
        return {
            "schema_version": 1,
            "artifact_type": "query-embeddings",
            "created_at": "2026-01-01T00:00:00Z",
            "identity": self.identity,
            "data_filename": "query_embeddings.jsonl",
            "data_sha256": hashlib.sha256(data).hexdigest(),
            "record_count": 2,
            "total": 2,
            "complete": 2,
            "missing": 0,
        }

    def download_file(self, reference, filename, destination):
        del reference
        path = Path(destination) / filename
        path.write_text('{"id":"q1"}\n{"id":"q2"}\n', encoding="utf-8")


def make_job(tmp_path: Path, digest: str) -> StageJob:
    identity = JobIdentity.create(
        stage=StageName.QUERY_EMBED,
        contract_version=1,
        model="fake",
        model_sha256="a" * 64,
        input_sha256=digest,
        runtime_parameters={},
    )
    return StageJob(
        StageName.QUERY_EMBED,
        1,
        "fake",
        identity,
        tmp_path / "input",
        tmp_path / "output",
        tmp_path / "output" / "query_embeddings.jsonl",
        "query_embeddings.jsonl",
        2,
        "worker",
        {},
    )


def test_checkpoint_reference_is_scoped_by_job_identity(tmp_path: Path):
    service = CheckpointService(
        FakeDatasets(DatasetRemoteState(DatasetPresence.ABSENT)), "checkpoint"
    )
    assert service.reference(make_job(tmp_path, "a" * 64)) != service.reference(
        make_job(tmp_path, "b" * 64)
    )


def test_missing_checkpoint_returns_empty_state(tmp_path: Path):
    service = CheckpointService(
        FakeDatasets(DatasetRemoteState(DatasetPresence.ABSENT)), "checkpoint"
    )
    state = service.inspect(make_job(tmp_path, "a" * 64))
    assert state.reference is None
    assert state.completion.missing == 2


def test_inspect_ready_checkpoint_keeps_downloaded_paths(tmp_path: Path):
    job = make_job(tmp_path, "a" * 64)
    datasets = FakeDatasets(DatasetRemoteState(DatasetPresence.EXISTS, status="READY"))
    datasets.identity = {"job_sha256": job.identity.sha256}
    service = CheckpointService(datasets, "checkpoint")
    state = service.inspect(job)
    assert state.artifact is not None
    assert state.artifact.data_path.exists()
    assert state.artifact.manifest_path.exists()
