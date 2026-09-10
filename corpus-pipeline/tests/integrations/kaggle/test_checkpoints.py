import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from corpus_pipeline.integrations.kaggle.artifacts import ArtifactContractError
from corpus_pipeline.integrations.kaggle.checkpoints import CheckpointService
from corpus_pipeline.integrations.kaggle.dataset_service import DatasetPresence
from corpus_pipeline.integrations.kaggle.models import JobIdentity, StageName
from corpus_pipeline.integrations.kaggle.workers.runtime import artifact_from_output
from corpus_pipeline.runtime.catalog import MODEL_CATALOG


def _job(stage: StageName, model: str):
    return SimpleNamespace(
        stage=stage,
        model=model,
        identity=SimpleNamespace(reuse_sha256="12345678" + "a" * 56),
    )


def _identity() -> JobIdentity:
    return JobIdentity.create(
        stage=StageName.RERANK,
        contract_version=3,
        model="qwen3-reranker:0.6b-fp16",
        model_sha256="a" * 64,
        input_sha256="b" * 64,
        runtime_parameters={"request_contract_sha256": "c" * 64},
    )


class FakeDatasetService:
    def __init__(self, manifest, source_path):
        self.manifest = manifest
        self.source_path = Path(source_path)
        self.roots = []

    def inspect_state(self, reference, *, active_owner=None):
        return SimpleNamespace(presence=DatasetPresence.EXISTS, status="READY")

    def fetch_json(self, reference, filename, destination=None):
        return self.manifest

    def download_file(self, reference, filename, destination):
        destination = Path(destination)
        self.roots.append(destination)
        target = destination / filename
        target.write_bytes(self.source_path.read_bytes())
        return target

    def wait_for_dataset_ready(self, reference):
        return None


def test_checkpoint_inspect_uses_explicit_isolated_root(tmp_path):
    data = tmp_path / "rerank_scores.jsonl"
    data.write_text('{"pair":"score"}\n', encoding="utf-8")
    artifact = artifact_from_output(
        data,
        artifact_type="rerank_scores",
        identity=_identity(),
        total=2,
        complete=1,
    )
    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    manifest["data_filename"] = data.name
    service = FakeDatasetService(manifest, data)
    job = SimpleNamespace(
        stage=StageName.RERANK,
        model="qwen3-reranker:0.6b-fp16",
        identity=_identity(),
        output_dir=tmp_path / "output",
        data_filename=data.name,
        expected_total=2,
    )

    state = CheckpointService(service, "owner").inspect(
        job, download_root=tmp_path / "acc1"
    )

    assert state.artifact is not None
    assert state.artifact.data_path.is_relative_to(tmp_path / "acc1")
    assert state.completion.complete == 1
    assert service.roots == [tmp_path / "acc1"]


def test_checkpoint_inspect_rejects_wrong_artifact_type(tmp_path):
    data = tmp_path / "rerank_scores.jsonl"
    data.write_text('{"pair":"score"}\n', encoding="utf-8")
    artifact = artifact_from_output(
        data,
        artifact_type="query_embedding_cache",
        identity=_identity(),
        total=1,
        complete=1,
    )
    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    service = FakeDatasetService(manifest, data)
    job = SimpleNamespace(
        stage=StageName.RERANK,
        model="qwen3-reranker:0.6b-fp16",
        identity=_identity(),
        output_dir=tmp_path / "output",
        data_filename=data.name,
        expected_total=1,
    )

    with pytest.raises(ArtifactContractError, match="artifact type mismatch"):
        CheckpointService(service, "owner").inspect(
            job, download_root=tmp_path / "acc1"
        )


def test_rerank_checkpoint_reference_respects_kaggle_slug_limit():
    service = CheckpointService(SimpleNamespace(), "owner")

    reference = service.reference(_job(StageName.RERANK, "qwen3-reranker:0.6b-fp16"))
    owner, slug = reference.split("/", 1)

    assert owner == "owner"
    assert slug == "re-eval-rerank-qwen3-reranker-12345678-checkpoint"
    assert 6 <= len(slug) <= 50
    assert re.fullmatch(r"[a-z0-9-]+", slug)


PRODUCTION_STAGES = tuple(
    stage for stage in StageName if not stage.value.endswith("-benchmark")
)


@pytest.mark.parametrize("stage", PRODUCTION_STAGES)
@pytest.mark.parametrize("model", tuple(MODEL_CATALOG))
def test_production_checkpoint_references_are_bounded_and_deterministic(stage, model):
    service = CheckpointService(SimpleNamespace(), "owner")
    job = _job(stage, model)

    first = service.reference(job)
    second = service.reference(job)
    slug = first.split("/", 1)[1]

    assert first == second
    assert 6 <= len(slug) <= 50
    assert slug.startswith(f"re-eval-{stage.value}-")
    assert slug.endswith("-12345678-checkpoint")
    assert re.fullmatch(r"[a-z0-9-]+", slug)
