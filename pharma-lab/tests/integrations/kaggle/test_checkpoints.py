import json
import re
from pathlib import Path
from typing import Any

import pytest
from tests.integrations.kaggle.factories import stage_job

from pharma_lab.integrations.kaggle.artifacts import ArtifactContractError
from pharma_lab.integrations.kaggle.checkpoints import CheckpointService
from pharma_lab.integrations.kaggle.dataset_service import (
    DatasetPresence,
    DatasetRemoteState,
    PreparedDataset,
)
from pharma_lab.integrations.kaggle.models import JobIdentity, StageName
from pharma_lab.integrations.kaggle.workers.runtime import artifact_from_output
from pharma_lab.runtime.catalog import MODEL_CATALOG


def _identity() -> JobIdentity:
    return JobIdentity.create(
        stage=StageName.RERANK,
        contract_version=3,
        model="qwen3-reranker:0.6b-fp16",
        model_sha256="a" * 64,
        input_sha256="b" * 64,
        runtime_parameters={"request_contract_sha256": "c" * 64},
    )


class UnusedDatasets:
    """Dataset port for tests that must never reach Kaggle."""

    def inspect_state(
        self, reference: str, *, active_owner: str | None = None
    ) -> DatasetRemoteState:
        raise AssertionError(f"unexpected dataset inspection: {reference}")

    def wait_for_dataset_ready(self, reference: str) -> None:
        raise AssertionError(f"unexpected dataset wait: {reference}")

    def fetch_json(self, reference: str, filename: str) -> dict[str, Any]:
        raise AssertionError(f"unexpected dataset fetch: {reference}/{filename}")

    def download_file(self, reference: str, filename: str, destination: Path) -> Path:
        raise AssertionError(f"unexpected dataset download: {reference}/{filename}")

    def ensure_dataset(
        self,
        slug: str,
        title: str,
        path: Path,
        *,
        public: bool = False,
        active_owner: str | None = None,
    ) -> PreparedDataset:
        raise AssertionError(f"unexpected dataset publish: {slug}")


class FakeDatasetService(UnusedDatasets):
    def __init__(self, manifest: dict[str, Any], source_path: Path):
        self.manifest = manifest
        self.source_path = Path(source_path)
        self.roots: list[Path] = []

    def inspect_state(
        self, reference: str, *, active_owner: str | None = None
    ) -> DatasetRemoteState:
        return DatasetRemoteState(DatasetPresence.EXISTS, status="READY")

    def fetch_json(self, reference: str, filename: str) -> dict[str, Any]:
        return self.manifest

    def download_file(self, reference: str, filename: str, destination: Path) -> Path:
        destination = Path(destination)
        self.roots.append(destination)
        target = destination / filename
        target.write_bytes(self.source_path.read_bytes())
        return target

    def wait_for_dataset_ready(self, reference: str) -> None:
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
    job = stage_job(
        tmp_path, identity=_identity(), data_filename=data.name, expected_total=2
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
    job = stage_job(
        tmp_path, identity=_identity(), data_filename=data.name, expected_total=1
    )

    with pytest.raises(ArtifactContractError, match="artifact type mismatch"):
        CheckpointService(service, "owner").inspect(
            job, download_root=tmp_path / "acc1"
        )


def test_rerank_checkpoint_reference_respects_kaggle_slug_limit(tmp_path):
    service = CheckpointService(UnusedDatasets(), "owner")
    job = stage_job(tmp_path, stage=StageName.RERANK, model="qwen3-reranker:0.6b-fp16")
    reuse_prefix = job.identity.reuse_sha256[:8]

    reference = service.reference(job)
    owner, slug = reference.split("/", 1)

    assert owner == "owner"
    assert slug == f"re-eval-rerank-qwen3-reranker-{reuse_prefix}-checkpoint"
    assert 6 <= len(slug) <= 50
    assert re.fullmatch(r"[a-z0-9-]+", slug)


PRODUCTION_STAGES = tuple(
    stage for stage in StageName if not stage.value.endswith("-benchmark")
)


@pytest.mark.parametrize("stage", PRODUCTION_STAGES)
@pytest.mark.parametrize("model", tuple(MODEL_CATALOG))
def test_production_checkpoint_references_are_bounded_and_deterministic(
    stage, model, tmp_path
):
    service = CheckpointService(UnusedDatasets(), "owner")
    job = stage_job(tmp_path, stage=stage, model=model)
    reuse_prefix = job.identity.reuse_sha256[:8]

    first = service.reference(job)
    second = service.reference(job)
    slug = first.split("/", 1)[1]

    assert first == second
    assert 6 <= len(slug) <= 50
    assert slug.startswith(f"re-eval-{stage.value}-")
    assert slug.endswith(f"-{reuse_prefix}-checkpoint")
    assert re.fullmatch(r"[a-z0-9-]+", slug)
