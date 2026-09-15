import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from tests.integrations.kaggle.factories import (
    owners,
    rerank_runtime_profile,
    stage_job,
    stage_request,
)

from seed_pipeline.integrations.kaggle import dependencies
from seed_pipeline.integrations.kaggle.dataset_service import (
    DatasetPresence,
    DatasetRemoteState,
    DatasetService,
    PreparedDataset,
)
from seed_pipeline.integrations.kaggle.models import ActionVerb, StageJob, StageName
from seed_pipeline.integrations.kaggle.reconcile import DesiredDataset
from seed_pipeline.integrations.kaggle.stages import RerankStage


def _job(tmp_path: Path):
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
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1}) + "\n", encoding="utf-8")
    request = stage_request(
        StageName.RERANK,
        "qwen3-reranker:0.6b-fp16",
        candidates,
        runtime_profile=rerank_runtime_profile(),
    )
    return RerankStage().build_job(request), candidates, manifest


def test_input_dataset_materializes_every_bundle_file_and_manifest(
    tmp_path, monkeypatch
):
    job, candidates, manifest = _job(tmp_path)
    monkeypatch.setattr(
        dependencies,
        "resolve_publishable_artifact",
        lambda *_args: SimpleNamespace(
            dataset_slug="model",
            sha256="model-sha",
        ),
    )
    desired = dependencies.default_desired_datasets(job, owners(), tmp_path)
    input_dataset = next(item for item in desired if item.resource_kind == "input")
    staged = input_dataset.materialize(tmp_path / "staged")
    payload = json.loads(
        (staged / "dependency_manifest.json").read_text(encoding="utf-8")
    )

    assert input_dataset.fingerprint == job.input_bundle.sha256
    assert input_dataset.manifest_filename == "dependency_manifest.json"
    assert payload == {
        "schema_version": 2,
        "resource_kind": "input",
        "fingerprint": job.input_bundle.sha256,
        "files": job.input_bundle.descriptors(),
    }
    assert (staged / "candidates.jsonl").read_bytes() == candidates.read_bytes()
    assert (staged / "manifest.json").read_bytes() == manifest.read_bytes()


def test_desired_datasets_multi_owner_resolution(tmp_path, monkeypatch):
    job, _, _ = _job(tmp_path)
    monkeypatch.setattr(
        dependencies,
        "resolve_publishable_artifact",
        lambda *_args: SimpleNamespace(
            dataset_slug="model-slug",
            sha256="model-sha",
        ),
    )
    owner_configuration = owners(
        "worker-acc",
        runtime="runtime-acc",
        corpus="corpus-acc",
        checkpoint="worker-acc",
    )
    desired = dependencies.default_desired_datasets(job, owner_configuration, tmp_path)
    by_kind = {item.resource_kind: item for item in desired}

    assert by_kind["model"].reference == "runtime-acc/model-slug"
    assert by_kind["model"].public is True
    assert by_kind["input"].reference.startswith("corpus-acc/pipeline-input-")
    assert by_kind["input"].public is True


class UnusedRunner:
    def run(
        self,
        args: list[str],
        capture_output: bool = False,
        *,
        live_output: bool = False,
    ) -> str:
        raise AssertionError(f"unexpected Kaggle command: {args}")


class FakeKaggleDatasets:
    """Shared remote dataset state seen by every account's DatasetService."""

    def __init__(self, monkeypatch):
        self.monkeypatch = monkeypatch
        self.published: set[str] = set()
        self.created: list[tuple[str, str]] = []

    def service(self, owner: str) -> DatasetService:
        service = DatasetService(UnusedRunner(), owner)

        def inspect_state(reference, *, active_owner=None):
            if reference in self.published:
                return DatasetRemoteState(
                    DatasetPresence.EXISTS, status="READY", current_version=1
                )
            return DatasetRemoteState(DatasetPresence.ABSENT)

        def ensure_dataset(
            slug, title, path, *, public=False, active_owner=None, message=None
        ):
            reference = f"{owner}/{slug}"
            self.created.append((owner, reference))
            self.published.add(reference)
            return PreparedDataset(reference, True, 1)

        def wait_for_dataset_ready(
            reference, *, minimum_version=None, max_attempts=300
        ):
            return None

        self.monkeypatch.setattr(service, "inspect_state", inspect_state)
        self.monkeypatch.setattr(service, "ensure_dataset", ensure_dataset)
        self.monkeypatch.setattr(
            service, "wait_for_dataset_ready", wait_for_dataset_ready
        )
        return service


def _shared_model(job: StageJob, owner_configuration, workspace: Path):
    del job, workspace
    return (
        DesiredDataset(
            "model",
            f"{owner_configuration.runtime}/model-slug",
            "Kaggle Pipeline Model",
            "model-sha",
            "model_manifest.json",
            True,
            lambda root: root,
        ),
    )


def test_missing_shared_dataset_is_created_by_the_shared_owner_profile(
    tmp_path, monkeypatch
):
    kaggle = FakeKaggleDatasets(monkeypatch)
    service = dependencies.DependencyService(
        kaggle.service("tertiary-user"),
        _shared_model,
        publishers=(kaggle.service("primary-user"),),
    )

    actions = service.reconcile(
        stage_job(tmp_path),
        owners("tertiary-user", runtime="primary-user", corpus="primary-user"),
        tmp_path,
        force=False,
        check_only=False,
    )

    assert [action.verb for action in actions] == [ActionVerb.CREATE]
    assert kaggle.created == [("primary-user", "primary-user/model-slug")]


def test_missing_dataset_without_an_owning_profile_fails_before_upload(
    tmp_path, monkeypatch
):
    kaggle = FakeKaggleDatasets(monkeypatch)
    service = dependencies.DependencyService(
        kaggle.service("tertiary-user"), _shared_model
    )

    with pytest.raises(
        ValueError,
        match="No Kaggle account profile can publish primary-user/model-slug",
    ):
        service.reconcile(
            stage_job(tmp_path),
            owners("tertiary-user", runtime="primary-user"),
            tmp_path,
            force=False,
            check_only=False,
        )
    assert kaggle.created == []
