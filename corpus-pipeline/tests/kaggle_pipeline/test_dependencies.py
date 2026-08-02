from pathlib import Path

from corpus_pipeline.integrations.kaggle.config import OwnerConfiguration
from corpus_pipeline.integrations.kaggle.dataset_service import (
    DatasetPresence,
    DatasetRemoteState,
)
from corpus_pipeline.integrations.kaggle.dependencies import DependencyService
from corpus_pipeline.integrations.kaggle.models import ActionVerb
from corpus_pipeline.integrations.kaggle.reconcile import DesiredDataset


class FakeDatasets:
    def __init__(self, states):
        self.states = states
        self.mutations = []
        self.waits = []

    def inspect_state(self, reference, *, active_owner=None):
        del active_owner
        return self.states[reference]

    def fetch_json(self, reference, filename):
        del filename
        return self.states[reference].manifest or {}

    def ensure_dataset(self, slug, title, path, **kwargs):
        del title, path, kwargs
        self.mutations.append(slug)
        return type("Prepared", (), {"reference": f"owner/{slug}", "changed": True})()

    def wait_for_dataset_ready(self, reference):
        self.waits.append(reference)


def job(tmp_path):
    return type(
        "Job",
        (),
        {"stage": "query-embed", "model": "model", "input_path": tmp_path / "input"},
    )()


def owners():
    return OwnerConfiguration("owner", "runtime", "corpus", "checkpoint")


def make_desired(kind, reference, fingerprint):
    return DesiredDataset(
        resource_kind=kind,
        reference=reference,
        title=kind,
        fingerprint=fingerprint,
        manifest_filename="dependency_manifest.json",
        public=False,
        materialize=lambda root: Path(root),
    )


def test_matching_dependencies_are_reused_without_materialization(tmp_path):
    desired = [make_desired("runtime", "owner/runtime", "same")]
    datasets = FakeDatasets(
        {
            "owner/runtime": DatasetRemoteState(
                DatasetPresence.EXISTS, "READY", manifest={"fingerprint": "same"}
            )
        }
    )
    service = DependencyService(datasets, lambda _job, _owners, _workspace: desired)
    actions = service.reconcile(
        job(tmp_path), owners(), tmp_path, force=False, check_only=False
    )
    assert [action.verb for action in actions] == [ActionVerb.REUSE]
    assert datasets.mutations == []


def test_check_only_does_not_mutate_missing_dependency(tmp_path):
    desired = [make_desired("input", "owner/input", "new")]
    datasets = FakeDatasets({"owner/input": DatasetRemoteState(DatasetPresence.ABSENT)})
    service = DependencyService(datasets, lambda _job, _owners, _workspace: desired)
    actions = service.reconcile(
        job(tmp_path), owners(), tmp_path, force=False, check_only=True
    )
    assert [action.verb for action in actions] == [ActionVerb.CREATE]
    assert datasets.mutations == []
