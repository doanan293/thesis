from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

from corpus_pipeline.integrations.kaggle.artifacts import sha256_file
from corpus_pipeline.integrations.kaggle.config import OwnerConfiguration
from corpus_pipeline.integrations.kaggle.dataset_service import (
    DatasetInventory,
    DatasetService,
)
from corpus_pipeline.integrations.kaggle.model_artifacts import (
    resolve_publishable_artifact,
    stage_model_dataset,
)
from corpus_pipeline.integrations.kaggle.models import (
    ActionVerb,
    ReconcileAction,
    StageJob,
)
from corpus_pipeline.integrations.kaggle.reconcile import DesiredDataset, plan_dataset

DesiredBuilder = Callable[
    [StageJob, OwnerConfiguration, Path], Sequence[DesiredDataset]
]

CORPUS_INPUT_DATASET_SLUG = "corpus-pipeline-rag-final"


def input_dataset_slug(job: StageJob) -> str:
    if getattr(job.stage, "value", job.stage) == "corpus-embed":
        return CORPUS_INPUT_DATASET_SLUG
    return f"pipeline-input-{job.identity.sha256[:16]}"


def kaggle_input_root(job: StageJob) -> Path:
    return Path("/kaggle/input") / input_dataset_slug(job)


class DependencyService:
    def __init__(self, datasets: DatasetService, desired_builder: DesiredBuilder):
        self.datasets = datasets
        self.desired_builder = desired_builder

    def desired(
        self, job: StageJob, owners: OwnerConfiguration, workspace: Path
    ) -> Sequence[DesiredDataset]:
        return self.desired_builder(job, owners, workspace)

    def require_ready(self, reference: str) -> None:
        self.datasets.require_ready(
            reference,
            guidance="publish the Kaggle runtime dataset before running a stage",
        )

    def reconcile(
        self,
        job: StageJob,
        owners: OwnerConfiguration,
        workspace: Path,
        *,
        force: bool,
        check_only: bool,
    ) -> Sequence[ReconcileAction]:
        desired = tuple(self.desired(job, owners, workspace))
        resources = {item.reference: item.manifest_filename for item in desired}
        inventory = DatasetInventory.load(
            self.datasets, resources, active_owner=owners.execution
        )
        actions: list[ReconcileAction] = []
        for item in desired:
            owner, _, slug = item.reference.partition("/")
            item_force = force and (owner == self.datasets.owner)
            action = plan_dataset(item, inventory.get(item.reference), force=item_force)
            if action.verb is ActionVerb.WAIT and not check_only:
                self.datasets.wait_for_dataset_ready(item.reference)
                refreshed = self.datasets.inspect_state(
                    item.reference, active_owner=owners.execution
                )
                inventory.remember(item.reference, refreshed)
                action = plan_dataset(item, refreshed, force=item_force)
            if action.verb in {ActionVerb.CREATE, ActionVerb.UPDATE} and not check_only:
                with tempfile.TemporaryDirectory(
                    prefix=f"dependency-{item.resource_kind}-"
                ) as raw:
                    staged = item.materialize(Path(raw))
                    owner, _, slug = item.reference.partition("/")
                    if owner != self.datasets.owner:
                        raise ValueError(
                            f"Dependency owner {owner} does not match dataset service owner {self.datasets.owner}"
                        )
                    prepared = self.datasets.ensure_dataset(
                        slug,
                        item.title,
                        staged,
                        public=item.public,
                        active_owner=owners.execution,
                    )
                self.datasets.wait_for_dataset_ready(
                    item.reference, minimum_version=prepared.expected_version
                )
                inventory.remember(
                    item.reference,
                    self.datasets.inspect_state(
                        item.reference, active_owner=owners.execution
                    ),
                )
            actions.append(action)
        return actions


def default_desired_datasets(
    job: StageJob, owners: OwnerConfiguration, workspace: Path
) -> Sequence[DesiredDataset]:
    """Build the model and immutable input datasets consumed by a stage."""
    gguf_root_value = job.worker_config.get("gguf_root")
    if not isinstance(gguf_root_value, str) or not gguf_root_value:
        raise ValueError("stage job is missing gguf_root for model reconciliation")
    model = resolve_publishable_artifact(Path(gguf_root_value), job.model)
    model_owner = owners.runtime or owners.execution
    model_ref = f"{model_owner}/{model.dataset_slug}"

    def materialize_model(root: Path) -> Path:
        return stage_model_dataset(model, root, model_owner)

    input_digest = job.input_bundle.sha256
    input_slug = input_dataset_slug(job)
    input_owner = (
        (owners.corpus or owners.runtime or owners.execution)
        if input_slug == CORPUS_INPUT_DATASET_SLUG
        else owners.execution
    )
    input_ref = f"{input_owner}/{input_slug}"

    def materialize_input(root: Path) -> Path:
        target = Path(root)
        target.mkdir(parents=True, exist_ok=True)
        for input_file in job.input_bundle.files:
            actual_digest = sha256_file(input_file.source_path)
            if actual_digest != input_file.sha256:
                raise ValueError(
                    f"Input bundle file changed for {input_file.key}: "
                    f"{actual_digest} != {input_file.sha256}"
                )
            shutil.copy2(input_file.source_path, target / input_file.filename)
        (target / "dependency_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "resource_kind": "input",
                    "fingerprint": input_digest,
                    "files": job.input_bundle.descriptors(),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return target

    return (
        DesiredDataset(
            "model",
            model_ref,
            f"Kaggle Pipeline Model {job.model}"[:50],
            model.sha256,
            "model_manifest.json",
            True,
            materialize_model,
        ),
        DesiredDataset(
            "input",
            input_ref,
            f"Kaggle Pipeline Input {job.stage.value}"[:50],
            input_digest,
            "dependency_manifest.json",
            True,
            materialize_input,
        ),
    )
