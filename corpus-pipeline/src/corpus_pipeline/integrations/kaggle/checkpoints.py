from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from corpus_pipeline.integrations.kaggle.artifacts import load_cloud_artifact
from corpus_pipeline.integrations.kaggle.dataset_service import (
    DatasetPresence,
    DatasetService,
)
from corpus_pipeline.integrations.kaggle.models import (
    CloudArtifact,
    Completion,
    StageJob,
)


@dataclass(frozen=True)
class CheckpointState:
    reference: str | None
    artifact: CloudArtifact | None
    completion: Completion


class CheckpointService:
    def __init__(self, datasets: DatasetService, owner: str):
        self.datasets = datasets
        self.owner = owner

    def reference(self, job: StageJob) -> str:
        model_slug = "".join(
            char if char.isalnum() else "-" for char in job.model.casefold()
        ).strip("-")
        return f"{self.owner}/re-eval-{job.stage.value}-{model_slug[:24]}-{job.identity.sha256[:8]}-checkpoint"

    def inspect(self, job: StageJob) -> CheckpointState:
        reference = self.reference(job)
        state = self.datasets.inspect_state(reference, active_owner=self.owner)
        empty = Completion(job.expected_total, 0, job.expected_total)
        if state.presence is DatasetPresence.ABSENT:
            return CheckpointState(None, None, empty)
        if state.presence is DatasetPresence.UNKNOWN:
            raise RuntimeError(
                f"Checkpoint state is unknown: {reference}: {state.detail}"
            )
        if state.status == "PENDING":
            self.datasets.wait_for_dataset_ready(reference)
        if state.status != "READY":
            raise RuntimeError(
                f"Checkpoint dataset is not READY: {reference} ({state.status})"
            )
        # Keep downloaded files under the job directory: callers may need to
        # promote a complete checkpoint or resume from a partial one after
        # this method returns.  A TemporaryDirectory would invalidate those
        # paths as soon as inspect() exits.
        root = job.output_dir / ".checkpoint"
        root.mkdir(parents=True, exist_ok=True)
        manifest = self.datasets.fetch_json(reference, "manifest.json")
        manifest_path = root / "manifest.json"
        import json

        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        data_path = root / str(manifest["data_filename"])
        self.datasets.download_file(reference, data_path.name, root)
        artifact = load_cloud_artifact(data_path, manifest_path, job.identity)
        return CheckpointState(reference, artifact, artifact.completion)

    def publish(self, job: StageJob, artifact: CloudArtifact) -> str:
        reference = self.reference(job)
        with tempfile.TemporaryDirectory(prefix="checkpoint-publish-") as raw:
            root = Path(raw)
            shutil.copy2(artifact.data_path, root / artifact.data_path.name)
            shutil.copy2(artifact.manifest_path, root / "manifest.json")
            slug = reference.split("/", 1)[1]
            self.datasets.ensure_dataset(
                slug,
                f"Kaggle Pipeline Checkpoint {job.stage.value}",
                root,
                public=False,
                active_owner=self.owner,
            )
        self.datasets.wait_for_dataset_ready(reference)
        return reference
