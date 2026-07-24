from __future__ import annotations

import os
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

from kaggle_vector_cache.dataset_service import DatasetService
from kaggle_vector_cache.kaggle_api import (
    KaggleCommandRunner,
    dataset_download_command,
)
from kaggle_vector_cache.lock import RunResult
from kaggle_vector_cache.manifests import (
    ManifestError,
    load_json,
    validate_checkpoint_artifact,
    validate_checkpoint_manifest,
)
from kaggle_vector_cache.models import require_model
from kaggle_vector_cache.parsers import checkpoint_dataset_slug
from vector_store.ingest_vectors import model_slug


@dataclass(frozen=True)
class CheckpointArtifact:
    root: Path
    cache_path: Path
    manifest_path: Path
    manifest: dict


class CloudCheckpointService:
    def __init__(
        self,
        runner: KaggleCommandRunner,
        datasets: DatasetService,
        owner: str,
    ):
        self.runner = runner
        self.datasets = datasets
        self.owner = owner

    def reference(self, model: str) -> str:
        return f"{self.owner}/{checkpoint_dataset_slug(model)}"

    def inspect(
        self,
        model: str,
        destination: Path,
        *,
        corpus_sha256: str,
    ) -> RunResult:
        reference = self.reference(model)
        status = self.datasets.optional_status(reference)
        if status is None:
            return RunResult(model, 0, 0, 0, 0)
        if status != "READY":
            raise RuntimeError(
                f"Checkpoint dataset is not READY: {reference} ({status})"
            )
        manifest = self.datasets.fetch_json(reference, "manifest.json", destination)
        spec = require_model(model)
        validate_checkpoint_manifest(
            manifest,
            model=model,
            corpus_sha256=corpus_sha256,
            vector_dimension=int(spec.vector_dimension),
        )
        return self.result_from_manifest(manifest)

    @staticmethod
    def result_from_manifest(manifest: dict) -> RunResult:
        return RunResult(
            model=str(manifest["model"]),
            run_count=int(manifest.get("run_count", 0)),
            total=int(manifest["total"]),
            complete=int(manifest["complete"]),
            missing=int(manifest["missing"]),
        )

    def _artifact_from_pair(
        self,
        root: Path,
        manifest_path: Path,
        cache_path: Path,
        *,
        model: str,
        corpus_sha256: str,
    ) -> CheckpointArtifact:
        manifest = load_json(manifest_path)
        spec = require_model(model)
        validate_checkpoint_artifact(
            manifest,
            cache_path,
            model=model,
            corpus_sha256=corpus_sha256,
            vector_dimension=int(spec.vector_dimension),
        )
        return CheckpointArtifact(root, cache_path, manifest_path, manifest)

    def materialize_kernel_output(
        self,
        incoming: Path,
        destination: Path,
        *,
        model: str,
        corpus_sha256: str,
    ) -> CheckpointArtifact:
        archives = list(Path(incoming).rglob("checkpoint.zip"))
        if len(archives) != 1:
            raise ManifestError(f"Expected one checkpoint.zip, found {len(archives)}")
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=False)
        with zipfile.ZipFile(archives[0]) as archive:
            for filename in ("manifest.json", "checkpoint.jsonl"):
                try:
                    payload = archive.read(filename)
                except KeyError as exc:
                    raise ManifestError(
                        f"Checkpoint archive is missing {filename}"
                    ) from exc
                (destination / filename).write_bytes(payload)
        return self._artifact_from_pair(
            destination,
            destination / "manifest.json",
            destination / "checkpoint.jsonl",
            model=model,
            corpus_sha256=corpus_sha256,
        )

    def materialize_downloaded_dataset(
        self,
        incoming: Path,
        *,
        model: str,
        corpus_sha256: str,
    ) -> CheckpointArtifact:
        incoming = Path(incoming)
        manifests = [
            path
            for path in incoming.rglob("manifest.json")
            if "validated" not in path.parts
        ]
        caches = [
            path
            for path in incoming.rglob("checkpoint.jsonl")
            if "validated" not in path.parts
        ]
        if len(manifests) == 1 and len(caches) == 1:
            return self._artifact_from_pair(
                manifests[0].parent,
                manifests[0],
                caches[0],
                model=model,
                corpus_sha256=corpus_sha256,
            )
        return self.materialize_kernel_output(
            incoming,
            incoming / "validated",
            model=model,
            corpus_sha256=corpus_sha256,
        )

    def publish(
        self,
        artifact: CheckpointArtifact,
        *,
        model: str,
        previous_manifest: dict | None,
    ) -> RunResult:
        previous_complete = (
            int(previous_manifest["complete"]) if previous_manifest is not None else -1
        )
        current_complete = int(artifact.manifest["complete"])
        if current_complete <= previous_complete:
            raise RuntimeError(
                f"Checkpoint made no progress for {model}: "
                f"complete={current_complete}, previous={previous_complete}"
            )
        self.datasets.ensure_dataset(
            slug=checkpoint_dataset_slug(model),
            title=f"Vector Cache Checkpoint ({model})",
            path=artifact.root,
            public=False,
        )
        self.datasets.wait_for_dataset_ready(self.reference(model))
        return self.result_from_manifest(artifact.manifest)

    def download(
        self,
        model: str,
        incoming: Path,
        output_dir: Path,
        *,
        corpus_sha256: str,
    ) -> tuple[Path, Path]:
        incoming = Path(incoming)
        self.runner.run(dataset_download_command(self.reference(model), incoming))
        artifact = self.materialize_downloaded_dataset(
            incoming,
            model=model,
            corpus_sha256=corpus_sha256,
        )
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        slug = model_slug(model)
        cache_target = output_dir / f"{slug}.jsonl"
        manifest_target = output_dir / f"{slug}.manifest.json"
        cache_staging = output_dir / f".{slug}.jsonl.ready"
        manifest_staging = output_dir / f".{slug}.manifest.json.ready"
        try:
            shutil.copy2(artifact.cache_path, cache_staging)
            shutil.copy2(artifact.manifest_path, manifest_staging)
            os.replace(cache_staging, cache_target)
            os.replace(manifest_staging, manifest_target)
        finally:
            cache_staging.unlink(missing_ok=True)
            manifest_staging.unlink(missing_ok=True)
        return cache_target, manifest_target
