from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from kaggle_vector_cache.kaggle_api import dataset_metadata
from kaggle_vector_cache.manifests import (
    MANIFEST_SCHEMA_VERSION,
    ManifestError,
    file_sha256,
    write_json,
)
from model_runtime.catalog import EMBEDDING_MODELS, ModelSpec, require_model
from model_runtime.compose import ModelArtifactError


@dataclass(frozen=True)
class CanonicalModelArtifact:
    model: str
    source_path: Path
    canonical_filename: str
    dataset_slug: str
    byte_size: int
    sha256: str
    vector_dimension: int

    @property
    def spec(self) -> ModelSpec:
        return require_model(self.model)


def resolve_publishable_artifacts(
    gguf_root: Path,
) -> tuple[CanonicalModelArtifact, ...]:
    artifacts = []
    for spec in EMBEDDING_MODELS.values():
        source = Path(gguf_root) / spec.canonical_filename
        if not source.is_file() or source.is_symlink():
            raise ModelArtifactError(f"Embedding GGUF is missing or invalid: {source}")
        if source.stat().st_size != spec.byte_size:
            raise ModelArtifactError(f"Embedding GGUF size mismatch for {spec.name}")
        if file_sha256(source) != spec.sha256:
            raise ModelArtifactError(f"Embedding GGUF SHA-256 mismatch for {spec.name}")
        artifacts.append(
            CanonicalModelArtifact(
                spec.name,
                source,
                spec.canonical_filename,
                spec.gguf_dataset_slug,
                spec.byte_size,
                spec.sha256,
                int(spec.vector_dimension),
            )
        )
    return tuple(artifacts)


def build_model_manifest(
    artifact: CanonicalModelArtifact, owner: str | None = None
) -> dict:
    result = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "model": artifact.model,
        "filename": artifact.canonical_filename,
        "size": artifact.byte_size,
        "sha256": artifact.sha256,
        "vector_dimension": artifact.vector_dimension,
        "source_kind": "user_provided",
    }
    if owner:
        result["dataset_id"] = f"{owner}/{artifact.dataset_slug}"
    return result


def stage_model_dataset(
    artifact: CanonicalModelArtifact, staging_root: Path, owner: str
) -> Path:
    source = artifact.source_path
    if source.is_symlink() or not source.is_file():
        raise ManifestError(f"GGUF source is not a regular file: {source}")
    if (
        source.stat().st_size != artifact.byte_size
        or file_sha256(source) != artifact.sha256
    ):
        raise ManifestError(f"GGUF source changed after validation: {source}")
    target = Path(staging_root) / artifact.dataset_slug
    remove_model_staging(target)
    target.mkdir(parents=True)
    write_json(
        target / "dataset-metadata.json",
        dataset_metadata(
            owner,
            artifact.dataset_slug,
            f"Vector Cache GGUF {artifact.model}"[:50],
            public=True,
        ),
    )
    write_json(target / "model_manifest.json", build_model_manifest(artifact, owner))
    (target / artifact.canonical_filename).symlink_to(
        os.path.relpath(source, start=target)
    )
    return target


def remove_model_staging(staged: Path) -> None:
    staged = Path(staged)
    if not staged.exists():
        return
    for child in staged.iterdir():
        child.unlink()
    staged.rmdir()
