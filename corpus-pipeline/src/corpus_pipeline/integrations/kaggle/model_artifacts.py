from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from corpus_pipeline.integrations.kaggle.api import dataset_metadata
from corpus_pipeline.integrations.kaggle.artifacts import sha256_file
from corpus_pipeline.runtime.catalog import ModelKind, ModelSpec, require_model


@dataclass(frozen=True)
class CanonicalModelArtifact:
    model: str
    source_path: Path
    canonical_filename: str
    dataset_slug: str
    byte_size: int
    sha256: str
    vector_dimension: int | None

    @property
    def spec(self) -> ModelSpec:
        return require_model(self.model)

    @property
    def kind(self) -> ModelKind:
        return self.spec.kind

    @property
    def reranker_protocol(self) -> str | None:
        return self.spec.reranker_protocol


def resolve_publishable_artifact(gguf_root: Path, model: str) -> CanonicalModelArtifact:
    spec = require_model(model)
    source = Path(gguf_root) / spec.canonical_filename
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"GGUF is missing or invalid: {source}")
    if source.stat().st_size != spec.byte_size:
        raise ValueError(f"GGUF size mismatch for {spec.name}")
    digest = sha256_file(source)
    if digest != spec.sha256:
        raise ValueError(f"GGUF SHA-256 mismatch for {spec.name}")
    return CanonicalModelArtifact(
        spec.name,
        source,
        spec.canonical_filename,
        spec.gguf_dataset_slug,
        spec.byte_size,
        digest,
        spec.vector_dimension,
    )


def stage_model_dataset(
    artifact: CanonicalModelArtifact, staging_root: Path, owner: str
) -> Path:
    source = artifact.source_path
    if (
        source.stat().st_size != artifact.byte_size
        or sha256_file(source) != artifact.sha256
    ):
        raise ValueError(
            f"GGUF source changed after corpus_pipeline.corpus.validation: {source}"
        )
    target = Path(staging_root) / artifact.dataset_slug
    target.mkdir(parents=True, exist_ok=True)
    (target / "dataset-metadata.json").write_text(
        json_dumps(
            dataset_metadata(
                owner,
                artifact.dataset_slug,
                f"Kaggle Pipeline GGUF {artifact.model}"[:50],
                public=True,
            )
        ),
        encoding="utf-8",
    )
    (target / "model_manifest.json").write_text(
        json_dumps(
            {
                "model": artifact.model,
                "filename": artifact.canonical_filename,
                "size": artifact.byte_size,
                "sha256": artifact.sha256,
                "vector_dimension": artifact.vector_dimension,
                "kind": artifact.kind.value,
                "reranker_protocol": artifact.reranker_protocol,
            }
        ),
        encoding="utf-8",
    )
    (target / "dependency_manifest.json").write_text(
        json_dumps(
            {
                "schema_version": 1,
                "resource_kind": "model",
                "fingerprint": artifact.sha256,
            }
        ),
        encoding="utf-8",
    )
    (target / artifact.canonical_filename).symlink_to(
        os.path.relpath(source, start=target)
    )
    return target


def json_dumps(payload: dict) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
