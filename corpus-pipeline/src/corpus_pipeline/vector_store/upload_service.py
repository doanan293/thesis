from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from corpus_pipeline.artifacts.bundle import load_bundle
from corpus_pipeline.evaluation.artifact_contracts import (
    ArtifactContractError,
    sha256_file,
)
from corpus_pipeline.runtime.catalog import require_model
from corpus_pipeline.vector_store.ingest_vectors import (
    DEFAULT_QDRANT_BATCH_SIZE,
    ChunkEmbeddingCache,
    collect_complete_cached_embeddings,
    load_qdrant_dependencies,
    read_normalized_input_points,
    upsert_points_from_embeddings,
)


@dataclass(frozen=True)
class UploadVectorsRequest:
    chunks_path: Path
    embeddings_dir: Path
    model: str
    qdrant_url: str
    qdrant_batch_size: int = DEFAULT_QDRANT_BATCH_SIZE


@dataclass(frozen=True)
class UploadVectorsResult:
    collection_name: str
    point_count: int
    reused: bool = False
    physical_collection: str | None = None


def upload_vectors(request: UploadVectorsRequest) -> UploadVectorsResult:
    bundle = load_bundle(
        request.embeddings_dir,
        expected_type="chunk_embeddings",
        require_complete=True,
    )
    spec = require_model(request.model)
    if spec.vector_dimension is None:
        raise ValueError(f"Embedding dimension is missing for {request.model}")
    expected_identity = {
        "input_sha256": sha256_file(request.chunks_path),
        "model": request.model,
        "model_sha256": spec.sha256,
        "vector_dimension": spec.vector_dimension,
    }
    if bundle.manifest.identity != expected_identity:
        raise ArtifactContractError(
            "Embedding bundle identity does not match --chunks and --model"
        )
    parsed = urlparse(request.qdrant_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"Invalid Qdrant URL: {request.qdrant_url}")
    points = read_normalized_input_points(request.chunks_path)
    embeddings = collect_complete_cached_embeddings(
        points,
        request.model,
        ChunkEmbeddingCache(bundle.data_path, spec.vector_dimension),
    )
    dependencies = load_qdrant_dependencies()
    alias_name = "thesis_chunks_" + spec.slug
    digest = (
        f"{expected_identity['input_sha256'][:12]}{bundle.manifest.data_sha256[:12]}"
    )
    collection_name = f"{alias_name}_{digest}"
    client = dependencies.helper_type(
        host=parsed.hostname,
        port=parsed.port or (443 if parsed.scheme == "https" else 80),
        collection_name=collection_name,
        vector_size=spec.vector_dimension,
    )
    client.init_collection()
    if client.collection_exists():
        if hasattr(client, "switch_alias"):
            client.switch_alias(alias_name)
        return UploadVectorsResult(
            alias_name, 0, reused=True, physical_collection=collection_name
        )
    uploaded = upsert_points_from_embeddings(
        points,
        embeddings,
        client,
        request.qdrant_batch_size,
        False,
        dependencies,
    )
    if hasattr(client, "switch_alias"):
        client.switch_alias(alias_name)
    return UploadVectorsResult(
        alias_name, uploaded, physical_collection=collection_name
    )
