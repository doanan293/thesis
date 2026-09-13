from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from seed_pipeline.evaluation.artifact_contracts import sha256_file
from seed_pipeline.runtime.catalog import require_model
from seed_pipeline.vector_store.ingest_vectors import (
    DEFAULT_QDRANT_BATCH_SIZE,
    ChunkEmbeddingCache,
    collect_complete_cached_embeddings,
    load_qdrant_dependencies,
    read_normalized_input_points,
    summarize_embedding_cache_completion,
    upsert_points_from_embeddings,
)


@dataclass(frozen=True)
class UploadVectorsRequest:
    chunks_path: Path
    embeddings_path: Path
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
    spec = require_model(request.model)
    if spec.vector_dimension is None:
        raise ValueError(f"Embedding dimension is missing for {request.model}")
    points = read_normalized_input_points(request.chunks_path)
    if not points or not request.embeddings_path.is_file():
        raise RuntimeError("Embedding cache is incomplete")
    cache = ChunkEmbeddingCache(
        request.embeddings_path,
        spec.vector_dimension,
        model_sha256=spec.sha256,
    )
    completion = summarize_embedding_cache_completion(points, request.model, cache)
    if not completion.is_complete:
        raise RuntimeError(
            f"Embedding cache is incomplete: {completion.missing} missing records"
        )
    parsed = urlparse(request.qdrant_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"Invalid Qdrant URL: {request.qdrant_url}")
    embeddings = collect_complete_cached_embeddings(
        points,
        request.model,
        cache,
    )
    dependencies = load_qdrant_dependencies()
    alias_name = "thesis_chunks_" + spec.slug
    digest = f"{sha256_file(request.chunks_path)[:12]}{cache.subset_sha256(points, request.model)[:12]}"
    collection_name = f"{alias_name}_{digest}"
    client = dependencies.helper_type(
        host=parsed.hostname,
        port=parsed.port or (443 if parsed.scheme == "https" else 80),
        collection_name=collection_name,
        vector_size=spec.vector_dimension,
    )
    expected_count = len(points)
    target_exists = client.collection_exists()
    reused = target_exists and client.point_count() == expected_count
    if reused:
        uploaded = 0
    else:
        if target_exists:
            client.clear_collection()
        else:
            client.init_collection()
        uploaded = upsert_points_from_embeddings(
            points,
            embeddings,
            client,
            request.qdrant_batch_size,
            False,
            dependencies,
        )
        actual_count = client.point_count()
        if actual_count != expected_count:
            raise RuntimeError(
                f"Qdrant point count mismatch for '{collection_name}': "
                f"expected {expected_count}, got {actual_count}"
            )
    client.switch_alias(alias_name, delete_legacy_collection=True)
    return UploadVectorsResult(
        alias_name,
        uploaded,
        reused=reused,
        physical_collection=collection_name,
    )
