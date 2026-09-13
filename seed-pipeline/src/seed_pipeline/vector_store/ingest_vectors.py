from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from seed_pipeline.config.paths import (
    PROJECT_ROOT,
    RAG_FINAL_CHUNKS_PATH,
    VECTOR_EMBEDDING_CACHE_DIR,
)
from seed_pipeline.vector_store.embedding_core import (
    QDRANT_POINT_ID_NAMESPACE,
    ChunkEmbeddingCache,
    EmbeddingCacheCompletion,
    EmbeddingCacheError,
    EmbeddingCollectionResult,
    EmbeddingError,
    EmbeddingProgressReporter,
    VectorInputError,
    canonical_json_hash,
    canonical_metadata_payload,
    chunk_payload_hash,
    chunked,
    collect_complete_cached_embeddings,
    collect_or_create_embeddings,
    detect_cache_vector_dimension,
    expected_cache_keys,
    get_embedding,
    get_embeddings_from_clients,
    inspect_embedding_cache_completion,
    iter_normalized_input_points,
    model_slug,
    normalize_canonical_metadata_record,
    normalize_input_records,
    parse_server_urls,
    partition_indexed_texts,
    prune_embedding_cache,
    qdrant_point_id,
    read_normalized_input_points,
    runtime_budget_exhausted,
    runtime_stop_deadline,
    summarize_embedding_cache_completion,
    text_hash,
    validate_cached_embedding,
)

__all__ = (
    "BM25_MODEL_NAME",
    "BM25_SPARSE_VECTOR_NAME",
    "CANONICAL_METADATA_PATH",
    "DEFAULT_COMPOSE_FILE",
    "DEFAULT_GGUF_ROOT",
    "DEFAULT_QDRANT_BATCH_SIZE",
    "DEFAULT_REQUEST_TIMEOUT",
    "DENSE_VECTOR_NAME",
    "QDRANT_POINT_ID_NAMESPACE",
    "VECTOR_DIMENSION",
    "ChunkEmbeddingCache",
    "EmbeddingCacheCompletion",
    "EmbeddingCacheError",
    "EmbeddingCollectionResult",
    "EmbeddingError",
    "EmbeddingProgressReporter",
    "QdrantDependencies",
    "VectorInputError",
    "build_qdrant_point",
    "canonical_json_hash",
    "canonical_metadata_payload",
    "chunk_payload_hash",
    "chunked",
    "collect_complete_cached_embeddings",
    "collect_or_create_embeddings",
    "default_embedding_cache_path",
    "detect_cache_vector_dimension",
    "expected_cache_keys",
    "get_embedding",
    "get_embeddings_from_clients",
    "inspect_embedding_cache_completion",
    "iter_normalized_input_points",
    "load_qdrant_dependencies",
    "model_slug",
    "normalize_canonical_metadata_record",
    "normalize_input_records",
    "parse_server_urls",
    "partition_indexed_texts",
    "prune_embedding_cache",
    "qdrant_point_id",
    "read_normalized_input_points",
    "runtime_budget_exhausted",
    "runtime_stop_deadline",
    "summarize_embedding_cache_completion",
    "text_hash",
    "upsert_points_from_embeddings",
    "validate_cached_embedding",
)

CANONICAL_METADATA_PATH = str(RAG_FINAL_CHUNKS_PATH)
VECTOR_DIMENSION = 1024
DEFAULT_QDRANT_BATCH_SIZE = 1000
DEFAULT_REQUEST_TIMEOUT = 900
DEFAULT_COMPOSE_FILE = PROJECT_ROOT.parent / "docker-compose.yml"
DEFAULT_GGUF_ROOT = PROJECT_ROOT.parent / "ai-models" / "gguf"


def default_embedding_cache_path(model_name: str) -> Path:
    return VECTOR_EMBEDDING_CACHE_DIR / f"{model_slug(model_name)}.jsonl"


DENSE_VECTOR_NAME = "dense_vector"
BM25_SPARSE_VECTOR_NAME = "bm25_sparse_vector"
BM25_MODEL_NAME = "Qdrant/bm25"


@dataclass(frozen=True)
class QdrantDependencies:
    document_type: type
    point_struct_type: type
    helper_type: type


def load_qdrant_dependencies() -> QdrantDependencies:
    from qdrant_client.models import Document, PointStruct

    from seed_pipeline.vector_store.qdrant_client_helper import QdrantClientHelper

    return QdrantDependencies(
        document_type=Document,
        point_struct_type=PointStruct,
        helper_type=QdrantClientHelper,
    )


def build_qdrant_point(
    point: dict,
    vector_embedding: list[float],
    dependencies: QdrantDependencies | None = None,
):
    dependencies = dependencies or load_qdrant_dependencies()
    return dependencies.point_struct_type(
        id=point["point_id"],
        vector={
            DENSE_VECTOR_NAME: vector_embedding,
            BM25_SPARSE_VECTOR_NAME: dependencies.document_type(
                text=point["embedding_text"],
                model=BM25_MODEL_NAME,
            ),
        },
        payload=point["payload"],
    )


def upsert_points_from_embeddings(
    points: list[dict],
    embeddings_by_cache_key: dict[int, list[float]],
    q_client,
    qdrant_batch_size: int,
    mock: bool,
    dependencies: QdrantDependencies | None = None,
) -> int:
    dependencies = dependencies or load_qdrant_dependencies()
    pending_points = []
    upserted_count = 0

    def flush_points() -> None:
        nonlocal upserted_count
        if not pending_points:
            return
        q_client.insert_chunks_batch(pending_points)
        upserted_count += len(pending_points)
        pending_points.clear()

    for prepared_count, point in enumerate(points, start=1):
        cache_key = int(point["cache_key"])
        pending_points.append(
            build_qdrant_point(point, embeddings_by_cache_key[cache_key], dependencies)
        )

        if prepared_count % 100 == 0:
            mode_str = "Mock" if mock else "llama.cpp"
            print(f"  [{mode_str}] Prepared embeddings for {prepared_count} chunks...")

        if len(pending_points) >= qdrant_batch_size:
            flush_points()

    flush_points()
    return upserted_count
