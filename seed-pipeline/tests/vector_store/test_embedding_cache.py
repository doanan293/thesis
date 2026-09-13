from __future__ import annotations

import pytest

from seed_pipeline.vector_store import embedding_core
from seed_pipeline.vector_store.ingest_vectors import (
    ChunkEmbeddingCache,
    EmbeddingCacheError,
)


def test_ingest_vectors_preserves_qdrant_point_namespace_export():
    from seed_pipeline.vector_store import ingest_vectors

    assert (
        ingest_vectors.QDRANT_POINT_ID_NAMESPACE
        == embedding_core.QDRANT_POINT_ID_NAMESPACE
    )


def test_subset_sha256_reuses_content_match_after_chunk_reordering(tmp_path):
    cache = ChunkEmbeddingCache(tmp_path / "embeddings.jsonl", vector_dim=2)
    cache.set("embedding-model", 1, "same text", "same-payload", [0.25, 0.75])
    exact_points = [
        {
            "cache_key": 1,
            "embedding_text": "same text",
            "payload_hash": "same-payload",
        }
    ]
    reordered_points = [
        {
            "cache_key": 99,
            "embedding_text": "same text",
            "payload_hash": "same-payload",
        }
    ]

    assert cache.subset_sha256(
        reordered_points, "embedding-model"
    ) == cache.subset_sha256(exact_points, "embedding-model")


def test_subset_sha256_still_rejects_missing_content(tmp_path):
    cache = ChunkEmbeddingCache(tmp_path / "embeddings.jsonl", vector_dim=2)
    cache.set("embedding-model", 1, "original text", "payload", [0.25, 0.75])
    changed_points = [
        {
            "cache_key": 1,
            "embedding_text": "changed text",
            "payload_hash": "payload",
        }
    ]

    with pytest.raises(EmbeddingCacheError, match="Embedding cache is incomplete"):
        cache.subset_sha256(changed_points, "embedding-model")
