from pathlib import Path

from corpus_pipeline.vector_store.ingest_vectors import ChunkEmbeddingCache


def test_embedding_cache_reuses_vector_after_chunk_reordering(tmp_path: Path):
    cache = ChunkEmbeddingCache(tmp_path / "embeddings.jsonl", vector_dim=2)
    vector = cache.set("model", 11158, "same text", "payload-a", [0.1, 0.2])

    reloaded = ChunkEmbeddingCache(cache.path, vector_dim=2)

    assert reloaded.get("model", 14682, "same text", "payload-a") == vector
    assert reloaded.hits == 1
    assert reloaded.misses == 0


def test_embedding_cache_does_not_reuse_changed_content(tmp_path: Path):
    cache = ChunkEmbeddingCache(tmp_path / "embeddings.jsonl", vector_dim=2)
    cache.set("model", 11158, "same text", "payload-a", [0.1, 0.2])

    assert cache.get("model", 14682, "changed text", "payload-a") is None
    assert cache.get("model", 14682, "same text", "payload-b") is None
