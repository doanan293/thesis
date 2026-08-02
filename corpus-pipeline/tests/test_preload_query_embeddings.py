import json
from pathlib import Path

from corpus_pipeline.evaluation.preload_query_embeddings import preload_embeddings
from corpus_pipeline.evaluation.query_embedding_cache import QueryEmbeddingCache


def test_preload_embeddings_incremental_resume(tmp_path: Path):
    eval_file = tmp_path / "test_eval.jsonl"
    eval_rows = [
        {"query_id": "Q1", "query": "Sốt xuất huyết là gì?"},
        {"query_id": "Q2", "query": "Triệu chứng cúm A?"},
    ]
    with eval_file.open("w", encoding="utf-8") as f:
        for row in eval_rows:
            f.write(json.dumps(row) + "\n")

    cache_file = tmp_path / "cache.jsonl"
    cache = QueryEmbeddingCache(cache_file)
    # Pre-populate Q1
    cache.set("qwen3-embedding:0.6b-fp16", "Q1", "Sốt xuất huyết là gì?", [0.1, 0.2])

    def mock_embed_fn(_query):
        return [0.3, 0.4]

    stats = preload_embeddings(
        eval_path=eval_file,
        model_name="qwen3-embedding:0.6b-fp16",
        cache_path=cache_file,
        embed_fn=mock_embed_fn,
        force=False,
    )

    assert stats["total"] == 2
    assert stats["cached"] == 1
    assert stats["processed"] == 1

    # Verify both Q1 and Q2 are now in cache
    reloaded_cache = QueryEmbeddingCache(cache_file)
    assert reloaded_cache.get(
        "qwen3-embedding:0.6b-fp16", "Q1", "Sốt xuất huyết là gì?"
    ) == [0.1, 0.2]
    assert reloaded_cache.get(
        "qwen3-embedding:0.6b-fp16", "Q2", "Triệu chứng cúm A?"
    ) == [0.3, 0.4]


def test_preload_embeddings_force_overwrite(tmp_path: Path):
    eval_file = tmp_path / "test_eval.jsonl"
    eval_file.write_text(
        json.dumps({"query_id": "Q1", "query": "Sốt xuất huyết là gì?"}) + "\n"
    )

    cache_file = tmp_path / "cache.jsonl"
    cache = QueryEmbeddingCache(cache_file)
    cache.set("qwen3-embedding:0.6b-fp16", "Q1", "Sốt xuất huyết là gì?", [0.1, 0.2])

    def mock_embed_fn(_query):
        return [0.9, 0.9]

    stats = preload_embeddings(
        eval_path=eval_file,
        model_name="qwen3-embedding:0.6b-fp16",
        cache_path=cache_file,
        embed_fn=mock_embed_fn,
        force=True,
    )

    assert stats["processed"] == 1
    reloaded_cache = QueryEmbeddingCache(cache_file)
    assert reloaded_cache.get(
        "qwen3-embedding:0.6b-fp16", "Q1", "Sốt xuất huyết là gì?"
    ) == [0.9, 0.9]
