import json
import time
from collections.abc import Callable
from pathlib import Path

from seed_pipeline.config.paths import PROCESSED_EVALUATION_DIR
from seed_pipeline.evaluation.query_embedding_cache import (
    QueryEmbeddingCache,
    default_query_embedding_cache_path,
)

DEFAULT_EVAL_JSONL = PROCESSED_EVALUATION_DIR / "section_retrieval_eval.jsonl"


def preload_embeddings(
    eval_path: Path,
    model_name: str,
    cache_path: Path | None = None,
    embed_fn: Callable[[str], list[float]] | None = None,
    embed_batch_fn: Callable[[list[str]], list[list[float]]] | None = None,
    batch_size: int = 32,
    force: bool = False,
    vector_dim: int | None = None,
    model_sha256: str = "",
) -> dict:
    eval_path = Path(eval_path)
    if cache_path is None:
        cache_path = default_query_embedding_cache_path(eval_path, model_name)

    cache = QueryEmbeddingCache(
        cache_path,
        vector_dim=vector_dim,
        model_sha256=model_sha256,
    )

    rows = []
    with eval_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    total = len(rows)
    cached = 0
    processed = 0

    queries_to_embed = []
    for row in rows:
        qid = str(row.get("query_id") or "")
        qtext = str(row.get("query") or "")
        if not force and cache.get(model_name, qid, qtext) is not None:
            cached += 1
        else:
            queries_to_embed.append((qid, qtext))

    if not queries_to_embed:
        return {
            "total": total,
            "cached": cached,
            "processed": 0,
            "cache_path": str(cache_path),
        }

    if embed_fn is None and embed_batch_fn is None:
        raise ValueError(
            "embed_fn or embed_batch_fn is required when embeddings are missing"
        )

    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    start_time = time.time()
    for start in range(0, len(queries_to_embed), batch_size):
        batch = queries_to_embed[start : start + batch_size]
        if embed_batch_fn is not None:
            vectors = embed_batch_fn([text for _, text in batch])
        elif embed_fn is not None:
            vectors = [embed_fn(text) for _, text in batch]
        else:
            raise ValueError("embed_fn or embed_batch_fn is required")
        if len(vectors) != len(batch):
            raise ValueError("embedding batch returned an unexpected number of vectors")
        for (qid, qtext), embedding in zip(batch, vectors, strict=True):
            cache.set(model_name, qid, qtext, embedding)
            processed += 1

    elapsed = time.time() - start_time

    return {
        "total": total,
        "cached": cached,
        "processed": processed,
        "elapsed": elapsed,
        "cache_path": str(cache_path),
    }
