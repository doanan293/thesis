import json
from pathlib import Path

import pytest

from corpus_pipeline.evaluation.artifact_contracts import (
    ArtifactContractError,
    iter_jsonl_objects,
)
from corpus_pipeline.evaluation.query_embedding_artifact import (
    build_query_snapshot,
    finalize_query_cache,
    inspect_query_cache,
)
from corpus_pipeline.evaluation.query_embedding_cache import (
    QueryEmbeddingCache,
    QueryEmbeddingCacheError,
)

MODEL = "qwen3-embedding:8b-fp16"


def write_eval(path: Path) -> Path:
    path.write_text(
        json.dumps({"query_id": "q1", "query": "xin chào"}) + "\n",
        encoding="utf-8",
    )
    return path


def test_build_query_snapshot_preserves_exact_bytes(tmp_path: Path):
    source = write_eval(tmp_path / "eval.jsonl")
    snapshot = build_query_snapshot(source, tmp_path / "snapshot")
    assert snapshot.data_path.read_bytes() == source.read_bytes()
    assert snapshot.row_count == 1
    assert snapshot.manifest.identity["eval_sha256"] == snapshot.manifest.data_sha256


def test_finalize_query_cache_compacts_latest_record(tmp_path: Path):
    eval_path = write_eval(tmp_path / "eval.jsonl")
    cache_path = tmp_path / "cache.jsonl"
    cache = QueryEmbeddingCache(cache_path)
    cache.set(MODEL, "q1", "xin chào", [0.1, 0.2])
    cache.set(MODEL, "q1", "xin chào", [0.3, 0.4])

    bundle = finalize_query_cache(
        eval_path=eval_path,
        cache_path=cache_path,
        model=MODEL,
        gguf_sha256="a" * 64,
        vector_dimension=2,
        output_dir=tmp_path / "final",
    )

    rows = list(iter_jsonl_objects(bundle.data_path))
    assert len(rows) == 1
    assert rows[0]["embedding"] == [0.3, 0.4]
    assert bundle.completion.is_complete


def test_finalize_query_cache_allows_partial_checkpoint(tmp_path: Path):
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(
        json.dumps({"query_id": "q1", "query": "one"})
        + "\n"
        + json.dumps({"query_id": "q2", "query": "two"})
        + "\n",
        encoding="utf-8",
    )
    cache_path = tmp_path / "cache.jsonl"
    QueryEmbeddingCache(cache_path).set(MODEL, "q1", "one", [0.1, 0.2])
    bundle = finalize_query_cache(
        eval_path=eval_path,
        cache_path=cache_path,
        model=MODEL,
        gguf_sha256="a" * 64,
        vector_dimension=2,
        output_dir=tmp_path / "partial",
    )
    assert bundle.completion.total == 2
    assert bundle.completion.complete == 1
    assert bundle.completion.missing == 1


def test_finalize_query_cache_require_complete_rejects_missing(tmp_path: Path):
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(
        json.dumps({"query_id": "q1", "query": "one"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(QueryEmbeddingCacheError, match="missing"):
        finalize_query_cache(
            eval_path=eval_path,
            cache_path=tmp_path / "missing.jsonl",
            model=MODEL,
            gguf_sha256="a" * 64,
            vector_dimension=2,
            output_dir=tmp_path / "final",
            require_complete=True,
        )


def test_inspect_query_cache_rejects_wrong_dimension(tmp_path: Path):
    eval_path = write_eval(tmp_path / "eval.jsonl")
    cache_path = tmp_path / "cache.jsonl"
    QueryEmbeddingCache(cache_path).set(MODEL, "q1", "xin chào", [0.1])
    with pytest.raises(ArtifactContractError, match="dimension"):
        inspect_query_cache(
            eval_path=eval_path,
            cache_path=cache_path,
            model=MODEL,
            vector_dimension=2,
        )
