import json
from pathlib import Path

import pytest

from seed_pipeline.cache.jsonl_records import (
    CacheRecordError,
    append_record,
    load_records,
)
from seed_pipeline.evaluation.query_embedding_cache import (
    QueryEmbeddingCache,
    QueryEmbeddingCacheError,
)


def test_load_records_returns_sealed_records(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    append_record(path, {"value": 1}, schema="probe-v1")

    records = load_records(path)

    assert [record["value"] for record in records] == [1]


def test_unsealed_records_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    path.write_text(json.dumps({"value": 1}) + "\n", encoding="utf-8")

    with pytest.raises(CacheRecordError, match="missing"):
        load_records(path)


def test_query_cache_rejects_unsealed_records(tmp_path: Path) -> None:
    path = tmp_path / "queries.jsonl"
    path.write_text(
        json.dumps(
            {"model": "m", "query_id": "q", "query_hash": "h", "embedding": [0.1]}
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(QueryEmbeddingCacheError):
        QueryEmbeddingCache(path, vector_dim=1)
