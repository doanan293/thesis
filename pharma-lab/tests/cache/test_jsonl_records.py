import json
from pathlib import Path

import pytest

from pharma_lab.cache.jsonl_records import (
    CacheRecordError,
    append_record,
    iter_records,
    load_records,
)
from pharma_lab.evaluation.query_embedding_cache import (
    QueryEmbeddingCache,
    QueryEmbeddingCacheError,
)


def test_load_records_returns_sealed_records(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    append_record(path, {"value": 1}, schema="probe-v1")

    records = load_records(path)

    assert [record["value"] for record in records] == [1]


def test_iter_records_streams_and_cuts_a_torn_last_line(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    append_record(path, {"value": 1}, schema="probe-v1")
    append_record(path, {"value": 2}, schema="probe-v1")
    complete = path.read_bytes()
    with path.open("ab") as handle:
        handle.write(b'{"value": 3, "cache_sch')

    records = iter_records(path)

    assert next(records)["value"] == 1
    assert [record["value"] for record in records] == [2]
    assert path.read_bytes() == complete


def test_invalid_json_before_the_last_line_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    path.write_bytes(b'{"value": \n')
    append_record(path, {"value": 1}, schema="probe-v1")

    with pytest.raises(CacheRecordError, match="invalid JSON"):
        load_records(path)


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
