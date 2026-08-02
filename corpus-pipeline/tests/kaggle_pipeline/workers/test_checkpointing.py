from pathlib import Path

import pytest

from corpus_pipeline.integrations.kaggle.workers.checkpointing import (
    CheckpointIdentityError,
    CheckpointStore,
)


def test_checkpoint_store_resumes_only_missing_records(tmp_path: Path):
    path = tmp_path / "records.jsonl"
    store = CheckpointStore.open(path, {"job": "a"}, lambda row: row["id"])
    store.append_batch([{"id": "one", "value": 1}])
    resumed = CheckpointStore.open(path, {"job": "a"}, lambda row: row["id"])
    assert resumed.missing([{"id": "one"}, {"id": "two"}]) == [{"id": "two"}]


def test_checkpoint_store_rejects_identity_mismatch(tmp_path: Path):
    path = tmp_path / "records.jsonl"
    CheckpointStore.open(path, {"job": "a"}, lambda row: row["id"]).append_batch(
        [{"id": "one"}]
    )
    with pytest.raises(CheckpointIdentityError, match="identity"):
        CheckpointStore.open(path, {"job": "b"}, lambda row: row["id"])
