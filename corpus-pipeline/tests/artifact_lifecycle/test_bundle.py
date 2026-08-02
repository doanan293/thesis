import json
from pathlib import Path

import pytest

from corpus_pipeline.artifacts.bundle import (
    ArtifactCompletion,
    ArtifactContractError,
    load_bundle,
    publish_bundle,
)


def write_jsonl(path: Path, count: int) -> None:
    path.write_text(
        "".join(json.dumps({"id": index}) + "\n" for index in range(count)),
        encoding="utf-8",
    )


def test_load_bundle_validates_complete_manifest(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    write_jsonl(source, 2)
    bundle = publish_bundle(
        tmp_path / "bundle",
        artifact_type="chunk_embeddings",
        source_path=source,
        identity={"model": "test"},
        total=2,
        complete=2,
    )

    loaded = load_bundle(
        bundle.root, expected_type="chunk_embeddings", require_complete=True
    )

    assert loaded.data_path.is_file()
    assert loaded.completion == ArtifactCompletion(2, 2, 0)


def test_load_bundle_rejects_incomplete_input(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    write_jsonl(source, 1)
    bundle = publish_bundle(
        tmp_path / "bundle",
        artifact_type="query_embeddings",
        source_path=source,
        identity={"model": "test"},
        total=2,
        complete=1,
    )

    with pytest.raises(ArtifactContractError, match="incomplete"):
        load_bundle(
            bundle.root, expected_type="query_embeddings", require_complete=True
        )


def test_load_bundle_rejects_type_mismatch(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    write_jsonl(source, 1)
    bundle = publish_bundle(
        tmp_path / "bundle",
        artifact_type="rerank_scores",
        source_path=source,
        identity={},
        total=1,
        complete=1,
    )

    with pytest.raises(ArtifactContractError, match="type mismatch"):
        load_bundle(
            bundle.root, expected_type="chunk_embeddings", require_complete=True
        )
