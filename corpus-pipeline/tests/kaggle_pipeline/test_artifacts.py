import json
from pathlib import Path

import pytest

from corpus_pipeline.integrations.kaggle.artifacts import (
    ArtifactContractError,
    load_cloud_artifact,
    promote_complete_artifact,
)
from corpus_pipeline.integrations.kaggle.models import JobIdentity, StageName


def make_identity():
    return JobIdentity.create(
        stage=StageName.QUERY_EMBED,
        contract_version=1,
        model="qwen3-embedding:0.6b-fp16",
        model_sha256="a" * 64,
        input_sha256="b" * 64,
        runtime_parameters={"batch_size": 32},
    )


def write_artifact(tmp_path: Path, *, total: int, complete: int, missing: int):
    data = tmp_path / "query_embeddings.jsonl"
    data.write_text('{"query_id":"q1"}\n' * complete, encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "query_embedding_cache",
                "created_at": "2026-01-01T00:00:00+00:00",
                "data_filename": data.name,
                "data_sha256": __import__("hashlib")
                .sha256(data.read_bytes())
                .hexdigest(),
                "record_count": complete,
                "identity": {
                    "stage": "query-embed",
                    "contract_version": 1,
                    "job_sha256": make_identity().sha256,
                    "model": "qwen3-embedding:0.6b-fp16",
                    "model_sha256": "a" * 64,
                    "input_sha256": "b" * 64,
                },
                "total": total,
                "complete": complete,
                "missing": missing,
            }
        ),
        encoding="utf-8",
    )
    return data, manifest


def test_incomplete_artifact_cannot_replace_complete_cache(tmp_path):
    data, manifest = write_artifact(tmp_path, total=2, complete=1, missing=1)
    destination = tmp_path / "cache.jsonl"
    destination.write_text('{"old":true}\n', encoding="utf-8")
    artifact = load_cloud_artifact(data, manifest, make_identity(), allow_partial=True)
    with pytest.raises(ArtifactContractError, match="incomplete"):
        promote_complete_artifact(artifact, destination)
    assert destination.read_text(encoding="utf-8") == '{"old":true}\n'


def test_complete_artifact_promotes_data_and_manifest(tmp_path):
    data, manifest = write_artifact(tmp_path, total=1, complete=1, missing=0)
    destination = tmp_path / "cache.jsonl"
    artifact = load_cloud_artifact(data, manifest, make_identity())
    assert promote_complete_artifact(artifact, destination) == destination
    assert destination.read_text(encoding="utf-8") == data.read_text(encoding="utf-8")
    assert (
        json.loads(destination.with_name("manifest.json").read_text())["complete"] == 1
    )
