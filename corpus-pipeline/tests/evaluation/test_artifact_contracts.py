import json
from pathlib import Path

import pytest

from corpus_pipeline.evaluation.artifact_contracts import (
    ArtifactContractError,
    ArtifactManifest,
    atomic_promote,
    iter_jsonl_objects,
    sha256_file,
    validate_data_file,
)


def test_manifest_validates_exact_file_hash(tmp_path: Path):
    data = tmp_path / "rows.jsonl"
    data.write_text('{"id":"q1"}\n', encoding="utf-8")
    manifest = ArtifactManifest.create(
        artifact_type="query_snapshot",
        data_path=data,
        record_count=1,
        identity={"eval_sha256": sha256_file(data)},
    )
    validate_data_file(data, manifest)
    data.write_text('{"id":"changed"}\n', encoding="utf-8")
    with pytest.raises(ArtifactContractError, match="checksum mismatch"):
        validate_data_file(data, manifest)


def test_iter_jsonl_reports_line_number(tmp_path: Path):
    data = tmp_path / "bad.jsonl"
    data.write_text('{"ok":true}\nnot-json\n', encoding="utf-8")
    with pytest.raises(ArtifactContractError, match=r"bad.jsonl:2"):
        list(iter_jsonl_objects(data))


def test_iter_jsonl_requires_object_records(tmp_path: Path):
    data = tmp_path / "array.jsonl"
    data.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(ArtifactContractError, match="JSON object"):
        list(iter_jsonl_objects(data))


def test_atomic_promote_replaces_only_after_validation(tmp_path: Path):
    incoming = tmp_path / "incoming.jsonl"
    target = tmp_path / "cache" / "final.jsonl"
    incoming.write_text('{"id":"q1"}\n', encoding="utf-8")
    atomic_promote(incoming, target)
    assert target.read_text(encoding="utf-8") == '{"id":"q1"}\n'


def test_manifest_round_trips_as_json(tmp_path: Path):
    data = tmp_path / "rows.jsonl"
    data.write_text(json.dumps({"id": "q1"}) + "\n", encoding="utf-8")
    manifest = ArtifactManifest.create(
        artifact_type="query_snapshot",
        data_path=data,
        record_count=1,
        identity={"eval_sha256": sha256_file(data)},
    )
    payload = manifest.to_dict()
    assert payload["schema_version"] == 1
    assert payload["record_count"] == 1
