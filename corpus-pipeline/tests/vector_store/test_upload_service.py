from pathlib import Path
from types import SimpleNamespace

import pytest

from corpus_pipeline.vector_store import upload_service
from corpus_pipeline.vector_store.upload_service import UploadVectorsRequest


class FakeUploadHelper:
    instances = []
    initial_exists = False
    initial_count = 0

    def __init__(self, **kwargs):
        self.collection_name = kwargs["collection_name"]
        self.calls = []
        self.exists = self.__class__.initial_exists
        self.count = self.__class__.initial_count
        self.expected_count_after_upload = None
        self.__class__.instances.append(self)

    def collection_exists(self):
        self.calls.append("collection_exists")
        return self.exists

    def init_collection(self):
        self.calls.append("init_collection")
        self.exists = True

    def clear_collection(self):
        self.calls.append("clear_collection")
        self.count = 0

    def point_count(self):
        self.calls.append("point_count")
        return self.count

    def switch_alias(self, alias_name, *, delete_legacy_collection=False):
        self.calls.append(("switch_alias", delete_legacy_collection))


class FakeCache:
    def __init__(self, *args, **kwargs):
        pass

    def subset_sha256(self, points, model):
        return "embedding-digest"


def run_upload(monkeypatch, tmp_path: Path, *, existing, initial_count, final_count):
    FakeUploadHelper.instances.clear()
    embeddings_path = tmp_path / "embeddings.jsonl"
    embeddings_path.write_text("cache\n")
    points = [{"cache_key": 1}, {"cache_key": 2}]
    FakeUploadHelper.initial_exists = existing
    FakeUploadHelper.initial_count = initial_count

    monkeypatch.setattr(
        upload_service,
        "require_model",
        lambda model: SimpleNamespace(vector_dimension=3, sha256="model", slug="model"),
    )
    monkeypatch.setattr(
        upload_service, "read_normalized_input_points", lambda path: points
    )
    monkeypatch.setattr(upload_service, "ChunkEmbeddingCache", FakeCache)
    monkeypatch.setattr(
        upload_service,
        "summarize_embedding_cache_completion",
        lambda points, model, cache: SimpleNamespace(is_complete=True, missing=0),
    )
    monkeypatch.setattr(
        upload_service,
        "collect_complete_cached_embeddings",
        lambda points, model, cache: {1: [0.1], 2: [0.2]},
    )
    monkeypatch.setattr(upload_service, "sha256_file", lambda path: "chunk-digest")
    monkeypatch.setattr(
        upload_service,
        "load_qdrant_dependencies",
        lambda: SimpleNamespace(helper_type=FakeUploadHelper),
    )

    def fake_upload(points, embeddings, q_client, *args):
        q_client.calls.append("upload")
        q_client.count = final_count
        return len(points)

    monkeypatch.setattr(upload_service, "upsert_points_from_embeddings", fake_upload)
    result = upload_service.upload_vectors(
        UploadVectorsRequest(
            chunks_path=tmp_path / "chunks.jsonl",
            embeddings_path=embeddings_path,
            model="model",
            qdrant_url="http://localhost:6333",
        )
    )
    helper = FakeUploadHelper.instances[-1]
    return result, helper


def test_missing_target_is_initialized_uploaded_verified_then_migrated(
    monkeypatch, tmp_path
):
    result, helper = run_upload(
        monkeypatch, tmp_path, existing=False, initial_count=0, final_count=2
    )

    assert helper.calls == [
        "collection_exists",
        "init_collection",
        "upload",
        "point_count",
        ("switch_alias", True),
    ]
    assert result.point_count == 2
    assert result.reused is False


def test_complete_target_is_reused_without_upload(monkeypatch, tmp_path):
    result, helper = run_upload(
        monkeypatch, tmp_path, existing=True, initial_count=2, final_count=2
    )

    assert helper.calls == [
        "collection_exists",
        "point_count",
        ("switch_alias", True),
    ]
    assert result.point_count == 0
    assert result.reused is True


def test_incomplete_target_is_recreated_and_uploaded(monkeypatch, tmp_path):
    result, helper = run_upload(
        monkeypatch, tmp_path, existing=True, initial_count=0, final_count=2
    )

    assert helper.calls == [
        "collection_exists",
        "point_count",
        "clear_collection",
        "upload",
        "point_count",
        ("switch_alias", True),
    ]
    assert result.reused is False


def test_count_mismatch_stops_before_legacy_migration(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError, match="point count mismatch"):
        run_upload(
            monkeypatch, tmp_path, existing=False, initial_count=0, final_count=1
        )

    helper = FakeUploadHelper.instances[-1]
    assert not any(
        isinstance(call, tuple) and call[0] == "switch_alias" for call in helper.calls
    )
