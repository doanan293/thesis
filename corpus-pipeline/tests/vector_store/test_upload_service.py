import json
from pathlib import Path

import pytest

from corpus_pipeline.artifacts.bundle import publish_bundle
from corpus_pipeline.vector_store.upload_service import (
    UploadVectorsRequest,
    upload_vectors,
)


def test_upload_rejects_incomplete_embedding_bundle(tmp_path: Path):
    source = tmp_path / "embeddings.jsonl"
    source.write_text(json.dumps({"chunk_key": 1, "embedding": [0.1]}) + "\n")
    bundle = publish_bundle(
        tmp_path / "bundle",
        artifact_type="chunk_embeddings",
        source_path=source,
        identity={"model": "qwen3-embedding:0.6b-fp16"},
        total=2,
        complete=1,
    )

    with pytest.raises(RuntimeError, match="incomplete"):
        upload_vectors(
            UploadVectorsRequest(
                chunks_path=tmp_path / "chunks.jsonl",
                embeddings_dir=bundle.root,
                model="qwen3-embedding:0.6b-fp16",
                qdrant_url="http://localhost:6333",
            )
        )


def test_upload_rejects_bundle_for_different_corpus_before_qdrant(tmp_path: Path):
    source = tmp_path / "embeddings.jsonl"
    source.write_text(json.dumps({"chunk_key": 1, "embedding": [0.1]}) + "\n")
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text('{"chunk_id":"c1"}\n', encoding="utf-8")
    bundle = publish_bundle(
        tmp_path / "bundle",
        artifact_type="chunk_embeddings",
        source_path=source,
        identity={
            "input_sha256": "0" * 64,
            "model": "qwen3-embedding:0.6b-fp16",
            "model_sha256": "1" * 64,
            "vector_dimension": 1024,
        },
        total=1,
        complete=1,
    )

    with pytest.raises(RuntimeError, match="identity"):
        upload_vectors(
            UploadVectorsRequest(
                chunks_path=chunks,
                embeddings_dir=bundle.root,
                model="qwen3-embedding:0.6b-fp16",
                qdrant_url="http://localhost:6333",
            )
        )
