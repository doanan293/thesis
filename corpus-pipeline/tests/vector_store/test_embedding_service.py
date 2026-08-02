from dataclasses import replace
from pathlib import Path

from corpus_pipeline.vector_store.embedding_service import (
    ChunkEmbeddingRequest,
    EmbeddingStageResult,
    LocalChunkEmbeddingBackend,
    chunk_checkpoint_path,
    chunk_embedding_identity,
)


def test_embedding_request_is_backend_neutral(tmp_path: Path):
    request = ChunkEmbeddingRequest(
        chunks_path=tmp_path / "chunks.jsonl",
        model="qwen3-embedding:0.6b-fp16",
        output_dir=tmp_path / "bundle",
        force=False,
        dry_run=True,
        budget_seconds=60,
        request_timeout_seconds=10,
    )

    result = LocalChunkEmbeddingBackend().run(request)

    assert isinstance(result, EmbeddingStageResult)
    assert result.actions == ("dry-run",)


def test_chunk_identity_is_backend_neutral_and_input_bound(tmp_path: Path):
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text('{"chunk_id":"c1"}\n', encoding="utf-8")
    identity = chunk_embedding_identity(
        ChunkEmbeddingRequest(
            chunks, "qwen3-embedding:0.6b-fp16", tmp_path, False, False, 60, 10
        )
    )
    assert set(identity) == {
        "input_sha256",
        "model",
        "model_sha256",
        "vector_dimension",
    }


def test_chunk_checkpoint_is_scoped_to_input_identity(tmp_path: Path):
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text('{"chunk_id":"c1"}\n', encoding="utf-8")
    request = ChunkEmbeddingRequest(
        chunks, "qwen3-embedding:0.6b-fp16", tmp_path / "bundle", False, False, 60, 10
    )
    first = chunk_checkpoint_path(request)
    assert chunk_checkpoint_path(replace(request, model="bge-m3:567m-fp16")) != first
    chunks.write_text('{"chunk_id":"c2"}\n', encoding="utf-8")
    assert chunk_checkpoint_path(request) != first
