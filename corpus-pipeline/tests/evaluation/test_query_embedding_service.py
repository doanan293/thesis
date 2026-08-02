from dataclasses import replace
from pathlib import Path

from corpus_pipeline.evaluation.query_embedding_service import (
    LocalQueryEmbeddingBackend,
    QueryEmbeddingRequest,
    query_checkpoint_path,
    query_embedding_identity,
)


def test_query_embedding_dry_run_is_backend_neutral(tmp_path: Path):
    result = LocalQueryEmbeddingBackend().run(
        QueryEmbeddingRequest(
            evaluation_path=tmp_path / "eval.jsonl",
            model="qwen3-embedding:0.6b-fp16",
            output_dir=tmp_path / "bundle",
            force=False,
            dry_run=True,
            budget_seconds=60,
            request_timeout_seconds=10,
        )
    )

    assert result.actions == ("dry-run",)


def test_query_identity_is_backend_neutral_and_input_bound(tmp_path: Path):
    evaluation = tmp_path / "eval.jsonl"
    evaluation.write_text('{"query_id":"q1","query":"one"}\n', encoding="utf-8")
    identity = query_embedding_identity(
        QueryEmbeddingRequest(
            evaluation, "qwen3-embedding:0.6b-fp16", tmp_path, False, False, 60, 10
        )
    )
    assert identity["model"] == "qwen3-embedding:0.6b-fp16"
    assert identity["eval_sha256"]
    assert identity["vector_dimension"] == 1024
    assert identity["logical_sha256"]
    assert "backend" not in identity


def test_query_checkpoint_is_scoped_to_full_stage_identity(tmp_path: Path):
    evaluation = tmp_path / "eval.jsonl"
    evaluation.write_text('{"query_id":"q1","query":"one"}\n', encoding="utf-8")
    request = QueryEmbeddingRequest(
        evaluation,
        "qwen3-embedding:0.6b-fp16",
        tmp_path / "bundle",
        False,
        False,
        60,
        10,
    )
    first = query_checkpoint_path(request)

    assert query_checkpoint_path(replace(request, model="bge-m3:567m-fp16")) != first
    evaluation.write_text('{"query_id":"q2","query":"two"}\n', encoding="utf-8")
    assert query_checkpoint_path(request) != first


def test_local_query_embedding_reuses_checkpoint_on_second_run(
    tmp_path: Path, monkeypatch
):
    evaluation = tmp_path / "eval.jsonl"
    evaluation.write_text('{"query_id":"q1","query":"one"}\n', encoding="utf-8")
    calls = []

    class FakeClient:
        def __init__(self, _endpoint, timeout):
            assert timeout == 10

        def embed(self, texts, _model, dimension):
            calls.extend(texts)
            return [[0.1] * dimension for _ in texts]

    monkeypatch.setattr(
        "corpus_pipeline.evaluation.query_embedding_service.LlamaCppComposeManager",
        lambda _path: object(),
    )
    monkeypatch.setattr(
        "corpus_pipeline.evaluation.query_embedding_service.resolve_server",
        lambda *_args: ["http://model"],
    )
    monkeypatch.setattr(
        "corpus_pipeline.evaluation.query_embedding_service.LlamaCppClient", FakeClient
    )
    request = QueryEmbeddingRequest(
        evaluation,
        "qwen3-embedding:0.6b-fp16",
        tmp_path / "bundle",
        False,
        False,
        60,
        10,
    )
    backend = LocalQueryEmbeddingBackend()

    backend.run(request)
    backend.run(request)

    evaluation.write_text('{"query_id":"q2","query":"two"}\n', encoding="utf-8")
    backend.run(request)

    assert calls == ["one", "two"]
