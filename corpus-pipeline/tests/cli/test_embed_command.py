from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

import corpus_pipeline.cli.commands.embed as embed_command
from corpus_pipeline.cli.app import app

runner = CliRunner()


def test_query_embedding_benchmark_rejects_local_backend():
    result = runner.invoke(
        app,
        ["embed", "queries", "--backend", "local", "--benchmark"],
    )

    assert result.exit_code != 0
    assert result.exit_code == 2


def test_corpus_embedding_benchmark_rejects_local_backend():
    result = runner.invoke(
        app,
        ["embed", "chunks", "--backend", "local", "--benchmark"],
    )

    assert result.exit_code != 0
    assert result.exit_code == 2


def test_chunk_embedding_passes_kaggle_account_to_request(monkeypatch):
    captured = {}

    class FakeBackend:
        def run(self, request):
            captured["request"] = request
            return SimpleNamespace(
                cache_path=Path("cache.jsonl"),
                actions=(),
                incomplete=False,
            )

    monkeypatch.setattr(embed_command, "KaggleChunkEmbeddingBackend", FakeBackend)
    result = runner.invoke(
        app,
        ["embed", "chunks", "--backend", "kaggle", "--kaggle-account", "acc2"],
    )

    assert result.exit_code == 0
    assert captured["request"].kaggle_account == "acc2"


def test_query_embedding_passes_kaggle_account_to_request(monkeypatch):
    captured = {}

    class FakeBackend:
        def run(self, request):
            captured["request"] = request
            return SimpleNamespace(
                cache_path=Path("cache.jsonl"),
                subset_sha256=None,
                actions=(),
                incomplete=False,
            )

    monkeypatch.setattr(embed_command, "query_backend", lambda _backend: FakeBackend())
    result = runner.invoke(
        app,
        ["embed", "queries", "--backend", "kaggle", "--kaggle-account", "acc3"],
    )

    assert result.exit_code == 0
    assert captured["request"].kaggle_account == "acc3"
