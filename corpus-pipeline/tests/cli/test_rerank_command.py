from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

import corpus_pipeline.cli.commands.rerank as rerank_command
from corpus_pipeline.cli.app import app

runner = CliRunner()


def test_rerank_warns_and_uses_canonical_artifact(monkeypatch):
    captured = {}

    class FakeBackend:
        def run(self, request):
            captured["request"] = request
            return SimpleNamespace(
                artifact_dir=Path("run/rerank/model/variant"),
                variant_sha256="variant",
                subset_sha256="subset",
                actions=("reuse=complete variant",),
                incomplete=False,
            )

    monkeypatch.setattr(rerank_command, "LocalRerankBackend", FakeBackend)
    result = runner.invoke(
        app,
        [
            "rerank",
            "--run",
            "experiment",
            "--model",
            "qwen3-reranker:0.6b-fp16",
            "--output-dir",
            "/tmp/legacy-cache.jsonl",
        ],
    )

    assert result.exit_code == 0
    assert "deprecated" in result.stderr
    assert captured["request"].model == "qwen3-reranker:0.6b-fp16"
    assert "variant_sha256=variant" in result.stdout
