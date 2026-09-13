from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

import seed_pipeline.cli.commands.rerank as rerank_command
from seed_pipeline.cli.app import app
from seed_pipeline.config.paths import HEAVY_RETRIEVAL_EVAL_DIR

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
    assert captured["request"].artifact_root == HEAVY_RETRIEVAL_EVAL_DIR / "experiment"
    assert "variant_sha256=variant" in result.stdout


def test_rerank_benchmark_rejects_local_backend():
    result = runner.invoke(
        app,
        [
            "rerank",
            "--run",
            "experiment",
            "--backend",
            "local",
            "--benchmark",
        ],
    )

    assert result.exit_code != 0
    assert result.exit_code == 2


def test_rerank_passes_kaggle_account_to_request(monkeypatch):
    captured = {}

    class FakeBackend:
        def run(self, request):
            captured["request"] = request
            return SimpleNamespace(
                artifact_dir=None,
                variant_sha256="variant",
                subset_sha256=None,
                actions=(),
                incomplete=True,
            )

    monkeypatch.setattr(rerank_command, "KaggleRerankBackend", FakeBackend)
    result = runner.invoke(
        app,
        [
            "rerank",
            "--run",
            "experiment",
            "--backend",
            "kaggle",
            "--kaggle-account",
            "acc2",
        ],
    )

    assert result.exit_code == 3
    assert captured["request"].kaggle_account == "acc2"
