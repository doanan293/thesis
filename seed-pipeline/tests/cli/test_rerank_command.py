from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import seed_pipeline.cli.commands.rerank as rerank_command
from seed_pipeline.cli.app import app
from seed_pipeline.config.paths import run_dir

runner = CliRunner()


def fake_backend(captured: dict, *, incomplete: bool):
    class FakeBackend:
        def run(self, request):
            captured["request"] = request
            return SimpleNamespace(
                artifact_dir=None if incomplete else Path("run/rerank/model"),
                variant_sha256="variant",
                subset_sha256=None if incomplete else "subset",
                actions=("missing_pairs=0",),
                incomplete=incomplete,
                benchmark_report=None,
                benchmark_levels=0,
            )

    return FakeBackend


def test_rerank_reads_the_run_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(
        rerank_command, "LocalRerankBackend", fake_backend(captured, incomplete=False)
    )

    result = runner.invoke(
        app, ["rerank", "--run", "experiment", "--model", "qwen3-reranker:0.6b-fp16"]
    )

    assert result.exit_code == 0, result.output
    assert captured["request"].run_root == run_dir("experiment")
    assert captured["request"].model == "qwen3-reranker:0.6b-fp16"
    assert "variant_sha256=variant" in result.stdout


@pytest.mark.parametrize(
    "option", [["--output-dir", "/tmp/cache.jsonl"], ["--candidates", "/tmp/c"]]
)
def test_rerank_rejects_removed_options(option: list[str]) -> None:
    result = runner.invoke(app, ["rerank", "--run", "experiment", *option])

    assert result.exit_code == 2


def test_rerank_benchmark_rejects_local_backend() -> None:
    result = runner.invoke(
        app, ["rerank", "--run", "experiment", "--backend", "local", "--benchmark"]
    )

    assert result.exit_code == 2


def test_rerank_passes_kaggle_account_to_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}
    monkeypatch.setattr(
        rerank_command, "KaggleRerankBackend", fake_backend(captured, incomplete=True)
    )

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
