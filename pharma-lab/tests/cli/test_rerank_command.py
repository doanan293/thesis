from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import pharma_lab.cli.commands.rerank as rerank_command
from pharma_lab.cli.app import app
from pharma_lab.config.paths import rerank_log_path, run_dir

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
                quota=(),
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


def test_rerank_benchmark_reaches_the_local_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}
    monkeypatch.setattr(
        rerank_command, "LocalRerankBackend", fake_backend(captured, incomplete=False)
    )

    result = runner.invoke(
        app,
        [
            "rerank",
            "--run",
            "experiment",
            "--backend",
            "local",
            "--benchmark",
            "--model",
            "qwen3-reranker:4b-fp16",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["request"].benchmark is True


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


def test_rerank_passes_auto_account_and_max_runs(
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
            "auto",
            "--max-runs",
            "4",
        ],
    )

    assert result.exit_code == 3
    assert captured["request"].kaggle_account == "auto"
    assert captured["request"].max_runs == 4


def test_rerank_rejects_zero_max_runs() -> None:
    result = runner.invoke(
        app, ["rerank", "--run", "experiment", "--backend", "kaggle", "--max-runs", "0"]
    )

    assert result.exit_code == 2


def test_rerank_appends_command_lines_to_the_model_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}
    monkeypatch.setattr(
        rerank_command, "LocalRerankBackend", fake_backend(captured, incomplete=False)
    )

    result = runner.invoke(
        app, ["rerank", "--run", "experiment", "--model", "qwen3-reranker:0.6b-fp16"]
    )

    assert result.exit_code == 0, result.output
    log = rerank_log_path("qwen3-reranker:0.6b-fp16").read_text(encoding="utf-8")
    assert "command backend=local run=experiment" in log
    assert "command status=complete actions=missing_pairs=0" in log


def test_rerank_logs_a_failed_command(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingBackend:
        def run(self, request):
            raise RuntimeError("boom")

    monkeypatch.setattr(rerank_command, "LocalRerankBackend", FailingBackend)

    result = runner.invoke(
        app, ["rerank", "--run", "experiment", "--model", "qwen3-reranker:0.6b-fp16"]
    )

    assert result.exit_code == 1
    log = rerank_log_path("qwen3-reranker:0.6b-fp16").read_text(encoding="utf-8")
    assert "command error=RuntimeError: boom" in log


def test_rerank_passes_the_kernel_to_recover(monkeypatch: pytest.MonkeyPatch) -> None:
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
            "--recover-kernel",
            "doanvanan0209/rerank-5f22fcadeede1072",
        ],
    )

    assert result.exit_code == 3
    assert captured["request"].recover_kernel == "doanvanan0209/rerank-5f22fcadeede1072"
