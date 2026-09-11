from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

import corpus_pipeline.cli.commands.retrieve as retrieve_command
from corpus_pipeline.cli.app import app
from corpus_pipeline.config.paths import (
    HEAVY_RETRIEVAL_EVAL_DIR,
    RETRIEVAL_EVAL_DIR,
)

runner = CliRunner()


def test_retrieve_passes_split_metadata_and_heavy_roots(monkeypatch):
    captured = {}

    def fake_run(request):
        captured["request"] = request
        return SimpleNamespace(
            workspace=SimpleNamespace(root=Path("data/retrieval_eval/experiment")),
            artifact=SimpleNamespace(
                data_path=Path("data/heavy/candidates.jsonl"), query_count=1
            ),
        )

    monkeypatch.setattr(retrieve_command, "run_retrieval", fake_run)
    result = runner.invoke(
        app,
        [
            "--json",
            "retrieve",
            "--run",
            "experiment",
            "--retriever",
            "bm25",
            "--model",
            "embeddinggemma:300m",
        ],
    )

    assert result.exit_code == 0
    assert captured["request"].run_root == RETRIEVAL_EVAL_DIR / "experiment"
    assert captured["request"].artifact_root == HEAVY_RETRIEVAL_EVAL_DIR / "experiment"


def test_retrieve_defaults_to_qdrant_rrf_k(monkeypatch):
    captured = {}

    def fake_run(request):
        captured["request"] = request
        return SimpleNamespace(
            workspace=SimpleNamespace(root=Path("data/retrieval_eval/experiment")),
            artifact=SimpleNamespace(
                data_path=Path("data/heavy/candidates.jsonl"), query_count=1
            ),
        )

    monkeypatch.setattr(retrieve_command, "run_retrieval", fake_run)
    result = runner.invoke(
        app, ["retrieve", "--run", "experiment", "--retriever", "bm25"]
    )

    assert result.exit_code == 0
    assert captured["request"].rrf_k == 2


def test_retrieve_forwards_explicit_rrf_k(monkeypatch):
    captured = {}

    def fake_run(request):
        captured["request"] = request
        return SimpleNamespace(
            workspace=SimpleNamespace(root=Path("data/retrieval_eval/experiment")),
            artifact=SimpleNamespace(
                data_path=Path("data/heavy/candidates.jsonl"), query_count=1
            ),
        )

    monkeypatch.setattr(retrieve_command, "run_retrieval", fake_run)
    result = runner.invoke(
        app,
        [
            "retrieve",
            "--run",
            "experiment",
            "--retriever",
            "bm25",
            "--rrf-k",
            "60",
        ],
    )

    assert result.exit_code == 0
    assert captured["request"].rrf_k == 60
