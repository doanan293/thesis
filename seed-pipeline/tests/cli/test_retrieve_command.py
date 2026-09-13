from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

import seed_pipeline.cli.commands.retrieve as retrieve_command
from seed_pipeline.cli.app import app
from seed_pipeline.config.paths import (
    BACKEND_ENV_FILE,
    HEAVY_RETRIEVAL_EVAL_DIR,
    RETRIEVAL_EVAL_DIR,
)

runner = CliRunner()


def _fake_run(captured: dict):
    def fake_run(request):
        captured["request"] = request
        return SimpleNamespace(
            workspace=SimpleNamespace(
                root=Path("data/retrieval_eval/experiment"),
                identity=SimpleNamespace(release_id="release"),
            ),
            artifact=SimpleNamespace(
                data_path=Path("data/heavy/candidates.jsonl"), query_count=1
            ),
        )

    return fake_run


def test_retrieve_passes_split_roots_and_backend_defaults(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(retrieve_command, "run_retrieval", _fake_run(captured))

    result = runner.invoke(app, ["--json", "retrieve", "--run", "experiment"])

    assert result.exit_code == 0, result.output
    request = captured["request"]
    assert request.run_root == RETRIEVAL_EVAL_DIR / "experiment"
    assert request.artifact_root == HEAVY_RETRIEVAL_EVAL_DIR / "experiment"
    assert request.retriever == "hybrid"
    assert request.rrf_k == 2
    assert request.collection == "formulary"
    assert request.backend_env_file == BACKEND_ENV_FILE
    assert request.query_embeddings is None


def test_retrieve_forwards_bm25_k_values_and_query_cache(monkeypatch, tmp_path):
    captured: dict = {}
    monkeypatch.setattr(retrieve_command, "run_retrieval", _fake_run(captured))

    result = runner.invoke(
        app,
        [
            "retrieve",
            "--run",
            "experiment",
            "--retriever",
            "bm25",
            "--candidate-k",
            "30",
            "--rrf-k",
            "60",
            "--query-embeddings",
            str(tmp_path / "queries.jsonl"),
        ],
    )

    assert result.exit_code == 0, result.output
    request = captured["request"]
    assert (request.retriever, request.candidate_k, request.rrf_k) == ("bm25", 30, 60)
    assert request.query_embeddings == tmp_path / "queries.jsonl"


def test_retrieve_no_longer_accepts_qdrant_options():
    result = runner.invoke(
        app, ["retrieve", "--run", "experiment", "--qdrant-url", "http://x"]
    )

    assert result.exit_code == 2
