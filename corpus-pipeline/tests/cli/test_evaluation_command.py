import json
from pathlib import Path

from typer.testing import CliRunner

import corpus_pipeline.cli.commands.evaluation as evaluation_command
from corpus_pipeline.cli.app import app
from corpus_pipeline.evaluation.rejudge_service import RejudgeResult

runner = CliRunner()


def test_rejudge_current_defaults_to_dry_run(monkeypatch):
    captured = {}

    def fake_run(request):
        captured["request"] = request
        return RejudgeResult(
            "old",
            "new",
            {"changed": 1},
            (),
            False,
        )

    monkeypatch.setattr(evaluation_command, "run_rejudging", fake_run)
    result = runner.invoke(
        app,
        [
            "--json",
            "evaluation",
            "rejudge-current",
            "--dense-run",
            "dense-qwen4b-k30",
            "--hybrid-run",
            "hybrid-qwen4b-p50-k30-rrf2",
        ],
    )

    assert result.exit_code == 0
    assert captured["request"].apply is False
    payload = json.loads(result.stdout)
    assert payload["details"]["applied"] is False
    assert payload["details"]["new_evaluation_sha256"] == "new"


def test_rejudge_current_apply_forwards_apply_flag(monkeypatch):
    captured = {}

    def fake_run(request):
        captured["request"] = request
        return RejudgeResult("old", "new", {}, (Path("report.md"),), True)

    monkeypatch.setattr(evaluation_command, "run_rejudging", fake_run)
    result = runner.invoke(
        app,
        [
            "evaluation",
            "rejudge-current",
            "--dense-run",
            "dense-qwen4b-k30",
            "--hybrid-run",
            "hybrid-qwen4b-p50-k30-rrf2",
            "--apply",
        ],
    )

    assert result.exit_code == 0
    assert captured["request"].apply is True
    assert captured["request"].evaluation_path.name == "section_retrieval_eval.jsonl"
