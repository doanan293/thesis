import json
from pathlib import Path

from typer.testing import CliRunner

import seed_pipeline.cli.commands.evaluation as evaluation_command
from seed_pipeline.cli.app import app
from seed_pipeline.evaluation.rejudge_service import RejudgeResult

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


def test_evaluation_build_derives_chunks_from_the_bundle(tmp_path, monkeypatch):
    from seed_pipeline.bundle.export import ExportRequest, export_bundle
    from seed_pipeline.evaluation.build_dataset import EvaluationBuildResult

    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "rag_final_small"
    export_bundle(
        ExportRequest(
            rag_final_dir=fixture,
            glossary_path=fixture / "term_glossary.json",
            mappings_path=fixture / "colloquial_mappings.json",
            output_dir=tmp_path / "bundle",
        )
    )
    captured = {}

    def fake_build(request):
        captured["request"] = request
        return EvaluationBuildResult(
            tmp_path / "patient_queries.json", tmp_path / "evaluation.jsonl", 0, 0
        )

    monkeypatch.setattr(evaluation_command, "build_evaluation_dataset", fake_build)
    result = runner.invoke(
        app,
        [
            "--json",
            "evaluation",
            "build",
            "--sections",
            str(fixture / "sections.jsonl"),
            "--bundle",
            str(tmp_path / "bundle"),
            "--chunks-output",
            str(tmp_path / "chunks.jsonl"),
            "--output-dir",
            str(tmp_path / "evaluation"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["request"].chunks_path == tmp_path / "chunks.jsonl"
    assert len((tmp_path / "chunks.jsonl").read_text("utf-8").splitlines()) == 6
    assert json.loads(result.stdout)["details"]["chunks"] == 6
