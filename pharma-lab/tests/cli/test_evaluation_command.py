import json
from pathlib import Path

from typer.testing import CliRunner

import pharma_lab.cli.commands.evaluation as evaluation_command
from pharma_lab.cli.app import app

runner = CliRunner()


def test_evaluation_build_derives_chunks_from_the_bundle(tmp_path, monkeypatch):
    from pharma_lab.bundle.export import ExportRequest, export_bundle
    from pharma_lab.evaluation.build_dataset import EvaluationBuildResult

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

    def fake_judgments(evaluation_path, bundle_dir):
        captured["judgments"] = (evaluation_path, bundle_dir)
        return tmp_path / "evaluation.judgments.jsonl"

    monkeypatch.setattr(evaluation_command, "build_evaluation_dataset", fake_build)
    monkeypatch.setattr(
        evaluation_command, "build_evaluation_judgments", fake_judgments
    )
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
    details = json.loads(result.stdout)["details"]
    assert details["chunks"] == 6
    assert captured["judgments"] == (tmp_path / "evaluation.jsonl", tmp_path / "bundle")
    assert details["judgments"] == str(tmp_path / "evaluation.judgments.jsonl")


def test_evaluation_judgments_rebuild_only_the_judgments(tmp_path, monkeypatch):
    captured = {}

    def fake_judgments(evaluation_path, bundle_dir):
        captured["args"] = (evaluation_path, bundle_dir)
        return tmp_path / "gold.judgments.jsonl"

    monkeypatch.setattr(
        evaluation_command, "build_evaluation_judgments", fake_judgments
    )

    result = runner.invoke(
        app,
        [
            "--json",
            "evaluation",
            "judgments",
            "--evaluation",
            str(tmp_path / "gold.jsonl"),
            "--bundle",
            str(tmp_path / "bundle"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["args"] == (tmp_path / "gold.jsonl", tmp_path / "bundle")
    assert json.loads(result.stdout)["artifact"] == str(
        tmp_path / "gold.judgments.jsonl"
    )
