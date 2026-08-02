from pathlib import Path

from typer.testing import CliRunner

from corpus_pipeline.cli.app import app
from corpus_pipeline.cli.commands import build as build_command
from corpus_pipeline.cli.commands import evaluation as evaluation_command
from corpus_pipeline.cli.commands import validate as validate_command
from corpus_pipeline.evaluation.build_dataset import EvaluationBuildResult
from corpus_pipeline.orchestration.build_corpus import BuildConfig, BuildResult
from corpus_pipeline.orchestration.validation_service import ValidationResult

runner = CliRunner()


def test_build_handler_forwards_explicit_final_dir(monkeypatch, tmp_path: Path):
    observed: dict[str, BuildConfig] = {}
    config = BuildConfig(
        pdf_path=tmp_path / "pdf",
        snapshot_archive=tmp_path / "archive",
        snapshot_manifest=tmp_path / "manifest",
        curated_tables_path=tmp_path / "corpus_pipeline.corpus.tables",
        table_overrides_path=tmp_path / "overrides",
        mappings_path=tmp_path / "mappings",
        glossary_path=tmp_path / "glossary",
    )
    monkeypatch.setattr(build_command, "default_config", lambda: config)

    def fake_run_build(selected):
        observed["config"] = selected
        return BuildResult("id", selected.final_dir)

    monkeypatch.setattr(build_command, "run_build", fake_run_build)

    result = runner.invoke(app, ["build", "--final-dir", str(tmp_path / "final")])

    assert result.exit_code == 0
    assert observed["config"].final_dir == tmp_path / "final"


def test_evaluation_build_is_one_logical_command(monkeypatch, tmp_path: Path):
    def fake_build(request):
        request.output_dir.mkdir(parents=True, exist_ok=True)
        patient_path = request.output_dir / "patient_queries.json"
        evaluation_path = request.output_dir / "section_retrieval_eval.jsonl"
        patient_path.write_text("[]\n", encoding="utf-8")
        evaluation_path.write_text("{}\n", encoding="utf-8")
        return EvaluationBuildResult(patient_path, evaluation_path, 0, 1)

    monkeypatch.setattr(evaluation_command, "build_evaluation_dataset", fake_build)
    result = runner.invoke(app, ["evaluation", "build", "--output-dir", str(tmp_path)])

    assert result.exit_code == 0
    assert "patient_queries.json" in result.stdout
    assert "section_retrieval_eval.jsonl" in result.stdout


def test_validate_returns_failure_exit_code(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        validate_command,
        "run_validation",
        lambda request: ValidationResult(
            output_json=request.output_json,
            output_markdown=None,
            ok=False,
            errors=("bad corpus",),
            metrics={},
        ),
    )
    result = runner.invoke(app, ["validate", "--output-dir", str(tmp_path)])

    assert result.exit_code == 1
    assert "status=failed" in result.stdout
