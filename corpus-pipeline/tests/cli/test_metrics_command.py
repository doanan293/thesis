import json
from pathlib import Path

from typer.testing import CliRunner

import corpus_pipeline.cli.commands.metrics as metrics_command
from corpus_pipeline.cli.app import app
from corpus_pipeline.evaluation.metrics_artifacts import MetricsArtifactResult
from corpus_pipeline.evaluation.metrics_service import MetricsResult

runner = CliRunner()


def fake_metrics_result(*, two_rerank_reports: bool) -> MetricsResult:
    baseline = MetricsArtifactResult(
        artifact_dir=Path("reports/baseline/base"),
        report_path=Path("reports/baseline/base/report.md"),
        results_path=Path("reports/baseline/base/metrics.jsonl"),
        metrics_sha256="base",
    )
    reranked = tuple(
        MetricsArtifactResult(
            artifact_dir=Path(f"reports/rerank/model-a/variant-{index}"),
            report_path=Path(f"reports/rerank/model-a/variant-{index}/report.md"),
            results_path=Path(f"reports/rerank/model-a/variant-{index}/metrics.jsonl"),
            metrics_sha256=f"metrics-{index}",
            model="model-a",
            variant_sha256=f"variant-{index}",
        )
        for index in range(2 if two_rerank_reports else 1)
    )
    return MetricsResult(baseline=baseline, reranked=reranked)


def test_metrics_passes_model_filter_and_lists_all_reports(monkeypatch):
    captured = {}

    def fake_run(request):
        captured["request"] = request
        return fake_metrics_result(two_rerank_reports=True)

    monkeypatch.setattr(metrics_command, "run_metrics", fake_run)
    result = runner.invoke(
        app,
        ["--json", "metrics", "--run", "experiment", "--model", "model-a"],
    )

    assert result.exit_code == 0
    assert captured["request"].model == "model-a"
    assert len(json.loads(result.stdout)["details"]["reranked"]) == 2


def test_metrics_rejects_model_and_variant_together():
    result = runner.invoke(
        app,
        [
            "metrics",
            "--run",
            "experiment",
            "--model",
            "model-a",
            "--variant",
            "abcd",
        ],
    )

    assert result.exit_code != 0
    assert "mutually exclusive" in result.output


def test_metrics_warns_and_ignores_output_dir(monkeypatch):
    captured = {}

    def fake_run(request):
        captured["request"] = request
        return fake_metrics_result(two_rerank_reports=False)

    monkeypatch.setattr(metrics_command, "run_metrics", fake_run)
    result = runner.invoke(
        app,
        [
            "metrics",
            "--run",
            "experiment",
            "--output-dir",
            "/tmp/legacy-reports",
        ],
    )

    assert result.exit_code == 0
    assert "deprecated" in result.stderr
    assert not hasattr(captured["request"], "output_dir")
