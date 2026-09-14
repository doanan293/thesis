import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import seed_pipeline.cli.commands.metrics as metrics_command
from seed_pipeline.cli.app import app
from seed_pipeline.config.paths import run_dir
from seed_pipeline.evaluation.metrics_artifacts import MetricsArtifactResult
from seed_pipeline.evaluation.metrics_service import MetricsRequest, MetricsResult

runner = CliRunner()


def fake_metrics_result() -> MetricsResult:
    baseline = Path("reports/baseline/top10-window3")
    rerank = Path("reports/rerank/model_a/top10-window3")
    return MetricsResult(
        baseline=MetricsArtifactResult(
            baseline, baseline / "report.md", baseline / "metrics.jsonl", "base"
        ),
        reranked=(
            MetricsArtifactResult(
                rerank,
                rerank / "report.md",
                rerank / "metrics.jsonl",
                "reranked",
                model="model-a",
                variant_sha256="variant",
            ),
        ),
    )


def capture_request(
    monkeypatch: pytest.MonkeyPatch, captured: dict[str, MetricsRequest]
) -> None:
    def fake_run(request: MetricsRequest) -> MetricsResult:
        captured["request"] = request
        return fake_metrics_result()

    monkeypatch.setattr(metrics_command, "run_metrics", fake_run)


def test_metrics_reads_the_run_tree_and_lists_reports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, MetricsRequest] = {}
    capture_request(monkeypatch, captured)

    result = runner.invoke(
        app, ["--json", "metrics", "--run", "experiment", "--model", "model-a"]
    )

    assert result.exit_code == 0, result.output
    request = captured["request"]
    assert (request.run_root, request.model, request.force) == (
        run_dir("experiment"),
        "model-a",
        False,
    )
    assert len(json.loads(result.stdout)["details"]["reranked"]) == 1


def test_metrics_force_is_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, MetricsRequest] = {}
    capture_request(monkeypatch, captured)

    result = runner.invoke(app, ["metrics", "--run", "experiment", "--force"])

    assert result.exit_code == 0, result.output
    assert captured["request"].force is True


@pytest.mark.parametrize(
    "option", [["--variant", "abcd"], ["--output-dir", "/tmp/reports"]]
)
def test_metrics_rejects_removed_options(option: list[str]) -> None:
    result = runner.invoke(app, ["metrics", "--run", "experiment", *option])

    assert result.exit_code == 2
