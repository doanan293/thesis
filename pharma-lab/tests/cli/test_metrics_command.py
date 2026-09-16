import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import pharma_lab.cli.commands.metrics as metrics_command
from pharma_lab.cli.app import app
from pharma_lab.config.paths import run_dir
from pharma_lab.evaluation.metric_comparison import (
    BootstrapResult,
    MetricComparison,
    MetricComparisonRequest,
)
from pharma_lab.evaluation.metrics_artifacts import MetricsArtifactResult
from pharma_lab.evaluation.metrics_service import MetricsRequest, MetricsResult

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


def fake_comparison(
    monkeypatch: pytest.MonkeyPatch,
    captured: dict[str, MetricComparisonRequest],
    bootstrap: BootstrapResult,
) -> None:
    def fake_run(request: MetricComparisonRequest) -> MetricComparison:
        captured["request"] = request
        return MetricComparison("mrr", 1000, 0.8, 0.81, bootstrap)

    monkeypatch.setattr(metrics_command, "run_metric_comparison", fake_run)


COMPARE = [
    "--json",
    "metrics",
    "compare",
    "--run",
    "sample",
    "--baseline",
    "qwen3-reranker:0.6b-fp16",
    "--candidate",
    "qwen3-reranker:4b-fp16",
    "--top-k",
    "30",
]


def test_metrics_compare_adopts_the_candidate_above_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, MetricComparisonRequest] = {}
    fake_comparison(monkeypatch, captured, BootstrapResult(0.01, 0.002, 0.018, 10_000))

    result = runner.invoke(app, COMPARE)

    assert result.exit_code == 0, result.output
    assert captured["request"] == MetricComparisonRequest(
        run_dir("sample"),
        "qwen3-reranker:0.6b-fp16",
        "qwen3-reranker:4b-fp16",
        "mrr",
        30,
        3,
        10_000,
        0,
    )
    details = json.loads(result.stdout)["details"]
    assert details["decision"] == "qwen3-reranker:4b-fp16"
    assert (details["ci95_low"], details["ci95_high"]) == (0.002, 0.018)
    assert details["candidate"] == {"model": "qwen3-reranker:4b-fp16", "mean": 0.81}


def test_metrics_compare_keeps_the_baseline_when_zero_is_inside(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, MetricComparisonRequest] = {}
    fake_comparison(monkeypatch, captured, BootstrapResult(0.01, -0.001, 0.02, 10_000))

    result = runner.invoke(app, COMPARE)

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["details"]["decision"] == (
        "qwen3-reranker:0.6b-fp16"
    )


def test_metrics_without_run_or_subcommand_is_a_usage_error() -> None:
    result = runner.invoke(app, ["metrics"])

    assert result.exit_code == 2
