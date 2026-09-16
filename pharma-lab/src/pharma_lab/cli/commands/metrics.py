from __future__ import annotations

from typing import Annotated

import typer

from pharma_lab.cli.runtime import (
    CommandResult,
    CommandStatus,
    run_handler,
    state_from_context,
)
from pharma_lab.config.defaults import DEFAULT_TOP_K
from pharma_lab.config.paths import run_dir
from pharma_lab.evaluation.metric_comparison import (
    MetricComparisonRequest,
    run_metric_comparison,
)
from pharma_lab.evaluation.metrics_service import MetricsRequest, run_metrics

metrics_app = typer.Typer(add_completion=False, invoke_without_command=True)


@metrics_app.callback()
def metrics(
    ctx: typer.Context,
    run: Annotated[str | None, typer.Option("--run")] = None,
    top_k: Annotated[int, typer.Option("--top-k")] = DEFAULT_TOP_K,
    window_size: Annotated[int, typer.Option("--window-size")] = 3,
    model: Annotated[str | None, typer.Option("--model")] = None,
    force: Annotated[bool, typer.Option("--force")] = False,
) -> None:
    """Write metrics reports for a run; `compare` tests two rerankers."""
    if ctx.invoked_subcommand is not None:
        return
    if run is None:
        raise typer.BadParameter(
            "required unless a subcommand is given", param_hint="'--run'"
        )
    request = MetricsRequest(run_dir(run), top_k, window_size, model, force)
    run_handler(state_from_context(ctx), lambda: _run(request))


def _run(request: MetricsRequest) -> CommandResult:
    result = run_metrics(request)
    return CommandResult(
        "metrics",
        CommandStatus.COMPLETE,
        result.baseline.artifact_dir,
        {
            "baseline": {
                "artifact": str(result.baseline.artifact_dir),
                "metrics_sha256": result.baseline.metrics_sha256,
            },
            "reranked": [
                {
                    "model": item.model,
                    "variant_sha256": item.variant_sha256,
                    "metrics_sha256": item.metrics_sha256,
                    "artifact": str(item.artifact_dir),
                }
                for item in sorted(result.reranked, key=lambda item: item.model or "")
            ],
        },
    )


@metrics_app.command("compare")
def compare(
    ctx: typer.Context,
    run: Annotated[str, typer.Option("--run")],
    baseline: Annotated[
        str, typer.Option("--baseline", help="Reranker the decision starts from")
    ],
    candidate: Annotated[
        str, typer.Option("--candidate", help="Reranker adopted only if it wins")
    ],
    metric: Annotated[
        str, typer.Option("--metric", help="Per-query key of metrics.jsonl")
    ] = "mrr",
    top_k: Annotated[int, typer.Option("--top-k")] = DEFAULT_TOP_K,
    window_size: Annotated[int, typer.Option("--window-size")] = 3,
    resamples: Annotated[int, typer.Option("--resamples")] = 10_000,
    seed: Annotated[int, typer.Option("--seed")] = 0,
) -> None:
    """Paired bootstrap of candidate minus baseline over the run's queries.

    The candidate wins only when the 95% interval's lower bound is above 0.
    """
    request = MetricComparisonRequest(
        run_dir(run), baseline, candidate, metric, top_k, window_size, resamples, seed
    )
    run_handler(state_from_context(ctx), lambda: _compare(request))


def _compare(request: MetricComparisonRequest) -> CommandResult:
    comparison = run_metric_comparison(request)
    bootstrap = comparison.bootstrap
    winner = request.candidate_model if comparison.candidate_wins else None
    return CommandResult(
        "metrics compare",
        CommandStatus.COMPLETE,
        request.run_root,
        {
            "metric": comparison.metric,
            "queries": comparison.query_count,
            "baseline": {
                "model": request.baseline_model,
                "mean": comparison.baseline_mean,
            },
            "candidate": {
                "model": request.candidate_model,
                "mean": comparison.candidate_mean,
            },
            "mean_difference": bootstrap.mean_difference,
            "ci95_low": bootstrap.ci_low,
            "ci95_high": bootstrap.ci_high,
            "resamples": bootstrap.resamples,
            "seed": request.seed,
            "decision": winner or request.baseline_model,
        },
    )
