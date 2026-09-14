from __future__ import annotations

from typing import Annotated, cast

import typer

from seed_pipeline.cli.runtime import (
    CommandResult,
    CommandStatus,
    run_handler,
    state_from_context,
)
from seed_pipeline.config.defaults import DEFAULT_TOP_K
from seed_pipeline.config.paths import run_dir
from seed_pipeline.evaluation.metrics_service import MetricsRequest, run_metrics


def metrics(
    ctx: typer.Context,
    run: Annotated[str, typer.Option("--run")] = cast(str, ...),
    top_k: Annotated[int, typer.Option("--top-k")] = DEFAULT_TOP_K,
    window_size: Annotated[int, typer.Option("--window-size")] = 3,
    model: Annotated[str | None, typer.Option("--model")] = None,
    force: Annotated[bool, typer.Option("--force")] = False,
) -> None:
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
