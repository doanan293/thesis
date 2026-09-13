from __future__ import annotations

from pathlib import Path
from typing import Annotated, cast

import typer

from seed_pipeline.cli.runtime import (
    CommandResult,
    CommandStatus,
    run_handler,
    state_from_context,
)
from seed_pipeline.config.defaults import DEFAULT_TOP_K
from seed_pipeline.config.paths import retrieval_run_roots
from seed_pipeline.evaluation.metrics_service import MetricsRequest, run_metrics


def metrics(
    ctx: typer.Context,
    run: Annotated[str, typer.Option("--run")] = cast(str, ...),
    top_k: Annotated[int, typer.Option("--top-k")] = DEFAULT_TOP_K,
    window_size: Annotated[int, typer.Option("--window-size")] = 3,
    model: Annotated[str | None, typer.Option("--model")] = None,
    variant: Annotated[str | None, typer.Option("--variant")] = None,
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
) -> None:
    if model is not None and variant is not None:
        raise typer.BadParameter("--model and --variant are mutually exclusive")
    if output_dir is not None:
        typer.echo(
            "Warning: --output-dir is deprecated; metrics artifacts are stored under the run",
            err=True,
        )
    result = run_handler(
        state_from_context(ctx),
        lambda: _run(run, top_k, window_size, model, variant),
    )
    _ = result


def _run(run, top_k, window_size, model, variant):
    metadata_root, artifact_root = retrieval_run_roots(run)
    result = run_metrics(
        MetricsRequest(
            metadata_root,
            top_k,
            window_size,
            model,
            variant,
            artifact_root,
        )
    )
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
                for item in sorted(
                    result.reranked,
                    key=lambda item: (item.model or "", item.variant_sha256 or ""),
                )
            ],
        },
    )
