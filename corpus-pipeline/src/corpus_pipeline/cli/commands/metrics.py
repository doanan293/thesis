from __future__ import annotations

from pathlib import Path
from typing import Annotated, cast

import typer

from corpus_pipeline.cli.runtime import (
    CommandResult,
    CommandStatus,
    run_handler,
    state_from_context,
)
from corpus_pipeline.config.defaults import DEFAULT_TOP_K
from corpus_pipeline.config.paths import RETRIEVAL_EVAL_RUNS_DIR
from corpus_pipeline.evaluation.metrics_service import MetricsRequest, run_metrics


def metrics(
    ctx: typer.Context,
    run: Annotated[str, typer.Option("--run")] = cast(str, ...),
    top_k: Annotated[int, typer.Option("--top-k")] = DEFAULT_TOP_K,
    window_size: Annotated[int, typer.Option("--window-size")] = 3,
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
) -> None:
    result = run_handler(
        state_from_context(ctx), lambda: _run(run, top_k, window_size, output_dir)
    )
    _ = result


def _run(run, top_k, window_size, output_dir):
    result = run_metrics(
        MetricsRequest(
            RETRIEVAL_EVAL_RUNS_DIR / run, top_k, window_size, output_dir=output_dir
        )
    )
    return CommandResult(
        "metrics",
        CommandStatus.COMPLETE,
        result.baseline_report,
        {
            "baseline": str(result.baseline_report),
            "reranked": str(result.reranked_report) if result.reranked_report else None,
        },
    )
