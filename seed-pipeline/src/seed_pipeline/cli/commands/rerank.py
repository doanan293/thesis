from __future__ import annotations

from pathlib import Path
from typing import Annotated, cast

import typer

from seed_pipeline.cli.options import Backend
from seed_pipeline.cli.runtime import (
    CommandResult,
    CommandStatus,
    run_handler,
    state_from_context,
)
from seed_pipeline.config.defaults import (
    DEFAULT_BUDGET_SECONDS,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_RERANKER_MODEL,
)
from seed_pipeline.config.paths import retrieval_run_roots
from seed_pipeline.evaluation.rerank_service import (
    KaggleRerankBackend,
    LocalRerankBackend,
    RerankRequest,
)


def rerank(
    ctx: typer.Context,
    run: Annotated[str, typer.Option("--run")] = cast(str, ...),
    backend: Annotated[Backend, typer.Option("--backend")] = Backend.LOCAL,
    model: Annotated[str, typer.Option("--model")] = DEFAULT_RERANKER_MODEL,
    candidates: Annotated[Path | None, typer.Option("--candidates")] = None,
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
    force: Annotated[bool, typer.Option("--force")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    budget_seconds: Annotated[
        int, typer.Option("--budget-seconds")
    ] = DEFAULT_BUDGET_SECONDS,
    request_timeout_seconds: Annotated[
        float, typer.Option("--request-timeout-seconds")
    ] = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    kaggle_account: Annotated[str | None, typer.Option("--kaggle-account")] = None,
) -> None:
    run_root, artifact_root = retrieval_run_roots(run)
    if output_dir is not None:
        typer.echo(
            "Warning: --output-dir is deprecated; rerank artifacts are stored under the run",
            err=True,
        )
    adapter = (
        LocalRerankBackend() if backend is Backend.LOCAL else KaggleRerankBackend()
    )
    result = run_handler(
        state_from_context(ctx),
        lambda: _result(
            adapter.run(
                RerankRequest(
                    run_root=run_root,
                    candidates_dir=candidates,
                    model=model,
                    force=force,
                    dry_run=dry_run,
                    budget_seconds=budget_seconds,
                    request_timeout_seconds=request_timeout_seconds,
                    artifact_root=artifact_root,
                    kaggle_account=kaggle_account,
                )
            )
        ),
    )
    _ = result


def _result(result) -> CommandResult:
    return CommandResult(
        "rerank",
        CommandStatus.INCOMPLETE if result.incomplete else CommandStatus.COMPLETE,
        result.artifact_dir,
        {
            "actions": result.actions,
            "subset_sha256": result.subset_sha256,
            "variant_sha256": result.variant_sha256,
            "benchmark_report": getattr(result, "benchmark_report", None),
            "benchmark_levels": getattr(result, "benchmark_levels", 0),
        },
    )
