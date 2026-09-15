from __future__ import annotations

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
from seed_pipeline.config.paths import run_dir
from seed_pipeline.evaluation.rerank_service import (
    KaggleRerankBackend,
    LocalRerankBackend,
    RerankRequest,
    RerankStageResult,
)


def rerank(
    ctx: typer.Context,
    run: Annotated[str, typer.Option("--run")] = cast(str, ...),
    backend: Annotated[Backend, typer.Option("--backend")] = Backend.LOCAL,
    model: Annotated[str, typer.Option("--model")] = DEFAULT_RERANKER_MODEL,
    force: Annotated[bool, typer.Option("--force")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    budget_seconds: Annotated[
        int, typer.Option("--budget-seconds")
    ] = DEFAULT_BUDGET_SECONDS,
    request_timeout_seconds: Annotated[
        float, typer.Option("--request-timeout-seconds")
    ] = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    benchmark: Annotated[
        bool,
        typer.Option(
            "--benchmark",
            help="Measure local llama-reranker levels and store the lowest-p95 profile.",
        ),
    ] = False,
    kaggle_account: Annotated[str | None, typer.Option("--kaggle-account")] = None,
) -> None:
    adapter = (
        LocalRerankBackend() if backend is Backend.LOCAL else KaggleRerankBackend()
    )
    request = RerankRequest(
        run_root=run_dir(run),
        model=model,
        force=force,
        dry_run=dry_run,
        budget_seconds=budget_seconds,
        request_timeout_seconds=request_timeout_seconds,
        benchmark=benchmark,
        kaggle_account=kaggle_account,
    )
    run_handler(state_from_context(ctx), lambda: _result(adapter.run(request)))


def _result(result: RerankStageResult) -> CommandResult:
    return CommandResult(
        "rerank",
        CommandStatus.INCOMPLETE if result.incomplete else CommandStatus.COMPLETE,
        result.artifact_dir,
        {
            "actions": result.actions,
            "subset_sha256": result.subset_sha256,
            "variant_sha256": result.variant_sha256,
            "benchmark_report": result.benchmark_report,
            "benchmark_levels": result.benchmark_levels,
        },
    )
