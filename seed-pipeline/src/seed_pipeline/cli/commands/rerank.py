from __future__ import annotations

from collections.abc import Callable
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
from seed_pipeline.evaluation.rerank_log import open_rerank_log
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
    kaggle_account: Annotated[
        str | None,
        typer.Option(
            "--kaggle-account",
            help=(
                "Kaggle account profile accN, or 'auto' to choose the account "
                "with the most GPU quota before every session."
            ),
        ),
    ] = None,
    max_runs: Annotated[
        int | None,
        typer.Option(
            "--max-runs",
            min=1,
            help="Stop after this many Kaggle sessions (default: until complete "
            "or out of quota).",
        ),
    ] = None,
    recover_kernel: Annotated[
        str | None,
        typer.Option(
            "--recover-kernel",
            help="OWNER/KERNEL-SLUG of a finished Kaggle kernel whose scores are "
            "merged into the local score cache (kaggle backend).",
        ),
    ] = None,
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
        max_runs=max_runs,
        recover_kernel=recover_kernel,
    )
    run_handler(state_from_context(ctx), lambda: _logged(backend, request, adapter.run))


def _logged(
    backend: Backend,
    request: RerankRequest,
    run: Callable[[RerankRequest], RerankStageResult],
) -> CommandResult:
    log = open_rerank_log(request.model, echo=False)
    log(
        f"command backend={backend.value} run={request.run_root.name} "
        f"model={request.model} kaggle_account={request.kaggle_account or 'default'} "
        f"max_runs={request.max_runs or 'unlimited'} force={request.force} "
        f"dry_run={request.dry_run} benchmark={request.benchmark}"
    )
    try:
        result = run(request)
    except BaseException as error:
        log(f"command error={type(error).__name__}: {error}")
        raise
    status = "incomplete" if result.incomplete else "complete"
    log(f"command status={status} actions={' | '.join(result.actions)}")
    return _result(result)


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
            "quota": result.quota,
        },
    )
