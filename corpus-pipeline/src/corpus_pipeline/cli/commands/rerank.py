from __future__ import annotations

from pathlib import Path
from typing import Annotated, cast

import typer

from corpus_pipeline.cli.options import Backend
from corpus_pipeline.cli.runtime import (
    CommandResult,
    CommandStatus,
    run_handler,
    state_from_context,
)
from corpus_pipeline.config.defaults import (
    DEFAULT_BUDGET_SECONDS,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_RERANKER_MODEL,
)
from corpus_pipeline.config.paths import RETRIEVAL_EVAL_RUNS_DIR
from corpus_pipeline.evaluation.rerank_service import (
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
) -> None:
    run_root = RETRIEVAL_EVAL_RUNS_DIR / run
    adapter = (
        LocalRerankBackend() if backend is Backend.LOCAL else KaggleRerankBackend()
    )
    result = run_handler(
        state_from_context(ctx),
        lambda: _result(
            adapter.run(
                RerankRequest(
                    run_root,
                    candidates,
                    output_dir,
                    model,
                    force,
                    dry_run,
                    budget_seconds,
                    request_timeout_seconds,
                )
            )
        ),
    )
    _ = result


def _result(result) -> CommandResult:
    return CommandResult(
        "rerank",
        CommandStatus.INCOMPLETE if result.incomplete else CommandStatus.COMPLETE,
        result.cache_path,
        {"actions": result.actions, "subset_sha256": result.subset_sha256},
    )
