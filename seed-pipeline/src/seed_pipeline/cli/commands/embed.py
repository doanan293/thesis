from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from seed_pipeline.cli.options import Backend
from seed_pipeline.cli.runtime import (
    CommandResult,
    CommandStatus,
    run_handler,
    state_from_context,
)
from seed_pipeline.config.defaults import DEFAULT_EMBEDDING_MODEL
from seed_pipeline.config.paths import (
    GOLD_DIR,
    query_embedding_cache_path,
)
from seed_pipeline.evaluation.query_embedding_service import (
    KaggleQueryEmbeddingBackend,
    LocalQueryEmbeddingBackend,
    QueryEmbeddingBackend,
    QueryEmbeddingRequest,
    QueryEmbeddingStageResult,
)
from seed_pipeline.runtime.catalog import require_model

embed_app = typer.Typer(no_args_is_help=True, add_completion=False)


@embed_app.command("queries")
def embed_queries(
    ctx: typer.Context,
    backend: Annotated[Backend, typer.Option("--backend")] = Backend.LOCAL,
    evaluation: Annotated[Path, typer.Option("--evaluation")] = GOLD_DIR
    / "section_retrieval_eval.jsonl",
    model: Annotated[str, typer.Option("--model")] = DEFAULT_EMBEDDING_MODEL,
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
    force: Annotated[bool, typer.Option("--force")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    budget_seconds: Annotated[int, typer.Option("--budget-seconds")] = 21_600,
    request_timeout_seconds: Annotated[
        float, typer.Option("--request-timeout-seconds")
    ] = 900.0,
    kaggle_account: Annotated[str | None, typer.Option("--kaggle-account")] = None,
) -> None:
    require_model(model)
    resolved_output = output_dir or query_embedding_cache_path(model)
    run_handler(
        state_from_context(ctx),
        lambda: _query_embedding_result(
            query_backend(backend).run(
                QueryEmbeddingRequest(
                    evaluation_path=evaluation,
                    model=model,
                    output_dir=resolved_output,
                    force=force,
                    dry_run=dry_run,
                    budget_seconds=budget_seconds,
                    request_timeout_seconds=request_timeout_seconds,
                    kaggle_account=kaggle_account,
                )
            )
        ),
    )


def query_backend(backend: Backend) -> QueryEmbeddingBackend:
    if backend is Backend.LOCAL:
        return LocalQueryEmbeddingBackend()
    if backend is Backend.KAGGLE:
        return KaggleQueryEmbeddingBackend()
    raise ValueError(f"Unsupported embedding backend: {backend}")


def _query_embedding_result(result: QueryEmbeddingStageResult) -> CommandResult:
    return CommandResult(
        "embed queries",
        CommandStatus.INCOMPLETE if result.incomplete else CommandStatus.COMPLETE,
        result.cache_path,
        {
            "actions": result.actions,
            "subset_sha256": result.subset_sha256,
            "benchmark_report": getattr(result, "benchmark_report", None),
            "benchmark_levels": getattr(result, "benchmark_levels", 0),
        },
    )
