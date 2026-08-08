from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from corpus_pipeline.cli.options import Backend
from corpus_pipeline.cli.runtime import (
    CommandResult,
    CommandStatus,
    run_handler,
    state_from_context,
)
from corpus_pipeline.config.defaults import DEFAULT_EMBEDDING_MODEL
from corpus_pipeline.config.paths import (
    PROCESSED_EVALUATION_DIR,
    RAG_FINAL_CHUNKS_PATH,
    query_embedding_cache_path,
)
from corpus_pipeline.evaluation.query_embedding_service import (
    KaggleQueryEmbeddingBackend,
    LocalQueryEmbeddingBackend,
    QueryEmbeddingBackend,
    QueryEmbeddingRequest,
)
from corpus_pipeline.runtime.catalog import require_model
from corpus_pipeline.vector_store.embedding_service import (
    ChunkEmbeddingRequest,
    KaggleChunkEmbeddingBackend,
    LocalChunkEmbeddingBackend,
)
from corpus_pipeline.vector_store.ingest_vectors import default_embedding_cache_path

embed_app = typer.Typer(no_args_is_help=True, add_completion=False)


def embed_chunks_command(
    *,
    backend: Backend = Backend.LOCAL,
    chunks: Path = RAG_FINAL_CHUNKS_PATH,
    model: str = DEFAULT_EMBEDDING_MODEL,
    output: Path | None = None,
    force: bool = False,
    dry_run: bool = False,
    budget_seconds: int = 21_600,
    request_timeout_seconds: float = 900.0,
) -> CommandResult:
    require_model(model)
    resolved_output = output or default_embedding_cache_path(model)
    adapter = (
        LocalChunkEmbeddingBackend()
        if backend is Backend.LOCAL
        else KaggleChunkEmbeddingBackend()
    )
    result = adapter.run(
        ChunkEmbeddingRequest(
            chunks_path=chunks,
            model=model,
            cache_path=resolved_output,
            force=force,
            dry_run=dry_run,
            budget_seconds=budget_seconds,
            request_timeout_seconds=request_timeout_seconds,
        )
    )
    status = CommandStatus.INCOMPLETE if result.incomplete else CommandStatus.COMPLETE
    return CommandResult(
        "embed chunks",
        status,
        result.cache_path,
        {"actions": result.actions},
    )


@embed_app.command("chunks")
def embed_chunks(
    ctx: typer.Context,
    backend: Annotated[Backend, typer.Option("--backend")] = Backend.LOCAL,
    chunks: Annotated[Path, typer.Option("--chunks")] = RAG_FINAL_CHUNKS_PATH,
    model: Annotated[str, typer.Option("--model")] = DEFAULT_EMBEDDING_MODEL,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "--output-dir",
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
    force: Annotated[bool, typer.Option("--force")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    budget_seconds: Annotated[int, typer.Option("--budget-seconds")] = 21_600,
    request_timeout_seconds: Annotated[
        float, typer.Option("--request-timeout-seconds")
    ] = 900.0,
) -> None:
    run_handler(
        state_from_context(ctx),
        lambda: embed_chunks_command(
            backend=backend,
            chunks=chunks,
            model=model,
            output=output,
            force=force,
            dry_run=dry_run,
            budget_seconds=budget_seconds,
            request_timeout_seconds=request_timeout_seconds,
        ),
    )


@embed_app.command("queries")
def embed_queries(
    ctx: typer.Context,
    backend: Annotated[Backend, typer.Option("--backend")] = Backend.LOCAL,
    evaluation: Annotated[Path, typer.Option("--evaluation")] = PROCESSED_EVALUATION_DIR
    / "section_retrieval_eval.jsonl",
    model: Annotated[str, typer.Option("--model")] = DEFAULT_EMBEDDING_MODEL,
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
    force: Annotated[bool, typer.Option("--force")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    budget_seconds: Annotated[int, typer.Option("--budget-seconds")] = 21_600,
    request_timeout_seconds: Annotated[
        float, typer.Option("--request-timeout-seconds")
    ] = 900.0,
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


def _query_embedding_result(result) -> CommandResult:
    return CommandResult(
        "embed queries",
        CommandStatus.INCOMPLETE if result.incomplete else CommandStatus.COMPLETE,
        result.cache_path,
        {"actions": result.actions, "subset_sha256": result.subset_sha256},
    )
