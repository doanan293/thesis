from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from corpus_pipeline.cli.runtime import (
    CommandResult,
    CommandStatus,
    run_handler,
    state_from_context,
)
from corpus_pipeline.config.defaults import DEFAULT_EMBEDDING_MODEL, DEFAULT_QDRANT_URL
from corpus_pipeline.config.paths import RAG_FINAL_CHUNKS_PATH
from corpus_pipeline.vector_store.ingest_vectors import default_embedding_cache_path
from corpus_pipeline.vector_store.upload_service import (
    UploadVectorsRequest,
    upload_vectors,
)

vectors_app = typer.Typer(no_args_is_help=True, add_completion=False)


@vectors_app.command("upload")
def upload(
    ctx: typer.Context,
    embeddings: Annotated[Path | None, typer.Option("--embeddings")] = None,
    chunks: Annotated[Path, typer.Option("--chunks")] = RAG_FINAL_CHUNKS_PATH,
    model: Annotated[str, typer.Option("--model")] = DEFAULT_EMBEDDING_MODEL,
    qdrant_url: Annotated[str, typer.Option("--qdrant-url")] = DEFAULT_QDRANT_URL,
    qdrant_batch_size: Annotated[int, typer.Option("--qdrant-batch-size")] = 1000,
) -> None:
    resolved_embeddings = embeddings or default_embedding_cache_path(model)

    def run() -> CommandResult:
        result = upload_vectors(
            UploadVectorsRequest(
                chunks_path=chunks,
                embeddings_path=resolved_embeddings,
                model=model,
                qdrant_url=qdrant_url,
                qdrant_batch_size=qdrant_batch_size,
            )
        )
        return CommandResult(
            "vectors upload",
            CommandStatus.COMPLETE,
            details={
                "collection": result.collection_name,
                "points": result.point_count,
            },
        )

    run_handler(state_from_context(ctx), run)
