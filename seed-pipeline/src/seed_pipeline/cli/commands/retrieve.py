from __future__ import annotations

from pathlib import Path
from typing import Annotated, cast

import typer

from seed_pipeline.cli.options import positive_int
from seed_pipeline.cli.runtime import (
    CommandResult,
    CommandStatus,
    run_handler,
    state_from_context,
)
from seed_pipeline.config.defaults import (
    DEFAULT_CANDIDATE_K,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_QDRANT_URL,
    DEFAULT_RETRIEVER,
    DEFAULT_RRF_K,
)
from seed_pipeline.config.paths import (
    PROCESSED_EVALUATION_DIR,
    query_embedding_cache_path,
    retrieval_run_roots,
)
from seed_pipeline.evaluation.retrieval_service import RetrieveRequest, run_retrieval


def resolve_query_embeddings_dir(
    evaluation: Path,
    model: str,
    retriever: str,
    explicit: Path | None,
) -> Path | None:
    if explicit is not None or retriever == "bm25":
        return explicit
    del evaluation
    return query_embedding_cache_path(model)


def retrieve(
    ctx: typer.Context,
    evaluation: Annotated[Path, typer.Option("--evaluation")] = PROCESSED_EVALUATION_DIR
    / "section_retrieval_eval.jsonl",
    query_embeddings: Annotated[Path | None, typer.Option("--query-embeddings")] = None,
    run: Annotated[str, typer.Option("--run")] = cast(str, ...),
    embedding_model: Annotated[str, typer.Option("--model")] = DEFAULT_EMBEDDING_MODEL,
    qdrant_url: Annotated[str, typer.Option("--qdrant-url")] = DEFAULT_QDRANT_URL,
    retriever: Annotated[str, typer.Option("--retriever")] = DEFAULT_RETRIEVER,
    candidate_k: Annotated[
        int,
        typer.Option("--candidate-k", callback=lambda _c, _p, v: positive_int(str(v))),
    ] = DEFAULT_CANDIDATE_K,
    prefetch_k: Annotated[
        int | None,
        typer.Option(
            "--prefetch-k",
            callback=lambda _c, _p, value: (
                None if value is None else positive_int(str(value))
            ),
        ),
    ] = None,
    rrf_k: Annotated[
        int, typer.Option("--rrf-k", callback=lambda _c, _p, v: positive_int(str(v)))
    ] = DEFAULT_RRF_K,
    limit: Annotated[int | None, typer.Option("--limit")] = None,
    force: Annotated[bool, typer.Option("--force")] = False,
) -> None:
    result = run_handler(
        state_from_context(ctx),
        lambda: _run(
            evaluation,
            query_embeddings,
            run,
            embedding_model,
            qdrant_url,
            retriever,
            candidate_k,
            prefetch_k,
            rrf_k,
            limit,
            force,
        ),
    )
    _ = result


def _run(
    evaluation,
    query_embeddings,
    run,
    model,
    qdrant_url,
    retriever,
    candidate_k,
    prefetch_k,
    rrf_k,
    limit,
    force,
) -> CommandResult:
    metadata_root, artifact_root = retrieval_run_roots(run)
    result = run_retrieval(
        RetrieveRequest(
            evaluation_path=evaluation,
            query_embeddings_dir=resolve_query_embeddings_dir(
                evaluation, model, retriever, query_embeddings
            ),
            run_root=metadata_root,
            embedding_model=model,
            qdrant_url=qdrant_url,
            retriever=retriever,
            candidate_k=candidate_k,
            prefetch_k=prefetch_k,
            rrf_k=rrf_k,
            limit=limit,
            force=force,
            artifact_root=artifact_root,
        )
    )
    return CommandResult(
        "retrieve",
        CommandStatus.COMPLETE,
        result.workspace.root,
        {
            "artifact": str(result.artifact.data_path),
            "queries": result.artifact.query_count,
        },
    )
