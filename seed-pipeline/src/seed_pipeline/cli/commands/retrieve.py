from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from seed_pipeline.bundle.export import COLLECTION_KEY
from seed_pipeline.cli.options import positive_int
from seed_pipeline.cli.runtime import (
    CommandResult,
    CommandStatus,
    run_handler,
    state_from_context,
)
from seed_pipeline.config.defaults import (
    DEFAULT_CANDIDATE_K,
    DEFAULT_RETRIEVER,
    DEFAULT_RRF_K,
)
from seed_pipeline.config.paths import BACKEND_ENV_FILE, GOLD_DIR, run_dir
from seed_pipeline.evaluation.backend_retrieval import RetrieveRequest, run_retrieval


def retrieve(
    ctx: typer.Context,
    run: Annotated[str, typer.Option("--run")],
    evaluation: Annotated[Path, typer.Option("--evaluation")] = GOLD_DIR
    / "section_retrieval_eval.jsonl",
    retriever: Annotated[
        str, typer.Option("--retriever", help="bm25, dense or hybrid")
    ] = DEFAULT_RETRIEVER,
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
    limit: Annotated[
        int | None,
        typer.Option("--limit", help="First N rows of the evaluation file"),
    ] = None,
    sample: Annotated[
        int | None,
        typer.Option(
            "--sample",
            help="Stratified sample of N rows, proportional per eval_group",
            callback=lambda _c, _p, value: (
                None if value is None else positive_int(str(value))
            ),
        ),
    ] = None,
    sample_seed: Annotated[
        int, typer.Option("--sample-seed", help="Random seed for --sample")
    ] = 0,
    collection: Annotated[str, typer.Option("--collection")] = COLLECTION_KEY,
    query_embeddings: Annotated[
        Path | None,
        typer.Option(
            "--query-embeddings",
            dir_okay=False,
            help="Query embedding cache from `seed embed queries` (dense and hybrid)",
        ),
    ] = None,
    backend_env_file: Annotated[
        Path, typer.Option("--backend-env-file", dir_okay=False)
    ] = BACKEND_ENV_FILE,
    force: Annotated[bool, typer.Option("--force")] = False,
) -> None:
    if limit is not None and sample is not None:
        raise typer.BadParameter("--limit and --sample are mutually exclusive")
    request = RetrieveRequest(
        evaluation_path=evaluation,
        run_root=run_dir(run),
        retriever=retriever,
        candidate_k=candidate_k,
        prefetch_k=prefetch_k,
        rrf_k=rrf_k,
        limit=limit,
        force=force,
        collection=collection,
        backend_env_file=backend_env_file,
        query_embeddings=query_embeddings,
        sample=sample,
        sample_seed=sample_seed,
    )
    run_handler(state_from_context(ctx), lambda: _run(request))


def _run(request: RetrieveRequest) -> CommandResult:
    result = run_retrieval(request)
    return CommandResult(
        "retrieve",
        CommandStatus.COMPLETE,
        result.workspace.root,
        {
            "artifact": str(result.artifact.data_path),
            "queries": result.artifact.query_count,
            "release_id": result.workspace.identity.release_id,
        },
    )
