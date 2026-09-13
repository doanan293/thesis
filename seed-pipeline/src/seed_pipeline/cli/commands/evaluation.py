from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from pharma_agent.domain.corpus.bundle import BundleValidationError, read_bundle

from seed_pipeline.bundle.evaluation_chunks import write_evaluation_chunks
from seed_pipeline.cli.commands.bundle import invalid_bundle
from seed_pipeline.cli.runtime import (
    CommandResult,
    CommandStatus,
    run_handler,
    state_from_context,
)
from seed_pipeline.config.paths import (
    DEFAULT_BUNDLE_DIR,
    EVALUATION_CHUNKS_PATH,
    PROCESSED_EVALUATION_DIR,
    RAG_FINAL_SECTIONS_PATH,
    retrieval_run_roots,
)
from seed_pipeline.evaluation.build_dataset import (
    EvaluationBuildRequest,
    build_evaluation_dataset,
)
from seed_pipeline.evaluation.rejudge_service import (
    RejudgeRequest,
    run_rejudging,
)

evaluation_app = typer.Typer(no_args_is_help=True, add_completion=False)


def evaluation_build_command(
    *,
    sections: Path = RAG_FINAL_SECTIONS_PATH,
    bundle: Path = DEFAULT_BUNDLE_DIR,
    chunks_output: Path = EVALUATION_CHUNKS_PATH,
    output_dir: Path = PROCESSED_EVALUATION_DIR,
    patient_query_count: int = 500,
    evaluation_row_count: int = 10_000,
) -> CommandResult:
    try:
        knowledge = read_bundle(bundle)
    except BundleValidationError as exc:
        raise invalid_bundle(exc) from exc
    chunk_count = write_evaluation_chunks(knowledge, chunks_output)
    result = build_evaluation_dataset(
        EvaluationBuildRequest(
            sections_path=sections,
            chunks_path=chunks_output,
            output_dir=output_dir,
            patient_query_count=patient_query_count,
            evaluation_row_count=evaluation_row_count,
        )
    )
    return CommandResult(
        command="evaluation build",
        status=CommandStatus.COMPLETE,
        artifact=result.evaluation_path,
        details={
            "chunks": chunk_count,
            "patient_queries": str(result.patient_queries_path),
            "patient_query_count": result.patient_query_count,
            "evaluation_row_count": result.evaluation_row_count,
        },
    )


@evaluation_app.command("build")
def evaluation_build(
    ctx: typer.Context,
    sections: Annotated[Path, typer.Option("--sections")] = RAG_FINAL_SECTIONS_PATH,
    bundle: Annotated[
        Path, typer.Option("--bundle", file_okay=False)
    ] = DEFAULT_BUNDLE_DIR,
    chunks_output: Annotated[
        Path, typer.Option("--chunks-output", dir_okay=False)
    ] = EVALUATION_CHUNKS_PATH,
    output_dir: Annotated[
        Path, typer.Option("--output-dir")
    ] = PROCESSED_EVALUATION_DIR,
    patient_query_count: Annotated[int, typer.Option("--patient-query-count")] = 500,
    evaluation_row_count: Annotated[
        int, typer.Option("--evaluation-row-count")
    ] = 10_000,
) -> None:
    run_handler(
        state_from_context(ctx),
        lambda: evaluation_build_command(
            sections=sections,
            bundle=bundle,
            chunks_output=chunks_output,
            output_dir=output_dir,
            patient_query_count=patient_query_count,
            evaluation_row_count=evaluation_row_count,
        ),
    )


@evaluation_app.command("rejudge-current")
def rejudge_current(
    ctx: typer.Context,
    dense_run: Annotated[str, typer.Option("--dense-run")],
    hybrid_run: Annotated[str, typer.Option("--hybrid-run")],
    apply: Annotated[
        bool, typer.Option("--apply", help="Replace the current evaluation and reports")
    ] = False,
) -> None:
    dense_metadata, dense_artifacts = retrieval_run_roots(dense_run)
    hybrid_metadata, hybrid_artifacts = retrieval_run_roots(hybrid_run)
    result = run_handler(
        state_from_context(ctx),
        lambda: _rejudge_current(
            dense_metadata,
            dense_artifacts,
            hybrid_metadata,
            hybrid_artifacts,
            apply,
        ),
    )
    _ = result


def _rejudge_current(
    dense_metadata: Path,
    dense_artifacts: Path,
    hybrid_metadata: Path,
    hybrid_artifacts: Path,
    apply: bool,
) -> CommandResult:
    result = run_rejudging(
        RejudgeRequest(
            evaluation_path=PROCESSED_EVALUATION_DIR / "section_retrieval_eval.jsonl",
            dense_run_root=dense_metadata,
            dense_artifact_root=dense_artifacts,
            hybrid_run_root=hybrid_metadata,
            hybrid_artifact_root=hybrid_artifacts,
            apply=apply,
        )
    )
    return CommandResult(
        "evaluation rejudge-current",
        CommandStatus.COMPLETE,
        PROCESSED_EVALUATION_DIR / "section_retrieval_eval.jsonl",
        {
            "applied": result.applied,
            "old_evaluation_sha256": result.old_evaluation_sha256,
            "new_evaluation_sha256": result.new_evaluation_sha256,
            "summary": result.summary,
            "reports": [str(path) for path in result.reports],
        },
    )
