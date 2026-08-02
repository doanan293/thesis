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
from corpus_pipeline.config.paths import (
    PROCESSED_EVALUATION_DIR,
    RAG_FINAL_CHUNKS_PATH,
    RAG_FINAL_SECTIONS_PATH,
)
from corpus_pipeline.evaluation.build_dataset import (
    EvaluationBuildRequest,
    build_evaluation_dataset,
)

evaluation_app = typer.Typer(no_args_is_help=True, add_completion=False)


def evaluation_build_command(
    *,
    sections: Path = RAG_FINAL_SECTIONS_PATH,
    chunks: Path = RAG_FINAL_CHUNKS_PATH,
    output_dir: Path = PROCESSED_EVALUATION_DIR,
    patient_query_count: int = 500,
    evaluation_row_count: int = 10_000,
) -> CommandResult:
    result = build_evaluation_dataset(
        EvaluationBuildRequest(
            sections_path=sections,
            chunks_path=chunks,
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
            "patient_queries": str(result.patient_queries_path),
            "patient_query_count": result.patient_query_count,
            "evaluation_row_count": result.evaluation_row_count,
        },
    )


@evaluation_app.command("build")
def evaluation_build(
    ctx: typer.Context,
    sections: Annotated[Path, typer.Option("--sections")] = RAG_FINAL_SECTIONS_PATH,
    chunks: Annotated[Path, typer.Option("--chunks")] = RAG_FINAL_CHUNKS_PATH,
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
            chunks=chunks,
            output_dir=output_dir,
            patient_query_count=patient_query_count,
            evaluation_row_count=evaluation_row_count,
        ),
    )
