from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from pharma_agent.domain.corpus.bundle import BundleValidationError, read_bundle

from pharma_lab.bundle.evaluation_chunks import write_evaluation_chunks
from pharma_lab.cli.commands.bundle import invalid_bundle
from pharma_lab.cli.runtime import (
    CommandResult,
    CommandStatus,
    run_handler,
    state_from_context,
)
from pharma_lab.config.paths import (
    BUNDLE_DIR,
    EVALUATION_CHUNKS_PATH,
    GOLD_DIR,
    RAG_FINAL_SECTIONS_PATH,
)
from pharma_lab.evaluation.build_dataset import (
    EvaluationBuildRequest,
    build_evaluation_dataset,
)
from pharma_lab.evaluation.relevance_judgments import build_evaluation_judgments

evaluation_app = typer.Typer(no_args_is_help=True, add_completion=False)
GOLD_EVALUATION_PATH = GOLD_DIR / "section_retrieval_eval.jsonl"


def evaluation_build_command(
    *,
    sections: Path = RAG_FINAL_SECTIONS_PATH,
    bundle: Path = BUNDLE_DIR,
    chunks_output: Path = EVALUATION_CHUNKS_PATH,
    output_dir: Path = GOLD_DIR,
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
    judgments = build_evaluation_judgments(result.evaluation_path, bundle)
    return CommandResult(
        command="evaluation build",
        status=CommandStatus.COMPLETE,
        artifact=result.evaluation_path,
        details={
            "chunks": chunk_count,
            "patient_queries": str(result.patient_queries_path),
            "patient_query_count": result.patient_query_count,
            "evaluation_row_count": result.evaluation_row_count,
            "judgments": str(judgments),
        },
    )


@evaluation_app.command("build")
def evaluation_build(
    ctx: typer.Context,
    sections: Annotated[Path, typer.Option("--sections")] = RAG_FINAL_SECTIONS_PATH,
    bundle: Annotated[Path, typer.Option("--bundle", file_okay=False)] = BUNDLE_DIR,
    chunks_output: Annotated[
        Path, typer.Option("--chunks-output", dir_okay=False)
    ] = EVALUATION_CHUNKS_PATH,
    output_dir: Annotated[Path, typer.Option("--output-dir")] = GOLD_DIR,
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


def evaluation_judgments_command(
    *, evaluation: Path = GOLD_EVALUATION_PATH, bundle: Path = BUNDLE_DIR
) -> CommandResult:
    try:
        path = build_evaluation_judgments(evaluation, bundle)
    except BundleValidationError as exc:
        raise invalid_bundle(exc) from exc
    return CommandResult(
        command="evaluation judgments",
        status=CommandStatus.COMPLETE,
        artifact=path,
        details={},
    )


@evaluation_app.command("judgments")
def evaluation_judgments(
    ctx: typer.Context,
    evaluation: Annotated[
        Path, typer.Option("--evaluation", dir_okay=False)
    ] = GOLD_EVALUATION_PATH,
    bundle: Annotated[Path, typer.Option("--bundle", file_okay=False)] = BUNDLE_DIR,
) -> None:
    """Rebuild the relevance judgments of a gold file; runs keep their candidates."""
    run_handler(
        state_from_context(ctx),
        lambda: evaluation_judgments_command(evaluation=evaluation, bundle=bundle),
    )
