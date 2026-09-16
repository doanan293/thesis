from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from pharma_lab.cli.runtime import (
    CommandResult,
    CommandStatus,
    run_handler,
    state_from_context,
)
from pharma_lab.config.paths import RAG_FINAL_DIR
from pharma_lab.orchestration.validation_service import (
    ValidationRequest,
    run_validation,
)


def validate_command(
    *,
    output_dir: Path = RAG_FINAL_DIR,
    output_json: Path | None = None,
    output_markdown: Path | None = None,
) -> CommandResult:
    report = run_validation(
        ValidationRequest(
            final_dir=output_dir,
            output_json=output_json,
            output_markdown=output_markdown,
        )
    )
    return CommandResult(
        command="validate",
        status=CommandStatus.COMPLETE if report.ok else CommandStatus.FAILED,
        artifact=report.output_json or output_dir,
        details={
            "errors": list(report.errors),
            "metrics": report.metrics,
            "output_markdown": str(report.output_markdown)
            if report.output_markdown
            else None,
        },
    )


def validate(
    ctx: typer.Context,
    output_dir: Annotated[
        Path, typer.Option("--output-dir", help="Published corpus directory.")
    ] = RAG_FINAL_DIR,
    output_json: Annotated[Path | None, typer.Option("--output-json")] = None,
    output_markdown: Annotated[Path | None, typer.Option("--output-markdown")] = None,
) -> None:
    run_handler(
        state_from_context(ctx),
        lambda: validate_command(
            output_dir=output_dir,
            output_json=output_json,
            output_markdown=output_markdown,
        ),
    )
