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
from corpus_pipeline.config.paths import PROJECT_ROOT
from corpus_pipeline.orchestration.preflight import run_preflight

DEFAULT_ENV_FILE = PROJECT_ROOT.parent / ".env"


def doctor_command(
    backend: Backend,
    env_file: Path,
    json_output: bool,
    debug: bool,
) -> CommandResult:
    report = run_preflight(backend, project_root=PROJECT_ROOT, env_file=env_file)
    return CommandResult(
        command="doctor",
        status=CommandStatus.COMPLETE if report.ok else CommandStatus.FAILED,
        details={
            "backend": report.backend.value,
            "issues": [
                {
                    "check": issue.check,
                    "message": issue.message,
                    "resource": issue.resource,
                }
                for issue in report.issues
            ],
            "issue_count": len(report.issues),
        },
    )


def doctor(
    ctx: typer.Context,
    backend: Annotated[Backend, typer.Option("--backend")] = Backend.LOCAL,
    env_file: Annotated[Path, typer.Option("--env-file")] = DEFAULT_ENV_FILE,
) -> None:
    run_handler(
        state_from_context(ctx),
        lambda: doctor_command(backend, env_file, False, False),
    )
