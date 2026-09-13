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
from seed_pipeline.config.environment import PROJECT_ENV_FILE
from seed_pipeline.config.paths import PROJECT_ROOT
from seed_pipeline.orchestration.preflight import run_preflight

DEFAULT_ENV_FILE = PROJECT_ENV_FILE


def doctor_command(
    backend: Backend,
    env_file: Path,
    json_output: bool,
    debug: bool,
    kaggle_account: str | None = None,
) -> CommandResult:
    del json_output, debug
    report = run_preflight(
        backend,
        project_root=PROJECT_ROOT,
        env_file=env_file,
        kaggle_account=kaggle_account,
    )
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
    kaggle_account: Annotated[str | None, typer.Option("--kaggle-account")] = None,
) -> None:
    run_handler(
        state_from_context(ctx),
        lambda: doctor_command(backend, env_file, False, False, kaggle_account),
    )
