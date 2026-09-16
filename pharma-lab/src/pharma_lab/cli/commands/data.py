from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

import typer

from pharma_lab.cli.runtime import (
    CommandResult,
    CommandStatus,
    run_handler,
    state_from_context,
)
from pharma_lab.config.paths import PROJECT_ROOT
from pharma_lab.data_archive.service import pull_data, push_data
from pharma_lab.integrations.kaggle.dataset_service import DatasetService
from pharma_lab.integrations.kaggle.service import resolve_execution_context

data_app = typer.Typer(
    help="Store Git-ignored data/ files in a private Kaggle dataset and restore them.",
    no_args_is_help=True,
)


def _dataset(kaggle_account: str | None) -> DatasetService:
    context = resolve_execution_context(kaggle_account)
    return DatasetService(context.runner, context.owners.execution)


@data_app.command("push")
def push(
    ctx: typer.Context,
    kaggle_account: Annotated[str | None, typer.Option("--kaggle-account")] = None,
    message: Annotated[str, typer.Option("--message")] = "Update pharma-lab data",
) -> None:
    def run() -> CommandResult:
        result = push_data(
            project_root=PROJECT_ROOT,
            dataset=_dataset(kaggle_account),
            message=message,
            now=datetime.now(UTC),
        )
        return CommandResult(
            "data push",
            CommandStatus.COMPLETE,
            None,
            {
                "dataset": result.reference,
                "version": result.version,
                "files": result.file_count,
                "parts": result.part_count,
            },
        )

    run_handler(state_from_context(ctx), run)


@data_app.command("pull")
def pull(
    ctx: typer.Context,
    kaggle_account: Annotated[str | None, typer.Option("--kaggle-account")] = None,
    force: Annotated[bool, typer.Option("--force")] = False,
) -> None:
    def run() -> CommandResult:
        result = pull_data(
            project_root=PROJECT_ROOT, dataset=_dataset(kaggle_account), force=force
        )
        return CommandResult(
            "data pull",
            CommandStatus.COMPLETE,
            PROJECT_ROOT / "data",
            {
                "dataset": result.reference,
                "files": result.file_count,
                "written": result.written,
                "unchanged": result.unchanged,
                "downloaded_parts": result.downloaded_parts,
            },
        )

    run_handler(state_from_context(ctx), run)
