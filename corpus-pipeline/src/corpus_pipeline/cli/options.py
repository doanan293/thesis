from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from corpus_pipeline.config.defaults import (
    DEFAULT_BUDGET_SECONDS,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
)
from corpus_pipeline.config.enums import Backend


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise ValueError("value must be >= 1")
    return parsed


BackendOption = Annotated[
    Backend,
    typer.Option("--backend", help="Where model work executes."),
]
OutputDirOption = Annotated[
    Path | None,
    typer.Option("--output-dir", file_okay=False, resolve_path=True),
]
ForceOption = Annotated[bool, typer.Option("--force")]
DryRunOption = Annotated[bool, typer.Option("--dry-run")]
BudgetOption = Annotated[
    int,
    typer.Option("--budget-seconds", help="Maximum stage wall-clock budget."),
]
RequestTimeoutOption = Annotated[
    float,
    typer.Option("--request-timeout-seconds", help="Timeout for one request."),
]

DEFAULT_BUDGET = DEFAULT_BUDGET_SECONDS
DEFAULT_REQUEST_TIMEOUT = DEFAULT_REQUEST_TIMEOUT_SECONDS
