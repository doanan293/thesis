from __future__ import annotations

import json
import shlex
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, NoReturn

import typer

from seed_pipeline.integrations.kaggle.errors import KaggleDetached


class CommandStatus(StrEnum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    FAILED = "failed"


@dataclass(frozen=True)
class CommandResult:
    command: str
    status: CommandStatus
    artifact: Path | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CliState:
    json_output: bool = False
    debug: bool = False


def state_from_context(ctx: typer.Context) -> CliState:
    return ctx.ensure_object(CliState)


class ResumableIncomplete(RuntimeError):
    def __init__(self, result: CommandResult):
        super().__init__(result.command)
        self.result = result


def result_payload(result: CommandResult) -> dict[str, Any]:
    payload: dict[str, Any] = asdict(result)
    payload["status"] = result.status.value
    payload["artifact"] = str(result.artifact) if result.artifact else None
    return payload


def render_result(result: CommandResult, *, json_output: bool) -> None:
    if json_output:
        typer.echo(
            json.dumps(result_payload(result), ensure_ascii=False, sort_keys=True)
        )
        return
    typer.echo(f"status={result.status.value}")
    if result.artifact is not None:
        typer.echo(f"artifact={result.artifact}")
    for key, value in sorted(result.details.items()):
        typer.echo(f"{key}={value}")


def fail(message: str, code: int) -> NoReturn:
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(code=code)


def run_handler(state: CliState, handler: Callable[[], CommandResult]) -> None:
    try:
        result = handler()
    except KaggleDetached as exc:
        command = shlex.join(["uv", "run", "seed", *sys.argv[1:]])
        fail(
            f"detached from Kaggle kernel {exc.reference}; it is still running. "
            f"Re-run to attach and recover: {command}",
            130,
        )
    except KeyboardInterrupt:
        fail("interrupted", 130)
    except ResumableIncomplete as exc:
        render_result(exc.result, json_output=state.json_output)
        raise typer.Exit(code=3) from None
    except (OSError, RuntimeError, ValueError) as exc:
        if state.debug:
            raise
        fail(str(exc), 1)
    render_result(result, json_output=state.json_output)
    if result.status is CommandStatus.INCOMPLETE:
        raise typer.Exit(code=3)
    if result.status is CommandStatus.FAILED:
        raise typer.Exit(code=1)
