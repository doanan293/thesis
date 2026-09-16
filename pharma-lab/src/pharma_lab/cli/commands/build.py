from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Annotated

import typer

from pharma_lab.cli.runtime import (
    CommandResult,
    CommandStatus,
    run_handler,
    state_from_context,
)
from pharma_lab.orchestration.build_corpus import default_config, run_build


def build_command(
    *,
    pdf: Path | None = None,
    leaflets_dir: Path | None = None,
    curated_tables: Path | None = None,
    table_overrides: Path | None = None,
    mappings: Path | None = None,
    glossary: Path | None = None,
    work_root: Path | None = None,
    final_dir: Path | None = None,
    max_chars: int | None = None,
) -> CommandResult:
    config = default_config()
    config = replace(
        config,
        pdf_path=pdf if pdf is not None else config.pdf_path,
        leaflets_dir=leaflets_dir if leaflets_dir is not None else config.leaflets_dir,
        curated_tables_path=(
            curated_tables if curated_tables is not None else config.curated_tables_path
        ),
        table_overrides_path=(
            table_overrides
            if table_overrides is not None
            else config.table_overrides_path
        ),
        mappings_path=mappings if mappings is not None else config.mappings_path,
        glossary_path=glossary if glossary is not None else config.glossary_path,
        work_root=work_root if work_root is not None else config.work_root,
        final_dir=final_dir if final_dir is not None else config.final_dir,
        max_chars=max_chars if max_chars is not None else config.max_chars,
    )
    result = run_build(config)
    return CommandResult(
        command="build",
        status=CommandStatus.COMPLETE,
        artifact=result.final_dir,
        details={"build_id": result.build_id},
    )


def build(
    ctx: typer.Context,
    pdf: Annotated[Path | None, typer.Option("--pdf")] = None,
    leaflets_dir: Annotated[Path | None, typer.Option("--leaflets-dir")] = None,
    curated_tables: Annotated[Path | None, typer.Option("--curated-tables")] = None,
    table_overrides: Annotated[Path | None, typer.Option("--table-overrides")] = None,
    mappings: Annotated[Path | None, typer.Option("--mappings")] = None,
    glossary: Annotated[Path | None, typer.Option("--glossary")] = None,
    work_root: Annotated[Path | None, typer.Option("--work-root")] = None,
    final_dir: Annotated[Path | None, typer.Option("--final-dir")] = None,
    max_chars: Annotated[int | None, typer.Option("--max-chars")] = None,
) -> None:
    run_handler(
        state_from_context(ctx),
        lambda: build_command(
            pdf=pdf,
            leaflets_dir=leaflets_dir,
            curated_tables=curated_tables,
            table_overrides=table_overrides,
            mappings=mappings,
            glossary=glossary,
            work_root=work_root,
            final_dir=final_dir,
            max_chars=max_chars,
        ),
    )
