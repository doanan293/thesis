from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from pharma_agent.domain.corpus.bundle import BundleValidationError

from seed_pipeline.bundle.export import ExportRequest, export_bundle
from seed_pipeline.cli.runtime import (
    CommandResult,
    CommandStatus,
    run_handler,
    state_from_context,
)
from seed_pipeline.config.paths import RAG_FINAL_DIR, RESOURCES_DIR

bundle_app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Export, check and embed knowledge bundles.",
)


def invalid_bundle(exc: BundleValidationError) -> ValueError:
    return ValueError("knowledge bundle is invalid: " + "; ".join(exc.problems[:20]))


def export_command(
    *,
    output: Path,
    rag_final_dir: Path = RAG_FINAL_DIR,
    glossary: Path = RESOURCES_DIR / "term_glossary.json",
    mappings: Path = RESOURCES_DIR / "colloquial_mappings.json",
    force: bool = False,
) -> CommandResult:
    try:
        result = export_bundle(
            ExportRequest(
                rag_final_dir=rag_final_dir,
                glossary_path=glossary,
                mappings_path=mappings,
                output_dir=output,
                force=force,
            )
        )
    except BundleValidationError as exc:
        raise invalid_bundle(exc) from exc
    return CommandResult(
        command="bundle export",
        status=CommandStatus.COMPLETE,
        artifact=output,
        details={
            "documents": result.manifest.document_count,
            "sections": result.manifest.section_count,
            "skipped_sections": list(result.skipped_sections),
        },
    )


@bundle_app.command("export")
def export(
    ctx: typer.Context,
    output: Annotated[
        Path, typer.Option("--output", file_okay=False, resolve_path=True)
    ],
    rag_final_dir: Annotated[Path, typer.Option("--rag-final-dir")] = RAG_FINAL_DIR,
    glossary: Annotated[Path, typer.Option("--glossary")] = RESOURCES_DIR
    / "term_glossary.json",
    mappings: Annotated[Path, typer.Option("--mappings")] = RESOURCES_DIR
    / "colloquial_mappings.json",
    force: Annotated[bool, typer.Option("--force")] = False,
) -> None:
    run_handler(
        state_from_context(ctx),
        lambda: export_command(
            output=output,
            rag_final_dir=rag_final_dir,
            glossary=glossary,
            mappings=mappings,
            force=force,
        ),
    )
