from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer
from pharma_agent.domain.corpus.bundle import BundleValidationError, read_bundle
from pharma_agent.domain.corpus.chunking import MAX_CHUNK_CHARS

from seed_pipeline.bundle.export import ExportRequest, export_bundle
from seed_pipeline.bundle.parity import check_chunk_parity
from seed_pipeline.cli.runtime import (
    CommandResult,
    CommandStatus,
    run_handler,
    state_from_context,
)
from seed_pipeline.config.paths import RAG_FINAL_DIR, RESOURCES_DIR
from seed_pipeline.evaluation.artifact_contracts import iter_jsonl_objects, write_json

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


def parity_command(
    *,
    bundle: Path,
    old_chunks: Path,
    report: Path | None = None,
    max_chars: int = MAX_CHUNK_CHARS,
) -> CommandResult:
    try:
        knowledge = read_bundle(bundle)
    except BundleValidationError as exc:
        raise invalid_bundle(exc) from exc
    result = check_chunk_parity(
        knowledge, iter_jsonl_objects(old_chunks), max_chars=max_chars
    )
    if report is not None:
        write_json(report, result.to_dict())
    return CommandResult(
        command="bundle parity",
        status=CommandStatus.COMPLETE if result.ok else CommandStatus.FAILED,
        artifact=report,
        details={
            "sections": result.sections_checked,
            "chunks": result.chunks_checked,
            "mismatches": result.mismatch_count,
            "first_mismatches": [asdict(item) for item in result.mismatches[:5]],
        },
    )


@bundle_app.command("parity")
def parity(
    ctx: typer.Context,
    bundle: Annotated[
        Path, typer.Option("--bundle", file_okay=False, resolve_path=True)
    ],
    old_chunks: Annotated[
        Path, typer.Option("--old-chunks", dir_okay=False, resolve_path=True)
    ],
    report: Annotated[Path | None, typer.Option("--report", dir_okay=False)] = None,
    max_chars: Annotated[int, typer.Option("--max-chars")] = MAX_CHUNK_CHARS,
) -> None:
    run_handler(
        state_from_context(ctx),
        lambda: parity_command(
            bundle=bundle, old_chunks=old_chunks, report=report, max_chars=max_chars
        ),
    )
