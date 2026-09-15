from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from pharma_agent.domain.corpus.bundle import BundleValidationError

from seed_pipeline.bundle.embed import BundleEmbedRequest, embed_bundle
from seed_pipeline.bundle.export import ExportRequest, export_bundle
from seed_pipeline.cli.runtime import (
    CommandResult,
    CommandStatus,
    run_handler,
    state_from_context,
)
from seed_pipeline.config.defaults import (
    DEFAULT_BUDGET_SECONDS,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
)
from seed_pipeline.config.enums import Backend
from seed_pipeline.config.paths import (
    BUNDLE_EMBED_WORK_DIR,
    RAG_FINAL_DIR,
    SOURCES_DIR,
)
from seed_pipeline.embeddings.service import (
    KaggleTextEmbeddingBackend,
    LocalTextEmbeddingBackend,
    TextEmbeddingBackend,
)

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
    glossary: Path = SOURCES_DIR / "term_glossary.json",
    mappings: Path = SOURCES_DIR / "colloquial_mappings.json",
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
    glossary: Annotated[Path, typer.Option("--glossary")] = SOURCES_DIR
    / "term_glossary.json",
    mappings: Annotated[Path, typer.Option("--mappings")] = SOURCES_DIR
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


def embed_command(
    *,
    bundle: Path,
    backend: Backend,
    model: str,
    cache: Path | None = None,
    work_dir: Path = BUNDLE_EMBED_WORK_DIR,
    force: bool = False,
    dry_run: bool = False,
    budget_seconds: int = DEFAULT_BUDGET_SECONDS,
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    kaggle_account: str | None = None,
) -> CommandResult:
    adapter: TextEmbeddingBackend = (
        LocalTextEmbeddingBackend()
        if backend is Backend.LOCAL
        else KaggleTextEmbeddingBackend()
    )
    try:
        result = embed_bundle(
            BundleEmbedRequest(
                bundle_dir=bundle,
                model=model,
                force=force,
                dry_run=dry_run,
                budget_seconds=budget_seconds,
                request_timeout_seconds=request_timeout_seconds,
                kaggle_account=kaggle_account,
                cache_path=cache,
                work_dir=work_dir,
            ),
            adapter,
        )
    except BundleValidationError as exc:
        raise invalid_bundle(exc) from exc
    return CommandResult(
        command="bundle embed",
        status=CommandStatus.INCOMPLETE
        if result.incomplete
        else CommandStatus.COMPLETE,
        artifact=bundle,
        details={
            "inputs": result.inputs,
            "vectors": result.vectors,
            "embeddings_file": result.embeddings_file,
            "actions": list(result.actions),
        },
    )


@bundle_app.command("embed")
def embed(
    ctx: typer.Context,
    bundle: Annotated[
        Path, typer.Option("--bundle", file_okay=False, resolve_path=True)
    ],
    backend: Annotated[Backend, typer.Option("--backend")],
    model: Annotated[str, typer.Option("--model")],
    cache: Annotated[Path | None, typer.Option("--cache", dir_okay=False)] = None,
    work_dir: Annotated[Path, typer.Option("--work-dir")] = BUNDLE_EMBED_WORK_DIR,
    force: Annotated[bool, typer.Option("--force")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    budget_seconds: Annotated[
        int, typer.Option("--budget-seconds")
    ] = DEFAULT_BUDGET_SECONDS,
    request_timeout_seconds: Annotated[
        float, typer.Option("--request-timeout-seconds")
    ] = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    kaggle_account: Annotated[str | None, typer.Option("--kaggle-account")] = None,
) -> None:
    run_handler(
        state_from_context(ctx),
        lambda: embed_command(
            bundle=bundle,
            backend=backend,
            model=model,
            cache=cache,
            work_dir=work_dir,
            force=force,
            dry_run=dry_run,
            budget_seconds=budget_seconds,
            request_timeout_seconds=request_timeout_seconds,
            kaggle_account=kaggle_account,
        ),
    )
