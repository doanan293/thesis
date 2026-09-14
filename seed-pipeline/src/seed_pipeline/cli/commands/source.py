from __future__ import annotations

import http.client
from datetime import date
from pathlib import Path
from typing import Annotated
from urllib.request import Request, urlopen

import typer

from seed_pipeline.cli.runtime import (
    CommandResult,
    CommandStatus,
    run_handler,
    state_from_context,
)
from seed_pipeline.config.paths import (
    DOCLING_INTERIM_DIR,
    FORMULARY_PDF_PATH,
    LEAFLETS_DIR,
    RAG_FINAL_SECTIONS_PATH,
    TEXT_INTERIM_DIR,
)
from seed_pipeline.corpus.sources.crawl import (
    CrawlFetchError,
    CrawlRequest,
    crawl_leaflets,
)
from seed_pipeline.corpus.sources.tables import (
    TableCurationRequest,
    TableExtractionRequest,
)
from seed_pipeline.corpus.sources.tables import (
    curate_tables as run_table_curation,
)
from seed_pipeline.corpus.sources.tables import (
    extract_tables as run_table_extraction,
)

source_app = typer.Typer(no_args_is_help=True, add_completion=False)


@source_app.command("crawl")
def crawl(
    ctx: typer.Context,
    leaflets_dir: Annotated[Path, typer.Option("--leaflets-dir")] = LEAFLETS_DIR,
    sitemap_url: Annotated[str | None, typer.Option("--sitemap-url")] = None,
    workers: Annotated[int, typer.Option("--workers")] = 8,
    request_timeout_seconds: Annotated[
        float, typer.Option("--request-timeout-seconds")
    ] = 10.0,
    force: Annotated[bool, typer.Option("--force")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    def run() -> CommandResult:
        result = crawl_leaflets(
            CrawlRequest(
                leaflets_dir,
                sitemap_url,
                workers,
                request_timeout_seconds,
                force,
                dry_run,
            ),
            _fetch,
            today=date.today(),
        )
        return CommandResult(
            "source crawl",
            CommandStatus.COMPLETE,
            result.manifest_path,
            {
                "total": result.total,
                "downloaded": result.downloaded,
                "skipped": result.skipped,
                "failed": result.failed,
            },
        )

    run_handler(state_from_context(ctx), run)


def _fetch(url: str, timeout: float) -> bytes:
    request = Request(url, headers={"User-Agent": "seed-pipeline"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.read()
    except (OSError, http.client.HTTPException, ValueError) as exc:
        raise CrawlFetchError(f"{url}: {exc}") from exc


@source_app.command("extract-tables")
def extract_tables(
    ctx: typer.Context,
    pdf: Annotated[Path, typer.Option("--pdf")] = FORMULARY_PDF_PATH,
    markdown: Annotated[Path, typer.Option("--markdown")] = TEXT_INTERIM_DIR
    / "full.cleaned.md",
    output_dir: Annotated[Path, typer.Option("--output-dir")] = DOCLING_INTERIM_DIR,
    ocr: Annotated[bool, typer.Option("--ocr")] = False,
    batch_size: Annotated[int, typer.Option("--batch-size")] = 50,
) -> None:
    def run() -> CommandResult:
        outputs = run_table_extraction(
            TableExtractionRequest(
                pdf, markdown, output_dir, ocr_enabled=ocr, batch_size=batch_size
            )
        )
        return CommandResult(
            "source extract-tables",
            CommandStatus.COMPLETE,
            details={k: str(v) for k, v in outputs.items()},
        )

    run_handler(state_from_context(ctx), run)


@source_app.command("curate-tables")
def curate_tables(
    ctx: typer.Context,
    input_dir: Annotated[Path, typer.Option("--input-dir")] = DOCLING_INTERIM_DIR,
    sections: Annotated[Path, typer.Option("--sections")] = RAG_FINAL_SECTIONS_PATH,
) -> None:
    def run() -> CommandResult:
        outputs = run_table_curation(TableCurationRequest(input_dir, sections))
        return CommandResult(
            "source curate-tables",
            CommandStatus.COMPLETE,
            details={k: str(v) for k, v in outputs.items()},
        )

    run_handler(state_from_context(ctx), run)
