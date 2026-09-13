from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from seed_pipeline.cli.runtime import (
    CommandResult,
    CommandStatus,
    run_handler,
    state_from_context,
)
from seed_pipeline.config.paths import (
    DOCLING_INTERIM_DIR,
    RAG_FINAL_SECTIONS_PATH,
    RAW_DIR,
    TEXT_INTERIM_DIR,
)
from seed_pipeline.corpus.sources.crawl import CrawlRequest, crawl_source
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
    sitemap_url: Annotated[
        str, typer.Option("--sitemap-url")
    ] = "https://www.nhathuocankhang.com/sitemap-sanpham.xml",
    output_dir: Annotated[Path, typer.Option("--output-dir")] = RAW_DIR
    / "ankhang"
    / "html",
    workers: Annotated[int, typer.Option("--workers")] = 8,
    request_timeout_seconds: Annotated[
        float, typer.Option("--request-timeout-seconds")
    ] = 10.0,
    force: Annotated[bool, typer.Option("--force")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    def run() -> CommandResult:
        from urllib.request import Request, urlopen

        def fetch(url: str, timeout: float) -> bytes:
            with urlopen(
                Request(url, headers={"User-Agent": "seed-pipeline"}), timeout=timeout
            ) as response:
                return response.read()

        result = crawl_source(
            CrawlRequest(
                sitemap_url,
                output_dir,
                workers,
                request_timeout_seconds,
                force,
                dry_run,
            ),
            fetch,
        )
        return CommandResult(
            "source crawl",
            CommandStatus.COMPLETE,
            details={
                "total": result.total,
                "downloaded": result.downloaded,
                "skipped": result.skipped,
                "failed": result.failed,
            },
        )

    run_handler(state_from_context(ctx), run)


@source_app.command("extract-tables")
def extract_tables(
    ctx: typer.Context,
    pdf: Annotated[Path, typer.Option("--pdf")] = RAW_DIR
    / "duoc-thu-quoc-gia-viet-nam.pdf",
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
