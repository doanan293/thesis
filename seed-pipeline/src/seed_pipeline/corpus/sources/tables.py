from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from seed_pipeline.corpus.tables.curate_docling_tables import process_table_curation
from seed_pipeline.corpus.tables.extract_docling_tables import process_tables


@dataclass(frozen=True)
class TableExtractionRequest:
    pdf_path: Path
    cleaned_markdown_path: Path
    output_dir: Path
    ocr_enabled: bool = False
    batch_size: int = 50
    dry_run: bool = False


@dataclass(frozen=True)
class TableCurationRequest:
    input_dir: Path
    sections_path: Path
    dry_run: bool = False


def extract_tables(request: TableExtractionRequest) -> dict[str, Path]:
    if request.batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if request.dry_run:
        return {}
    return process_tables(
        pdf_path=request.pdf_path,
        cleaned_markdown_path=request.cleaned_markdown_path,
        output_dir=request.output_dir,
        ocr_enabled=request.ocr_enabled,
        batch_size=request.batch_size,
    )


def curate_tables(request: TableCurationRequest) -> dict[str, Path]:
    if request.dry_run:
        return {}
    return process_table_curation(
        input_dir=request.input_dir,
        sections_path=request.sections_path,
    )
