#!/usr/bin/env python3
from __future__ import annotations

import concurrent.futures
import json
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from seed_pipeline.config.paths import (
    DOCLING_INTERIM_DIR,
    FORMULARY_PDF_PATH,
    TEXT_INTERIM_DIR,
)

DEFAULT_PDF = FORMULARY_PDF_PATH
DEFAULT_CLEANED_MD = TEXT_INTERIM_DIR / "full.cleaned.md"
DEFAULT_OUTPUT_DIR = DOCLING_INTERIM_DIR
PAGE_MARKER_RE = re.compile(r"<!--\s*page\s+(\d{4})\s*-->")
TABLE_CAPTION_RE = re.compile(r"\bBảng\s+\d+[.:]?\s+.+", re.IGNORECASE)
MARKDOWN_TABLE_SEPARATOR_RE = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$"
)
DOCLING_PAGE_MARKER_RE = re.compile(
    r"^\s*(?:<!--\s*page\s+\d+\s*-->|<!--\s*PageBreak\s*-->|---\s*$)",
    re.IGNORECASE,
)
KNOWN_DRUG_ACRONYMS = (
    "ABC",
    "ARV",
    "AZT",
    "CD4",
    "D4T",
    "DDI",
    "EFV",
    "HGB",
    "HIV",
    "LPV",
    "NRTI",
    "NVP",
    "TDF",
)


@dataclass
class TableCaption:
    page: int
    caption: str
    line_number: int


@dataclass
class TableRecord:
    table_id: str
    page: int
    table_index: int
    caption: str | None
    markdown: str
    source: str
    extraction_mode: str = "text_layer"
    context_heading: str | None = None
    flags: list[str] | None = None


def clean_markdown_inline(text: str) -> str:
    cleaned = text.strip()
    cleaned = cleaned.replace("\\*", "*").replace("\\_", "_")
    cleaned = re.sub(r"\*{1,3}", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def find_table_captions(markdown_text: str) -> list[TableCaption]:
    captions: list[TableCaption] = []
    current_page: int | None = None
    for line_number, raw_line in enumerate(markdown_text.splitlines(), start=1):
        marker = PAGE_MARKER_RE.search(raw_line)
        if marker:
            current_page = int(marker.group(1))
            continue
        cleaned = clean_markdown_inline(raw_line)
        match = TABLE_CAPTION_RE.search(cleaned)
        if not match or current_page is None:
            continue
        if not raw_line.lstrip().startswith(
            ("*", "**", "***")
        ) and not cleaned.startswith("Bảng "):
            continue
        captions.append(
            TableCaption(
                page=current_page,
                caption=match.group(0).strip(),
                line_number=line_number,
            )
        )
    return captions


def parse_page_list(value: str) -> list[int]:
    pages: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if start > end:
                raise ValueError(f"Invalid descending page range: {part}")
            pages.update(range(start, end + 1))
            continue
        pages.add(int(part))
    return sorted(pages)


def chunk_pages(pages: list[int], *, batch_size: int) -> list[list[int]]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    batches: list[list[int]] = []
    current: list[int] = []
    for page in sorted(set(pages)):
        starts_new_batch = (
            not current or page != current[-1] + 1 or len(current) >= batch_size
        )
        if starts_new_batch:
            if current:
                batches.append(current)
            current = [page]
            continue
        current.append(page)
    if current:
        batches.append(current)
    return batches


def get_pdf_page_count(pdf_path: Path) -> int:
    if shutil.which("pdfinfo") is None:
        raise RuntimeError("pdfinfo is required to count PDF pages but was not found.")
    result = subprocess.run(
        ["pdfinfo", str(pdf_path)],
        check=True,
        text=True,
        capture_output=True,
    )
    for line in result.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    raise RuntimeError(
        f"Could not determine page count from pdfinfo output for {pdf_path}"
    )


def select_pages(
    *,
    explicit_pages: list[int] | None,
    caption_pages_only: bool,
    captions: list[TableCaption],
    pdf_page_count: int,
) -> list[int]:
    if explicit_pages is not None:
        return sorted(set(explicit_pages))
    if caption_pages_only:
        return sorted({caption.page for caption in captions})
    return list(range(1, pdf_page_count + 1))


def is_markdown_table_block(lines: list[str]) -> bool:
    if len(lines) < 2:
        return False
    return any(MARKDOWN_TABLE_SEPARATOR_RE.match(line) for line in lines[:3])


def extract_markdown_tables(markdown_text: str) -> list[str]:
    tables: list[str] = []
    current: list[str] = []
    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()
        if line.lstrip().startswith("|") and "|" in line.strip()[1:]:
            current.append(line)
            continue
        if current:
            if is_markdown_table_block(current):
                tables.append("\n".join(current).strip())
            current = []
    if current and is_markdown_table_block(current):
        tables.append("\n".join(current).strip())
    return tables


def clean_table_markdown(markdown: str) -> str:
    replacements = {
        "&gt;": ">",
        "&lt;": "<",
        "&amp;": "&",
        "VitaminB 12": "Vitamin B12",
        "vitaminB 12": "vitamin B12",
        "Vitamin B 12": "Vitamin B12",
        "vitamin B 12": "vitamin B12",
        "mm 3": "mm3",
    }

    cleaned_lines: list[str] = []
    for line in markdown.splitlines():
        cleaned = line
        for bad, good in replacements.items():
            cleaned = cleaned.replace(bad, good)

        cleaned = re.sub(r"(?<=\w)([<>]=?)\s*", r" \1 ", cleaned)
        for acronym in KNOWN_DRUG_ACRONYMS:
            cleaned = re.sub(rf"(?<=[a-zà-ỹ]){acronym}\b", f" {acronym}", cleaned)
            cleaned = re.sub(rf"\b{acronym}(?=[a-zà-ỹ])", f"{acronym} ", cleaned)

        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r"[ \t]+\|", " |", cleaned)
        cleaned = re.sub(r"\|[ \t]+", "| ", cleaned)
        cleaned_lines.append(cleaned)
    return "\n".join(cleaned_lines)


def caption_from_line(line: str) -> str | None:
    cleaned = clean_markdown_inline(line)
    match = TABLE_CAPTION_RE.search(cleaned)
    return match.group(0).strip() if match else None


def heading_from_line(line: str) -> str | None:
    stripped = line.strip()
    if not stripped.startswith("#"):
        return None
    return clean_markdown_inline(re.sub(r"^#+\s*", "", stripped))


def table_has_continuation_like_rows(markdown: str) -> bool:
    for line in markdown.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue
        if MARKDOWN_TABLE_SEPARATOR_RE.match(line):
            continue
        first_cell = cells[0]
        if re.fullmatch(
            r"(?:vấn|tư|bằng|hoặc|và|tiếp)\b.*", first_cell, flags=re.IGNORECASE
        ):
            return True
    return False


def extract_page_table_records(
    *,
    page: int,
    markdown_text: str,
    source: str,
    extraction_mode: str = "text_layer",
) -> list[TableRecord]:
    records: list[TableRecord] = []
    current_lines: list[str] = []
    current_caption: str | None = None
    current_heading: str | None = None
    table_caption: str | None = None
    table_heading: str | None = None

    def flush_table() -> None:
        nonlocal current_lines, table_caption, table_heading
        if current_lines and is_markdown_table_block(current_lines):
            index = len(records) + 1
            cleaned_markdown = clean_table_markdown("\n".join(current_lines).strip())
            flags: list[str] = []
            if table_caption is None:
                flags.append("missing_caption")
            if table_has_continuation_like_rows(cleaned_markdown):
                flags.append("possible_continuation_rows")
            records.append(
                TableRecord(
                    table_id=f"page-{page:04d}-table-{index:03d}",
                    page=page,
                    table_index=index,
                    caption=table_caption,
                    markdown=cleaned_markdown,
                    source=source,
                    extraction_mode=extraction_mode,
                    context_heading=table_heading,
                    flags=flags,
                )
            )
        current_lines = []
        table_caption = None
        table_heading = None

    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()
        if line.lstrip().startswith("|") and "|" in line.strip()[1:]:
            if not current_lines:
                table_caption = current_caption
                table_heading = current_heading
            current_lines.append(line)
            continue
        flush_table()
        caption = caption_from_line(line)
        if caption:
            current_caption = caption
        heading = heading_from_line(line)
        if heading:
            current_heading = heading

    flush_table()
    seen_captions: set[str] = set()
    for record in records:
        if not record.caption:
            continue
        if record.caption in seen_captions:
            record.flags = [*(record.flags or []), "duplicate_caption"]
        seen_captions.add(record.caption)
    return records


def captions_by_page(captions: list[TableCaption]) -> dict[int, list[TableCaption]]:
    grouped: dict[int, list[TableCaption]] = {}
    for caption in captions:
        grouped.setdefault(caption.page, []).append(caption)
    return grouped


def require_docling_converter():
    try:
        from docling.document_converter import DocumentConverter
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Docling is not installed. Install it with: python3 -m pip install docling"
        ) from exc
    return DocumentConverter


def create_docling_converter(*, ocr_enabled: bool):
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = ocr_enabled
    pipeline_options.do_table_structure = True
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )


def extract_page_range_pdf(
    source_pdf: Path, pages: list[int], output_pdf: Path
) -> None:
    if shutil.which("qpdf") is None:
        raise RuntimeError(
            "qpdf is required for selective page extraction but was not found."
        )
    if not pages:
        raise ValueError("pages must not be empty")
    page_range = str(pages[0]) if len(pages) == 1 else f"{pages[0]}-{pages[-1]}"
    subprocess.run(
        [
            "qpdf",
            str(source_pdf),
            "--pages",
            ".",
            page_range,
            "--",
            str(output_pdf),
        ],
        check=True,
    )


def extract_single_page_pdf(source_pdf: Path, page: int, output_pdf: Path) -> None:
    extract_page_range_pdf(source_pdf, [page], output_pdf)


def convert_pdf_to_markdown_with_converter(converter, pdf_path: Path) -> str:
    return convert_pdf_to_document_with_converter(
        converter, pdf_path
    ).export_to_markdown()


def convert_pdf_to_document_with_converter(converter, pdf_path: Path):
    result = converter.convert(str(pdf_path))
    return result.document


def convert_pdf_to_markdown(pdf_path: Path, *, ocr_enabled: bool) -> str:
    require_docling_converter()
    converter = create_docling_converter(ocr_enabled=ocr_enabled)
    return convert_pdf_to_markdown_with_converter(converter, pdf_path)


def convert_pdf_to_markdown_with_progress(
    converter,
    pdf_path: Path,
    *,
    progress: Callable[[str], None] | None,
    progress_label: str,
    progress_interval: float = 30.0,
) -> str:
    if progress is None:
        return convert_pdf_to_markdown_with_converter(converter, pdf_path)

    progress(f"[docling] {progress_label}: Docling convert started")
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            convert_pdf_to_markdown_with_converter, converter, pdf_path
        )
        while True:
            try:
                return future.result(timeout=progress_interval)
            except concurrent.futures.TimeoutError:
                elapsed = time.monotonic() - started
                progress(
                    f"[docling] {progress_label}: still converting after {elapsed:.0f}s"
                )


def convert_pdf_to_document_with_progress(
    converter,
    pdf_path: Path,
    *,
    progress: Callable[[str], None] | None,
    progress_label: str,
    progress_interval: float = 30.0,
):
    if progress is None:
        return convert_pdf_to_document_with_converter(converter, pdf_path)

    progress(f"[docling] {progress_label}: Docling convert started")
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            convert_pdf_to_document_with_converter, converter, pdf_path
        )
        while True:
            try:
                return future.result(timeout=progress_interval)
            except concurrent.futures.TimeoutError:
                elapsed = time.monotonic() - started
                progress(
                    f"[docling] {progress_label}: still converting after {elapsed:.0f}s"
                )


def item_label_value(item) -> str:
    label = getattr(item, "label", "")
    return str(getattr(label, "value", label))


def item_page_no(item) -> int | None:
    provenance = getattr(item, "prov", None)
    if not provenance:
        return None
    return getattr(provenance[0], "page_no", None)


def parent_ref(item) -> str:
    parent = getattr(item, "parent", None)
    return str(getattr(parent, "cref", parent or ""))


def item_to_page_markdown(item, document) -> str:
    label = item_label_value(item)
    if label == "table":
        return item.export_to_markdown(doc=document).strip()
    if label == "caption" and parent_ref(item).startswith("#/tables/"):
        return ""
    if label in {"page_header", "page_footer"}:
        return ""

    text = str(getattr(item, "text", "")).strip()
    if not text:
        return ""
    if label == "title":
        return f"# {text}"
    if label == "section_header":
        return f"## {text}"
    if label == "list_item":
        return f"- {text}"
    return text


def docling_document_to_page_markdown(document, pages: list[int]) -> dict[int, str]:
    relative_to_actual_page = dict(enumerate(pages, start=1))
    page_parts: dict[int, list[str]] = {page: [] for page in pages}

    for item, _level in document.iterate_items():
        relative_page = item_page_no(item)
        if relative_page is None:
            continue
        actual_page = relative_to_actual_page.get(relative_page)
        if actual_page is None:
            continue
        markdown = item_to_page_markdown(item, document)
        if markdown:
            page_parts[actual_page].append(markdown)

    return {page: "\n\n".join(parts).strip() for page, parts in page_parts.items()}


def split_markdown_pages(markdown_text: str, pages: list[int]) -> dict[int, str] | None:
    if len(pages) == 1:
        return {pages[0]: markdown_text.strip()}

    parts: list[list[str]] = []
    current: list[str] = []
    saw_marker = False
    for raw_line in markdown_text.splitlines():
        if DOCLING_PAGE_MARKER_RE.match(raw_line):
            saw_marker = True
            if current or parts:
                parts.append(current)
                current = []
            continue
        current.append(raw_line)
    if current or parts:
        parts.append(current)

    parts = [part for part in parts if "\n".join(part).strip()]
    if not saw_marker or len(parts) != len(pages):
        return None
    return {
        page: "\n".join(part).strip() for page, part in zip(pages, parts, strict=True)
    }


def extract_pages_with_docling(
    source_pdf: Path,
    pages: list[int],
    *,
    ocr_enabled: bool,
    batch_size: int = 50,
    progress: Callable[[str], None] | None = None,
) -> dict[int, str]:
    page_markdown: dict[int, str] = {}
    require_docling_converter()
    converter = create_docling_converter(ocr_enabled=ocr_enabled)
    batches = chunk_pages(pages, batch_size=batch_size)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for batch_index, batch_pages in enumerate(batches, start=1):
            batch_label = f"batch {batch_index}/{len(batches)} pages {batch_pages[0]}-{batch_pages[-1]}"
            if progress:
                progress(f"[docling] {batch_label}: start")
            batch_started = time.monotonic()
            batch_pdf = (
                tmp_dir
                / f"batch-{batch_index:04d}-{batch_pages[0]:04d}-{batch_pages[-1]:04d}.pdf"
            )
            extract_page_range_pdf(source_pdf, batch_pages, batch_pdf)
            batch_document = convert_pdf_to_document_with_progress(
                converter,
                batch_pdf,
                progress=progress,
                progress_label=batch_label,
            )
            page_markdown.update(
                docling_document_to_page_markdown(batch_document, batch_pages)
            )
            if progress:
                elapsed = time.monotonic() - batch_started
                progress(f"[docling] {batch_label}: converted in {elapsed:.1f}s")
    return page_markdown


def build_table_records(
    *,
    source_pdf: Path,
    captions: list[TableCaption],
    page_markdown: dict[int, str],
    extraction_mode: str = "text_layer",
) -> list[TableRecord]:
    records: list[TableRecord] = []
    previous_context_heading: str | None = None
    for page in sorted(page_markdown):
        page_records = extract_page_table_records(
            page=page,
            markdown_text=page_markdown[page],
            source=str(source_pdf),
            extraction_mode=extraction_mode,
        )
        for record in page_records:
            if record.context_heading:
                previous_context_heading = record.context_heading
                continue
            if record.caption is None and previous_context_heading:
                record.context_heading = previous_context_heading
                record.flags = [*(record.flags or []), "continued_context"]
        if page_records:
            records.extend(page_records)
    return records


def write_table_outputs(
    *,
    output_dir: Path,
    pages: list[int],
    captions: list[TableCaption],
    records: list[TableRecord],
    page_markdown: dict[int, str],
    extraction_mode: str = "text_layer",
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pages_dir = output_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    table_pages_path = output_dir / "table_pages.txt"
    captions_path = output_dir / "captions.jsonl"
    tables_jsonl_path = output_dir / "tables.jsonl"
    tables_md_path = output_dir / "tables.md"
    audit_path = output_dir / "table_audit.json"

    table_pages_path.write_text(
        "\n".join(str(page) for page in pages) + ("\n" if pages else ""),
        encoding="utf-8",
    )
    captions_path.write_text(
        "\n".join(
            json.dumps(asdict(caption), ensure_ascii=False, sort_keys=True)
            for caption in captions
        )
        + ("\n" if captions else ""),
        encoding="utf-8",
    )
    tables_jsonl_path.write_text(
        "\n".join(
            json.dumps(asdict(record), ensure_ascii=False, sort_keys=True)
            for record in records
        )
        + ("\n" if records else ""),
        encoding="utf-8",
    )

    table_md_parts: list[str] = []
    for record in records:
        table_md_parts.append(f"## {record.table_id}")
        table_md_parts.append(f"- Page: {record.page}")
        if record.caption:
            table_md_parts.append(f"- Caption: {record.caption}")
        if record.context_heading:
            table_md_parts.append(f"- Context: {record.context_heading}")
        if record.flags:
            table_md_parts.append(f"- Flags: {', '.join(record.flags)}")
        table_md_parts.append("")
        table_md_parts.append(record.markdown)
        table_md_parts.append("")
    tables_md_path.write_text("\n".join(table_md_parts), encoding="utf-8")

    for page, markdown in sorted(page_markdown.items()):
        (pages_dir / f"page-{page:04d}.docling.md").write_text(
            markdown.rstrip() + "\n", encoding="utf-8"
        )

    audit = {
        "pages": pages,
        "caption_count": len(captions),
        "table_count": len(records),
        "extraction_mode": extraction_mode,
        "flag_counts": {},
        "pages_without_markdown_tables": [
            page
            for page in pages
            if not extract_markdown_tables(page_markdown.get(page, ""))
        ],
    }
    for record in records:
        for flag in record.flags or []:
            audit["flag_counts"][flag] = audit["flag_counts"].get(flag, 0) + 1
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return {
        "table_pages": table_pages_path,
        "captions_jsonl": captions_path,
        "tables_jsonl": tables_jsonl_path,
        "tables_md": tables_md_path,
        "audit": audit_path,
    }


def process_tables(
    *,
    pdf_path: Path,
    cleaned_markdown_path: Path,
    output_dir: Path,
    pages: list[int] | None = None,
    ocr_enabled: bool = False,
    caption_pages_only: bool = False,
    batch_size: int = 50,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Path]:
    cleaned_text = cleaned_markdown_path.read_text(encoding="utf-8")
    captions = find_table_captions(cleaned_text)
    selected_pages = select_pages(
        explicit_pages=pages,
        caption_pages_only=caption_pages_only,
        captions=captions,
        pdf_page_count=get_pdf_page_count(pdf_path),
    )
    selected_captions = [
        caption for caption in captions if caption.page in set(selected_pages)
    ]
    extraction_mode = "ocr" if ocr_enabled else "text_layer"
    if progress:
        progress(
            f"[docling] selected {len(selected_pages)} pages; "
            f"batch_size={batch_size}; mode={extraction_mode}"
        )
    page_markdown = extract_pages_with_docling(
        pdf_path,
        selected_pages,
        ocr_enabled=ocr_enabled,
        batch_size=batch_size,
        progress=progress,
    )
    records = build_table_records(
        source_pdf=pdf_path,
        captions=selected_captions,
        page_markdown=page_markdown,
        extraction_mode=extraction_mode,
    )
    return write_table_outputs(
        output_dir=output_dir,
        pages=selected_pages,
        captions=selected_captions,
        records=records,
        page_markdown=page_markdown,
        extraction_mode=extraction_mode,
    )
