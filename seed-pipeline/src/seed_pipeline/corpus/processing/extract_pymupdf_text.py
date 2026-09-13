from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import fitz

from seed_pipeline.config.paths import RAW_DIR, TEXT_INTERIM_DIR

DEFAULT_PDF = RAW_DIR / "duoc-thu-quoc-gia-viet-nam.pdf"
DEFAULT_OUTPUT = TEXT_INTERIM_DIR / "full.md"

REGION_TITLES = {"CÁC CHUYÊN LUẬN CHUNG", "CÁC CHUYÊN LUẬN THUỐC", "CÁC PHỤ LỤC"}
FIELD_HEADINGS = {
    "Tên chung quốc tế",
    "Mã ATC",
    "Loại thuốc",
    "Dạng thuốc và hàm lượng",
    "Dược lý và cơ chế tác dụng",
    "Chỉ định",
    "Chống chỉ định",
    "Thận trọng",
    "Thời kỳ mang thai",
    "Thời kỳ cho con bú",
    "Tác dụng không mong muốn (ADR)",
    "Hướng dẫn cách xử trí ADR",
    "Liều lượng và cách dùng",
    "Tương tác thuốc",
    "Độ ổn định và bảo quản",
    "Tương kỵ",
    "Quá liều và xử trí",
}


@dataclass(frozen=True)
class LineRecord:
    page: int
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    bold: bool
    size: float


def normalize_line_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text.replace("\u00a0", " ")).strip()
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = re.sub(r"([([])\s+", r"\1", text)
    text = re.sub(r"\s+([])])", r"\1", text)
    return text


def parse_page_selection(value: str | None, page_count: int) -> list[int]:
    if value is None:
        return list(range(1, page_count + 1))
    pages: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            pages.update(range(start, end + 1))
        else:
            pages.add(int(part))
    ordered = sorted(pages)
    if not ordered or ordered[0] < 1 or ordered[-1] > page_count:
        raise ValueError(
            f"Page selection {value!r} outside PDF page range 1-{page_count}"
        )
    return ordered


def is_running_header(record: LineRecord) -> bool:
    text = record.text
    if record.y0 >= 50:
        return False
    return text == "DTQGVN 2" or text.isdigit() or len(text) <= 80


def line_records_from_page(page: fitz.Page, page_number: int) -> list[LineRecord]:
    records: list[LineRecord] = []
    page_dict = page.get_text("dict")
    if not isinstance(page_dict, dict):
        raise TypeError(f"PyMuPDF returned {type(page_dict).__name__} for text dict")
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = normalize_line_text("".join(span.get("text", "") for span in spans))
            if not text:
                continue
            x0, y0, x1, y1 = line["bbox"]
            record = LineRecord(
                page=page_number,
                text=text,
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                bold=any("bold" in span.get("font", "").lower() for span in spans),
                size=max((float(span.get("size", 0)) for span in spans), default=0.0),
            )
            if not is_running_header(record):
                records.append(record)
    return records


def ordered_body_lines(
    records: Iterable[LineRecord], page_width: float
) -> list[LineRecord]:
    records = list(records)
    if not records:
        return []
    left: list[LineRecord] = []
    right: list[LineRecord] = []
    midpoint = page_width / 2
    for record in records:
        center = (record.x0 + record.x1) / 2
        if center < midpoint:
            left.append(record)
        else:
            right.append(record)

    def key(item):
        return (round(item.y0, 1), round(item.x0, 1))

    if left and right:
        return sorted(left, key=key) + sorted(right, key=key)
    return sorted(records, key=key)


def is_upper_heading(text: str) -> bool:
    letters = [char for char in text if char.isalpha()]
    if len(letters) < 3:
        return False
    return sum(char.isupper() for char in letters) / len(letters) >= 0.8


def format_line(record: LineRecord) -> str:
    text = record.text
    if text in REGION_TITLES:
        return f"# {text}"
    if is_upper_heading(text) and (record.bold or record.size >= 11):
        return f"# {text}"
    if record.bold:
        stripped = text.rstrip(":")
        if stripped in FIELD_HEADINGS:
            return f"**{stripped}**"
        for heading in FIELD_HEADINGS:
            prefix = f"{heading}:"
            if text.startswith(prefix):
                rest = text[len(prefix) :].strip()
                return f"**{prefix}** {rest}".rstrip()
    return text


def extract_pdf_text(
    pdf_path: Path = DEFAULT_PDF, pages: list[int] | None = None
) -> str:
    parts: list[str] = []
    with fitz.open(pdf_path) as doc:
        selected_pages = (
            pages if pages is not None else list(range(1, doc.page_count + 1))
        )
        for page_number in selected_pages:
            page = doc.load_page(page_number - 1)
            parts.append(f"<!-- page {page_number:04d} -->")
            records = line_records_from_page(page, page_number)
            parts.extend(
                format_line(record)
                for record in ordered_body_lines(records, page.rect.width)
            )
            parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def process_pdf(
    pdf_path: Path = DEFAULT_PDF,
    output_path: Path = DEFAULT_OUTPUT,
    pages: list[int] | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(extract_pdf_text(pdf_path, pages=pages), encoding="utf-8")
    return output_path
