#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from seed_pipeline.config.paths import DOCLING_INTERIM_DIR, RAG_INTERIM_DIR
from seed_pipeline.corpus.processing.clean_markdown_corpus import (
    OCR_TEXT_REPAIRS,
    SPLIT_WORDS_REPAIRS,
    repair_split_syllables,
)

DEFAULT_INPUT_DIR = DOCLING_INTERIM_DIR
DEFAULT_SECTIONS = RAG_INTERIM_DIR / "sections.jsonl"
DEFAULT_CONTINUATION_OVERRIDES = DEFAULT_INPUT_DIR / "table_continuation_overrides.json"


@dataclass
class RawTable:
    table_id: str
    page: int
    table_index: int
    caption: str | None
    markdown: str
    source: str | None = None
    extraction_mode: str = "text_layer"
    context_heading: str | None = None
    flags: list[str] | None = None


@dataclass
class CaptionRecord:
    page: int
    caption: str
    line_number: int


@dataclass
class SectionRecord:
    id: str
    title: str
    section: str
    text: str
    start_page: int
    end_page: int
    content_type: str
    context_path: list[str]


@dataclass
class CuratedTable:
    table_id: str
    source_table_ids: list[str]
    page_start: int
    page_end: int
    caption: str | None
    canonical_caption: str | None
    context_heading: str | None
    table_type: str
    section_id: str | None
    section_match_method: str
    section_match_confidence: float
    merge_status: str
    markdown: str
    quality_flags: list[str]


@dataclass
class MergeGroup:
    raw: RawTable
    source_table_ids: list[str]
    page_start: int
    page_end: int
    merge_status: str


@dataclass
class ContinuationDecision:
    decision: str
    reason: str


@dataclass
class CurationResult:
    records: list[CuratedTable]
    continuation_merge_events: list[dict[str, Any]]
    continuation_review_items: list[dict[str, Any]]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, records: list[Any]) -> None:
    path.write_text(
        "".join(
            json.dumps(asdict(record), ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def load_raw_tables(input_dir: Path) -> list[RawTable]:
    return [RawTable(**item) for item in read_jsonl(input_dir / "tables.jsonl")]


def load_captions(input_dir: Path) -> list[CaptionRecord]:
    return [CaptionRecord(**item) for item in read_jsonl(input_dir / "captions.jsonl")]


def load_sections(path: Path) -> list[SectionRecord]:
    sections: list[SectionRecord] = []
    for item in read_jsonl(path):
        sections.append(
            SectionRecord(
                id=item["id"],
                title=item.get("title", ""),
                section=item.get("section", ""),
                text=item.get("text", ""),
                start_page=int(item.get("start_page") or 0),
                end_page=int(item.get("end_page") or 0),
                content_type=item.get("content_type", ""),
                context_path=list(item.get("context_path") or []),
            )
        )
    return sections


def load_continuation_overrides(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Continuation overrides must be a JSON object: {path}")
    overrides: dict[str, dict[str, Any]] = {}
    for table_id, value in payload.items():
        if not isinstance(value, dict):
            raise ValueError(f"Continuation override for {table_id} must be an object")
        action = value.get("action")
        if action not in {"merge_with_next", "separate"}:
            raise ValueError(
                f"Unsupported continuation override action for {table_id}: {action}"
            )
        overrides[str(table_id)] = dict(value)
    return overrides


def normalize_text(value: str | None) -> str:
    text = re.sub(r"\s+", " ", (value or "").strip().lower())
    decomposed = unicodedata.normalize("NFD", text)
    without_marks = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return without_marks.replace("đ", "d")


COMMON_MATCH_TOKENS = {
    "bang",
    "cac",
    "cho",
    "cua",
    "de",
    "duoc",
    "hoac",
    "khi",
    "la",
    "mot",
    "neu",
    "sau",
    "theo",
    "truoc",
    "va",
    "voi",
}


def content_tokens(value: str | None) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", normalize_text(value))
        if len(token) > 1 and token not in COMMON_MATCH_TOKENS
    }


def table_rows(markdown: str) -> list[str]:
    # Strip Unicode PUA chars (OCR bullet artifacts like U+F0B3) from entire markdown
    markdown = re.sub(r"[\uE000-\uF8FF]", "", markdown)
    return [
        line.rstrip() for line in markdown.splitlines() if line.strip().startswith("|")
    ]


def row_cells(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped.startswith("|"):
        return []
    return [cell.strip() for cell in stripped.strip("|").split("|")]


def normalize_markdown_row(line: str) -> str:
    cells = row_cells(line)
    if not cells:
        return line.rstrip()
    # Strip Unicode Private Use Area chars (OCR bullet artifacts) and orphaned ** / *** markers
    cleaned = []
    for cell in cells:
        cell = re.sub(r"[\uE000-\uF8FF]", "", cell)  # PUA chars
        cell = re.sub(r"\s*\*{2,}\s*$", "", cell)  # trailing ** or *** in cell
        cell = cell.strip()
        cleaned.append(cell)
    return "| " + " | ".join(cleaned) + " |"


def alpha_count(value: str) -> int:
    return len(re.findall(r"[A-Za-zÀ-ỹ]", value))


def digit_count(value: str) -> int:
    return len(re.findall(r"\d", value))


def is_separator_row(line: str) -> bool:
    return bool(
        re.fullmatch(r"\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*", line)
    )


def looks_like_data_row(line: str) -> bool:
    if is_separator_row(line):
        return False
    cells = [cell for cell in row_cells(line) if cell]
    if len(cells) < 2:
        return False
    joined = " ".join(cells)
    normalized = normalize_text(joined)
    if digit_count(joined) >= 2:
        return True
    data_terms = {
        "kg",
        "ml",
        "mg",
        "microgam",
        "phut",
        "gio",
        "ngay",
        "lan",
        "tiem",
        "uong",
        "truyen",
        "duong",
    }
    if any(term in normalized for term in data_terms) and digit_count(joined) >= 1:
        return True
    first = normalize_text(cells[0])
    return first.startswith(("-", "van", "tu", "hoac", "tiep", "duong"))


def column_count(markdown: str) -> int:
    for line in table_rows(markdown):
        if not is_separator_row(line):
            return len(row_cells(line))
    return 0


def data_rows_without_header(markdown: str) -> list[str]:
    rows = table_rows(markdown)
    if len(rows) <= 2:
        return []
    return [
        normalize_markdown_row(row) for row in rows[2:] if not is_separator_row(row)
    ]


def continuation_data_rows(markdown: str) -> list[str]:
    rows = table_rows(markdown)
    if not rows:
        return []
    first_row = rows[0]
    if looks_like_data_row(first_row):
        return [
            normalize_markdown_row(row) for row in rows if not is_separator_row(row)
        ]
    return data_rows_without_header(markdown)


def first_table_cells(markdown: str) -> list[str]:
    for row in table_rows(markdown):
        if not is_separator_row(row):
            return [cell for cell in row_cells(row) if cell]
    return []


def starts_with_new_table_header(raw: RawTable) -> bool:
    rows = table_rows(raw.markdown)
    first_row = rows[0] if rows else ""
    if first_row and looks_like_data_row(first_row):
        return False

    cells = first_table_cells(raw.markdown)
    if len(cells) < 2:
        return False
    first_cell = normalize_text(cells[0])
    if first_cell in {
        "chi dinh",
        "doi tuong",
        "tinh trang",
        "do thanh thai creatinin",
        "cl cr",
        "tuoi",
    }:
        return True
    alpha_cells = [cell for cell in cells if re.search(r"[A-Za-zÀ-ỹ]", cell)]
    numeric_heavy_cells = [
        cell for cell in cells if digit_count(cell) > alpha_count(cell)
    ]
    return (
        len(alpha_cells) >= 2
        and len(numeric_heavy_cells) == 0
        and len(" ".join(cells)) <= 180
    )


def compatible_columns(left: RawTable, right: RawTable) -> bool:
    left_count = column_count(left.markdown)
    right_count = column_count(right.markdown)
    return left_count > 0 and left_count == right_count


def markdown_preview(markdown: str, limit: int = 240) -> str:
    return re.sub(r"\s+", " ", markdown.strip())[:limit]


def continuation_event(
    left: RawTable, right: RawTable, decision: ContinuationDecision
) -> dict[str, Any]:
    return {
        "source_table_ids": [left.table_id, right.table_id],
        "page_start": min(left.page, right.page),
        "page_end": max(left.page, right.page),
        "context_heading": left.context_heading or right.context_heading,
        "decision": decision.decision,
        "reason": decision.reason,
        "left_preview": markdown_preview(left.markdown),
        "right_preview": markdown_preview(right.markdown),
    }


def allows_chained_continuation(raw: RawTable) -> bool:
    context = normalize_text(raw.context_heading)
    index_context_terms = (
        "cau truc ma atc",
        "danh muc cac chuyen luan thuoc",
    )
    return not any(term in context for term in index_context_terms)


def override_decision(
    left: RawTable,
    right: RawTable,
    overrides: dict[str, dict[str, Any]],
) -> ContinuationDecision | None:
    override = overrides.get(left.table_id)
    if not override:
        return None
    action = override.get("action")
    if action == "separate":
        return ContinuationDecision("separate", "override_forced_separate")
    if action == "merge_with_next":
        expected_next = override.get("next_table_id")
        if expected_next != right.table_id:
            raise ValueError(
                f"Continuation override for {left.table_id} must target the next adjacent table; "
                f"expected {right.table_id}, got {expected_next}"
            )
        if right.page not in {left.page, left.page + 1}:
            raise ValueError(
                f"Continuation override for {left.table_id} must target the next adjacent table by page; "
                f"got pages {left.page} and {right.page}"
            )
        return ContinuationDecision("merge", "override_forced_merge")
    return None


def continuation_decision(
    left: RawTable,
    right: RawTable,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> ContinuationDecision:
    forced = override_decision(left, right, overrides or {})
    if forced is not None:
        return forced

    if right.page not in {left.page, left.page + 1}:
        return ContinuationDecision("separate", "non_adjacent_page")
    if not compatible_columns(left, right):
        return ContinuationDecision("separate", "column_mismatch")

    left_caption = normalize_text(left.caption)
    right_caption = normalize_text(right.caption)
    right_flags = set(right.flags or [])
    if left_caption and left_caption == right_caption:
        return ContinuationDecision("merge", "same_caption")
    if "duplicate_caption" in right_flags:
        return ContinuationDecision("merge", "duplicate_caption")
    if (
        normalize_text(left.context_heading) == normalize_text(right.context_heading)
        and not right.caption
        and "continued_context" not in right_flags
    ):
        return ContinuationDecision("review", "same_context_missing_caption")
    if starts_with_new_table_header(right):
        return ContinuationDecision("separate", "new_header_detected")
    if "continued_context" in right_flags and normalize_text(
        left.context_heading
    ) == normalize_text(right.context_heading):
        rows = table_rows(right.markdown)
        if looks_like_data_row(rows[0] if rows else ""):
            return ContinuationDecision("merge", "numeric_data_continuation")
        return ContinuationDecision("merge", "continued_context_same_heading")

    rows = table_rows(right.markdown)
    first_row = rows[0] if rows else ""
    first_cell = row_cells(first_row)[0].strip().lower() if row_cells(first_row) else ""
    if first_cell.startswith(("-", "vấn", "tư", "hoặc", "tiếp", "đường")):
        return ContinuationDecision("merge", "leading_continuation_marker")
    return ContinuationDecision("separate", "no_continuation_signal")


def should_merge(
    left: RawTable, right: RawTable, overrides: dict[str, dict[str, Any]] | None = None
) -> bool:
    return continuation_decision(left, right, overrides).decision == "merge"


def merge_markdown(left: str, right: str) -> str:
    left_rows = [normalize_markdown_row(row) for row in table_rows(left)]
    right_rows = table_rows(right)
    if not right_rows:
        return "\n".join(left_rows).strip()

    first_right_row = right_rows[0]
    header_tokens = set(re.findall(r"\w+", normalize_text(left_rows[0])))
    right_row_tokens = set(re.findall(r"\w+", normalize_text(first_right_row)))

    is_replicated_header = False
    if (
        header_tokens
        and len(right_row_tokens & header_tokens) / len(header_tokens) > 0.7
    ):
        is_replicated_header = True

    if is_replicated_header:
        start_index = (
            2 if len(right_rows) > 1 and is_separator_row(right_rows[1]) else 1
        )
        right_data_rows = right_rows[start_index:]
    else:
        if len(right_rows) > 1 and is_separator_row(right_rows[1]):
            right_data_rows = [right_rows[0], *right_rows[2:]]
        else:
            right_data_rows = right_rows

    normalized_right_rows = [
        normalize_markdown_row(row)
        for row in right_data_rows
        if not is_separator_row(row)
    ]
    return "\n".join([*left_rows, *normalized_right_rows]).strip()


def merge_raw_tables(
    raw_tables: list[RawTable],
    continuation_overrides: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[MergeGroup], list[dict[str, Any]], list[dict[str, Any]]]:
    groups: list[MergeGroup] = []
    merge_events: list[dict[str, Any]] = []
    review_items: list[dict[str, Any]] = []
    consumed_override_ids: set[str] = set()
    index = 0
    while index < len(raw_tables):
        current = RawTable(**asdict(raw_tables[index]))
        source_ids = [current.table_id]
        page_start = current.page
        page_end = current.page
        merged_flags = list(current.flags or [])
        index += 1

        while index < len(raw_tables):
            pair_overrides = continuation_overrides
            if continuation_overrides and current.table_id in consumed_override_ids:
                pair_overrides = dict(continuation_overrides)
                pair_overrides.pop(current.table_id, None)
            decision = continuation_decision(current, raw_tables[index], pair_overrides)
            if decision.decision == "review":
                review_items.append(
                    continuation_event(current, raw_tables[index], decision)
                )
                break
            if decision.decision != "merge":
                break

            next_table = raw_tables[index]
            merge_events.append(continuation_event(current, next_table, decision))
            if decision.reason == "override_forced_merge":
                consumed_override_ids.add(current.table_id)
            source_ids.append(next_table.table_id)
            current.markdown = merge_markdown(current.markdown, next_table.markdown)
            page_start = min(page_start, next_table.page)
            page_end = max(page_end, next_table.page)
            if allows_chained_continuation(current):
                current.page = next_table.page
            merged_flags.extend(next_table.flags or [])
            merged_flags.append("merged_fragment")
            current.flags = sorted(set(merged_flags))
            index += 1

        groups.append(
            MergeGroup(
                raw=current,
                source_table_ids=source_ids,
                page_start=page_start,
                page_end=page_end,
                merge_status="merged" if len(source_ids) > 1 else "single",
            )
        )
    return groups, merge_events, review_items


def is_appendix_reference_context(value: str | None) -> bool:
    normalized = normalize_text(value)
    appendix_terms = (
        "bang tinh dien tich be mat than the",
        "phu luc",
        "muc luc",
        "danh muc",
    )
    return any(term in normalized for term in appendix_terms)


def classify_table(raw: RawTable, canonical_caption: str | None) -> str:
    context = normalize_text(raw.context_heading)
    caption = normalize_text(canonical_caption or raw.caption)
    header = normalize_text(
        raw.markdown.splitlines()[0] if raw.markdown.splitlines() else ""
    )
    if raw.page < 40 or "cong tac vien" in context or "ky hieu chu viet tat" in context:
        return "front_matter_table"
    if "danh muc cac chuyen luan thuoc" in context or "cau truc ma atc" in context:
        return "index_table"
    if is_appendix_reference_context(raw.context_heading) and not raw.caption:
        return "review_table"
    clinical_terms = (
        "lieu",
        "xu tri",
        "adr",
        "doc tinh",
        "inr",
        "hgb",
        "creatinin",
        "child-pugh",
        "phac do",
        "can nang",
        "the tich",
        "chi dinh",
        "muc do",
        "that bai",
        "dieu tri",
        "virus",
        "mien dich",
        "lam sang",
        "duoc ly",
        "duoc dong hoc",
        "dang thuoc",
        "ham luong",
        "do on dinh",
        "bao quan",
        "pho khang khuan",
        "do nhay cam",
        "mic",
        "rni",
        "dang bao che",
        "khoi dau tac dung",
        "thoi gian tac dung",
        "nong do",
        "dung moi",
        "ti le nhay cam",
        "ty le nhay cam",
        "tac dung toi da",
        "tac dung keo dai",
        "phan loai",
        "pho tac dung",
        "khang thuoc",
        "doi khang",
        "giai doc",
        "chat gay doc",
        "ngo doc",
        "cl cr",
        "clcr",
        "nua doi thai tru",
        "thai tru",
        "hbsag",
        "phoi nhiem",
        "tiem phong",
        "khang the",
        "viem gan b",
    )
    if any(
        term in context or term in caption or term in header for term in clinical_terms
    ):
        return "clinical_table"
    return "review_table"


def normalize_caption(
    raw: RawTable, captions: list[CaptionRecord], page_order: int
) -> str | None:
    same_page = [caption for caption in captions if caption.page == raw.page]
    if 0 <= page_order < len(same_page):
        return same_page[page_order].caption

    raw_caption = raw.caption or ""
    if raw_caption.strip().lower().startswith("bảng"):
        match = re.search(
            r"\bBảng\s+\d+[.:]?(?:\s+[^|]+)?", raw_caption, flags=re.IGNORECASE
        )
        if match:
            return re.sub(r"\s+", " ", match.group(0)).strip()
    return raw.caption


def table_text_overlap_match(
    raw: RawTable, sections: list[SectionRecord]
) -> tuple[str | None, float]:
    table_tokens = content_tokens(raw.markdown)
    if not table_tokens:
        return None, 0.0

    scored: list[tuple[int, float, SectionRecord]] = []
    for section in sections:
        overlap_count = len(table_tokens & content_tokens(section.text))
        overlap_ratio = overlap_count / len(table_tokens)
        scored.append((overlap_count, overlap_ratio, section))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    if not scored:
        return None, 0.0

    best_count, best_ratio, best_section = scored[0]
    second_count = scored[1][0] if len(scored) > 1 else 0
    if (
        best_count >= 8
        and best_ratio >= 0.35
        and (best_ratio >= 0.55 or best_count - second_count >= 4)
    ):
        return best_section.id, min(0.92, 0.65 + best_ratio * 0.25)
    return None, 0.0


def map_section(
    raw: RawTable, canonical_caption: str | None, sections: list[SectionRecord]
) -> tuple[str | None, str, float]:
    same_page = [
        section
        for section in sections
        if section.start_page <= raw.page <= section.end_page
    ]
    caption_norm = normalize_text(canonical_caption)
    if caption_norm:
        for section in same_page:
            if caption_norm in normalize_text(section.text):
                return section.id, "caption_text", 0.95

    overlap_section_id, overlap_confidence = table_text_overlap_match(raw, same_page)
    if overlap_section_id:
        return overlap_section_id, "table_text_overlap", overlap_confidence

    context_norm = normalize_text(raw.context_heading)
    if context_norm:
        for section in same_page:
            section_labels = normalize_text(
                " ".join([section.section, *section.context_path])
            )
            if context_norm in section_labels or section_labels in context_norm:
                return section.id, "context_heading", 0.85

    header_terms = [
        cell.strip()
        for cell in (raw.markdown.splitlines()[0] if raw.markdown.splitlines() else "")
        .strip("|")
        .split("|")
    ]
    header_norm = normalize_text(" ".join(header_terms))
    for section in same_page:
        if header_norm and header_norm in normalize_text(section.text):
            return section.id, "table_header_text", 0.75

    if same_page:
        best = min(
            same_page,
            key=lambda item: (item.end_page - item.start_page, len(item.text)),
        )
        return best.id, "page_range", 0.5
    return None, "unmapped", 0.0


def clean_table_markdown(markdown: str) -> str:
    cleaned = markdown
    cleaned = re.sub(r"[\uE000-\uF8FF]", "", cleaned)

    # 1. Stuck words with acronyms
    cleaned = re.sub(
        r"(?<=[a-zA-Zà-ỹ])(ALT|MAC|O|O2|NVP|EFV|TDF|LPV|AZT|CD4|HGB|HIV)\b",
        r" \1",
        cleaned,
    )
    cleaned = re.sub(
        r"\b(ALT|MAC|O|O2|NVP|EFV|TDF|LPV|AZT|CD4|HGB|HIV)(?=[a-zA-Zà-ỹ])",
        r"\1 ",
        cleaned,
    )

    # 2. Specific typos
    cleaned = cleaned.replace("NGỪNGngay", "NGỪNG ngay")
    cleaned = cleaned.replace("MACtrung", "MAC trung")
    cleaned = cleaned.replace("trịMAC", "trị MAC")
    cleaned = cleaned.replace("Giá trịMAC", "Giá trị MAC")
    cleaned = cleaned.replace("trongO 2", "trong O2")
    cleaned = cleaned.replace("khiALT", "khi ALT")
    cleaned = cleaned.replace("MA O", "MAO")
    cleaned = cleaned.replace("IMA O", "IMAO")
    cleaned = cleaned.replace("Natr i", "Natri")
    cleaned = cleaned.replace("N at r i", "Natri")
    cleaned = cleaned.replace("clo rid", "clorid")
    cleaned = cleaned.replace("clo r id", "clorid")
    for bad, good in OCR_TEXT_REPAIRS.items():
        cleaned = cleaned.replace(bad, good)
    cleaned = cleaned.replace("nông độ", "nồng độ")

    # 3. Space punctuation cleanups
    cleaned = re.sub(r"(?<=\w)\.(?=[A-ZÀ-Ỵ])", r". ", cleaned)
    cleaned = re.sub(r"(?<=\w),(?=[a-zA-Zà-ỹ])", r", ", cleaned)
    cleaned = re.sub(r"(?<=[A-Za-zÀ-Ỵà-ỹ]):(?=[A-Za-zÀ-Ỵà-ỹ])", r": ", cleaned)

    # 4. Normalize oC/o C/OC/O C to °C
    cleaned = re.sub(r"(\d+)\s*[oO]\s*C\b", r"\1 °C", cleaned)
    cleaned = re.sub(r"(\d+)\s*[oO]\n\s*C\b", r"\1 °C", cleaned)

    # Normalize unbalanced number ranges like 1- 2 or 1 -2 to 1 - 2
    cleaned = re.sub(r"(\b\d+(?:,\d+)?)-\s+(\d+(?:,\d+)?\b)", r"\1 - \2", cleaned)
    cleaned = re.sub(r"(\b\d+(?:,\d+)?)\s+-(\d+(?:,\d+)?\b)", r"\1 - \2", cleaned)

    # 5. Standard space cleanups
    has_trailing_newline = markdown.endswith("\n")
    cleaned_lines = []
    for line in cleaned.splitlines():
        row = re.sub(r"[ \t]{2,}", " ", line)
        row = re.sub(r"[ \t]+\|", " |", row)
        row = re.sub(r"\|[ \t]+", "| ", row)
        cleaned_lines.append(row)
    cleaned = "\n".join(cleaned_lines)
    if has_trailing_newline:
        cleaned += "\n"

    # 6. Normalize split words
    for bad, good in SPLIT_WORDS_REPAIRS.items():
        cleaned = cleaned.replace(bad, good)

    # 7. Unified space-split syllables repair
    cleaned = repair_split_syllables(cleaned)
    cleaned = re.sub(r"(\b\d+)(ngày|giờ|lần|tháng|năm|tuổi)\b", r"\1 \2", cleaned)

    # 8. Strip orphaned bold/footnote markers (2 or more stars)
    cleaned = re.sub(r"\*{2,}", "", cleaned)
    cleaned = re.sub(r"[ \t]+\|", " |", cleaned)

    return cleaned


def curate_table_result(
    raw_tables: list[RawTable],
    captions: list[CaptionRecord],
    sections: list[SectionRecord],
    continuation_overrides: dict[str, dict[str, Any]] | None = None,
) -> CurationResult:
    page_seen: dict[int, int] = {}
    curated: list[CuratedTable] = []

    merge_groups, merge_events, review_items = merge_raw_tables(
        raw_tables, continuation_overrides
    )
    for group in merge_groups:
        raw = group.raw
        raw.page = group.page_start
        page_order = page_seen.get(raw.page, 0)
        page_seen[raw.page] = page_order + 1
        canonical_caption = normalize_caption(raw, captions, page_order)
        table_type = classify_table(raw, canonical_caption)
        section_id, match_method, confidence = map_section(
            raw, canonical_caption, sections
        )
        curated.append(
            CuratedTable(
                table_id=f"curated-table-{group.page_start:04d}-{raw.table_index:03d}",
                source_table_ids=group.source_table_ids,
                page_start=group.page_start,
                page_end=group.page_end,
                caption=raw.caption,
                canonical_caption=canonical_caption,
                context_heading=raw.context_heading,
                table_type=table_type,
                section_id=section_id,
                section_match_method=match_method,
                section_match_confidence=confidence,
                merge_status=group.merge_status,
                markdown=clean_table_markdown(raw.markdown),
                quality_flags=list(raw.flags or []),
            )
        )
    return CurationResult(
        records=curated,
        continuation_merge_events=merge_events,
        continuation_review_items=review_items,
    )


def curate_tables(
    raw_tables: list[RawTable],
    captions: list[CaptionRecord],
    sections: list[SectionRecord],
) -> list[CuratedTable]:
    return curate_table_result(raw_tables, captions, sections).records


def write_curated_markdown(path: Path, records: list[CuratedTable]) -> None:
    parts: list[str] = []
    for record in records:
        parts.append(f"## {record.table_id}")
        parts.append(f"- Pages: {record.page_start}-{record.page_end}")
        parts.append(f"- Type: {record.table_type}")
        parts.append(f"- Section: {record.section_id or 'unmapped'}")
        if record.canonical_caption:
            parts.append(f"- Caption: {record.canonical_caption}")
        if record.quality_flags:
            parts.append(f"- Flags: {', '.join(record.quality_flags)}")
        parts.append("")
        parts.append(record.markdown)
        parts.append("")
    path.write_text("\n".join(parts), encoding="utf-8")


def review_table_reason(record: CuratedTable) -> str:
    if is_appendix_reference_context(record.context_heading) and not record.section_id:
        return "appendix_reference_table_not_rag_eligible"
    if not record.section_id:
        return "unmapped_review_table"
    return "requires_manual_review"


def build_audit(raw_count: int, result: CurationResult) -> dict[str, Any]:
    records = result.records
    type_counts: dict[str, int] = {}
    flag_counts: dict[str, int] = {}
    for record in records:
        type_counts[record.table_type] = type_counts.get(record.table_type, 0) + 1
        for flag in record.quality_flags:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
    remaining_review_tables = [
        record for record in records if record.table_type == "review_table"
    ]
    remaining_review_table_ids = [record.table_id for record in remaining_review_tables]
    remaining_review_table_items = [
        {
            "table_id": record.table_id,
            "page_start": record.page_start,
            "page_end": record.page_end,
            "section_id": record.section_id,
            "context_heading": record.context_heading,
            "reason": review_table_reason(record),
        }
        for record in remaining_review_tables
    ]
    promoted_review_table_count = sum(
        1
        for record in records
        if record.table_type == "clinical_table"
        and record.section_id
        and record.page_start >= 40
        and not record.caption
    )

    return {
        "raw_table_count": raw_count,
        "curated_table_count": len(records),
        "merge_count": sum(1 for record in records if record.merge_status == "merged"),
        "promoted_review_table_count": promoted_review_table_count,
        "remaining_review_table_count": len(remaining_review_table_ids),
        "remaining_review_table_ids": remaining_review_table_ids,
        "remaining_review_table_items": remaining_review_table_items,
        "continuation_merge_count": len(result.continuation_merge_events),
        "continuation_review_count": len(result.continuation_review_items),
        "continuation_merge_events": result.continuation_merge_events,
        "continuation_review_items": result.continuation_review_items,
        "table_type_counts": type_counts,
        "quality_flag_counts": flag_counts,
        "unmapped_table_count": sum(1 for record in records if not record.section_id),
        "review_table_ids": remaining_review_table_ids,
    }


def process_table_curation(
    *,
    input_dir: Path = DEFAULT_INPUT_DIR,
    sections_path: Path = DEFAULT_SECTIONS,
    continuation_overrides_path: Path | None = None,
) -> dict[str, Path]:
    input_dir = Path(input_dir)
    raw_tables = load_raw_tables(input_dir)
    captions = load_captions(input_dir)
    sections = load_sections(Path(sections_path))
    if continuation_overrides_path is None:
        continuation_overrides_path = input_dir / "table_continuation_overrides.json"
    continuation_overrides = load_continuation_overrides(continuation_overrides_path)
    result = curate_table_result(
        raw_tables, captions, sections, continuation_overrides=continuation_overrides
    )
    curated = result.records

    curated_jsonl = input_dir / "tables.curated.jsonl"
    curated_md = input_dir / "tables.curated.md"
    audit_path = input_dir / "table_curation_audit.json"
    write_jsonl(curated_jsonl, curated)
    write_curated_markdown(curated_md, curated)
    audit_path.write_text(
        json.dumps(
            build_audit(len(raw_tables), result),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "curated_jsonl": curated_jsonl,
        "curated_md": curated_md,
        "audit": audit_path,
    }
