import json
import os
import re
import unicodedata
from pathlib import Path

from corpus_pipeline.config.chunking import DEFAULT_CHUNK_MAX_CHARS
from corpus_pipeline.config.paths import (
    ANKHANG_MARKDOWN_INTERIM_DIR,
    RAG_FINAL_DIR,
    RAW_DIR,
)
from corpus_pipeline.corpus.metadata.payload_layers import compact_colloquial_mapping

SECTIONS_PATH = RAG_FINAL_DIR / "sections.jsonl"
CHUNKS_PATH = RAG_FINAL_DIR / "chunks.jsonl"
AUDIT_PATH = RAG_FINAL_DIR / "audit.json"
MANIFEST_PATH = RAG_FINAL_DIR / "manifest.json"
MAPPINGS_PATH = RAW_DIR / "colloquial_mappings.json"
MARKDOWN_DIR = ANKHANG_MARKDOWN_INTERIM_DIR

FULL_SECTION_MAX_CHARS = 16000
INSTRUCTION_LEAFLET_SOURCE = "Tờ hướng dẫn sử dụng"


def load_mappings(path: Path = MAPPINGS_PATH):
    if Path(path).exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def resolve_colloquial_mapping(slug: str, mappings: dict | None) -> dict:
    slug = str(slug or "").strip()
    empty = {"aliases": [], "visual_sign": "", "mapping_key": ""}
    if not slug or not mappings:
        return empty

    if slug in mappings:
        value = mappings.get(slug) or {}
        return {
            "aliases": [str(item) for item in value.get("aliases", [])],
            "visual_sign": str(value.get("visual_sign") or ""),
            "mapping_key": slug,
        }

    prefix_matches = [
        str(key)
        for key in mappings
        if str(key).startswith(f"{slug}-") or slug.startswith(f"{key}-")
    ]
    if len(prefix_matches) != 1:
        return empty

    key = prefix_matches[0]
    value = mappings.get(key) or {}
    return {
        "aliases": [str(item) for item in value.get("aliases", [])],
        "visual_sign": str(value.get("visual_sign") or ""),
        "mapping_key": key,
    }


def build_context_header(title: str) -> str:
    return f"{str(title or '').strip()}\n> Thông tin chi tiết".strip()


def normalize_inline_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def title_without_parenthetical(title: str) -> str:
    return normalize_inline_spaces(re.sub(r"\s*\([^)]*\)\s*$", "", title or ""))


def unique_text_values(values: list[str]) -> list[str]:
    seen = set()
    output = []
    for value in values:
        cleaned = normalize_inline_spaces(value).strip(" .;:-")
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        output.append(cleaned)
    return output


def product_names_from_title(title: str) -> list[str]:
    base = title_without_parenthetical(title)
    shortened = re.sub(
        r"\b(?:giảm|trị|điều trị|hỗ trợ)\b.*$",
        "",
        base,
        flags=re.IGNORECASE,
    ).strip()
    return unique_text_values([base, shortened]) or unique_text_values([title])


def split_table(table_text: str, max_chars: int) -> list[str]:
    lines = table_text.strip().split("\n")
    if len(lines) < 3 or not (
        lines[0].strip().startswith("|") and lines[1].strip().startswith("|")
    ):
        return [table_text]

    header_text = f"{lines[0]}\n{lines[1]}"
    header_len = len(header_text) + 1

    chunks = []
    current_rows = []
    current_len = 0

    for row in lines[2:]:
        row_len = len(row) + 1
        if current_rows and (header_len + current_len + row_len > max_chars):
            chunks.append(f"{header_text}\n" + "\n".join(current_rows))
            current_rows = [row]
            current_len = row_len
        else:
            current_rows.append(row)
            current_len += row_len

    if current_rows:
        chunks.append(f"{header_text}\n" + "\n".join(current_rows))
    return chunks


def split_oversized_paragraph(paragraph: str, max_chars: int) -> list[str]:
    parts = []
    remaining = paragraph.strip()

    while len(remaining) > max_chars:
        window = remaining[: max_chars + 1]
        split_at = -1

        sentence_matches = list(re.finditer(r"[.!?;:…]\s+", window))
        if sentence_matches:
            split_at = sentence_matches[-1].end()

        if split_at <= 0:
            newline_at = window.rfind("\n", 0, max_chars + 1)
            if newline_at > 0:
                split_at = newline_at + 1

        if split_at <= 0:
            whitespace_match = re.search(r"\s+\S*$", remaining[: max_chars + 1])
            if whitespace_match and whitespace_match.start() > 0:
                split_at = whitespace_match.start()

        if split_at <= 0:
            split_at = max_chars

        chunk = remaining[:split_at].strip()
        if chunk:
            parts.append(chunk)
        remaining = remaining[split_at:].strip()

    if remaining:
        parts.append(remaining)
    return parts


def chunk_text(text, max_chars=DEFAULT_CHUNK_MAX_CHARS):
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    current = ""

    for paragraph in paragraphs:
        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = ""
        if len(paragraph) <= max_chars:
            current = paragraph
            continue

        if paragraph.strip().startswith("|"):
            chunks.extend(split_table(paragraph, max_chars))
        else:
            chunks.extend(split_oversized_paragraph(paragraph, max_chars))

    if current:
        chunks.append(current)

    # Tự động sửa lỗi Markdown chưa đóng thẻ in đậm
    sanitized_chunks = []
    for chunk in chunks:
        chunk = strip_table_bold_artifacts(chunk)
        chunk = strip_orphan_bold_artifacts(chunk)
        chunk = strip_bold_markers(chunk)
        if chunk.count("**") % 2 != 0 and not chunk.rstrip().endswith("|"):
            chunk = chunk + "**"
        chunk = strip_orphan_bold_artifacts(chunk)
        chunk = strip_bold_markers(chunk)
        chunk = re.sub(r"\n\s*\*\*\s*$", "", chunk).strip()
        sanitized_chunks.append(chunk)

    return sanitized_chunks


def fix_missing_leading_pipes(text: str) -> str:
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if "|" in stripped and not stripped.startswith("|"):
            line = "| " + line.lstrip()
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def separate_malformed_table_lines(text: str) -> str:
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped == "":
            cleaned_lines.append(line)
            continue
        if not stripped.startswith("|"):
            if cleaned_lines and cleaned_lines[-1].strip().startswith("|"):
                cleaned_lines.append("")
            cleaned_lines.append(line)
        else:
            if (
                cleaned_lines
                and cleaned_lines[-1].strip() != ""
                and not cleaned_lines[-1].strip().startswith("|")
            ):
                cleaned_lines.append("")
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def convert_single_column_tables_to_text(text: str) -> str:
    text = fix_missing_leading_pipes(text)
    text = separate_malformed_table_lines(text)
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        if line.strip().startswith("|") and len(line) > 1500:
            cells = [c.strip() for c in line.split("|") if c.strip()]
            cleaned_lines.append("")
            for cell in cells:
                cleaned_lines.append(cell)
                cleaned_lines.append("")
        else:
            cleaned_lines.append(line)

    lines = cleaned_lines
    new_lines = []
    in_table = False
    table_lines = []

    for line in lines:
        if line.strip().startswith("|"):
            in_table = True
            table_lines.append(line)
        else:
            if in_table:
                new_lines.extend(process_table(table_lines))
                table_lines = []
                in_table = False
            new_lines.append(line)

    if in_table:
        new_lines.extend(process_table(table_lines))

    return strip_table_bold_artifacts("\n".join(new_lines))


def process_table_clean(table_lines: list[str]) -> list[str]:
    if len(table_lines) <= 2:
        has_sep = any(
            re.match(r"^\s*\|(?:\s*:?-+:?\s*\|)+\s*$", l) for l in table_lines
        )
        if has_sep:
            return []
    return table_lines


def strip_table_bold_artifacts(text: str) -> str:
    cleaned_lines = []
    for line in text.splitlines():
        if line.strip().startswith("|"):
            line = strip_bold_markers(line)
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def strip_orphan_bold_artifacts(text: str) -> str:
    cleaned_lines = []
    for line in text.splitlines():
        if line.strip() == "**":
            continue
        indent_len = len(line) - len(line.lstrip())
        indent = line[:indent_len]
        body = line[indent_len:]
        if body.startswith("** "):
            line = indent + body[3:]
        stripped = line.rstrip()
        if (
            stripped.endswith("**")
            and stripped.count("**") == 1
            and not stripped.lstrip().startswith("**")
        ):
            line = stripped[:-2].rstrip()
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def strip_bold_markers(text: str) -> str:
    text = re.sub(r"(?<=[0-9A-Za-zÀ-ỹĐđ])\*{2,}(?=[0-9A-Za-zÀ-ỹĐđ])", " ", text)
    return re.sub(r"\*{2,}", "", text)


def split_markdown_table_row(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped.startswith("|"):
        return []
    if stripped.endswith("**"):
        stripped = stripped[:-2].rstrip()
    stripped = stripped[1:-1] if stripped.endswith("|") else stripped[1:]
    return [cell.strip() for cell in stripped.split("|")]


def is_separator_cells(cells: list[str]) -> bool:
    return bool(cells) and all(re.match(r"^:?-{3,}:?$", cell.strip()) for cell in cells)


def format_markdown_table_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def normalize_multicolumn_table_rows(table_lines: list[str]) -> list[str]:
    widest_column_count = 0
    for line in table_lines:
        cells = split_markdown_table_row(line)
        widest_column_count = max(widest_column_count, len(cells))
    if widest_column_count <= 1:
        return table_lines

    normalized = []
    for line in table_lines:
        cells = split_markdown_table_row(line)
        if not cells:
            normalized.append(line)
            continue
        if is_separator_cells(cells):
            normalized.append(format_markdown_table_row(["---"] * widest_column_count))
            continue
        if len(cells) < widest_column_count:
            cells = [*cells, *([""] * (widest_column_count - len(cells)))]
        normalized.append(format_markdown_table_row(cells))
    return normalized


def process_table(table_lines: list[str]) -> list[str]:
    if len(table_lines) == 1:
        cells = split_markdown_table_row(table_lines[0])
        if is_separator_cells(cells):
            return []
        if len(cells) <= 1:
            content = (
                cells[0].strip() if cells else table_lines[0].strip().strip("|").strip()
            )
            return [content] if content else []
        return [format_markdown_table_row(cells)]
    if len(table_lines) < 2:
        return table_lines

    if (
        len(table_lines) >= 3
        and len(split_markdown_table_row(table_lines[0])) == 1
        and is_separator_cells(split_markdown_table_row(table_lines[1]))
        and len(split_markdown_table_row(table_lines[1])) == 1
        and any(len(split_markdown_table_row(line)) > 1 for line in table_lines[2:])
    ):
        caption = split_markdown_table_row(table_lines[0])[0]
        return [caption, "", *normalize_multicolumn_table_rows(table_lines[2:])]

    if all(
        line.strip().startswith("|") and line.count("|") <= 2 for line in table_lines
    ):
        plain_lines = []
        for line in table_lines:
            content = line.strip().strip("|").strip()
            if re.match(r"^:?-+:?$", content):
                continue
            if content:
                plain_lines.append(content)
        return plain_lines

    split_idx = -1
    for idx, line in enumerate(table_lines):
        if line.count("|") == 2 and len(line) > 1000:
            split_idx = idx
            break

    if split_idx != -1:
        before = table_lines[:split_idx]
        giant_row = table_lines[split_idx]
        after = table_lines[split_idx + 1 :]

        result_lines = []
        if before:
            result_lines.extend(process_table(before))

        content = giant_row.strip().strip("|").strip()
        if content:
            result_lines.append("")
            result_lines.append(content)
            result_lines.append("")

        if after:
            result_lines.extend(process_table(after))

        return result_lines
    else:
        is_single_col = True
        for line in table_lines:
            if line.count("|") != 2:
                is_single_col = False
                break

        if is_single_col:
            plain_lines = []
            for line in table_lines:
                content = line.strip().strip("|").strip()
                if re.match(r"^:?-+:?$", content):
                    continue
                if content:
                    plain_lines.append(content)
            return plain_lines
        else:
            return normalize_multicolumn_table_rows(table_lines)


def separate_ankhang_heading_runs(text: str) -> str:
    list_headings = [
        "Thận trọng khi sử dụng",
        "Thai kỳ và cho con bú",
        "Phụ nữ có thai và cho con bú",
        "Khả năng lái xe và vận hành máy móc",
        "Tương tác thuốc",
        "Tương kỵ của thuốc",
        "Bảo quản",
        "Hạn dùng",
        "Nhà sản xuất",
        "Thương hiệu",
    ]
    inline_headings = [
        "Mang thai",
        "Phụ nữ có thai",
        "Phụ nữ cho con bú",
        "Tương tác của thuốc",
        "Tương kỵ của thuốc",
        "Dược lực học",
        "Dược động học",
        "Cho con bú",
        "Phân loại dược lý",
        "Mã ATC",
        "Cơ chế tác dụng",
        "Tác dụng dược lý",
        "Các phối hợp không được khuyến cáo",
        "Phối hợp yêu cầu thận trọng",
        "Các phối hợp nên cân nhắc",
        "Liên quan đến Perindopril",
        "Liên quan đến Amlodipin",
    ]
    for heading in list_headings:
        text = re.sub(rf"(?<!\n)\s*-\s*{re.escape(heading)}", f"\n- {heading}", text)
        text = re.sub(rf"(- {re.escape(heading)})(?=[A-ZÀ-ỴĐ])", r"\1\n", text)
        text = re.sub(rf"({re.escape(heading)})(?=[A-ZÀ-ỴĐ])", r"\1\n", text)
    for heading in inline_headings:
        text = re.sub(rf"({re.escape(heading)})(?=[A-ZÀ-ỴĐ])", r"\1\n", text)
    return text


def separate_stuck_drug_and_vietnamese_words(text: str) -> str:
    chars = list(text)
    output = []
    for index, char in enumerate(chars):
        if index:
            prev = chars[index - 1]
            next_char = chars[index + 1] if index + 1 < len(chars) else ""
            if (prev.islower() and char.isupper() and next_char.islower()) or (
                prev.isdigit() and char.isupper() and next_char.islower()
            ):
                output.append(" ")
        output.append(char)
    text = "".join(output)

    boundary_words = [
        "Không",
        "Chưa",
        "Dự",
        "Dữ",
        "Đánh",
        "Dùng",
        "Khuyến",
        "Tương",
        "Các",
        "Chất",
        "Thuốc",
        "Kháng",
        "Ngoài",
        "Nếu",
        "Nhiễm",
        "Những",
        "Sử",
        "Mỗi",
        "Cả",
        "Cần",
        "Việc",
        "Tăng",
        "Giảm",
        "Rối",
    ]
    for word in boundary_words:
        text = re.sub(
            rf"(?<=[^\W_])(?={re.escape(word)}\b)", " ", text, flags=re.IGNORECASE
        )

    replacements = {
        "MedDRAPhân": "MedDRA Phân",
        "loạitheo": "loại theo",
        "bơquan": "cơ quan",
        "cơquan": "cơ quan",
        "MedDRA Phân loạitheo": "MedDRA Phân loại theo",
        "MedDRA Phân loại theo hệ bơquan": "MedDRA Phân loại theo hệ cơ quan",
        "MedDRA Phân loại theo hệ cơquan": "MedDRA Phân loại theo hệ cơ quan",
        "Rấtthườngxuyên": "Rất thường xuyên",
        "Thườngxuyên": "Thường xuyên",
        "khángcholinesterase": "kháng cholinesterase",
        "thảicreatinin": "thải creatinin",
        "tạithận": "tại thận",
        "CYP3M": "CYP3A4",
        "aP3A4": "CYP3A4",
        "phenobarţbital": "phenobarbital",
        "Phenobarţbital": "Phenobarbital",
        "hydroxypropỵl": "hydroxypropyl",
        "Hydroxypropỵl": "Hydroxypropyl",
        "acenocốumarốl": "acenocoumarol",
        "methylceíulose": "methylcellulose",
        "Methylceíulose": "Methylcellulose",
        "Actinobacỉllus": "Actinobacillus",
        "acetylsalicỵlic": "acetylsalicylic",
        "ỉurothiomalat": "aurothiomalat",
        "hydrochloriơnatri": "hydrochloric/natri",
        "loạnchuyểnhoá": "loạn chuyển hoá",
        "khửNS": "khử NS ",
        "sửlfamethoxazole": "sulfamethoxazole",
        "RO3AК07": "R03AK07",
        "Mä ATC": "Mã ATC",
        "CYPЗA4": "CYP3A4",
        "Cтax": "Cmax",
        "PNMTvàtrẻ": "PNMT và trẻ",
        "ThayATV": "Thay ATV",
        "điều trịARVvà": "điều trị ARV và",
        "bêndưới": "bên dưới",
        "tụỳ": "tụy",
        "phẳn": "phản",
        "tiếptheo": "tiếp theo",
        "hộichứng": "hội chứng",
        "bắt dau dieu tri bång": "bắt đầu điều trị bằng",
        "bång": "bằng",
        "nån": "nản",
        "Lån": "Lần",
        "Čhild": "Child",
        "chặt chē": "chặt chẽ",
        "Kēm": "Kẽm",
        "riêng rē": "riêng rẽ",
        "nhę": "nhẹ",
        "mę": "mẹ",
        "bác sī": "bác sĩ",
        "ĺch mẫu": "ích mẫu",
        "ĺt gặp": "Ít gặp",
        "benzył": "benzyl",
        "růi ro": "rủi ro",
        "AUCƮ": "AUCτ",
        "ǎn": "ăn",
        "lȧ": "là",
        "Bȧng": "Bảng",
        "Pḥòng": "Phòng",
        "ADṚ": "ADR",
        "thuỐC": "thuốc",
        "flo Voxamin": "fluvoxamin",
        "planzapine": "olanzapine",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    stuck_prefixes = [
        "như",
        "với",
        "tuổi",
        "thuốc",
        "giật",
        "cảm",
        "nấm",
        "uống",
        "toàn",
        "dẫn",
        "và",
        "bởi",
        "hợp",
        "dụng",
        "bằng",
        "của",
        "tăng",
        "giảm",
        "thải",
        "khác",
        "máu",
        "thư",
        "sản",
        "truyền",
        "triển",
        "thương",
        "đầu",
        "học",
        "áp",
        "dục",
        "chế",
        "các",
        "dùng",
        "giới",
        "kháng",
        "khuẩn",
    ]
    for prefix in stuck_prefixes:
        text = re.sub(
            rf"\b({prefix})(?=[A-Za-z][A-Za-z0-9]{{3,}})",
            r"\1 ",
            text,
            flags=re.IGNORECASE,
        )

    stuck_suffixes = [
        "không",
        "chưa",
        "các",
        "dữ",
        "tăng",
        "giảm",
        "dẫn",
        "nhạy",
        "toàn",
        "huyết",
        "hàng",
        "mỗi",
        "máu",
        "hoặc",
        "thuốc",
        "và",
        "chưa",
        "axít",
    ]
    for suffix in stuck_suffixes:
        text = re.sub(
            rf"(?<=[A-Za-z0-9])(?={suffix}\b)", " ", text, flags=re.IGNORECASE
        )

    text = re.sub(r"(?<=[A-Za-z0-9])và(?=[A-Za-z0-9])", " và ", text)
    text = re.sub(r"(?<=[A-Za-z])(?=N=)", " ", text)
    text = re.sub(r"(?<=CYP\d[A-Z]\d)(?=[A-Z]{2,}\b)", " ", text)
    text = re.sub(r"(?<=ức chế)(?=CYP)", " ", text, flags=re.IGNORECASE)
    return text


def remove_link_placeholder_sentences(text: str) -> str:
    placeholder_patterns = [
        r"(?im)^\s*(?:[-*]\s*)?(?:Xem thêm tại hướng dẫn sử dụng của thuốc:\s*Tại đây|Xem thêm(?: thông tin tại)? Hướng dẫn sử dụng của thuốc|Xem thêm tại đây|Tham khảo thêm tại đây|Vui lòng tham khảo Hướng dẫn sử dụng tại đây)\.\s*$",
        r"(?im)^\s*(?:[-*]\s*)?(?:Thử nghiệm lâm sàng|Các nghiên cứu lâm sàng|Hiệu quả lâm sàng và độ an toàn|Các số liệu nghiên cứu về tính an toàn tiền lâm sàng|Dữ liệu lâm sàng):\s*(?:Vui lòng\s*)?(?:xem(?: thêm| thông tin chi tiết)?|tham khảo(?: thêm)?(?: trong)?)[^.]*?tại đây\.\s*$",
        r"(?im)^\s*Vui lòng xem Hiệu quả lâm sàng của Symbicort đối với Hen suyển và COPD ở HDSD tại đây\.\s*$",
        r"(?im)^\s*Vui lòng tham khảo thêm trong Hướng dẫn sử dụng của thuốc tại đây\.\s*$",
        r"Bảng 3:\s*Tóm\s+tắt hiệu quả theo HAM-A:\s*xem tại đây\.\s*",
        r"Thời gian xảy ra các cơn lo âu tái phát,\s*đường cong Kaplan-Meier:\s*xem tại đây\.\s*",
        r"An toàn lâm sàng:\s*xem tại đây\.\s*",
    ]
    for pattern in placeholder_patterns:
        text = re.sub(pattern, "", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_pharmacokinetic_spacing(text: str) -> str:
    replacements = {
        "Adefovirdipivoxil": "Adefovir dipivoxil",
        "AUCinff": "AUCinf",
        "C max": "Cmax",
        "C min": "Cmin",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"(?<=[a-zà-ỹ])(?=(?:AUC(?:inf|tau)?|Cmax|Cmin)\b)", " ", text)
    text = re.sub(r"(?<=:)(?=[↔↑↓])", " ", text)
    text = re.sub(r"(?<=[A-Za-zÀ-ỹ0-9)])(?=[↔↑↓])", " ", text)
    text = re.sub(r"([↔↑↓])(?=\S)", r"\1 ", text)
    text = re.sub(r"(?<=\d%)(?=[A-Za-zÀ-ỹ(])", " ", text)
    text = re.sub(r"(?<=\))(?=[A-ZÀ-ỴĐ])", " ", text)
    text = re.sub(r"\b(Warfarin)\(", r"\1 (", text)
    return text


def clean_text_quality(text):
    # Normalize to NFC (composed form) to fix literal combining mark errors
    text = unicodedata.normalize("NFC", text)

    # Fix Icelandic Eth typo in Vietnamese text
    text = text.replace("Ð", "Đ").replace("ð", "đ")
    text = text.replace("৹", "°").replace("˚", "°")
    text = text.replace("\u0334", "≥")
    text = re.sub(r"\(\s*≥\s*", "(≥ ", text)

    # Fix forbidden/known text artifacts
    replacements = {
        "Nông độ": "Nồng độ",
        "nông độ": "nồng độ",
        "ºC": "°C",
        " dến ": " đến ",
        "mgngày": "mg/ngày",
        "thu ốc": "thuốc",
        "ho ạt": "hoạt",
        "huy ết": "huyết",
        "củ a": "của",
        "điề u tr ị": "điều trị",
        "creatnin": "creatinin",
        "dã chứng minh": "đã chứng minh",
        "dầy dủ": "đầy đủ",
        "dầy sừng": "dày sừng",
        "dầy lên": "dày lên",
        "kiển soát": "kiểm soát",
        "oxi hóa khứ": "oxi hóa khử",
        "mơ trước": "mg trước",
        "cóthể": "có thể",
        "Cóthể": "Có thể",
        "khảnăng": "khả năng",
        "chứcnăng": "chức năng",
        "bệnhnhân": "bệnh nhân",
        "cónguy": "có nguy",
        "nguycơ": "nguy cơ",
        "phảnứng": "phản ứng",
        "tủyxương": "tủy xương",
        "ngừngđiều": "ngừng điều",
        "hoặcngừng": "hoặc ngừng",
        "khingừng": "khi ngừng",
        "khôngmong": "không mong",
        "xácđịnh": "xác định",
        "hiếmgặp": "hiếm gặp",
        "baogồm": "bao gồm",
        "liềucao": "liều cao",
        "hôhấp": "hô hấp",
        "lồngngực": "lồng ngực",
        "vàtrung": "và trung",
        "trungthất": "trung thất",
        "ganmật": "gan mật",
        "sinhsản": "sinh sản",
        "tuyếnvú": "tuyến vú",
        "toànthân": "toàn thân",
        "tạichỗ": "tại chỗ",
        "dướida": "dưới da",
        "hệthần": "hệ thần",
        "hệmiễn": "hệ miễn",
        "hệsinh": "hệ sinh",
        "cơxương": "cơ xương",
        "môliên": "mô liên",
        "bạchhuyết": "bạch huyết",
        "thầnkinh": "thần kinh",
        "ngoạibiên": "ngoại biên",
        "huyếtthanh": "huyết thanh",
        "tếbào": "tế bào",
        "cáctác": "các tác",
        "dohạ": "do hạ",
        "đivệ": "đi vệ",
        "tấtcả": "tất cả",
        "tìnhtrạng": "tình trạng",
        "mổ hôi": "mồ hôi",
        "mồhôi": "mồ hôi",
        "mangnesihuyết": "magnesi huyết",
        "Mangnesi": "Magnesi",
        "mangnesi": "magnesi",
        "cảnhbáo": "cảnh báo",
        "thậntrọng": "thận trọng",
        "suytế": "suy tế",
        "ảnhhưởng": "ảnh hưởng",
        "sửdụng": "sử dụng",
        "bàitiết": "bài tiết",
        "dungnạp": "dung nạp",
        "mậtđộ": "mật độ",
        "C ảnh": "Cảnh",
        "C ẢNH": "CẢNH",
        "Thuốc hoặc hoạt chấtTương tác thuốcẢnh hưởng": "Thuốc hoặc hoạt chất\nTương tác thuốc\nẢnh hưởng",
        "đốm đầu từ trực tràng": "đốm dầu từ trực tràng",
        "thực quảng": "thực quản",
        "biểu kiển": "biểu kiến",
        "sử chuyển hóa": "sự chuyển hóa",
        "chấtlà": "chất là",
        "mỗ hôi": "mồ hôi",
        "Nêu liệu pháp": "Nếu liệu pháp",
        "dữ liêu": "dữ liệu",
        "hiểm gặp": "hiếm gặp",
        " đen <": " đến <",
        "thị giáca": "thị giác a",
        "phổid": "phổi d",
        "chảye": "chảy e",
        "nône": "nôn e",
        "Nône": "Nôn e",
        "bụngf": "bụng f",
        "với gani": "với gan i",
        "mỏim": "mỏi m",
        "Aldo-Uniόn": "Aldo-Unión",
        "polymerase ϒ": "polymerase γ",
        "Ϗ": "κ",
        "Cmax„": "Cmax",
        "lsomalt,:Maltodextrin": "Isomalt, Maltodextrin",
        "Phản ứng da.chân-tay": "Phản ứng da-chân tay",
        "khó th ở": "khó thở",
        "Khó th ở": "Khó thở",
        "tán suất": "tần suất",
        "tắn suất": "tần suất",
        "tẩn suất": "tần suất",
        "tẩn xuất": "tần suất",
        "Tán suat": "Tần suất",
        "tán suat": "tần suất",
        "tầnsuất": "tần suất",
        "Tầng bạch cáu": "Tăng bạch cầu",
        "tầng bạch cáu": "tăng bạch cầu",
        "Hổng ban": "Hồng ban",
        "phat ban": "phát ban",
        "miển nam": "miền nam",
        "A'lele": "Allele",
        "tròng vòng": "trong vòng",
        "chuyểnhóa": "chuyển hóa",
        "dạnguống": "dạng uống",
        "loạnmạch": "loạn mạch",
        "miễndịch": "miễn dịch",
        "miêndịch": "miễn dịch",
        "nhiễmkhuẩn": "nhiễm khuẩn",
        "Nhiễmkhuẩn": "Nhiễm khuẩn",
        "kýsinh": "ký sinh",
        "giảmbạch": "giảm bạch",
        "kinhngoại": "kinh ngoại",
        "loạnchung": "loạn chung",
        "loạntrên": "loạn trên",
        "mongmuốn": "mong muốn",
        "bụngtrên": "bụng trên",
        "candidathực": "candida thực",
        "Candidathực": "Candida thực",
        "Candidamiệng": "Candida miệng",
        "Candidasinh": "Candida sinh",
        "Candidađường": "Candida đường",
        "candidađường": "candida đường",
        "chặtđường": "chặt đường",
        "dụngđồng": "dụng đồng",
        "giáckích": "giác kích",
        "huyếtđường": "huyết đường",
        "loạnchuyển": "loạn chuyển",
        "loạnthận": "loạn thận",
        "nghiêmtrọng": "nghiêm trọng",
        "chấtchuyển": "chất chuyển",
        "cầutrung": "cầu trung",
        "dịchnghiêm": "dịch nghiêm",
        "dịchtiêm": "dịch tiêm",
        "liênquan": "liên quan",
        "loétđường": "loét đường",
        "chốngđông": "chống đông",
        "bónATiêu": "bón A Tiêu",
        "BónATiêu": "Bón A Tiêu",
        "cầuDARZALEX": "cầu DARZALEX",
        "chuyểnhoá": "chuyển hoá",
        "chứngBẢNG": "chứng BẢNG",
        "chứngđứng": "chứng đứng",
        "giácngon": "giác ngon",
        "Khôngphổ": "Không phổ",
        "trongthời": "trong thời",
        "chobệnh": "cho bệnh",
        "đến400": "đến 400",
        "vàongày": "vào ngày",
        "chođến": "cho đến",
        "choángváng": "choáng váng",
        "cựckhoái": "cực khoái",
        "tượngchoáng": "tượng choáng",
        "hammuốn": "ham muốn",
        "móc,cho": "móc, cho",
        "Ngất*Choáng": "Ngất Choáng",
        "chẩm dứt": "chấm dứt",
        "Nồng độ đỉnh trong dịch không thay đổi khi điều trị dài ngược sau khi uống thuốc 2 - 8 giờ": "Nồng độ đỉnh trong dịch não tủy đạt được sau khi uống thuốc 2 - 8 giờ",
        "loạnthính": "loạn thính",
        "thểhuyết": "thể huyết",
        "thốngmiễn": "thống miễn",
        "trongđiều": "trong điều",
        "biếnchứng": "biến chứng",
        "chứngkhác": "chứng khác",
        "chứngminh": "chứng minh",
        "chứngmắt": "chứng mắt",
        "chứngngoại": "chứng ngoại",
        "chứngngưng": "chứng ngưng",
        "dạngmethyl": "dạng methyl",
        "giáclạnh": "giác lạnh",
        "giảmtrọng": "giảm trọng",
        "huyếtcầu": "huyết cầu",
        "huyếthọc": "huyết học",
        "huyếtnghiêm": "huyết nghiêm",
        "huyếttrên": "huyết trên",
        "hóalipid": "hóa lipid",
        "hơnnhiều": "hơn nhiều",
        "hạchbạch": "hạch bạch",
        "khuẩnnhư": "khuẩn như",
        "khuẩnđường": "khuẩn đường",
        "khôngchọn": "không chọn",
        "khônggồm": "không gồm",
        "khôngphù": "không phù",
        "khôngphổ": "không phổ",
        "khôngthể": "không thể",
        "khôngđòi": "không đòi",
        "khôngđược": "không được",
        "khứugiác": "khứu giác",
        "kinhnguyệt": "kinh nguyệt",
        "kèmchứng": "kèm chứng",
        "loạndinh": "loạn dinh",
        "loạnkhác": "loạn khác",
        "loạnthần": "loạn thần",
        "loạntiêu": "loạn tiêu",
        "loạntiền": "loạn tiền",
        "lượngbạch": "lượng bạch",
        "lượngđiều": "lượng điều",
        "mạchthần": "mạch thần",
        "ngoạitháp": "ngoại tháp",
        "ngoạithất": "ngoại thất",
        "nhiềuthành": "nhiều thành",
        "nhiễmnấm": "nhiễm nấm",
        "nhiễmphổi": "nhiễm phổi",
        "nhiễmtoan": "nhiễm toan",
        "nhiễmtrùng": "nhiễm trùng",
        "nhiễmđộc": "nhiễm độc",
        "nhânthiếu": "nhân thiếu",
        "nhịpnhanh": "nhịp nhanh",
        "năngmiễn": "năng miễn",
        "nấmnhiễm": "nấm nhiễm",
        "phùquanh": "phù quanh",
        "theođường": "theo đường",
        "thiếuhụt": "thiếu hụt",
        "thángđiều": "tháng điều",
        "tháođường": "tháo đường",
        "thậngiai": "thận giai",
        "thểthiết": "thể thiết",
        "thốngthần": "thống thần",
        "tiêuchảy": "tiêu chảy",
        "trungtính": "trung tính",
        "trướngbụng": "trướng bụng",
        "trầmtrọng": "trầm trọng",
        "trọngkhi": "trọng khi",
        "viêmthần": "viêm thần",
        "viêmthận": "viêm thận",
        "vớinhiều": "với nhiều",
        "điềukiện": "điều kiện",
        "điềutrị": "điều trị",
        "đườngtiêm": "đường tiêm",
        "đườngtiêu": "đường tiêu",
        "rốiloạn": "rối loạn",
        "Rốiloạn": "Rối loạn",
        "hệbạch": "hệ bạch",
        "bạchcầu": "bạch cầu",
        "dạdày": "dạ dày",
        "nhịptim": "nhịp tim",
        "phảnvệ": "phản vệ",
        "phảnvệc": "phản vệ c",
        "ganứ": "gan ứ",
        "toànthể": "toàn thể",
        "thậnFORXIGA": "thận FORXIGA",
        "thậnTOPAMAX": "thận TOPAMAX",
        "CODEINELiên": "CODEINE Liên",
        "urichuyếtd": "uric huyết d",
        "HCITrọng lượng": "HCl. Trọng lượng",
        "HCItrọng lượng": "HCl. Trọng lượng",
        "với phẩn N": "với phần N",
        "xem phẩn": "xem phần",
        "Xem phẩn": "Xem phần",
        "thành phẩn": "thành phần",
        "Thành phẩn": "Thành phần",
        "một phẩn": "một phần",
        "ở phẩn là": "ở phân là",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = remove_link_placeholder_sentences(text)
    text = separate_ankhang_heading_runs(text)
    text = re.sub(r"(Chất (?:cảm ứng|ức chế))(?=CYP)", r"\1 ", text)
    text = re.sub(r"(CYP\d[A-Z]\d)(?=Chất)", r"\1\n", text)
    text = re.sub(r"(?<=:)(?=\d)", " ", text)
    text = re.sub(r"(\d)dến", r"\1 đến", text)
    text = separate_stuck_drug_and_vietnamese_words(text)
    text = normalize_pharmacokinetic_spacing(text)
    text = text.replace("M ức", "Mức").replace("Ch ức", "Chức").replace("Th ức", "Thức")
    text = text.replace("Tr ở", "Trở").replace("Nh ức", "Nhức")
    text = re.sub(r"(?<!\w)m ức\b", "mức", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<!\w)ch ức\b", "chức", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<!\w)th ức\b", "thức", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<!\w)tr ở\b", "trở", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<!\w)nh ức\b", "nhức", text, flags=re.IGNORECASE)
    text = re.sub(r"\buông\b", "uống", text, flags=re.IGNORECASE)
    text = text.replace("m Eq", "mEq")
    text = re.sub(r",{2,}", ",", text)
    text = re.sub(r"\.{4,}", "...", text)

    # Fix joined numeric units (e.g., 6mg -> 6 mg, 3ml -> 3 ml)
    # We remove the lookbehind so it also matches and fixes cases like 'Sitagliptin100mg'
    units = ["microgam", "mg", "ml", "kg", "cm2", "m2", "g"]
    for unit in units:
        pattern = re.compile(rf"(\d+)({unit})(?=\b|/)")
        text = pattern.sub(r"\1 \2", text)

    # Fix missing space after punctuation followed by capital letter (e.g., liều.Tuy -> liều. Tuy)
    text = re.sub(r"\.\.\.(?=[A-Za-zÀ-ỹ])", "... ", text)
    text = re.sub(r"\*+([.!?])(?=[A-ZÀ-ỴĐ])", r"\1 ", text)
    text = re.sub(r"(?<=\S)\s+([,;:!?])", r"\1", text)
    text = re.sub(r"(?<=\S)\s+\.(?=\s|$)", ".", text)
    text = re.sub(r"(?<=\S)\s+\.(?=[A-Za-zÀ-ỹ])", ". ", text)
    text = re.sub(r"(?<=\S)\s+([.!?])\s+(?=[A-ZÀ-ỴĐ])", r"\1 ", text)
    text = re.sub(r"(?<=\S)\s+([.!?])(\*+)(?=[A-ZÀ-ỴĐ])", r"\1\2 ", text)
    text = re.sub(r"(?<=\S)\s+([.!?])(?=[A-ZÀ-ỴĐ])", r"\1 ", text)
    text = re.sub(r"(?<=%)([:;,.!?])(\*+)(?=[A-ZÀ-ỴĐ])", r"\1\2 ", text)
    text = re.sub(r"(?<=%)([:;,.!?])(?=[A-ZÀ-ỴĐ])", r"\1 ", text)
    text = re.sub(r"(?<=[^\W_])([:;,.!?])(\*+)(?=[A-ZÀ-ỴĐ])", r"\1\2 ", text)
    text = re.sub(r"(?<=\))([:;,.!?])(\*+)(?=[A-ZÀ-ỴĐ])", r"\1\2 ", text)
    text = re.sub(r"(?<=[^\W_])([:;,.!?])(?=[A-ZÀ-ỴĐ])", r"\1 ", text)
    text = re.sub(r"(?<=\))([:;,.!?])(?=[A-ZÀ-ỴĐ])", r"\1 ", text)

    return text


def clean_post_table_punctuation_artifacts(text: str) -> str:
    text = re.sub(r"(?<=\|)\s*[:;,.!?]\s*(?=\|)", "  ", text)
    text = re.sub(r"(?<=\|)\s*[:;,.!?]\s+(?=[^|\n]*\|)", " ", text)
    text = re.sub(r"(?<=[.!?])\s*:\s+", " ", text)
    text = re.sub(r"(?<=\S)\s+([,;:!?])", r"\1", text)
    text = re.sub(r"(?<=\S)\s+\.(?=\s|$)", ".", text)
    return text


def remove_null_placeholder_lines(text: str) -> str:
    text = re.sub(r"(?im)^\s*null\s*$", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def has_meaningful_section_body(content: str) -> bool:
    body = re.sub(r"^# .*(?:\n|$)", "", content, count=1).strip()
    body = re.sub(r"(?m)^#{1,6}\s+.*$", "", body)
    return bool(re.search(r"[0-9A-Za-zÀ-ỹĐđ]{2,}", body))


def integrate_drug_file(file_path, *, markdown_root: Path, mappings=None):
    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    # Clean unicode and text quality typos before processing
    content = clean_unicode_text(content)
    content = clean_text_quality(content)

    # Convert single-column layout tables to plain text to prevent oversized chunks
    content = convert_single_column_tables_to_text(content)
    content = strip_bold_markers(strip_orphan_bold_artifacts(content))
    content = clean_post_table_punctuation_artifacts(content)
    content = remove_null_placeholder_lines(content)

    # Extract category and slug from path
    rel_path = os.path.relpath(file_path, markdown_root)
    category = os.path.dirname(rel_path)
    slug = os.path.splitext(os.path.basename(rel_path))[0]

    # Extract title
    title_match = re.match(r"^# (.*?)(?:\n|$)", content)
    title = title_match.group(1).strip() if title_match else "Không rõ tên thuốc"
    if not has_meaningful_section_body(content):
        return None, []

    colloquial = resolve_colloquial_mapping(slug, mappings)
    aliases = colloquial["aliases"]
    visual_sign = colloquial["visual_sign"]
    colloquial_mapping_key = colloquial["mapping_key"]

    section_id = f"brand:ankhang:{category}:{slug}"

    # Determine hydration strategy based on length of content
    strategy = (
        "chunk_window" if len(content) > FULL_SECTION_MAX_CHARS else "full_section"
    )
    context_path = ["Thông tin chi tiết"]
    context_header = build_context_header(title)
    colloquial_mapping = compact_colloquial_mapping(
        {
            "colloquial_mapping_key": colloquial_mapping_key,
            "product_aliases": aliases,
            "visual_sign": visual_sign,
            "product_names": product_names_from_title(title) or [title],
        }
    )

    # Create Section Record
    section_record = {
        "id": section_id,
        "content_type": "brand_page",
        "title": title,
        "section": "Thông tin chi tiết",
        "text": content,
        "source": INSTRUCTION_LEAFLET_SOURCE,
        "context_path": context_path,
        "context_header": context_header,
        "colloquial_mapping": colloquial_mapping,
        "start_page": 0,
        "end_page": 0,
        "warnings": [],
        "hydrate_strategy": strategy,
        "section_char_count": len(content),
    }

    # Create Chunk Records
    text_chunks = chunk_text(content)
    chunk_records = []
    for i, chunk_text_content in enumerate(text_chunks, 1):
        chunk_records.append(
            {
                "id": f"{section_id}:chunk-{i:03d}",
                "section_id": section_id,
                "content_type": "brand_page",
                "title": title,
                "section": "Thông tin chi tiết",
                "text": chunk_text_content,
                "context_path": context_path,
                "context_header": context_header,
                "colloquial_mapping": colloquial_mapping,
                "chunk_role": "prose",
                "chunk_content_type": "paragraph",
                "chunk_index": i,
                "start_page": 0,
                "end_page": 0,
                "warnings": [],
                "hydrate_strategy": strategy,
                "section_char_count": len(content),
                "source": INSTRUCTION_LEAFLET_SOURCE,
            }
        )

    return section_record, chunk_records


def clean_unicode_text(text):
    if not isinstance(text, str):
        return text
    import unicodedata

    # Normalize to NFC to merge combining marks
    text = unicodedata.normalize("NFC", text)
    replacements = {
        # Quotes
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        # Ellipsis
        "\u2026": "...",
        # Dashes / Hyphens
        "\u2013": "-",
        "\u2014": "-",
        "\u2011": "-",
        "\u2212": "-",
        # Spaces / Invisible characters
        "\u00a0": " ",
        "\u200b": "",
        "\u00ad": "",
        "\u200e": "",
        "\u200f": "",
        "\u2003": " ",
        # Bullets / Symbols
        "\u2022": "*",
        "\u2122": "(TM)",
        # Degree Celsius
        "\u2103": "°C",
        # Cyrillic lookalikes (homoglyphs) to Latin
        "\u0421": "C",  # Capital ES -> C
        "\u0430": "a",  # Small A -> a
        "\u0445": "x",  # Small HA -> x
        "\u0410": "A",  # Capital A -> A
        "\u0423": "Y",  # Capital U -> Y
        "\u0420": "P",  # Capital ER -> P
        "\u0422": "T",  # Capital TE -> T
        "\u0415": "E",  # Capital IE -> E
        # Greek lookalikes to Latin
        "\u039c": "M",  # Capital MU -> M
        "\u0392": "B",  # Capital BETA -> B
        # Dotless i
        "\u0131": "i",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def clean_unicode_value(val):
    if isinstance(val, str):
        return clean_unicode_text(val)
    elif isinstance(val, dict):
        return {
            k: unicodedata.normalize("NFC", v)
            if k == "source" and isinstance(v, str)
            else clean_unicode_value(v)
            for k, v in val.items()
        }
    elif isinstance(val, list):
        return [clean_unicode_value(v) for v in val]
    return val


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(clean_unicode_value(record), ensure_ascii=False) + "\n")


def is_ankhang_record(record: dict) -> bool:
    return str(record.get("id") or "").startswith("brand:ankhang:")


def integrate_ankhang_corpus(
    *,
    markdown_dir: Path,
    sections_path: Path,
    chunks_path: Path,
    mappings_path: Path,
    output_sections_path: Path,
    output_chunks_path: Path,
) -> tuple[int, int]:
    markdown_dir = Path(markdown_dir)
    if not markdown_dir.is_dir():
        raise FileNotFoundError(f"Markdown directory is missing: {markdown_dir}")
    mappings = load_mappings(mappings_path)
    md_files = sorted(markdown_dir.glob("**/*.md"))
    new_sections = []
    new_chunks = []

    for file_path in md_files:
        sec, chks = integrate_drug_file(
            file_path, markdown_root=markdown_dir, mappings=mappings
        )
        if sec is None:
            continue
        new_sections.append(sec)
        new_chunks.extend(chks)

    existing_sections = read_jsonl(sections_path)
    existing_chunks = read_jsonl(chunks_path)
    kept_sections = [
        section for section in existing_sections if not is_ankhang_record(section)
    ]
    kept_chunks = [chunk for chunk in existing_chunks if not is_ankhang_record(chunk)]
    final_sections = kept_sections + new_sections
    final_chunks = kept_chunks + new_chunks

    write_jsonl(output_sections_path, final_sections)
    write_jsonl(output_chunks_path, final_chunks)
    return len(new_sections), len(new_chunks)
