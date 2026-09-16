#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from pharma_lab.config.chunking import DEFAULT_CHUNK_MAX_CHARS
from pharma_lab.config.paths import RAG_INTERIM_DIR, TEXT_INTERIM_DIR
from pharma_lab.corpus.processing.clean_markdown_corpus import SPLIT_WORDS_REPAIRS

SOURCE_NAME = (
    "Dược thư Quốc gia Việt Nam (Xuất bản lần thứ 2) – Nhà xuất bản Y học, Hà Nội, 2018"
)


DEFAULT_INPUT = TEXT_INTERIM_DIR / "full.cleaned.md"
DEFAULT_OUTPUT_DIR = RAG_INTERIM_DIR


@dataclass
class SectionRecord:
    id: str
    content_type: str
    title: str
    section: str
    text: str
    source: str = SOURCE_NAME
    context_path: list[str] = field(default_factory=list)
    context_header: str = ""
    start_page: int | None = None
    end_page: int | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class ChunkRecord:
    id: str
    section_id: str
    content_type: str
    title: str
    section: str
    text: str
    source: str = SOURCE_NAME
    context_path: list[str] = field(default_factory=list)
    context_header: str = ""
    chunk_index: int = 1
    start_page: int | None = None
    end_page: int | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class AuditReport:
    drug_monographs: int
    general_monographs: int
    sections: int
    chunks: int
    longest_sections: list[dict]
    shortest_monographs: list[dict]
    unknown_section_labels: list[str]
    known_source_exceptions: list[dict]
    warnings: list[str]


PAGE_MARKER_RE = re.compile(r"<!--\s*page\s+(\d{4})\s*-->")
OCR_SPACING_REPAIRS = {
    "D ược": "Dược",
    "Dư ợc": "Dược",
    "H ướng": "Hướng",
    "Liều l ượng": "Liều lượng",
    "Liều lư ợng": "Liều lượng",
    "T ương": "Tương",
    "Tư ơng": "Tương",
    "hàm l ượng": "hàm lượng",
    "th ượng": "thượng",
    "chuyên giay - dược": "chuyên gia y - dược",
    "SỬ DỤNGAN": "SỬ DỤNG AN",
    "CHONGƯỜI": "CHO NGƯỜI",
    "\uf0ae": "→",
    "\uf0af": "↓",
}


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    without_marks = "".join(
        char for char in normalized if unicodedata.category(char) != "Mn"
    )
    return (
        without_marks.replace("Đ", "D")
        .replace("đ", "d")
        .replace("Ð", "D")
        .replace("ð", "d")
    )


def slugify(text: str) -> str:
    ascii_text = strip_accents(text).lower()
    ascii_text = re.sub(r"[^a-z0-9]+", "-", ascii_text)
    return ascii_text.strip("-")


def clean_label(line: str) -> str:
    cleaned = line.strip()
    cleaned = re.sub(r"^#{1,6}\s*", "", cleaned)
    cleaned = cleaned.replace("\\*", "*").replace("\\_", "_")
    cleaned = re.sub(r"^\*{1,3}", "", cleaned)
    cleaned = re.sub(r"\*{1,3}:", ":", cleaned)
    cleaned = re.sub(r":\*{1,3}", ":", cleaned)
    cleaned = re.sub(r"\*{1,3}$", "", cleaned)
    cleaned = cleaned.replace("*", "")
    for bad, good in OCR_SPACING_REPAIRS.items():
        cleaned = cleaned.replace(bad, good)
    cleaned = cleaned.strip()
    if cleaned.startswith("**") and cleaned.endswith("**"):
        cleaned = cleaned[2:-2].strip()
    return cleaned


def iter_page_lines(text: str) -> Iterable[tuple[int | None, str]]:
    current_page: int | None = None
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        marker = PAGE_MARKER_RE.search(raw_line)
        if marker:
            current_page = int(marker.group(1))
            continue
        line = raw_line.strip()
        if line:
            yield current_page, line


def record_to_json(record: SectionRecord | ChunkRecord | AuditReport) -> dict:
    return asdict(record)


SECTION_FIELD_PREFIXES = {
    "ten chung quoc te",
    "ma atc",
    "ma act",
    "loai thuoc",
    "dang thuoc va ham luong",
    "duoc ly va co che tac dung",
    "chi dinh",
    "chong chi dinh",
    "than trong",
    "thoi ky mang thai",
    "thoi ky cho con bu",
    "tac dung khong mong muon",
    "huong dan cach xu tri adr",
    "lieu luong va cach dung",
    "tuong tac thuoc",
    "do on dinh va bao quan",
    "tuong ky",
    "qua lieu va xu tri",
    "thong tin qui che",
    "ten thuong mai",
}
SOURCE_VERIFIED_PLAIN_DRUG_DESCRIPTORS = {
    ("SALBUTAMOL", "(Dùng trong hô hấp)"): "SALBUTAMOL (Dùng trong hô hấp)",
    ("SALBUTAMOL", "(Dùng trong sản khoa)"): "SALBUTAMOL (Dùng trong sản khoa)",
}

NON_DRUG_HEADINGS = {
    "cac chuyen luan chung",
    "cac chuyen luan thuoc",
    "cac phu luc",
    "muc luc tra cuu",
    "noi dung",
}
VIETNAMESE_ACCENTED_RE = re.compile(r"[À-Ỵà-ỵ]")


def normalized_label(line: str) -> str:
    return slugify(clean_label(line)).replace("-", " ")


def uppercase_ratio(text: str) -> float:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    uppercase = [char for char in letters if char.upper() == char]
    return len(uppercase) / len(letters)


def is_heading_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("#") or (
        stripped.startswith("**") and stripped.endswith("**") and len(stripped) <= 120
    )


def is_drug_title(line: str) -> bool:
    label = clean_label(line)
    normalized = normalized_label(label)
    title_core = re.sub(r"\s*\([^)]*\)\s*$", "", label).strip()
    if not is_heading_line(line):
        return False
    if "*" in label:
        return False
    if normalized in NON_DRUG_HEADINGS:
        return False
    if any(normalized.startswith(prefix) for prefix in SECTION_FIELD_PREFIXES):
        return False
    if label.count("(") != label.count(")"):
        return False
    if len(label) < 3 or len(label) > 90:
        return False
    alpha_count = sum(1 for char in title_core if char.isalpha())
    if alpha_count <= 4 and not VIETNAMESE_ACCENTED_RE.search(label):
        return False
    return uppercase_ratio(title_core) >= 0.75


def extract_main_lines(text: str) -> list[tuple[str, int | None, str]]:
    region: str | None = None
    output: list[tuple[str, int | None, str]] = []

    for page, line in iter_page_lines(text):
        normalized = normalized_label(line)
        if normalized == "cac chuyen luan chung":
            region = "general"
            continue
        if normalized == "cac chuyen luan thuoc":
            region = "drug"
            continue
        if normalized in {"cac phu luc", "muc luc tra cuu"}:
            region = None
            continue
        if region in {"general", "drug"}:
            output.append((region, page, line))

    lines = merge_split_official_general_titles(output)
    return merge_split_drug_titles(lines)


def merge_split_official_general_titles(
    lines: list[tuple[str, int | None, str]],
) -> list[tuple[str, int | None, str]]:
    merged: list[tuple[str, int | None, str]] = []
    index = 0
    while index < len(lines):
        region, page, line = lines[index]
        if (
            region == "general"
            and index + 1 < len(lines)
            and lines[index + 1][0] == "general"
            and is_heading_line(line)
            and is_heading_line(lines[index + 1][2])
        ):
            combined_label = clean_label(f"{line} {clean_label(lines[index + 1][2])}")
            if combined_label in OFFICIAL_GENERAL_MONOGRAPHS:
                merged.append((region, page, f"# {combined_label}"))
                index += 2
                continue
        merged.append((region, page, line))
        index += 1
    return merged


def merge_split_drug_titles(
    lines: list[tuple[str, int | None, str]],
) -> list[tuple[str, int | None, str]]:
    KNOWN_SPLIT_DRUGS = {
        (
            "VẮC XIN HAEMOPHILUS INFLUENZAE TYP B",
            "CỘNG HỢP",
        ): "VẮC XIN HAEMOPHILUS INFLUENZAE TYP B CỘNG HỢP",
        (
            "GLOBULIN MIỄN DỊCH CHỐNG UỐN VÁN VÀ",
            "HUYẾT THANH CHỐNG UỐN VÁN (NGỰA)",
        ): "GLOBULIN MIỄN DỊCH CHỐNG UỐN VÁN VÀ HUYẾT THANH CHỐNG UỐN VÁN (NGỰA)",
        (
            "GLOBULIN MIỄN DỊCH KHÁNG DẠI",
            "VÀ HUYẾT THANH KHÁNG DẠI",
        ): "GLOBULIN MIỄN DỊCH KHÁNG DẠI VÀ HUYẾT THANH KHÁNG DẠI",
        (
            "THUỐC TƯƠNG TỰ HORMON GIẢI PHÓNG",
            "GONADOTROPIN",
        ): "THUỐC TƯƠNG TỰ HORMON GIẢI PHÓNG GONADOTROPIN",
    }

    merged: list[tuple[str, int | None, str]] = []
    index = 0
    while index < len(lines):
        region, page, line = lines[index]
        if (
            region == "drug"
            and index + 1 < len(lines)
            and lines[index + 1][0] == "drug"
        ):
            h1 = clean_label(line)
            h2 = clean_label(lines[index + 1][2])
            if (
                is_heading_line(line)
                and is_heading_line(lines[index + 1][2])
                and (h1, h2) in KNOWN_SPLIT_DRUGS
            ):
                combined_title = KNOWN_SPLIT_DRUGS[(h1, h2)]
                merged.append((region, page, f"# {combined_title}"))
                index += 2
                continue
            if (
                is_drug_title(line)
                and (h1, h2) in SOURCE_VERIFIED_PLAIN_DRUG_DESCRIPTORS
            ):
                combined_title = SOURCE_VERIFIED_PLAIN_DRUG_DESCRIPTORS[(h1, h2)]
                merged.append((region, page, f"# {combined_title}"))
                index += 2
                continue
        merged.append((region, page, line))
        index += 1
    return merged


CANONICAL_SECTION_LABELS = {
    "ten chung quoc te": "Tên chung quốc tế",
    "ma atc": "Mã ATC",
    "ma act": "Mã ATC",
    "loai thuoc": "Loại thuốc",
    "dang thuoc va ham luong": "Dạng thuốc và hàm lượng",
    "duoc ly va co che tac dung": "Dược lý và cơ chế tác dụng",
    "chi dinh": "Chỉ định",
    "chong chi dinh": "Chống chỉ định",
    "than trong": "Thận trọng",
    "thoi ky mang thai": "Thời kỳ mang thai",
    "thoi ky cho con bu": "Thời kỳ cho con bú",
    "tac dung khong mong muon adr": "Tác dụng không mong muốn (ADR)",
    "tac dung khong mong muon": "Tác dụng không mong muốn (ADR)",
    "huong dan cach xu tri adr": "Hướng dẫn cách xử trí ADR",
    "lieu luong va cach dung": "Liều lượng và cách dùng",
    "lieu luong va cach su dung": "Liều lượng và cách dùng",
    "lieu luong cach dung": "Liều lượng và cách dùng",
    "cach dung va lieu luong": "Liều lượng và cách dùng",
    "lieu dung va cach dung": "Liều lượng và cách dùng",
    "tuong tac thuoc": "Tương tác thuốc",
    "do on dinh va bao quan": "Độ ổn định và bảo quản",
    "bao quan": "Độ ổn định và bảo quản",
    "tuong ky": "Tương kỵ",
    "qua lieu va xu tri": "Quá liều và xử trí",
    "thong tin qui che": "Thông tin qui chế",
    "ten thuong mai": "Tên thương mại",
}
INLINE_SECTION_MARKER_RE = re.compile(
    r"(\*{2,3}(?:"
    + "|".join(
        re.escape(label)
        for label in sorted(
            set(CANONICAL_SECTION_LABELS.values()), key=len, reverse=True
        )
    )
    + r")\*{2,3})"
)
INLINE_NUMBERED_HEADING_RE = re.compile(
    r"(\*{2,3}\d+(?:\.\d+)*\.\s+[^*\n]{2,160}\*{2,3})"
)
LEADING_BOLD_HEADING_RE = re.compile(r"^(\*{2,3}[^*\n]{2,120}\*{2,3})\s+(.+)$")


def split_inline_section_markers(line: str) -> list[str]:
    parts = [line]
    for splitter in (INLINE_SECTION_MARKER_RE, INLINE_NUMBERED_HEADING_RE):
        split_parts: list[str] = []
        for part in parts:
            split_parts.extend(splitter.split(part))
        parts = split_parts
    split_parts = []
    for part in parts:
        match = LEADING_BOLD_HEADING_RE.match(part.strip())
        if match:
            remainder = match.group(2).strip()
            if remainder.startswith(":"):
                split_parts.append(f"{match.group(1)} {remainder}")
            else:
                split_parts.extend([match.group(1), remainder])
        else:
            split_parts.append(part)
    parts = split_parts
    parts = [part.strip() for part in parts if part.strip()]
    recombined: list[str] = []
    index = 0
    while index < len(parts):
        current = parts[index]
        if (
            index + 1 < len(parts)
            and normalized_label(clean_label(current)) in CANONICAL_SECTION_LABELS
            and parts[index + 1].lstrip().startswith(":")
        ):
            recombined.append(f"{current} {parts[index + 1].lstrip()}")
            index += 2
            continue
        recombined.append(current)
        index += 1
    parts = recombined
    return parts or [line]


def section_label_from_line(line: str) -> tuple[str | None, str]:
    if line.strip().startswith("|"):
        return None, line

    label = clean_label(line)
    if not label:
        return None, line

    stripped = line.strip()
    is_markdown_heading = (
        stripped.startswith("#")
        or stripped.startswith("**")
        or stripped.startswith("***")
    )

    normalized = normalized_label(label)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    is_exact_label = normalized in CANONICAL_SECTION_LABELS

    # Strict validation for plain text lines
    if not is_markdown_heading:
        # If it's not a markdown heading, we ONLY allow exact matches
        if not is_exact_label:
            return None, line
        # The exact match must not end with a period and must start with an uppercase letter
        if label.endswith("."):
            return None, line
        first_char = label[0]
        if first_char.isalpha() and first_char.islower():
            return None, line

    if normalized in CANONICAL_SECTION_LABELS:
        return CANONICAL_SECTION_LABELS[normalized], ""

    for key, canonical in CANONICAL_SECTION_LABELS.items():
        if normalized.startswith(key + " "):
            remainder = re.sub(r"^[^:]{1,80}:\**\s*", "", label).strip()
            return canonical, remainder

    return None, line


def is_general_section_heading(line: str) -> bool:
    label = clean_label(line)
    if not is_heading_line(line) and not is_plain_general_hierarchy_heading(label):
        return False
    if is_drug_title(line) and not is_numbered_general_heading_label(label):
        return False
    return not (len(label) < 3 or len(label) > 120)


def is_general_monograph_title(line: str) -> bool:
    stripped = line.strip()
    label = clean_label(line)
    if not stripped.startswith("#"):
        return False
    if is_parenthetical_acronym_heading(label):
        return False
    # Ignore titles that are not general monographs
    normalized = normalized_label(label)
    if normalized in NON_DRUG_HEADINGS:
        return False
    return label in OFFICIAL_GENERAL_MONOGRAPHS


def is_parenthetical_acronym_heading(label: str) -> bool:
    return bool(re.fullmatch(r"\([A-Z0-9/ +.-]{2,}\)", label))


def is_numbered_general_heading_label(label: str) -> bool:
    return bool(re.match(r"^(?:[A-ZĐ]|[IVXLCDM]+|\d+(?:\.\d+)*)\.\s+", label))


def is_plain_general_hierarchy_heading(label: str) -> bool:
    return is_numbered_general_heading_label(label) and uppercase_ratio(label) >= 0.65


def general_heading_level(label: str, hierarchy: list[tuple[int, str]]) -> int:
    if is_parenthetical_acronym_heading(label):
        return hierarchy[-1][0] + 1 if hierarchy else 4
    if re.match(r"^[IVXLCDM]+\.\s+", label):
        return 2
    if re.match(r"^[A-ZĐ]\.\s+", label):
        return 1
    numbered = re.match(r"^(\d+(?:\.\d+)*)\.\s+", label)
    if numbered:
        return 2 + len(numbered.group(1).split("."))
    return 3 if hierarchy else 1


def build_context_header(title: str, context_path: list[str]) -> str:
    lines = [title, *context_path]
    return "\n> ".join(line for line in lines if line)


def clean_section_line(line: str) -> str:
    cleaned = line.strip()
    if cleaned.startswith("#"):
        cleaned = re.sub(r"^#{1,6}\s*", "", cleaned).strip()
    cleaned = re.sub(r"\s+#{1,6}\s+", " ", cleaned)
    cleaned = cleaned.replace("\\*", "")
    cleaned = cleaned.replace("*", "")
    for bad, good in OCR_SPACING_REPAIRS.items():
        cleaned = cleaned.replace(bad, good)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    # Fix missing space after punctuation followed by capital letter (e.g., chứng:Cục -> chứng: Cục)
    cleaned = re.sub(r"([A-Za-zÀ-Ỵà-ỵ0-9][:;,.])([A-ZÀ-ỴĐ])", r"\1 \2", cleaned)
    return cleaned.strip()


def clean_section_text(lines: list[str]) -> str:
    cleaned = [clean_section_line(line) for line in lines if line.strip()]
    cleaned = [line for line in cleaned if line]
    text = "\n".join(cleaned).strip()
    for bad, good in SPLIT_WORDS_REPAIRS.items():
        text = text.replace(bad, good)
    text = repair_targeted_punctuation_artifacts(text)
    return text


def repair_targeted_punctuation_artifacts(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    repairs = {
        "khí:Các": "khí: Các",
        "thu:Thuốc": "thu: Thuốc",
        "tuổi:Uống": "tuổi: Uống",
        "Hô hấp:Viêm": "Hô hấp: Viêm",
        "50%).Vì": "50%). Vì",
        "Suy thận:Không": "Suy thận: Không",
        "hen:Viêm": "hen: Viêm",
        "lít).Tuy nhiên": "lít). Tuy nhiên",
        "mắt...có": "mắt... có",
        "v.v...Các": "v.v... Các",
        "hấp...Vì": "hấp... Vì",
        "1mg/ngày).Các": "1 mg/ngày). Các",
        "Aviranz;Efavula": "Aviranz; Efavula",
        "20 ml;1 g/2 ml": "20 ml; 1 g/2 ml",
        "%);. Gan:": "%). Gan:",
        "ºC": "°C",
        " dến ": " đến ",
        "creatnin": "creatinin",
        "sentivity": "sensitivity",
        "Hight density": "High density",
        "Đích 37;": "Đích 3,7;",
        "mgngày": "mg ngày",
        "uèng": "uống",
        "tiªm": "tiêm",
        "d\u00adíi": "dưới",
        "tÜnh": "tĩnh",
        "m¹ch": "mạch",
        "Dtĩnh mạch": "D tĩnh mạch",
        "AUCtĩnh mạch": "AUC tĩnh mạch",
        "AUCtĩnh": "AUC tĩnh",
        "AUCuống": "AUC uống",
        "HCO-\n3+ H+ → H2 CO3 → CO2 + H2 O": "HCO3- + H+ → H2CO3 → CO2 + H2O",
        "H2 CO3": "H2CO3",
        "H2 O": "H2O",
        "Na2S2 O3": "Na2S2O3",
        "Na2 SO3": "Na2SO3",
        "O meprazole": "Omeprazole",
        "O meprazol": "Omeprazol",
        "O floxacin": "Ofloxacin",
        "O xcarbazepin": "Oxcarbazepin",
        "O htahara": "Ohtahara",
        "O rlistat": "Orlistat",
        "thân trong sốc nhiễm khuẩn": "thận trọng trong sốc nhiễm khuẩn",
        "PNMTvàtrẻ": "PNMT và trẻ",
        "ThayATV": "Thay ATV",
        "điều trịARVvà": "điều trị ARV và",
    }
    for bad, good in repairs.items():
        text = text.replace(bad, good)

    text = re.sub(r"(?:\.\s*){4,}", "...", text)
    text = re.sub(r"(?<=\S)\s+\.{3}", "...", text)
    text = re.sub(r"\.\.\.(?=[A-Za-zÀ-ỹ])", "... ", text)
    text = re.sub(r"(?<=\S)\s+([,;:!?])", r"\1", text)
    text = re.sub(r"(?<=\S)\s+\.(?=\s|$)", ".", text)
    text = re.sub(r"(?<=\S)\s+\.(?=[A-Za-zÀ-ỹ])", ". ", text)
    text = re.sub(r"\bO\s+([a-zà-ỹ]{2,})", r"O\1", text)
    text = re.sub(r"(?<=\d)(microgam|mg|ml|kg|cm2|m2|g)(?=\b|/)", r" \1", text)

    species_abbreviations = ("E", "H", "K", "S", "B")
    for abbreviation in species_abbreviations:
        text = re.sub(rf"\b{abbreviation}\.([a-z])", rf"{abbreviation}. \1", text)
    text = re.sub(r"\bStaph\.([a-z])", r"Staph. \1", text)
    return text


def make_section_id(
    content_type: str,
    title: str,
    section: str,
    *,
    context_path: list[str] | None = None,
) -> str:
    prefix = "drug" if content_type == "drug_monograph" else "general"
    if content_type == "general_monograph" and context_path:
        section_slug = ":".join(slugify(part) for part in context_path if slugify(part))
    else:
        section_slug = slugify(section)
    return f"{prefix}:{slugify(title)}:{section_slug}"


def uniquify_section_ids(records: list[SectionRecord]) -> list[SectionRecord]:
    total_by_id: dict[str, int] = {}
    for record in records:
        total_by_id[record.id] = total_by_id.get(record.id, 0) + 1

    seen_by_id: dict[str, int] = {}
    for record in records:
        if total_by_id[record.id] == 1:
            continue
        seen_by_id[record.id] = seen_by_id.get(record.id, 0) + 1
        occurrence = seen_by_id[record.id]
        if occurrence > 1:
            record.id = f"{record.id}:part-{occurrence:03d}"
    return records


def append_abbreviations_section(
    records: list[SectionRecord], text: str
) -> list[SectionRecord]:
    lines = text.splitlines()
    start_idx = None
    end_idx = None
    for idx, line in enumerate(lines):
        val = line.strip()
        if val == "# KÝ HIỆU CHỮ VIẾT TẮT":
            start_idx = idx
        elif val == "# CÁC CHUYÊN LUẬN CHUNG" and start_idx is not None:
            end_idx = idx
            break

    if start_idx is None or end_idx is None:
        return records

    abbrev_lines = []
    for i in range(start_idx + 1, end_idx):
        line = lines[i]
        if PAGE_MARKER_RE.search(line):
            continue
        abbrev_lines.append(line)

    text_value = clean_section_text(abbrev_lines)
    if not text_value:
        return records

    title = "KÝ HIỆU CHỮ VIẾT TẮT"
    section = "Danh mục chữ viết tắt"
    context_path = [section]
    record = SectionRecord(
        id=make_section_id(
            "general_monograph", title, section, context_path=context_path
        ),
        content_type="general_monograph",
        title=title,
        section=section,
        text=text_value,
        context_path=context_path,
        context_header=build_context_header(title, context_path),
        start_page=33,
        end_page=36,
    )

    if any(r.id == record.id for r in records):
        return records
    return [*records, record]


def append_admixture_appendix_section(
    records: list[SectionRecord], text: str
) -> list[SectionRecord]:
    lines = text.splitlines()
    start_idx = None
    end_idx = None
    for idx, line in enumerate(lines):
        val = line.strip()
        if val.startswith("# PHỤ LỤC 2:"):
            start_idx = idx
        elif val.startswith("# PHỤ LỤC 3:") and start_idx is not None:
            end_idx = idx
            break

    if start_idx is None or end_idx is None:
        return records

    appendix_lines = []
    for i in range(start_idx, end_idx):
        line = lines[i]
        if PAGE_MARKER_RE.search(line):
            continue
        appendix_lines.append(line)

    # Nhận diện các tiểu mục
    headings = [
        ("1. Khái niệm dịch truyền tĩnh mạch", ["1. Khái niệm dịch truyền tĩnh mạch"]),
        (
            "2. Những nguyên tắc chung khi pha thêm thuốc vào dịch truyền tĩnh mạch",
            ["2. Những nguyên tắc chung khi pha thêm thuốc vào dịch truyền tĩnh mạch"],
        ),
        (
            "3. Các vấn đề cần chú ý khi thực hành pha thêm thuốc vào dịch truyền tĩnh mạch",
            [
                "3. Các vấn đề cần chú ý khi thực hành pha thêm thuốc vào dịch truyền tĩnh mạch"
            ],
        ),
        (
            "4. Các yếu tố cơ bản ảnh hưởng đến tương hợp, tương kỵ cần lưu ý khi pha thêm thuốc vào dịch truyền",
            [
                "4. Các yếu tố cơ bản ảnh hưởng đến tương hợp, tương kỵ cần lưu ý khi pha thêm thuốc vào dịch truyền"
            ],
        ),
        (
            "5. Thực hành pha thêm thuốc vào dịch truyền tĩnh mạch",
            ["5. Thực hành pha thêm thuốc vào dịch truyền tĩnh mạch"],
        ),
        (
            "6. Các phương pháp đưa vào đường truyền tĩnh mạch",
            ["6. Các phương pháp đưa vào đường truyền tĩnh mạch"],
        ),
        (
            "7. Các thuốc đưa vào bằng đường truyền tĩnh mạch",
            ["7. Các thuốc đưa vào bằng đường truyền tĩnh mạch"],
        ),
    ]

    heading_indices = []
    for heading_text, context_path in headings:
        # Tìm dòng chứa tiêu đề
        found_idx = None
        for idx, line in enumerate(appendix_lines):
            clean_line = clean_label(line)
            if heading_text in clean_line or (
                heading_text.split(".")[0] + "." in clean_line
                and heading_text.split(".")[1][:15] in clean_line
            ):
                found_idx = idx
                break
        if found_idx is not None:
            heading_indices.append((found_idx, heading_text, context_path))

    heading_indices.sort()

    title = "PHỤ LỤC 2: PHA THÊM THUỐC TIÊM VÀO DỊCH TRUYỀN TĨNH MẠCH"
    for i, (idx_in_app, heading_text, context_path) in enumerate(heading_indices):
        start_line_idx = idx_in_app
        end_line_idx = (
            heading_indices[i + 1][0]
            if i + 1 < len(heading_indices)
            else len(appendix_lines)
        )

        section_lines = appendix_lines[start_line_idx:end_line_idx]
        text_value = clean_section_text(section_lines)
        if not text_value:
            continue

        record = SectionRecord(
            id=make_section_id(
                "general_monograph", title, heading_text, context_path=context_path
            ),
            content_type="general_monograph",
            title=title,
            section=heading_text,
            text=text_value,
            context_path=context_path,
            context_header=build_context_header(title, context_path),
            start_page=1500,
            end_page=1509,
        )
        if not any(r.id == record.id for r in records):
            records.append(record)

    return records


def append_atc_appendix_section(
    records: list[SectionRecord], text: str
) -> list[SectionRecord]:
    lines = text.splitlines()
    start_idx = None
    end_idx = None
    for idx, line in enumerate(lines):
        val = line.strip()
        if val.startswith("# PHỤ LỤC 3:"):
            start_idx = idx
        elif val.startswith("# MỤC LỤC TRA CỨU") and start_idx is not None:
            end_idx = idx
            break

    if start_idx is None or end_idx is None:
        return records

    atc_lines = []
    for i in range(start_idx + 1, end_idx):
        line = lines[i]
        if PAGE_MARKER_RE.search(line):
            continue
        atc_lines.append(line)

    text_value = clean_section_text(atc_lines)
    if not text_value:
        return records

    title = "PHỤ LỤC 3: DANH MỤC THUỐC PHÂN LOẠI THEO MÃ ATC"
    section = "Bảng phân loại ATC"
    context_path = [section]
    record = SectionRecord(
        id=make_section_id(
            "general_monograph", title, section, context_path=context_path
        ),
        content_type="general_monograph",
        title=title,
        section=section,
        text=text_value,
        context_path=context_path,
        context_header=build_context_header(title, context_path),
        start_page=1510,
        end_page=1528,
    )

    if any(r.id == record.id for r in records):
        return records
    return [*records, record]


def append_brand_mappings_section(
    records: list[SectionRecord], text: str
) -> list[SectionRecord]:
    lines = text.splitlines()
    start_idx = None
    end_idx = None
    for idx, line in enumerate(lines):
        val = line.strip()
        if val == "# MỤC LỤC TRA CỨU":
            start_idx = idx
        elif val.startswith("# NHÀ XUẤT BẢN Y HỌC") and start_idx is not None:
            end_idx = idx
            break

    if start_idx is None or end_idx is None:
        return records

    index_lines = []
    for i in range(start_idx + 1, end_idx):
        line = lines[i]
        if PAGE_MARKER_RE.search(line):
            continue
        index_lines.append(line)

    mappings = []
    for line in index_lines:
        line_clean = line.strip()
        # Match pattern: Brand - Generic, Page
        match = re.match(r"^(?:##\s*)?([^-]+)\s*-\s*([^,]+),\s*(\d+)$", line_clean)
        if match:
            brand = match.group(1).strip()
            generic = match.group(2).strip()
            page = match.group(3).strip()

            # Dọn dẹp ký tự thừa của OCR và định dạng
            brand = brand.split("##")[-1].strip()
            brand = re.sub(r",\s*\d+$", "", brand).strip()
            brand = repair_targeted_punctuation_artifacts(brand)
            generic = generic.split("##")[-1].strip()
            generic = re.sub(r",\s*\d+$", "", generic).strip()
            generic = repair_targeted_punctuation_artifacts(generic)

            # Tránh trùng lặp hoạt chất gốc với biệt dược
            if brand.lower() != generic.lower() and len(brand) > 2 and len(generic) > 2:
                mappings.append(
                    f"- **{brand}**: biệt dược chứa hoạt chất **{generic}** (Trang {page})"
                )

    if not mappings:
        return records

    text_value = "\n".join(mappings)
    title = "MỤC LỤC TRA CỨU BIỆT DƯỢC VÀ HOẠT CHẤT"
    section = "Bảng tra cứu biệt dược"
    context_path = [section]

    record = SectionRecord(
        id=make_section_id(
            "general_monograph", title, section, context_path=context_path
        ),
        content_type="general_monograph",
        title=title,
        section=section,
        text=text_value,
        context_path=context_path,
        context_header=build_context_header(title, context_path),
        start_page=1529,
        end_page=1600,
    )

    if any(r.id == record.id for r in records):
        return records
    return [*records, record]


def parse_sections(text: str) -> list[SectionRecord]:
    records: list[SectionRecord] = []
    current_type: str | None = None
    current_title: str | None = None
    current_section = "Nội dung"
    current_lines: list[str] = []
    current_hierarchy: list[tuple[int, str]] = []
    start_page: int | None = None
    last_page: int | None = None
    table_heading_lines_remaining = 0

    def context_path_for_current_section() -> list[str]:
        if current_type == "general_monograph":
            hierarchy_labels = [label for _, label in current_hierarchy]
            if current_section == "Nội dung":
                return hierarchy_labels or [current_section]
            if hierarchy_labels and hierarchy_labels[-1] == current_section:
                return hierarchy_labels
            return [*hierarchy_labels, current_section]
        return [current_section]

    def flush() -> None:
        nonlocal current_lines, start_page, last_page
        if not current_type or not current_title:
            current_lines = []
            return
        text_value = clean_section_text(current_lines)
        if not text_value:
            current_lines = []
            return
        context_path = context_path_for_current_section()
        records.append(
            SectionRecord(
                id=make_section_id(
                    current_type,
                    current_title,
                    current_section,
                    context_path=context_path,
                ),
                content_type=current_type,
                title=current_title,
                section=current_section,
                text=text_value,
                context_path=context_path,
                context_header=build_context_header(current_title, context_path),
                start_page=start_page,
                end_page=last_page,
            )
        )
        current_lines = []

    def set_general_hierarchy_section(label: str, page: int | None) -> None:
        nonlocal \
            current_section, \
            start_page, \
            current_hierarchy, \
            table_heading_lines_remaining
        flush()
        level = general_heading_level(label, current_hierarchy)
        current_hierarchy = [
            (item_level, item_label)
            for item_level, item_label in current_hierarchy
            if item_level < level
        ]
        current_hierarchy.append((level, label))
        current_section = label
        start_page = page
        table_heading_lines_remaining = 0

    for region, page, raw_line in extract_main_lines(text):
        last_page = page
        for line in split_inline_section_markers(raw_line):
            last_page = page
            if region == "drug" and is_drug_title(line):
                flush()
                current_type = "drug_monograph"
                current_title = clean_label(line)
                current_section = "Nội dung"
                current_hierarchy = []
                start_page = page
                table_heading_lines_remaining = 0
                continue

            if region == "general" and is_general_monograph_title(line):
                flush()
                current_type = "general_monograph"
                current_title = clean_label(line)
                current_section = "Nội dung"
                current_hierarchy = []
                start_page = page
                table_heading_lines_remaining = 0
                continue

            section, remainder = (
                section_label_from_line(line)
                if current_type == "drug_monograph"
                else (None, line)
            )
            if section and current_title:
                if current_type == "drug_monograph" and section in {
                    "Tên chung quốc tế",
                    "Mã ATC",
                    "Loại thuốc",
                    "Dạng thuốc và hàm lượng",
                }:
                    if current_section != "Thông tin chung":
                        flush()
                        current_section = "Thông tin chung"
                        start_page = page
                    table_heading_lines_remaining = 0
                    current_lines.append(
                        f"**{section}:** {remainder}" if remainder else f"**{section}**"
                    )
                    continue
                else:
                    flush()
                    current_section = section
                    start_page = page
                    table_heading_lines_remaining = 0
                    if remainder:
                        current_lines.append(remainder)
                    continue

            if (
                region == "general"
                and current_title
                and table_heading_lines_remaining > 0
                and is_heading_line(line)
                and not is_numbered_general_heading_label(clean_label(line))
            ):
                current_lines.append(line)
                table_heading_lines_remaining -= 1
                continue

            if (
                region == "general"
                and current_title
                and is_general_section_heading(line)
            ):
                set_general_hierarchy_section(clean_label(line), page)
                continue

            if current_title:
                if not current_lines:
                    start_page = page
                current_lines.append(remainder)
                if region == "general" and clean_label(line).startswith("Bảng "):
                    table_heading_lines_remaining = 3

    flush()
    records = append_abbreviations_section(records, text)
    records = append_admixture_appendix_section(records, text)
    records = append_atc_appendix_section(records, text)
    records = append_brand_mappings_section(records, text)
    records = append_bsa_appendix_formula_section(records, text)
    return uniquify_section_ids(records)


def bsa_appendix_formula_section(text: str) -> SectionRecord | None:
    lines = list(iter_page_lines(text))
    title_index: int | None = None
    title_page: int | None = None
    for index, (page, line) in enumerate(lines):
        header_norm = re.sub(r"\s+", " ", clean_label(line).upper())
        if (
            header_norm
            == "PHỤ LỤC 1: XÁC ĐỊNH DIỆN TÍCH BỀ MẶT THÂN THỂ NGƯỜI TỪ CHIỀU CAO VÀ CÂN NẶNG"
        ):
            title_index = index
            title_page = page
            break
    if title_index is None:
        return None

    safe_lines: list[str] = []
    example_line: str | None = None
    for _, line in lines[title_index + 1 :]:
        label = clean_label(line)
        if label.startswith("BẢNG TÍNH DIỆN TÍCH BỀ MẶT THÂN THỂ NGƯỜI"):
            break
        safe_lines.append(line)

    for offset, (_, line) in enumerate(lines[title_index + 1 :]):
        if "Ví dụ:" not in line:
            continue
        example_lines = [line]
        for _, continuation in lines[title_index + offset + 2 :]:
            label = clean_label(continuation)
            if not label or re.fullmatch(r"[\d,\s.]+", label):
                break
            example_lines.append(continuation)
            if "m2" in continuation:
                break
        example_line = " ".join(example_lines)
        break

    text_lines = [line for line in safe_lines if clean_label(line) != "Trong đó:"]
    if example_line:
        text_lines.append(example_line)
    text_value = clean_section_text(text_lines)
    text_norm = re.sub(r"\s+", "", text_value)
    if not text_value or "S=W0,425×H0,725×71,84" not in text_norm:
        return None

    title = (
        "PHỤ LỤC 1: XÁC ĐỊNH DIỆN TÍCH BỀ MẶT THÂN THỂ NGƯỜI TỪ CHIỀU CAO VÀ CÂN NẶNG"
    )
    section = "Công thức DuBois và DuBois"
    context_path = [section]
    return SectionRecord(
        id=make_section_id(
            "general_monograph", title, section, context_path=context_path
        ),
        content_type="general_monograph",
        title=title,
        section=section,
        text=text_value,
        context_path=context_path,
        context_header=build_context_header(title, context_path),
        start_page=title_page,
        end_page=title_page,
    )


def append_bsa_appendix_formula_section(
    records: list[SectionRecord], text: str
) -> list[SectionRecord]:
    section = bsa_appendix_formula_section(text)
    if section is None:
        return records
    if any(record.id == section.id for record in records):
        return records
    return [*records, section]


OFFICIAL_GENERAL_MONOGRAPHS = {
    "DƯỢC ĐỘNG HỌC VÀ CÁC THÔNG SỐ CHÍNH",
    "DỊ ỨNG THUỐC",
    "HƯỚNG DẪN SỬ DỤNG DƯỢC THƯ QUỐC GIA VIỆT NAM",
    "KÊ ĐƠN THUỐC",
    "NGỘ ĐỘC VÀ THUỐC GIẢI ĐỘC",
    "PHÒNG BỆNH VIÊM GAN B VÀ SỬ DỤNG HỢP LÝ THUỐC ĐIỀU TRỊ VIÊM GAN B MẠN TÍNH",
    "PHÒNG NGỪA VÀ XỬ TRÍ PHẢN ỨNG CÓ HẠI CỦA THUỐC",
    "SỬ DỤNG AN TOÀN THUỐC GIẢM ĐAU",
    "SỬ DỤNG HỢP LÝ CÁC THUỐC ĐIỀU TRỊ BỆNH HEN PHẾ QUẢN",
    "SỬ DỤNG HỢP LÝ THUỐC KHÁNG HIV CHO NGƯỜI BỆNH HIV/AIDS",
    "SỬ DỤNG HỢP LÝ THUỐC KHÁNG SINH",
    "SỬ DỤNG HỢP LÝ THUỐC KHÁNG SINH CEPHALOSPORIN",
    "SỬ DỤNG HỢP LÝ THUỐC KHÁNG ĐỘNG KINH",
    "SỬ DỤNG THUỐC TRONG THỜI KỲ MANG THAI VÀ CHO CON BÚ",
    "SỬ DỤNG THUỐC Ở NGƯỜI CAO TUỔI",
    "SỬ DỤNG THUỐC Ở NGƯỜI SUY GIẢM CHỨC NĂNG GAN, THẬN",
    "SỬ DỤNG THUỐC Ở TRẺ EM",
    "THUỐC CHỐNG LOẠN THẦN, XỬ TRÍ CÁC TÁC DỤNG KHÔNG MONG MUỐN",
    "TÌNH HÌNH BỆNH LAO, LAO KHÁNG THUỐC VÀ SỬ DỤNG HỢP LÝ THUỐC CHỐNG LAO",
    "TƯƠNG TÁC THUỐC",
}


def is_known_malformed_duplicate_section(record: SectionRecord) -> bool:
    if record.id != "drug:warfarin:chi-dinh:part-003":
        return False
    if record.title != "WARFARIN" or record.section != "Chỉ định":
        return False
    return all(
        marker in record.text
        for marker in (
            "thời gian điều trị",
            "biến chứng:Cục",
            "Đích 2,5; INR 2 - 3;",
            "máu đông trong nội tâm mạc",
            "1 - 3 tháng",
        )
    )


def filter_sections_for_rag(records: list[SectionRecord]) -> list[SectionRecord]:
    sections: list[SectionRecord] = []
    for record in records:
        if is_known_malformed_duplicate_section(record):
            continue
        if record.content_type == "general_monograph":
            is_official = record.title in OFFICIAL_GENERAL_MONOGRAPHS
            is_appendix_or_index = any(
                record.title.startswith(prefix)
                for prefix in ("PHỤ LỤC", "KÝ HIỆU", "MỤC LỤC")
            )
            if not (is_official or is_appendix_or_index):
                continue
        sections.append(record)
    return sections


def split_oversized_paragraph(paragraph: str, max_chars: int) -> list[str]:
    parts: list[str] = []
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


def split_long_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]
    chunks: list[str] = []
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
        chunks.extend(split_oversized_paragraph(paragraph, max_chars))

    if current:
        chunks.append(current)
    return chunks


def chunk_sections(
    records: list[SectionRecord], *, max_chars: int = DEFAULT_CHUNK_MAX_CHARS
) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    for record in records:
        parts = split_long_text(record.text, max_chars)
        context_path = record.context_path or [record.section]
        context_header = record.context_header or build_context_header(
            record.title, context_path
        )
        for index, part in enumerate(parts, start=1):
            chunks.append(
                ChunkRecord(
                    id=f"{record.id}:chunk-{index:03d}",
                    section_id=record.id,
                    content_type=record.content_type,
                    title=record.title,
                    section=record.section,
                    text=part,
                    context_path=context_path,
                    context_header=context_header,
                    chunk_index=index,
                    start_page=record.start_page,
                    end_page=record.end_page,
                    warnings=list(record.warnings),
                )
            )
    return chunks


IMPORTANT_DRUG_SECTIONS = {"Chỉ định", "Chống chỉ định", "Liều lượng và cách dùng"}
EXPECTED_MISSING_DRUG_SECTIONS: dict[tuple[str, str], str] = {
    (
        "ATORVASTATIN",
        "Chỉ định",
    ): "Verified in source PDF: statin summary entry omits this standalone section.",
    (
        "ATORVASTATIN",
        "Chống chỉ định",
    ): "Verified in source PDF: statin summary entry omits this standalone section.",
    (
        "FLUVASTATIN",
        "Chỉ định",
    ): "Verified in source PDF: statin summary entry omits this standalone section.",
    (
        "FLUVASTATIN",
        "Chống chỉ định",
    ): "Verified in source PDF: statin summary entry omits this standalone section.",
    (
        "LOVASTATIN",
        "Chỉ định",
    ): "Verified in source PDF: statin summary entry omits this standalone section.",
    (
        "LOVASTATIN",
        "Chống chỉ định",
    ): "Verified in source PDF: statin summary entry omits this standalone section.",
    (
        "PRAVASTATIN",
        "Chỉ định",
    ): "Verified in source PDF: statin summary entry omits this standalone section.",
    (
        "PRAVASTATIN",
        "Chống chỉ định",
    ): "Verified in source PDF: statin summary entry omits this standalone section.",
    (
        "SIMVASTATIN",
        "Chỉ định",
    ): "Verified in source PDF: statin summary entry omits this standalone section.",
    (
        "SIMVASTATIN",
        "Chống chỉ định",
    ): "Verified in source PDF: statin summary entry omits this standalone section.",
    (
        "EPERISON HYDROCLORID",
        "Chống chỉ định",
    ): "Verified in source PDF: monograph omits this standalone section.",
    (
        "NATRI THIOSULFAT",
        "Chống chỉ định",
    ): "Verified in source PDF: monograph omits this standalone section.",
    (
        "THUỐC PHIỆN - OPIAT - OPIOID",
        "Chống chỉ định",
    ): "Verified in source PDF: group monograph directs readers to individual opioid monographs for this section.",
    (
        "THUỐC PHIỆN - OPIAT - OPIOID",
        "Liều lượng và cách dùng",
    ): "Verified in source PDF: group monograph directs readers to individual opioid monographs for this section.",
}
EMBEDDED_HEADING_RE = re.compile(r"#\s*([A-ZÀ-Ỵ0-9][A-ZÀ-Ỵ0-9 ()/.,-]{2,90})")


def drug_title_candidates_in_text(text: str) -> set[str]:
    candidates: set[str] = set()
    for line in text.splitlines():
        label = clean_label(line)
        if uppercase_ratio(label) >= 0.75 and 3 <= len(label) <= 90:
            candidates.add(label)
    for match in EMBEDDED_HEADING_RE.finditer(text):
        candidates.add(clean_label(match.group(1)))
    return candidates


def build_audit(
    sections: list[SectionRecord], chunks: list[ChunkRecord]
) -> AuditReport:
    drug_titles = sorted(
        {record.title for record in sections if record.content_type == "drug_monograph"}
    )
    drug_title_set = set(drug_titles)
    general_titles = sorted(
        {
            record.title
            for record in sections
            if record.content_type == "general_monograph"
        }
    )
    warnings: list[str] = []
    known_source_exceptions: list[dict] = []

    by_drug: dict[str, set[str]] = {}
    monograph_lengths: dict[str, int] = {}
    for record in sections:
        if record.content_type != "drug_monograph":
            continue
        by_drug.setdefault(record.title, set()).add(record.section)
        monograph_lengths[record.title] = monograph_lengths.get(record.title, 0) + len(
            record.text
        )

    for title, section_names in sorted(by_drug.items()):
        for required_section in sorted(IMPORTANT_DRUG_SECTIONS):
            if required_section not in section_names:
                reason = EXPECTED_MISSING_DRUG_SECTIONS.get((title, required_section))
                if reason:
                    known_source_exceptions.append(
                        {
                            "title": title,
                            "section": required_section,
                            "reason": reason,
                        }
                    )
                    continue
                warnings.append(f"Drug {title} lacks section: {required_section}")

    for title, total_chars in sorted(monograph_lengths.items()):
        if total_chars < 300:
            warnings.append(f"Drug {title} is unusually short: {total_chars} chars")

    for record in sections:
        if len(record.text) > 20000:
            warnings.append(
                f"Section {record.id} is unusually long: {len(record.text)} chars"
            )

    for chunk in chunks:
        for title in sorted(drug_title_candidates_in_text(chunk.text) & drug_title_set):
            if title != chunk.title:
                warnings.append(f"Chunk {chunk.id} may contain next drug title {title}")
                break

    longest_sections = [
        {
            "id": record.id,
            "title": record.title,
            "section": record.section,
            "chars": len(record.text),
        }
        for record in sorted(sections, key=lambda item: len(item.text), reverse=True)[
            :10
        ]
    ]
    shortest_monographs = [
        {"title": title, "chars": chars}
        for title, chars in sorted(monograph_lengths.items(), key=lambda item: item[1])[
            :10
        ]
    ]

    return AuditReport(
        drug_monographs=len(drug_titles),
        general_monographs=len(general_titles),
        sections=len(sections),
        chunks=len(chunks),
        longest_sections=longest_sections,
        shortest_monographs=shortest_monographs,
        unknown_section_labels=[],
        known_source_exceptions=known_source_exceptions,
        warnings=warnings,
    )


def write_jsonl(path: Path, records: list[SectionRecord] | list[ChunkRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(record_to_json(record), ensure_ascii=False, sort_keys=True)
        for record in records
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_json(path: Path, report: AuditReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(record_to_json(report), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def process_corpus(
    *, input_path: Path, output_dir: Path, max_chars: int = DEFAULT_CHUNK_MAX_CHARS
) -> dict[str, Path]:
    text = input_path.read_text(encoding="utf-8")
    raw_sections = parse_sections(text)
    sections = filter_sections_for_rag(raw_sections)

    chunks = chunk_sections(sections, max_chars=max_chars)
    audit = build_audit(sections, chunks)

    sections_path = output_dir / "sections.jsonl"
    chunks_path = output_dir / "chunks.jsonl"
    audit_path = output_dir / "audit.json"

    write_jsonl(sections_path, sections)
    write_jsonl(chunks_path, chunks)
    write_json(audit_path, audit)

    return {"sections": sections_path, "chunks": chunks_path, "audit": audit_path}
