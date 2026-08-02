#!/usr/bin/env python3
from __future__ import annotations

import itertools
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from corpus_pipeline.config.chunking import DEFAULT_CHUNK_MAX_CHARS
from corpus_pipeline.config.paths import (
    ANKHANG_MARKDOWN_INTERIM_DIR,
    CANONICAL_INTERIM_DIR,
    DOCLING_INTERIM_DIR,
    RAG_FINAL_DIR,
    RAG_INTERIM_DIR,
)
from corpus_pipeline.corpus.metadata.qdrant_payload_contract import (
    QDRANT_RUNTIME_PAYLOAD_FIELDS,
    validate_runtime_payload,
)
from corpus_pipeline.corpus.processing.preprocess_rag_corpus import (
    OFFICIAL_GENERAL_MONOGRAPHS,
)

DEFAULT_FINAL_DIR = RAG_FINAL_DIR
DEFAULT_CANONICAL_DIR = CANONICAL_INTERIM_DIR
DEFAULT_RAG_INTERIM_DIR = RAG_INTERIM_DIR
DEFAULT_DOCLING_DIR = DOCLING_INTERIM_DIR
DEFAULT_ANKHANG_MARKDOWN_DIR = ANKHANG_MARKDOWN_INTERIM_DIR
DEFAULT_REPORT_JSON = RAG_FINAL_DIR / "final_validation_report.json"
DEFAULT_REPORT_MD = RAG_FINAL_DIR / "final_validation_report.md"

BSA_SECTION_ID = (
    "general:phu-luc-1-xac-dinh-dien-tich-be-mat-than-the-nguoi-tu-chieu-cao-va-can-nang:"
    "cong-thuc-dubois-va-dubois"
)
VANCOMYCIN_SECTION_ID = "drug:vancomycin:duoc-ly-va-co-che-tac-dung"
VANCOMYCIN_TABLE_ID = "curated-table-1456-001"
FORBIDDEN_TEXT_ARTIFACTS = [
    "nông độ",
    "thu ốc",
    "ho ạt",
    "huy ết",
    "củ a",
    "điề u tr ị",
    "StevensJohnson",
    "EpsteinBarr",
]
KNOWN_TEXT_ARTIFACTS = [
    "O meprazol",
    "O meprazole",
    "O floxacin",
    "O xcarbazepin",
    "O htahara",
    "O rlistat",
    "O xytocin",
    "O nchocerca",
    "O xaliplatin",
    "O ctreotid",
    "O lanzapin",
    "O xacilin",
    "O rciprenalin",
    "O seltamivir",
    "O ndansetron",
    "O xybutynin",
    "O pitz",
    "SỬ DỤNGAN",
    "CHONGƯỜI",
    "chuyên giay - dược",
    "\uf0ae",
    "\uf0af",
    "ºC",
    " dến ",
    "creatnin",
    "sentivity",
    "Hight density",
    "mgngày",
    "H2 CO3",
    "H2 O",
    "Na2S2 O3",
    "Na2 SO3",
]
TEXT_QUALITY_SENTINELS = {
    "text_quality_nong_do": "nồng độ",
    "text_quality_stevens_johnson": "Stevens-Johnson",
    "text_quality_epstein_barr": "Epstein-Barr",
}
MAX_CHARS = DEFAULT_CHUNK_MAX_CHARS
HYDRATE_STRATEGIES = {"full_section", "chunk_window", "search_only"}
BRAND_INDEX_SECTION_ID = (
    "general:muc-luc-tra-cuu-biet-duoc-va-hoat-chat:bang-tra-cuu-biet-duoc"
)
CHUNK_ROLES = {"prose", "table", "index_entry", "appendix_list"}
CONTENT_PAGE_RANGE = range(37, 1601)
ALLOWED_MISSING_CONTENT_PAGES = {37, 38, 1497, 1498}
JOINED_NUMERIC_UNIT_RE = re.compile(
    r"(?<![A-Za-zÀ-Ỵà-ỵ])\d+(?:microgam|mg|ml|kg|cm2|m2|g)(?=\b|/)"
)
PUNCT_NO_SPACE_RE = re.compile(r"[A-Za-zÀ-Ỵà-ỵ0-9][:;,.][A-ZÀ-ỴĐ]")
BRAND_INDEX_ENTRY_START_RE = re.compile(r"^- \*\*[^*]+\*\*: .+")
BRAND_INDEX_DANGLING_RE = re.compile(r"- \*\*[^*]+\*\*:\s*$")
BRAND_INDEX_CONTINUATION_RE = re.compile(r"^biệt dược chứa hoạt chất \*\*")
SEVERITIES = {"blocking", "auto_fixable", "suspect", "accepted"}
NON_ANKHANG_SOURCE = (
    "Dược thư Quốc gia Việt Nam (Xuất bản lần thứ 2) – Nhà xuất bản Y học, Hà Nội, 2018"
)
ANKHANG_SOURCE = "Tờ hướng dẫn sử dụng"
SALBUTAMOL_DESCRIPTOR_SECTIONS = {
    "drug:salbutamol:noi-dung:part-002",
    "drug:salbutamol:thong-tin-chung:part-002",
    "drug:salbutamol:duoc-ly-va-co-che-tac-dung:part-002",
    "drug:salbutamol:chi-dinh:part-002",
    "drug:salbutamol:chong-chi-dinh:part-002",
    "drug:salbutamol:than-trong:part-002",
    "drug:salbutamol:thoi-ky-mang-thai:part-002",
    "drug:salbutamol:thoi-ky-cho-con-bu:part-002",
    "drug:salbutamol:tac-dung-khong-mong-muon-adr:part-002",
    "drug:salbutamol:lieu-luong-va-cach-dung:part-002",
    "drug:salbutamol:tuong-tac-thuoc:part-002",
    "drug:salbutamol:do-on-dinh-va-bao-quan:part-002",
    "drug:salbutamol:tuong-ky:part-002",
    "drug:salbutamol:qua-lieu-va-xu-tri:part-002",
    "drug:salbutamol:thong-tin-qui-che:part-002",
    "drug:salbutamol:ten-thuong-mai:part-002",
}
ACCEPTED_DUPLICATE_SECTIONS = {
    "Thông tin qui chế",
    "Độ ổn định và bảo quản",
    "Hướng dẫn cách xử trí ADR",
    "Quá liều và xử trí",
}
ACCEPTED_DUPLICATE_SECTION_GROUPS = {
    frozenset(
        {
            "drug:atapulgit:thoi-ky-mang-thai",
            "drug:atapulgit:thoi-ky-cho-con-bu",
        }
    ),
    frozenset(
        {
            "drug:benzathin-penicilin-g:chong-chi-dinh",
            "drug:phenoxymethylpenicilin:chong-chi-dinh",
        }
    ),
    frozenset(
        {
            "drug:bexaroten:thoi-ky-cho-con-bu",
            "drug:salmeterol:thoi-ky-cho-con-bu",
        }
    ),
    frozenset(
        {
            "drug:biotin:thoi-ky-mang-thai",
            "drug:biotin:thoi-ky-cho-con-bu",
        }
    ),
    frozenset(
        {
            "drug:calci-clorid:thoi-ky-mang-thai",
            "drug:calci-clorid:thoi-ky-cho-con-bu",
        }
    ),
    frozenset(
        {
            "drug:codein-phosphat:thoi-ky-cho-con-bu",
            "drug:cycloserin:thoi-ky-mang-thai",
            "drug:triamcinolon:thoi-ky-mang-thai",
        }
    ),
    frozenset(
        {
            "drug:codein-phosphat:thoi-ky-mang-thai",
            "drug:lomustin:thoi-ky-mang-thai",
        }
    ),
    frozenset(
        {
            "drug:dextran-40:thoi-ky-cho-con-bu",
            "drug:dextran-70:thoi-ky-cho-con-bu",
        }
    ),
    frozenset(
        {
            "drug:estriol:tuong-ky",
            "drug:estron:tuong-ky",
        }
    ),
    frozenset(
        {
            "drug:fluorouracil:thoi-ky-mang-thai",
            "drug:fluorouracil:thoi-ky-cho-con-bu",
        }
    ),
    frozenset(
        {
            "drug:naphazolin:tuong-ky",
            "drug:oxymetazolin-hydroclorid:tuong-ky",
        }
    ),
    frozenset(
        {
            "drug:palivizumab:thoi-ky-mang-thai",
            "drug:palivizumab:thoi-ky-cho-con-bu",
        }
    ),
    frozenset(
        {
            "drug:pancrelipase:thoi-ky-cho-con-bu",
            "drug:sat-ii-sulfat:thoi-ky-cho-con-bu",
        }
    ),
    frozenset(
        {
            "drug:salbutamol-dung-trong-ho-hap:ten-thuong-mai",
            "drug:salbutamol-dung-trong-san-khoa:ten-thuong-mai",
        }
    ),
    frozenset(
        {
            "drug:salbutamol-dung-trong-ho-hap:thoi-ky-cho-con-bu",
            "drug:salbutamol-dung-trong-san-khoa:thoi-ky-cho-con-bu",
        }
    ),
    frozenset(
        {
            "drug:saxagliptin:thoi-ky-mang-thai",
            "drug:sitagliptin:thoi-ky-mang-thai",
        }
    ),
    frozenset(
        {
            "drug:vac-xin-rubella:ten-thuong-mai",
            "drug:vac-xin-soi-quai-bi-rubella:ten-thuong-mai",
        }
    ),
}
HTML_LEFTOVER_RE = re.compile(r"</?[a-z][a-z0-9]*(?:\s+[^>]*)?>", re.IGNORECASE)
PART_SUFFIX_RE = re.compile(r":part-\d{3}$")
CONTROL_CATEGORY_PREFIXES = {"Cc", "Cf"}


@dataclass(frozen=True)
class AuditFinding:
    record_id: str
    record_type: str
    source: str
    category: str
    severity: str
    message: str
    snippet: str
    source_evidence: str = ""
    auto_fixable: bool = False
    recommended_action: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if payload["severity"] not in SEVERITIES:
            raise ValueError(f"Invalid severity: {payload['severity']}")
        return payload


@dataclass
class ValidationReport:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    findings: list[AuditFinding] = field(default_factory=list)

    @property
    def blocking_count(self) -> int:
        return sum(1 for finding in self.findings if finding.severity == "blocking")

    @property
    def auto_fixable_count(self) -> int:
        return sum(1 for finding in self.findings if finding.severity == "auto_fixable")

    @property
    def suspect_count(self) -> int:
        return sum(1 for finding in self.findings if finding.severity == "suspect")

    @property
    def accepted_count(self) -> int:
        return sum(1 for finding in self.findings if finding.severity == "accepted")

    @property
    def severity_counts(self) -> dict[str, int]:
        counts = Counter(finding.severity for finding in self.findings)
        return {severity: counts.get(severity, 0) for severity in sorted(SEVERITIES)}

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "metrics": self.metrics,
            "errors": self.errors,
            "warnings": self.warnings,
            "severity_counts": self.severity_counts,
            "findings": [finding.to_dict() for finding in self.findings],
        }


@dataclass
class CorpusInputs:
    final_sections: list[dict[str, Any]]
    final_chunks: list[dict[str, Any]]
    canonical_blocks: list[dict[str, Any]]
    canonical_sections: list[dict[str, Any]]
    rag_sections: list[dict[str, Any]]
    rag_chunks: list[dict[str, Any]]
    curated_tables: list[dict[str, Any]]
    ankhang_markdown_files: list[Path]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def chunk_record_id(record: dict[str, Any]) -> str:
    return str(record.get("chunk_id") or record.get("id") or "")


def chunk_record_text(record: dict[str, Any]) -> str:
    return str(record.get("chunk_text") or record.get("text") or "")


def duplicate_ids(records: list[dict[str, Any]], key: str) -> list[str]:
    counts = Counter(str(record.get(key) or "") for record in records)
    return sorted(item for item, count in counts.items() if item and count > 1)


def table_blocks(blocks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    tables: dict[str, dict[str, Any]] = {}
    for block in blocks:
        table_id = block.get("table_id")
        if block.get("content_type") == "table" and table_id:
            tables[str(table_id)] = block
    return tables


def validate_counts(
    *,
    final_audit: dict[str, Any],
    canonical_audit: dict[str, Any],
    sections: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    table_block_count: int,
    errors: list[str],
) -> None:
    expected = {
        "section_count": len(sections),
        "chunk_count": len(chunks),
        "table_block_count": table_block_count,
    }
    for key, actual in expected.items():
        audit_value = final_audit.get(key)
        if audit_value != actual:
            errors.append(
                f"Final audit {key}={audit_value} does not match actual {actual}"
            )

    if final_audit.get("duplicate_review_count") != 0:
        errors.append(
            f"Final audit duplicate_review_count must be 0, got {final_audit.get('duplicate_review_count')}"
        )
    if canonical_audit.get("duplicate_review_count") != 0:
        errors.append(
            f"Canonical audit duplicate_review_count must be 0, got {canonical_audit.get('duplicate_review_count')}"
        )


def validate_references(
    *,
    sections: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    blocks: list[dict[str, Any]],
    tables_by_id: dict[str, dict[str, Any]],
    errors: list[str],
) -> None:
    section_ids = {str(section["id"]) for section in sections}
    block_ids = {str(block["block_id"]) for block in blocks}

    for chunk in chunks:
        section_id = str(chunk.get("section_id") or "")
        if section_id not in section_ids:
            errors.append(
                f"Chunk {chunk.get('id')} references missing section_id {section_id}"
            )

    chunk_section_ids = {str(chunk.get("section_id") or "") for chunk in chunks}
    for section in sections:
        section_id = str(section["id"])
        if section_id not in chunk_section_ids:
            errors.append(f"Section {section_id} has no chunk")

    for section in sections:
        section_id = str(section["id"])
        for block_id in section.get("block_ids") or []:
            if str(block_id) not in block_ids:
                errors.append(
                    f"Section {section_id} references missing block_id {block_id}"
                )
        for table_id in section.get("table_ids") or []:
            table = tables_by_id.get(str(table_id))
            if table is None:
                errors.append(
                    f"Section {section_id} references missing table_id {table_id}"
                )
                continue
            if table.get("section_id") != section_id:
                errors.append(
                    f"Section {section_id} references table_id {table_id} "
                    f"but table block belongs to {table.get('section_id')}"
                )

    for block in blocks:
        section_id = str(block.get("section_id") or "")
        if section_id not in section_ids:
            errors.append(
                f"Block {block.get('block_id')} references missing section_id {section_id}"
            )


def validate_chunk_source_blocks(
    *,
    chunks: list[dict[str, Any]],
    blocks: list[dict[str, Any]],
    errors: list[str],
) -> None:
    blocks_by_id = {str(block.get("block_id") or ""): block for block in blocks}
    chunks_by_section: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        chunks_by_section.setdefault(str(chunk.get("section_id") or ""), []).append(
            chunk
        )
        if chunk.get("source") == ANKHANG_SOURCE:
            continue
        chunk_id = str(chunk.get("id") or "")
        source_block_id = str(chunk.get("source_block_id") or "")
        if not source_block_id:
            errors.append(f"Chunk {chunk_id} is missing source_block_id")
            continue
        block = blocks_by_id.get(source_block_id)
        if block is None:
            errors.append(
                f"Chunk {chunk_id} references missing source_block_id {source_block_id}"
            )
            continue
        if block.get("section_id") != chunk.get("section_id"):
            errors.append(
                f"Chunk {chunk_id} source_block_id {source_block_id} belongs to section {block.get('section_id')}"
            )
        if chunk.get("chunk_content_type") and chunk.get(
            "chunk_content_type"
        ) != block.get("content_type"):
            errors.append(
                f"Chunk {chunk_id} chunk_content_type {chunk.get('chunk_content_type')} "
                f"does not match source block content_type {block.get('content_type')}"
            )
        if chunk.get("chunk_role") == "table" and chunk.get("table_id") != block.get(
            "table_id"
        ):
            errors.append(
                f"Chunk {chunk_id} table_id does not match source block {source_block_id}"
            )

    for section_id, section_chunks in chunks_by_section.items():
        indexes = sorted(int(chunk.get("chunk_index") or 0) for chunk in section_chunks)
        if indexes and indexes != list(range(1, len(indexes) + 1)):
            errors.append(
                f"Section {section_id} has non-contiguous chunk_index values: {indexes}"
            )


def validate_sentinels(
    sections: list[dict[str, Any]],
    tables_by_id: dict[str, dict[str, Any]],
    errors: list[str],
) -> None:
    sections_by_id = {str(section["id"]): section for section in sections}

    bsa = sections_by_id.get(BSA_SECTION_ID)
    if not bsa:
        errors.append(f"Sentinel bsa_formula missing section: {BSA_SECTION_ID}")
    else:
        bsa_text = str(bsa.get("text") or "")
        bsa_text_norm = re.sub(r"\s+", "", bsa_text)
        bsa_checks_norm = [
            ("S=W0,425×H0,725×71,84", "S = W0,425 × H0,725 × 71,84"),
            ("Wlàkhốilượngcơthể(kg).", "W là khối lượng cơ thể (kg)."),
            ("Hlàchiềucaocơthể(cm).", "H là chiều cao cơ thể (cm)."),
            ("1,66m2", "1,66 m2"),
        ]
        for expected_norm, expected_display in bsa_checks_norm:
            if expected_norm not in bsa_text_norm:
                errors.append(
                    f"Sentinel bsa_formula missing expected text: {expected_display}"
                )
        if "curated-table-1499-001" in json.dumps(bsa, ensure_ascii=False):
            errors.append(
                "Sentinel bsa_formula must not reference malformed curated-table-1499-001"
            )
        if "Chiều cao (cm)" in bsa_text:
            errors.append(
                "Sentinel bsa_formula contains malformed table text marker: Chiều cao (cm)"
            )

    vancomycin = sections_by_id.get(VANCOMYCIN_SECTION_ID)
    if not vancomycin:
        errors.append(
            f"Sentinel vancomycin_mic missing section: {VANCOMYCIN_SECTION_ID}"
        )
    else:
        table_ids = set(vancomycin.get("table_ids") or [])
        if VANCOMYCIN_TABLE_ID not in table_ids:
            errors.append(
                f"Sentinel vancomycin_mic missing table_id: {VANCOMYCIN_TABLE_ID}"
            )
        table = tables_by_id.get(VANCOMYCIN_TABLE_ID)
        markdown = str((table or {}).get("markdown") or "")
        for expected in ("MIC", "Enterococcus"):
            if expected not in markdown:
                errors.append(
                    f"Sentinel vancomycin_mic table missing expected text: {expected}"
                )


def validate_duplicate_review_file(canonical_dir: Path, errors: list[str]) -> None:
    review_path = canonical_dir / "table_duplicate_review.jsonl"
    rows = read_jsonl(review_path)
    if rows:
        errors.append(
            f"table_duplicate_review.jsonl must be empty, got {len(rows)} rows"
        )


def collect_text(records: list[dict[str, Any]], fields: tuple[str, ...]) -> str:
    values: list[str] = []
    for record in records:
        for f in fields:
            value = record.get(f)
            if isinstance(value, str):
                values.append(value)
    return "\n".join(values)


def is_markdown_table_row(line: str) -> bool:
    stripped = line.strip()
    return (
        stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2
    )


def is_markdown_separator_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped.startswith("|") or "-" not in stripped:
        return False
    return stripped.replace("|", "").replace("-", "").replace(":", "").strip() == ""


def validate_text_quality(
    sections: list[dict[str, Any]], chunks: list[dict[str, Any]], errors: list[str]
) -> None:
    section_text = collect_text(sections, ("text", "context_header"))
    chunk_text = collect_text(chunks, ("text", "context_header"))
    for artifact in FORBIDDEN_TEXT_ARTIFACTS:
        if artifact in section_text:
            errors.append(
                f"Forbidden text artifact {artifact!r} found in sections.jsonl"
            )
        if artifact in chunk_text:
            errors.append(f"Forbidden text artifact {artifact!r} found in chunks.jsonl")
    for name, expected in TEXT_QUALITY_SENTINELS.items():
        if expected not in section_text:
            errors.append(f"Sentinel {name} missing expected text: {expected}")

    for artifact in KNOWN_TEXT_ARTIFACTS:
        if artifact in section_text:
            errors.append(f"Known text artifact {artifact!r} found in sections.jsonl")
        if artifact in chunk_text:
            errors.append(f"Known text artifact {artifact!r} found in chunks.jsonl")


def validate_quality_gate(
    sections: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    errors: list[str],
    *,
    max_chars: int = MAX_CHARS,
) -> None:
    for section in sections:
        if section.get("source") == ANKHANG_SOURCE:
            continue
        section_id = str(section.get("id") or "")
        warnings = [str(warning) for warning in (section.get("warnings") or [])]
        if warnings:
            errors.append(f"Section {section_id} has warnings: {'; '.join(warnings)}")
        text = str(section.get("text") or "")
        if any(unicodedata.category(char) == "Mn" for char in text):
            errors.append(f"Literal combining mark found in section {section_id}")
        for match in JOINED_NUMERIC_UNIT_RE.finditer(text):
            errors.append(
                f"Joined numeric unit {match.group(0)!r} found in section {section_id}"
            )
        for match in PUNCT_NO_SPACE_RE.finditer(text):
            token = match.group(0)
            if re.match(r"^[A-Z]\.([A-Z]|\d)$", token) or token in {
                "N,N",
                "g.L",
                "p.W",
            }:
                continue
            errors.append(
                f"Suspicious punctuation without space {token!r} found in section {section_id}"
            )

    for chunk in chunks:
        chunk_id = str(chunk.get("id") or "")
        text = str(chunk.get("text") or "")
        if len(text) > max_chars:
            errors.append(f"Chunk {chunk_id} text exceeds {max_chars} chars")

        if chunk.get("source") == ANKHANG_SOURCE:
            continue

        warnings = [str(warning) for warning in (chunk.get("warnings") or [])]
        if warnings:
            errors.append(f"Chunk {chunk_id} has warnings: {'; '.join(warnings)}")
        lines = text.splitlines()
        if any(is_markdown_table_row(line) for line in lines) and not any(
            is_markdown_separator_line(line) for line in lines
        ):
            errors.append(
                f"Chunk {chunk_id} contains table rows without a markdown separator/header"
            )

    chunks_by_section: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        chunks_by_section.setdefault(str(chunk.get("section_id") or ""), []).append(
            chunk
        )
    for section_chunks in chunks_by_section.values():
        ordered = sorted(
            section_chunks, key=lambda item: int(item.get("chunk_index") or 0)
        )
        for previous, current in itertools.pairwise(ordered):
            if previous.get("source") == ANKHANG_SOURCE:
                continue
            previous_text = str(previous.get("text") or "").rstrip()
            current_text = str(current.get("text") or "").lstrip()
            if (
                previous_text
                and current_text
                and previous_text[-1].islower()
                and current_text[0].islower()
            ):
                errors.append(
                    f"Possible word split between chunks {previous.get('id')} and {current.get('id')}"
                )


def validate_brand_index_chunks(
    chunks: list[dict[str, Any]], errors: list[str]
) -> None:
    for chunk in chunks:
        if str(chunk.get("section_id") or "") != BRAND_INDEX_SECTION_ID:
            continue
        chunk_id = str(chunk.get("id") or "")
        text = str(chunk.get("text") or "").strip()
        if not text:
            continue
        first_line = next(
            (line.strip() for line in text.splitlines() if line.strip()), ""
        )
        if BRAND_INDEX_CONTINUATION_RE.search(first_line):
            errors.append(
                f"Brand index chunk {chunk_id} starts with an entry continuation"
            )
        if BRAND_INDEX_DANGLING_RE.search(text):
            errors.append(
                f"Brand index chunk {chunk_id} ends with dangling entry marker"
            )


def validate_hydration_metadata(
    sections: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    errors: list[str],
) -> None:
    sections_by_id = {str(section.get("id") or ""): section for section in sections}
    for section in sections:
        section_id = str(section.get("id") or "")
        strategy = section.get("hydrate_strategy")
        if strategy not in HYDRATE_STRATEGIES:
            errors.append(
                f"Section {section_id} has invalid hydrate_strategy: {strategy}"
            )
        expected_count = len(str(section.get("text") or ""))
        if section.get("section_char_count") != expected_count:
            errors.append(
                f"Section {section_id} section_char_count={section.get('section_char_count')} "
                f"does not match text length {expected_count}"
            )

    for chunk in chunks:
        chunk_id = str(chunk.get("id") or "")
        context_header = str(chunk.get("context_header") or "").strip()
        raw_text = str(chunk.get("text") or "").strip()
        mapping = chunk.get("colloquial_mapping")
        if not context_header:
            errors.append(f"Chunk {chunk_id} is missing context_header")
        if "Tên gọi khác:" in context_header or "Dấu hiệu nhận biết:" in context_header:
            errors.append(
                f"Chunk {chunk_id} context_header contains rendered colloquial metadata"
            )
        if "context_text" in chunk:
            errors.append(f"Chunk {chunk_id} must not contain context_text")
        if not raw_text:
            errors.append(f"Chunk {chunk_id} is missing text")
        if mapping not in (None, {}) and not isinstance(mapping, dict):
            errors.append(f"Chunk {chunk_id} colloquial_mapping must be an object")

        strategy = chunk.get("hydrate_strategy")
        if strategy not in HYDRATE_STRATEGIES:
            errors.append(f"Chunk {chunk_id} has invalid hydrate_strategy: {strategy}")
        chunk_role = chunk.get("chunk_role")
        if chunk_role is not None and chunk_role not in CHUNK_ROLES:
            errors.append(f"Chunk {chunk_id} has invalid chunk_role: {chunk_role}")
        section = sections_by_id.get(str(chunk.get("section_id") or ""))
        if not section:
            continue
        if chunk.get("hydrate_strategy") != section.get("hydrate_strategy"):
            errors.append(
                f"Chunk {chunk_id} hydrate_strategy does not match section {section.get('id')}"
            )
        if chunk.get("section_char_count") != section.get("section_char_count"):
            errors.append(
                f"Chunk {chunk_id} section_char_count does not match section {section.get('id')}"
            )


def validate_full_coverage(sections: list[dict[str, Any]], errors: list[str]) -> None:
    covered_pages: set[int] = set()
    for section in sections:
        start_page = section.get("start_page")
        end_page = section.get("end_page")
        if start_page is None or end_page is None:
            errors.append(f"Section {section.get('id')} is missing page range")
            continue
        covered_pages.update(range(int(start_page), int(end_page) + 1))

    missing_pages = sorted(
        set(CONTENT_PAGE_RANGE) - covered_pages - ALLOWED_MISSING_CONTENT_PAGES
    )
    if missing_pages:
        errors.append(
            f"Content page coverage has unexpected missing pages: {format_page_ranges(missing_pages)}"
        )

    general_titles = {
        str(section.get("title") or "")
        for section in sections
        if section.get("content_type") == "general_monograph"
    }
    missing_general = sorted(OFFICIAL_GENERAL_MONOGRAPHS - general_titles)
    if missing_general:
        errors.append(
            f"Missing official general monographs: {', '.join(missing_general)}"
        )


def format_page_ranges(pages: list[int]) -> str:
    ranges: list[str] = []
    if not pages:
        return ""
    start = previous = pages[0]
    for page in pages[1:]:
        if page == previous + 1:
            previous = page
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = page
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ", ".join(ranges)


def load_inputs(
    *,
    final_dir: Path,
    canonical_dir: Path,
    rag_interim_dir: Path,
    docling_dir: Path,
    ankhang_markdown_dir: Path,
    final_sections_path: Path | None = None,
    final_chunks_path: Path | None = None,
    curated_tables_path: Path | None = None,
) -> CorpusInputs:
    ankhang_files = (
        sorted(ankhang_markdown_dir.glob("**/*.md"))
        if ankhang_markdown_dir.exists()
        else []
    )
    return CorpusInputs(
        final_sections=read_jsonl(final_sections_path or final_dir / "sections.jsonl"),
        final_chunks=read_jsonl(final_chunks_path or final_dir / "chunks.jsonl"),
        canonical_blocks=read_jsonl(canonical_dir / "blocks.jsonl"),
        canonical_sections=read_jsonl(canonical_dir / "sections.jsonl"),
        rag_sections=read_jsonl(rag_interim_dir / "sections.jsonl"),
        rag_chunks=read_jsonl(rag_interim_dir / "chunks.jsonl"),
        curated_tables=read_jsonl(
            curated_tables_path or docling_dir / "tables.curated.jsonl"
        ),
        ankhang_markdown_files=ankhang_files,
    )


def text_snippet(value: Any, limit: int = 220) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())[:limit]


def normalized_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def normalized_markdown_table_rows(value: Any) -> list[str]:
    rows: list[str] = []
    for line in str(value or "").splitlines():
        if is_markdown_table_row(line) and not is_markdown_separator_line(line):
            rows.append(normalized_text(line))
    return rows


def table_chunk_rows_are_in_source_block(chunk_text: Any, block_text: Any) -> bool:
    chunk_rows = normalized_markdown_table_rows(chunk_text)
    if not chunk_rows:
        return False
    block_rows = set(normalized_markdown_table_rows(block_text))
    return all(row in block_rows for row in chunk_rows)


def add_finding(report: ValidationReport, **kwargs: Any) -> None:
    report.findings.append(AuditFinding(**kwargs))


def duplicate_values(records: list[dict[str, Any]], key: str) -> list[str]:
    counts = Counter(str(record.get(key) or "") for record in records)
    return sorted(value for value, count in counts.items() if value and count > 1)


def audit_identity_and_references(
    inputs: CorpusInputs, report: ValidationReport
) -> None:
    sections_by_id = {
        str(section.get("id") or ""): section for section in inputs.final_sections
    }
    chunks_by_section: dict[str, list[dict[str, Any]]] = defaultdict(list)
    blocks_by_id = {
        str(block.get("block_id") or ""): block for block in inputs.canonical_blocks
    }
    block_ids = set(blocks_by_id)

    for duplicate in duplicate_values(inputs.final_sections, "id"):
        add_finding(
            report,
            record_id=duplicate,
            record_type="section",
            source="mixed",
            category="identity",
            severity="blocking",
            message="Duplicate final section id",
            snippet=duplicate,
            recommended_action="Fix the generator that emitted duplicate section ids and rebuild final artifacts.",
        )
    for duplicate in duplicate_values(inputs.final_chunks, "id"):
        add_finding(
            report,
            record_id=duplicate,
            record_type="chunk",
            source="mixed",
            category="identity",
            severity="blocking",
            message="Duplicate final chunk id",
            snippet=duplicate,
            recommended_action="Fix chunk id generation and rebuild final artifacts.",
        )

    for chunk in inputs.final_chunks:
        section_id = str(chunk.get("section_id") or "")
        chunk_id = str(chunk.get("id") or "")
        chunks_by_section[section_id].append(chunk)
        if section_id not in sections_by_id:
            add_finding(
                report,
                record_id=chunk_id,
                record_type="chunk",
                source=str(chunk.get("source") or ""),
                category="references",
                severity="blocking",
                message="Chunk references a missing section",
                snippet=section_id,
                recommended_action="Regenerate chunks from final sections after fixing the missing section reference.",
            )

    for section_id, section in sections_by_id.items():
        if section_id not in chunks_by_section:
            add_finding(
                report,
                record_id=section_id,
                record_type="section",
                source=str(section.get("source") or ""),
                category="references",
                severity="blocking",
                message="Section has no chunk",
                snippet=text_snippet(section.get("text")),
                recommended_action="Regenerate chunks for this section.",
            )
        for block_id in section.get("block_ids") or []:
            if str(block_id) not in block_ids:
                add_finding(
                    report,
                    record_id=section_id,
                    record_type="section",
                    source=str(section.get("source") or ""),
                    category="references",
                    severity="blocking",
                    message="Section references a missing canonical block",
                    snippet=str(block_id),
                    recommended_action="Fix canonical block generation and rebuild final artifacts.",
                )

    for section_id, section_chunks in chunks_by_section.items():
        indexes = sorted(int(chunk.get("chunk_index") or 0) for chunk in section_chunks)
        if indexes and indexes != list(range(1, len(indexes) + 1)):
            add_finding(
                report,
                record_id=section_id,
                record_type="section",
                source=str(sections_by_id.get(section_id, {}).get("source") or ""),
                category="chunking",
                severity="blocking",
                message="Chunk indexes are not contiguous",
                snippet=str(indexes[:20]),
                recommended_action="Regenerate chunks so chunk_index starts at 1 and has no gaps per section.",
            )


def audit_source_blocks(inputs: CorpusInputs, report: ValidationReport) -> None:
    blocks_by_id = {
        str(block.get("block_id") or ""): block for block in inputs.canonical_blocks
    }
    for chunk in inputs.final_chunks:
        if chunk.get("source") == ANKHANG_SOURCE:
            continue
        chunk_id = str(chunk.get("id") or "")
        source_block_id = str(chunk.get("source_block_id") or "")
        if not source_block_id:
            add_finding(
                report,
                record_id=chunk_id,
                record_type="chunk",
                source=str(chunk.get("source") or ""),
                category="source_mapping",
                severity="blocking",
                message="Non-An Khang chunk is missing source_block_id",
                snippet=text_snippet(chunk.get("text")),
                recommended_action="Rebuild final chunks from canonical blocks so every Dược thư chunk has source_block_id.",
            )
            continue
        block = blocks_by_id.get(source_block_id)
        if block is None:
            add_finding(
                report,
                record_id=chunk_id,
                record_type="chunk",
                source=str(chunk.get("source") or ""),
                category="source_mapping",
                severity="blocking",
                message="Chunk source_block_id does not exist in canonical blocks",
                snippet=source_block_id,
                recommended_action="Regenerate canonical blocks and chunks from the same build.",
            )
            continue
        if block.get("section_id") != chunk.get("section_id"):
            add_finding(
                report,
                record_id=chunk_id,
                record_type="chunk",
                source=str(chunk.get("source") or ""),
                category="source_mapping",
                severity="blocking",
                message="Chunk source block belongs to a different section",
                snippet=f"{source_block_id} -> {block.get('section_id')}",
                recommended_action="Fix chunk source_block_id assignment in canonical chunk generation.",
            )
        block_text = (
            block.get("markdown")
            if block.get("content_type") == "table"
            else block.get("text")
        )
        chunk_text = chunk.get("text")
        chunk_text_is_mapped = normalized_text(chunk_text) in normalized_text(
            block_text
        )
        if block.get(
            "content_type"
        ) == "table" and table_chunk_rows_are_in_source_block(chunk_text, block_text):
            chunk_text_is_mapped = True
        if normalized_text(chunk_text) and not chunk_text_is_mapped:
            add_finding(
                report,
                record_id=chunk_id,
                record_type="chunk",
                source=str(chunk.get("source") or ""),
                category="source_mapping",
                severity="suspect",
                message="Chunk text is not a normalized substring of its source block",
                snippet=text_snippet(chunk.get("text")),
                source_evidence=text_snippet(block_text),
                recommended_action="Inspect the chunk splitter and source block; preserve unchanged text unless source evidence proves corruption.",
            )


def audit_text_quality(inputs: CorpusInputs, report: ValidationReport) -> None:
    for record_type, records in (
        ("section", inputs.final_sections),
        ("chunk", inputs.final_chunks),
    ):
        for record in records:
            text = str(record.get("text") or "")
            record_id = str(record.get("id") or "")
            source = str(record.get("source") or "")
            for char in text:
                category = unicodedata.category(char)
                if category in CONTROL_CATEGORY_PREFIXES and char not in {"\n", "\t"}:
                    add_finding(
                        report,
                        record_id=record_id,
                        record_type=record_type,
                        source=source,
                        category="text_quality",
                        severity="blocking",
                        message="Text contains disallowed control or format character",
                        snippet=repr(char),
                        recommended_action="Normalize or remove the control character in the upstream cleaning layer.",
                    )
                    break
            if "\ufffd" in text or any("\ue000" <= char <= "\uf8ff" for char in text):
                add_finding(
                    report,
                    record_id=record_id,
                    record_type=record_type,
                    source=source,
                    category="text_quality",
                    severity="auto_fixable",
                    message="Text contains replacement or private-use character",
                    snippet=text_snippet(text),
                    auto_fixable=True,
                    recommended_action="Apply a deterministic Unicode cleanup in the source-specific cleaning layer and rebuild.",
                )


def audit_part_chains(inputs: CorpusInputs, report: ValidationReport) -> None:
    for section in inputs.final_sections:
        section_id = str(section.get("id") or "")
        if not PART_SUFFIX_RE.search(section_id):
            continue
        if section_id in SALBUTAMOL_DESCRIPTOR_SECTIONS:
            add_finding(
                report,
                record_id=section_id,
                record_type="section",
                source=str(section.get("source") or ""),
                category="semantic_structure",
                severity="auto_fixable",
                message="Salbutamol descriptor monograph was emitted as part-* section",
                snippet=text_snippet(section.get("text")),
                source_evidence="# SALBUTAMOL followed by plain parenthetical descriptor in data/interim/text/full.cleaned.md",
                auto_fixable=True,
                recommended_action="Merge the plain parenthetical descriptor into the title before parsing so ids use salbutamol-dung-trong-ho-hap or salbutamol-dung-trong-san-khoa.",
            )
        else:
            add_finding(
                report,
                record_id=section_id,
                record_type="section",
                source=str(section.get("source") or ""),
                category="semantic_structure",
                severity="suspect",
                message="Section id has part-* suffix",
                snippet=text_snippet(section.get("text")),
                recommended_action="Review upstream source and canonical tables; keep unchanged unless source evidence proves this is a split error.",
            )


def audit_duplicates(inputs: CorpusInputs, report: ValidationReport) -> None:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for section in inputs.final_sections:
        text_key = normalized_text(section.get("text")).lower()
        if text_key:
            buckets[(str(section.get("source") or ""), text_key)].append(section)

    for records in buckets.values():
        if len(records) < 2:
            continue
        first = records[0]
        ids = [str(record.get("id") or "") for record in records]
        is_reviewed_group = frozenset(ids) in ACCEPTED_DUPLICATE_SECTION_GROUPS
        is_reviewed_section_type = (
            first.get("section") in ACCEPTED_DUPLICATE_SECTIONS
            and len(str(first.get("text") or "")) <= 220
        )
        severity = (
            "accepted" if is_reviewed_group or is_reviewed_section_type else "suspect"
        )
        add_finding(
            report,
            record_id=ids[0],
            record_type="section_group",
            source=str(first.get("source") or ""),
            category="duplicates",
            severity=severity,
            message="Sections have exact duplicate normalized text",
            snippet=", ".join(ids[:8]),
            recommended_action=(
                "Accepted reviewed repeated medical/regulatory text; no corpus change required."
                if severity == "accepted"
                else "Inspect duplicate sections before removing; repeated medical language can be intentional."
            ),
        )


def audit_tables_and_chunks(inputs: CorpusInputs, report: ValidationReport) -> None:
    for chunk in inputs.final_chunks:
        chunk_id = str(chunk.get("id") or "")
        text = str(chunk.get("text") or "")
        lines = text.splitlines()
        has_table_row = any(is_markdown_table_row(line) for line in lines)
        has_separator = any(is_markdown_separator_line(line) for line in lines)
        if chunk.get("chunk_role") == "table" and not chunk.get("table_id"):
            add_finding(
                report,
                record_id=chunk_id,
                record_type="chunk",
                source=str(chunk.get("source") or ""),
                category="tables",
                severity="blocking",
                message="Table chunk is missing table_id",
                snippet=text_snippet(text),
                recommended_action="Fix canonical table chunk metadata and rebuild final chunks.",
            )
        if (
            has_table_row
            and not has_separator
            and chunk.get("source") != ANKHANG_SOURCE
        ):
            add_finding(
                report,
                record_id=chunk_id,
                record_type="chunk",
                source=str(chunk.get("source") or ""),
                category="tables",
                severity="suspect",
                message="Chunk contains Markdown table rows without a separator line",
                snippet=text_snippet(text),
                recommended_action="Inspect source block and table splitter before editing; some single-column An Khang layouts are intentionally flattened.",
            )


def audit_ankhang(inputs: CorpusInputs, report: ValidationReport) -> None:
    for section in inputs.final_sections:
        if section.get("source") != ANKHANG_SOURCE:
            continue
        section_id = str(section.get("id") or "")
        text = str(section.get("text") or "")
        if not text.lstrip().startswith("# "):
            add_finding(
                report,
                record_id=section_id,
                record_type="section",
                source=ANKHANG_SOURCE,
                category="ankhang",
                severity="suspect",
                message="An Khang page is missing expected H1 heading",
                snippet=text_snippet(text),
                recommended_action="Re-parse the source Markdown/HTML for this page before changing final text.",
            )
        if HTML_LEFTOVER_RE.search(text):
            add_finding(
                report,
                record_id=section_id,
                record_type="section",
                source=ANKHANG_SOURCE,
                category="ankhang",
                severity="suspect",
                message="An Khang text contains raw HTML-like markup",
                snippet=text_snippet(text),
                recommended_action="Fix the An Khang HTML-to-Markdown parser only after verifying the raw HTML source.",
            )


def deep_audit_metrics_for(inputs: CorpusInputs) -> dict[str, Any]:
    section_sources = Counter(
        str(section.get("source") or "") for section in inputs.final_sections
    )
    chunk_sources = Counter(
        str(chunk.get("source") or "") for chunk in inputs.final_chunks
    )
    return {
        "section_count": len(inputs.final_sections),
        "chunk_count": len(inputs.final_chunks),
        "canonical_block_count": len(inputs.canonical_blocks),
        "canonical_section_count": len(inputs.canonical_sections),
        "rag_interim_section_count": len(inputs.rag_sections),
        "rag_interim_chunk_count": len(inputs.rag_chunks),
        "curated_table_count": len(inputs.curated_tables),
        "ankhang_markdown_file_count": len(inputs.ankhang_markdown_files),
        "section_sources": dict(sorted(section_sources.items())),
        "chunk_sources": dict(sorted(chunk_sources.items())),
    }


def apply_deep_audit_checks(inputs: CorpusInputs, report: ValidationReport) -> None:
    audit_identity_and_references(inputs, report)
    audit_source_blocks(inputs, report)
    audit_text_quality(inputs, report)
    audit_part_chains(inputs, report)
    audit_duplicates(inputs, report)
    audit_tables_and_chunks(inputs, report)
    audit_ankhang(inputs, report)
    report.findings.sort(
        key=lambda item: (item.severity, item.category, item.record_id, item.message)
    )


def run_deep_audit(
    *,
    final_dir: Path = DEFAULT_FINAL_DIR,
    canonical_dir: Path = DEFAULT_CANONICAL_DIR,
    rag_interim_dir: Path = DEFAULT_RAG_INTERIM_DIR,
    docling_dir: Path = DEFAULT_DOCLING_DIR,
    ankhang_markdown_dir: Path = DEFAULT_ANKHANG_MARKDOWN_DIR,
    final_sections_path: Path | None = None,
    final_chunks_path: Path | None = None,
    curated_tables_path: Path | None = None,
) -> ValidationReport:
    inputs = load_inputs(
        final_dir=Path(final_dir),
        canonical_dir=Path(canonical_dir),
        rag_interim_dir=Path(rag_interim_dir),
        docling_dir=Path(docling_dir),
        ankhang_markdown_dir=Path(ankhang_markdown_dir),
        final_sections_path=final_sections_path,
        final_chunks_path=final_chunks_path,
        curated_tables_path=curated_tables_path,
    )
    report = ValidationReport(ok=True, metrics=deep_audit_metrics_for(inputs))
    apply_deep_audit_checks(inputs, report)
    report.ok = report.blocking_count == 0
    return report


def validate_final_rag(
    *,
    final_dir: Path = DEFAULT_FINAL_DIR,
    canonical_dir: Path = DEFAULT_CANONICAL_DIR,
    rag_interim_dir: Path = DEFAULT_RAG_INTERIM_DIR,
    docling_dir: Path = DEFAULT_DOCLING_DIR,
    ankhang_markdown_dir: Path = DEFAULT_ANKHANG_MARKDOWN_DIR,
    enforce_full_coverage: bool = True,
    include_deep_audit: bool = False,
    final_sections_path: Path | None = None,
    final_chunks_path: Path | None = None,
    curated_tables_path: Path | None = None,
) -> ValidationReport:
    final_dir = Path(final_dir)
    canonical_dir = Path(canonical_dir)
    errors: list[str] = []
    warnings: list[str] = []

    sections = read_jsonl(final_sections_path or final_dir / "sections.jsonl")
    chunks = read_jsonl(final_chunks_path or final_dir / "chunks.jsonl")
    blocks = read_jsonl(canonical_dir / "blocks.jsonl")
    final_audit = read_json(final_dir / "audit.json")
    canonical_audit = read_json(canonical_dir / "audit.json")
    tables_by_id = table_blocks(blocks)

    for name, records in (
        ("sections", sections),
        ("chunks", chunks),
        ("blocks", blocks),
    ):
        duplicates = duplicate_ids(records, "id" if name != "blocks" else "block_id")
        for duplicate in duplicates:
            errors.append(f"Duplicate {name} id: {duplicate}")

    validate_counts(
        final_audit=final_audit,
        canonical_audit=canonical_audit,
        sections=sections,
        chunks=chunks,
        table_block_count=len(tables_by_id),
        errors=errors,
    )
    validate_references(
        sections=sections,
        chunks=chunks,
        blocks=blocks,
        tables_by_id=tables_by_id,
        errors=errors,
    )
    validate_chunk_source_blocks(chunks=chunks, blocks=blocks, errors=errors)
    validate_sentinels(sections, tables_by_id, errors)
    validate_text_quality(sections, chunks, errors)
    validate_quality_gate(sections, chunks, errors)
    validate_brand_index_chunks(chunks, errors)
    validate_hydration_metadata(sections, chunks, errors)
    if enforce_full_coverage:
        validate_full_coverage(sections, errors)
    validate_duplicate_review_file(canonical_dir, errors)

    metrics = {
        "section_count": len(sections),
        "chunk_count": len(chunks),
        "canonical_block_count": len(blocks),
        "table_block_count": len(tables_by_id),
        "duplicate_review_count": int(final_audit.get("duplicate_review_count") or 0),
    }
    report = ValidationReport(
        ok=not errors, errors=errors, warnings=warnings, metrics=metrics
    )
    if include_deep_audit:
        inputs = load_inputs(
            final_dir=Path(final_dir),
            canonical_dir=Path(canonical_dir),
            rag_interim_dir=Path(rag_interim_dir),
            docling_dir=Path(docling_dir),
            ankhang_markdown_dir=Path(ankhang_markdown_dir),
            final_sections_path=final_sections_path,
            final_chunks_path=final_chunks_path,
            curated_tables_path=curated_tables_path,
        )
        report.metrics.update(deep_audit_metrics_for(inputs))
        apply_deep_audit_checks(inputs, report)
        report.ok = not report.errors and report.blocking_count == 0
    return report


def validate_unified_chunks(path: Path) -> ValidationReport:
    chunks = read_jsonl(Path(path))
    errors: list[str] = []
    warnings: list[str] = []
    seen: set[str] = set()
    for index, chunk in enumerate(chunks, start=1):
        chunk_id = chunk_record_id(chunk)
        if not chunk_id:
            errors.append(f"Unified chunk row {index} is missing chunk_id")
        elif chunk_id in seen:
            errors.append(f"Duplicate unified chunk_id: {chunk_id}")
        seen.add(chunk_id)
        missing = [
            field
            for field in ("section_id", "chunk_text", "embedding_text")
            if not str(chunk.get(field) or "").strip()
        ]
        if missing:
            errors.append(f"Unified chunk {chunk_id or index} missing: {missing}")
            continue
        runtime = {
            field: chunk[field]
            for field in QDRANT_RUNTIME_PAYLOAD_FIELDS
            if field in chunk
        }
        try:
            validate_runtime_payload(runtime, label=f"Unified chunk {chunk_id}")
        except ValueError as exc:
            errors.append(str(exc))
        if chunk.get("hydrate_strategy") not in HYDRATE_STRATEGIES:
            errors.append(
                f"Unified chunk {chunk_id} has invalid hydrate_strategy: "
                f"{chunk.get('hydrate_strategy')}"
            )
    return ValidationReport(
        ok=not errors,
        errors=errors,
        warnings=warnings,
        metrics={"chunk_count": len(chunks)},
    )


def combine_validation_reports(
    *,
    source_report: dict[str, Any],
    deep_report: dict[str, Any],
    unified_report: dict[str, Any],
) -> dict[str, Any]:
    parts = (source_report, deep_report, unified_report)
    return {
        "ok": all(bool(part.get("ok")) for part in parts),
        "metrics": {
            key: value
            for part in parts
            for key, value in (part.get("metrics") or {}).items()
        },
        "errors": [error for part in parts for error in part.get("errors", [])],
        "warnings": [warning for part in parts for warning in part.get("warnings", [])],
        "findings": [finding for part in parts for finding in part.get("findings", [])],
        "stages": {
            "source_validation": source_report,
            "deep_audit": deep_report,
            "unified_contract": unified_report,
        },
    }


def write_json_report(path: Path, report: ValidationReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def write_markdown_report(path: Path, report: ValidationReport) -> None:
    lines = [
        "# RAG Final Validation Report",
        "",
        f"- OK: `{report.ok}`",
        f"- Validation errors: `{len(report.errors)}`",
        f"- Warnings: `{len(report.warnings)}`",
        f"- Blocking findings: `{report.blocking_count}`",
        f"- Auto-fixable findings: `{report.auto_fixable_count}`",
        f"- Suspect findings: `{report.suspect_count}`",
        f"- Accepted findings: `{report.accepted_count}`",
        "",
        "## Metrics",
        "",
    ]
    for key, value in sorted(report.metrics.items()):
        lines.append(f"- `{key}`: `{value}`")

    lines.extend(["", "## Validation Errors", ""])
    if report.errors:
        lines.extend(f"- {error}" for error in report.errors)
    else:
        lines.append("No validation errors.")

    lines.extend(["", "## Warnings", ""])
    if report.warnings:
        lines.extend(f"- {warning}" for warning in report.warnings)
    else:
        lines.append("No warnings.")

    lines.extend(["", "## Audit Findings", ""])
    if not report.findings:
        lines.append("No findings.")
    for finding in report.findings:
        lines.extend(
            [
                f"### {finding.severity}: {finding.record_id}",
                "",
                f"- Record type: `{finding.record_type}`",
                f"- Source: `{finding.source}`",
                f"- Category: `{finding.category}`",
                f"- Message: {finding.message}",
                f"- Auto-fixable: `{finding.auto_fixable}`",
                f"- Snippet: {finding.snippet}",
                f"- Source evidence: {finding.source_evidence or ''}",
                f"- Recommended action: {finding.recommended_action}",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
