import json
from dataclasses import dataclass
from pathlib import Path

from seed_pipeline.config.paths import PROCESSED_EVALUATION_DIR, RAG_FINAL_DIR
from seed_pipeline.corpus.metadata.payload_layers import compact_colloquial_mapping
from seed_pipeline.evaluation.build_section_retrieval_eval import (
    ankhang_candidates,
    build_chunks_by_section,
    chunk_identifier,
    load_jsonl,
    normalize_spaces,
)

DEFAULT_SECTIONS_PATH = RAG_FINAL_DIR / "sections.jsonl"
DEFAULT_CHUNKS_PATH = RAG_FINAL_DIR / "chunks.jsonl"
DEFAULT_OUTPUT_PATH = PROCESSED_EVALUATION_DIR / "patient_queries.json"
DEFAULT_TARGET_COUNT = 500

ANKHANG_CATEGORY_ORDER = [
    "brand_ingredient",
    "brand_indication",
    "brand_dosage",
    "brand_contraindication",
    "brand_adr",
    "brand_precaution",
    "brand_interaction",
    "brand_storage",
    "brand_manufacturer",
    "brand_pharmacology",
    "brand_product",
]

PATIENT_QUERY_TEMPLATES = {
    "brand_ingredient": (
        "{product} có thành phần gì?",
        "Trong {product} có hoạt chất nào?",
        "{product} gồm những thành phần nào?",
    ),
    "brand_indication": (
        "{product} dùng để làm gì?",
        "{product} có công dụng gì?",
        "Khi nào có thể dùng {product}?",
    ),
    "brand_dosage": (
        "{product} dùng như thế nào?",
        "Cách dùng {product} ra sao?",
        "{product} dùng liều thế nào?",
    ),
    "brand_contraindication": (
        "Ai không nên dùng {product}?",
        "{product} chống chỉ định cho trường hợp nào?",
        "Trường hợp nào cần tránh dùng {product}?",
    ),
    "brand_adr": (
        "{product} có tác dụng phụ gì?",
        "Dùng {product} có thể gặp phản ứng bất lợi nào?",
        "{product} có gây tác dụng không mong muốn nào không?",
    ),
    "brand_precaution": (
        "Khi dùng {product} cần lưu ý gì?",
        "{product} có lưu ý gì khi sử dụng?",
        "Dùng {product} cần thận trọng điều gì?",
    ),
    "brand_interaction": (
        "{product} có tương tác thuốc nào cần tránh?",
        "Dùng {product} chung với thuốc khác có sao không?",
        "{product} có kỵ với thuốc nào không?",
    ),
    "brand_storage": (
        "{product} bảo quản thế nào?",
        "Nên bảo quản {product} ra sao?",
        "{product} cần điều kiện bảo quản gì?",
    ),
    "brand_manufacturer": (
        "{product} của hãng nào?",
        "Ai sản xuất {product}?",
        "{product} do công ty nào sản xuất?",
    ),
    "brand_pharmacology": (
        "Dược lý của {product} là gì?",
        "{product} có thông tin dược lực học gì?",
        "{product} có thông tin dược động học gì?",
    ),
    "brand_product": (
        "{product} là thuốc gì?",
        "Cho tôi thông tin về {product}",
        "{product} là sản phẩm dùng cho vấn đề gì?",
    ),
}

CATEGORY_DIFFICULTY = {
    "brand_ingredient": "easy",
    "brand_indication": "easy",
    "brand_product": "easy",
    "brand_dosage": "medium",
    "brand_contraindication": "medium",
    "brand_adr": "medium",
    "brand_precaution": "medium",
    "brand_interaction": "medium",
    "brand_storage": "medium",
    "brand_manufacturer": "medium",
    "brand_pharmacology": "hard",
}


@dataclass(frozen=True)
class PatientQueryCandidate:
    section_id: str
    product_name: str
    category: str
    chunk_id: str
    chunk_index: int


def unique_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    unique = []
    for value in values:
        cleaned = normalize_spaces(value or "")
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(cleaned)
    return unique


def product_names_for_section(section: dict) -> list[str]:
    mapping = compact_colloquial_mapping(section)
    values = [
        *(mapping.get("product_names") or []),
        *(mapping.get("aliases") or []),
        section.get("title", ""),
    ]
    return unique_preserve_order([str(value) for value in values])


def display_product_name(section: dict) -> str:
    names = product_names_for_section(section)
    if not names:
        return ""
    return min(names, key=lambda value: (len(value), value.lower()))


def is_ankhang_product_section(section: dict) -> bool:
    return section.get("content_type") == "brand_page" and str(
        section.get("id", "")
    ).startswith("brand:ankhang:")


def build_patient_query_candidates(
    sections: list[dict], chunks: list[dict]
) -> list[PatientQueryCandidate]:
    sections_by_id = {
        section["id"]: section
        for section in sections
        if section.get("id") and is_ankhang_product_section(section)
    }
    if not sections_by_id:
        raise RuntimeError("No An Khang product sections found in final RAG sections")

    chunks_by_section = build_chunks_by_section(chunks)
    raw_candidates = ankhang_candidates(sections_by_id, chunks_by_section)
    candidates: list[PatientQueryCandidate] = []
    seen: set[tuple[str, str]] = set()

    for section, chunk, category in raw_candidates:
        if category not in PATIENT_QUERY_TEMPLATES:
            continue
        product_name = display_product_name(section)
        if not product_name:
            continue
        key = (section["id"], category)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            PatientQueryCandidate(
                section_id=section["id"],
                product_name=product_name,
                category=category,
                chunk_id=chunk_identifier(chunk) if chunk else "",
                chunk_index=int(chunk.get("chunk_index", -1)) if chunk else -1,
            )
        )

    if not candidates:
        raise RuntimeError(
            "No An Khang product sections produced patient query candidates"
        )
    return candidates


def ordered_candidates_by_category(
    candidates: list[PatientQueryCandidate],
) -> list[PatientQueryCandidate]:
    by_category: dict[str, list[PatientQueryCandidate]] = {
        category: [] for category in ANKHANG_CATEGORY_ORDER
    }
    for candidate in candidates:
        by_category.setdefault(candidate.category, []).append(candidate)

    for category_items in by_category.values():
        category_items.sort(
            key=lambda item: (item.section_id, item.product_name, item.category)
        )

    ordered: list[PatientQueryCandidate] = []
    cursors = dict.fromkeys(by_category, 0)
    while True:
        progressed = False
        for category in ANKHANG_CATEGORY_ORDER:
            items = by_category.get(category, [])
            cursor = cursors.get(category, 0)
            if cursor >= len(items):
                continue
            ordered.append(items[cursor])
            cursors[category] = cursor + 1
            progressed = True
        if not progressed:
            break
    return ordered


def row_for_candidate(candidate: PatientQueryCandidate, template_index: int) -> dict:
    templates = PATIENT_QUERY_TEMPLATES[candidate.category]
    query = templates[template_index % len(templates)].format(
        product=candidate.product_name
    )
    return {
        "query": normalize_spaces(query),
        "category": candidate.category,
        "expected_section_id": candidate.section_id,
        "difficulty": CATEGORY_DIFFICULTY.get(candidate.category, "medium"),
        "notes": f"An Khang patient query for {candidate.category}",
    }


def generate_patient_queries(
    sections: list[dict],
    chunks: list[dict],
    target_count: int = DEFAULT_TARGET_COUNT,
) -> list[dict]:
    if target_count <= 0:
        raise ValueError("target_count must be positive")

    candidates = ordered_candidates_by_category(
        build_patient_query_candidates(sections, chunks)
    )
    rows: list[dict] = []
    seen_pairs: set[tuple[str, str]] = set()
    max_template_count = max(
        len(templates) for templates in PATIENT_QUERY_TEMPLATES.values()
    )

    for template_index in range(max_template_count):
        progressed = False
        for candidate in candidates:
            row = row_for_candidate(candidate, template_index)
            pair = (row["query"].lower(), row["expected_section_id"])
            if pair in seen_pairs:
                continue
            if not row["expected_section_id"].startswith("brand:ankhang:"):
                raise RuntimeError(
                    f"Generated non-An Khang patient row: {row['expected_section_id']}"
                )
            rows.append(row)
            seen_pairs.add(pair)
            progressed = True
            if len(rows) >= target_count:
                return rows
        if not progressed:
            break

    raise RuntimeError(
        f"Could only generate {len(rows)} unique An Khang patient queries for target {target_count}"
    )


def build_patient_queries(
    sections_path: Path = DEFAULT_SECTIONS_PATH,
    chunks_path: Path = DEFAULT_CHUNKS_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    target_count: int = DEFAULT_TARGET_COUNT,
) -> list[dict]:
    sections = load_jsonl(sections_path)
    chunks = load_jsonl(chunks_path)
    rows = generate_patient_queries(
        sections=sections, chunks=chunks, target_count=target_count
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return rows
