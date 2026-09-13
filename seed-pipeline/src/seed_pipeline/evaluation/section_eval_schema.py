import re
from dataclasses import dataclass

DEFAULT_TARGET_ROWS = 10000
DEFAULT_MIN_ROWS = DEFAULT_TARGET_ROWS
DEFAULT_MAX_ROWS = DEFAULT_TARGET_ROWS

DEFAULT_EVAL_GROUP_QUOTAS = {
    "formulary": 5000,
    "ankhang": 2500,
    "patient_natural": 500,
    "chunk_risk": 1000,
    "multi_intent": 500,
    "noisy_confuser": 500,
}

FORMULARY_QUERY_FORM_TARGETS = {
    "lexical": 500,
    "template": 500,
    "paraphrase": 1500,
    "natural": 1500,
    "clinical": 1000,
}

NOISY_CONFUSER_QUERY_FORM_TARGETS = {
    "noisy": 300,
    "confuser": 200,
}

ANKHANG_QUERY_FORM_TARGETS = {
    "brand_template": 2000,
    "colloquial_alias": 500,
}

ANKHANG_ALIAS_ANSWER_MODE_TARGETS = {
    "single": 400,
    "any_acceptable": 100,
}

EVAL_HEADER = [
    "query_id",
    "query",
    "eval_group",
    "source_family",
    "query_form",
    "intent_category",
    "source_subcategory",
    "expected_section_id",
    "expected_section_ids",
    "query_intent_count",
    "answer_mode",
    "expected_title",
    "expected_chunk_id",
    "expected_chunk_index",
    "expected_chunk_role",
    "retrieval_granularity",
    "difficulty",
    "eval_tags",
    "notes",
]

ALLOWED_EVAL_GROUPS = set(DEFAULT_EVAL_GROUP_QUOTAS)
ALLOWED_SOURCE_FAMILIES = {
    "drug_formulary",
    "ankhang_brand",
    "general_appendix",
    "general_guidance",
}
ALLOWED_QUERY_FORMS = {
    "lexical",
    "template",
    "paraphrase",
    "natural",
    "clinical",
    "noisy",
    "confuser",
    "patient",
    "brand_template",
    "colloquial_alias",
    "chunk_coverage",
    "multi_intent",
}
ALLOWED_INTENT_CATEGORIES = {
    "indication",
    "dosage",
    "contraindication",
    "adr",
    "interaction",
    "pregnancy_lactation",
    "ingredient",
    "precaution",
    "storage",
    "manufacturer",
    "pharmacology",
    "general_info",
    "brand_lookup",
    "atc_lookup",
    "table_lookup",
    "guidance",
    "multi_intent",
}
ALLOWED_ANSWER_MODES = {"single", "multi_required", "any_acceptable"}
ALLOWED_RETRIEVAL_GRANULARITIES = {
    "section",
    "chunk_window",
    "chunk_exact",
    "multi_section",
}
ALLOWED_DIFFICULTIES = {"easy", "medium", "hard"}
RETRIEVAL_AFFINITY_TAGS = {"hybrid_favored", "bm25_favored", "dense_favored"}

FORMULARY_CATEGORY_MAP = {
    "indication": ("indication", "formulary_indication"),
    "contraindication": ("contraindication", "formulary_contraindication"),
    "dosage": ("dosage", "formulary_dosage"),
    "adr": ("adr", "formulary_adr"),
    "interaction": ("interaction", "formulary_interaction"),
    "pregnancy": ("pregnancy_lactation", "formulary_pregnancy_lactation"),
    "general_guidance": ("guidance", "formulary_guidance"),
    "atc_appendix": ("atc_lookup", "appendix_atc"),
    "brand_lookup": ("brand_lookup", "appendix_brand_index"),
    "table_lookup": ("table_lookup", "formulary_table"),
    "drug_fact": ("general_info", "formulary_general_info"),
}

ANKHANG_CATEGORY_MAP = {
    "brand_ingredient": ("ingredient", "ankhang_ingredient"),
    "brand_indication": ("indication", "ankhang_indication"),
    "brand_dosage": ("dosage", "ankhang_dosage"),
    "brand_contraindication": ("contraindication", "ankhang_contraindication"),
    "brand_adr": ("adr", "ankhang_adr"),
    "brand_precaution": ("precaution", "ankhang_precaution"),
    "brand_interaction": ("interaction", "ankhang_interaction"),
    "brand_pharmacology": ("pharmacology", "ankhang_pharmacology"),
    "brand_storage": ("storage", "ankhang_storage"),
    "brand_manufacturer": ("manufacturer", "ankhang_manufacturer"),
    "brand_product": ("general_info", "ankhang_product"),
}

ANKHANG_SOURCE_SUBCATEGORY_PRIORITY = [
    "ankhang_dosage",
    "ankhang_indication",
    "ankhang_ingredient",
    "ankhang_contraindication",
    "ankhang_adr",
    "ankhang_precaution",
    "ankhang_interaction",
    "ankhang_storage",
    "ankhang_pharmacology",
    "ankhang_manufacturer",
    "ankhang_product",
]


@dataclass(frozen=True)
class EvalTaxonomy:
    intent_category: str
    source_subcategory: str


def taxonomy_for_source_category(category: str, is_ankhang: bool) -> tuple[str, str]:
    mapping = ANKHANG_CATEGORY_MAP if is_ankhang else FORMULARY_CATEGORY_MAP
    return mapping.get(
        category,
        ("general_info", "ankhang_product" if is_ankhang else "formulary_general_info"),
    )


def source_family_for_section(section: dict) -> str:
    section_id = str(section.get("id") or "")
    content_type = str(section.get("content_type") or "")
    title = str(section.get("title") or "").lower()
    if content_type == "brand_page" or section_id.startswith("brand:ankhang:"):
        return "ankhang_brand"
    if section_id.startswith("general:phu-luc") or "phụ lục" in title:
        return "general_appendix"
    if content_type == "general_monograph" or section_id.startswith("general:"):
        return "general_guidance"
    return "drug_formulary"


def retrieval_granularity_from_hydrate_strategy(
    strategy: str, chunk_role: str = ""
) -> str:
    if strategy == "chunk_window":
        return "chunk_window"
    if strategy == "search_only" or chunk_role in {"index_entry", "appendix_list"}:
        return "chunk_exact"
    return "section"


def answer_mode_for_expected_ids(
    expected_ids: list[str], multi_required: bool = False
) -> str:
    if multi_required:
        return "multi_required"
    if len(expected_ids) > 1:
        return "any_acceptable"
    return "single"


def normalize_eval_tags(tags: list[str] | set[str] | tuple[str, ...]) -> list[str]:
    return sorted({tag.strip() for tag in tags if tag.strip()})


def clean_query_label(value: str, max_chars: int = 72) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    if text.count("|") >= 2 or re.search(r"\|\s*-{2,}\s*\|", text):
        return ""
    text = re.sub(r"(?m)^#{1,6}\s*", "", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"[*_`>]+", " ", text)
    text = text.replace("|", " ")
    text = re.sub(r"\s+", " ", text).strip(" -:;")
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    truncated = text[: max_chars + 1].rsplit(" ", 1)[0].strip(" -:;")
    if len(truncated) < 24:
        return ""
    return f"{truncated}..."


def expected_sections_for_eval_row(row: dict) -> list[str]:
    raw_expected_ids = row.get("expected_section_ids")
    if isinstance(raw_expected_ids, list):
        return [str(section_id) for section_id in raw_expected_ids if section_id]
    if raw_expected_ids:
        return [
            section_id for section_id in str(raw_expected_ids).split("|") if section_id
        ]
    expected_id = str(row.get("expected_section_id") or "")
    return [expected_id] if expected_id else []


def validate_query_rows(rows: list[dict], headers: list[str] | None = None) -> None:
    if headers is not None:
        header_list = list(headers)
        duplicate_columns = sorted(
            {field for field in header_list if header_list.count(field) > 1}
        )
        if duplicate_columns:
            raise ValueError(f"duplicate eval column(s): {duplicate_columns}")

        missing_columns = [
            column for column in EVAL_HEADER if column not in header_list
        ]
        if missing_columns:
            raise ValueError(f"missing required eval column(s): {missing_columns}")

    for index, row in enumerate(rows, start=1):
        query_id = str(row.get("query_id") or f"row-{index}")

        expected_section_ids = row.get("expected_section_ids")
        if not isinstance(expected_section_ids, list):
            raise ValueError(
                f"row {index} {query_id} expected_section_ids must be a list"
            )

        eval_tags = row.get("eval_tags")
        if not isinstance(eval_tags, list):
            raise ValueError(f"row {index} {query_id} eval_tags must be a list")

        query_intent_count = row.get("query_intent_count")
        if not isinstance(query_intent_count, int) or isinstance(
            query_intent_count, bool
        ):
            raise ValueError(
                f"row {index} {query_id} query_intent_count must be an integer"
            )

        expected_chunk_index = row.get("expected_chunk_index")
        if expected_chunk_index is not None and (
            not isinstance(expected_chunk_index, int)
            or isinstance(expected_chunk_index, bool)
        ):
            raise ValueError(
                f"row {index} {query_id} expected_chunk_index must be an integer or None"
            )

        answer_mode = str(row.get("answer_mode") or "")
        query_form = str(row.get("query_form") or "")
        retrieval_granularity = str(row.get("retrieval_granularity") or "")
        expected_ids = expected_sections_for_eval_row(row)

        if answer_mode not in ALLOWED_ANSWER_MODES:
            raise ValueError(
                f"row {index} {query_id} has unknown answer_mode: {answer_mode}"
            )
        if query_form not in ALLOWED_QUERY_FORMS:
            raise ValueError(
                f"row {index} {query_id} has unknown query_form: {query_form}"
            )
        if retrieval_granularity not in ALLOWED_RETRIEVAL_GRANULARITIES:
            raise ValueError(
                f"row {index} {query_id} has unknown retrieval_granularity: {retrieval_granularity}"
            )
        if query_intent_count != len(expected_ids):
            raise ValueError(
                f"row {index} {query_id} query_intent_count {query_intent_count} "
                f"does not match {len(expected_ids)} expected sections"
            )
        if answer_mode == "single" and len(expected_ids) != 1:
            raise ValueError(
                f"row {index} {query_id} single answer_mode requires exactly one expected section"
            )
        if answer_mode == "any_acceptable" and len(expected_ids) < 2:
            raise ValueError(
                f"row {index} {query_id} any_acceptable answer_mode requires multiple expected sections"
            )
        if answer_mode == "multi_required" and len(expected_ids) < 2:
            raise ValueError(
                f"row {index} {query_id} multi_required answer_mode requires multiple expected sections"
            )
        if answer_mode == "multi_required" and retrieval_granularity != "multi_section":
            raise ValueError(
                f"row {index} {query_id} multi_required answer_mode requires multi_section retrieval"
            )
        if retrieval_granularity == "multi_section" and answer_mode != "multi_required":
            raise ValueError(
                f"row {index} {query_id} multi_section retrieval requires multi_required answer_mode"
            )
