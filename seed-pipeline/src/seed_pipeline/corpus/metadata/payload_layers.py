from __future__ import annotations

import re
from collections.abc import Iterable

COLLOQUIAL_ALIAS_LABEL = "Tên gọi khác"
VISUAL_SIGN_LABEL = "Dấu hiệu nhận biết"


def normalized_list(value) -> list[str]:
    if value in (None, "", [], {}):
        return []
    values = value if isinstance(value, list) else [value]
    output: list[str] = []
    seen = set()
    for raw in values:
        text = re.sub(r"\s+", " ", str(raw or "")).strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        output.append(text)
    return output


def clean_context_header(header: str) -> str:
    lines: list[str] = []
    for raw_line in (header or "").splitlines():
        line = raw_line.strip()
        lowered = line.casefold()
        if lowered.startswith(f"{COLLOQUIAL_ALIAS_LABEL}:".casefold()):
            continue
        if lowered.startswith(f"{VISUAL_SIGN_LABEL}:".casefold()):
            continue
        if line:
            lines.append(line)
    return "\n".join(lines).strip()


def compact_colloquial_mapping(record: dict | None) -> dict:
    record = record or {}
    existing = record.get("colloquial_mapping")
    if isinstance(existing, dict):
        source = existing
        key = str(source.get("key") or "").strip()
        aliases = normalized_list(source.get("aliases"))
        visual_sign = str(source.get("visual_sign") or "").strip()
        product_names = normalized_list(source.get("product_names"))
    else:
        key = str(record.get("colloquial_mapping_key") or "").strip()
        aliases = normalized_list(record.get("product_aliases"))
        visual_sign = str(record.get("visual_sign") or "").strip()
        product_names = normalized_list(record.get("product_names"))

    output: dict = {}
    if key:
        output["key"] = key
    if aliases:
        output["aliases"] = aliases
    if visual_sign:
        output["visual_sign"] = visual_sign
    if product_names:
        output["product_names"] = product_names
    return output


def term_annotations_from_enrichments(
    term_enrichments: Iterable[dict], limit: int = 8
) -> list[dict]:
    annotations: list[dict] = []
    seen = set()
    for enrichment in term_enrichments or []:
        term = str(enrichment.get("term") or "").strip()
        if not term:
            continue
        key = term.casefold()
        if key in seen:
            continue
        seen.add(key)
        annotation = {
            "term": term,
            "vi": normalized_list(enrichment.get("vietnamese_expansions")),
            "en": normalized_list(enrichment.get("english_expansions")),
        }
        if annotation["vi"] or annotation["en"]:
            annotations.append(annotation)
        if len(annotations) >= limit:
            break
    return annotations


def format_colloquial_mapping(mapping: dict, visible_text: str = "") -> str:
    visible = (visible_text or "").casefold()
    lines: list[str] = []
    aliases = [
        alias
        for alias in normalized_list(mapping.get("aliases"))
        if alias.casefold() not in visible
    ]
    if aliases:
        lines.append(f"{COLLOQUIAL_ALIAS_LABEL}: {', '.join(aliases)}")
    visual_sign = str(mapping.get("visual_sign") or "").strip()
    if visual_sign and visual_sign.casefold() not in visible:
        lines.append(f"{VISUAL_SIGN_LABEL}: {visual_sign}")
    return "\n".join(lines)


def format_term_annotations(
    term_annotations: Iterable[dict], visible_text: str = "", limit: int = 8
) -> str:
    visible = (visible_text or "").casefold()
    lines: list[str] = []
    seen = set()
    for annotation in term_annotations or []:
        term = str(annotation.get("term") or "").strip()
        if not term:
            continue
        key = term.casefold()
        if key in seen:
            continue
        seen.add(key)
        expansions = normalized_list(
            [*(annotation.get("vi") or []), *(annotation.get("en") or [])]
        )
        expansion_text = "; ".join(expansions)
        if not expansion_text:
            continue
        if term.casefold() in visible and expansion_text.casefold() in visible:
            continue
        lines.append(f"- {term}: {expansion_text}")
        if len(lines) >= limit:
            break
    return "Thuật ngữ trong đoạn:\n" + "\n".join(lines) if lines else ""


def merge_colloquial_mappings(mappings: Iterable[dict]) -> dict:
    keys: list[str] = []
    aliases: list[str] = []
    visual_signs: list[str] = []
    product_names: list[str] = []
    for mapping in mappings or []:
        if not isinstance(mapping, dict):
            continue
        keys.extend(normalized_list(mapping.get("key")))
        aliases.extend(normalized_list(mapping.get("aliases")))
        visual_signs.extend(normalized_list(mapping.get("visual_sign")))
        product_names.extend(normalized_list(mapping.get("product_names")))
    output: dict = {}
    if keys:
        output["key"] = keys[0]
    if aliases:
        output["aliases"] = normalized_list(aliases)
    if visual_signs:
        output["visual_sign"] = visual_signs[0]
    if product_names:
        output["product_names"] = normalized_list(product_names)
    return output


def compose_evidence_text(
    *,
    context_header: str,
    colloquial_mappings: Iterable[dict],
    chunk_texts: Iterable[str],
    term_annotations: Iterable[dict],
) -> str:
    parts: list[str] = []
    header = clean_context_header(context_header)
    if header:
        parts.append(header)
    body = "\n\n".join(
        (text or "").strip() for text in chunk_texts if (text or "").strip()
    )
    visible_before_mapping = "\n\n".join([*parts, body])
    mapping_text = format_colloquial_mapping(
        merge_colloquial_mappings(colloquial_mappings), visible_before_mapping
    )
    if mapping_text:
        parts.append(mapping_text)
    if body:
        parts.append(body)
    visible = "\n\n".join(parts)
    term_text = format_term_annotations(term_annotations, visible)
    if term_text:
        parts.append(term_text)
    return "\n\n".join(parts).strip()
