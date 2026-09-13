from __future__ import annotations

import re

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
