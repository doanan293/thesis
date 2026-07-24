from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from rag_metadata.payload_layers import compose_evidence_text


@dataclass(frozen=True)
class HydratedEvidence:
    text: str
    chunk_ids: list[str] = field(default_factory=list)
    section_id: str = ""
    incomplete: bool = False
    warnings: list[str] = field(default_factory=list)


def _payload(payload_or_point) -> dict:
    if isinstance(payload_or_point, dict) and isinstance(
        payload_or_point.get("payload"), dict
    ):
        return payload_or_point["payload"]
    if isinstance(payload_or_point, dict):
        return payload_or_point
    payload = getattr(payload_or_point, "payload", {})
    return payload if isinstance(payload, dict) else {}


def _chunk_index(row: dict) -> int:
    try:
        return int(row.get("chunk_index") or 0)
    except (TypeError, ValueError):
        return 0


def _chunk_text(row: dict) -> tuple[str, bool]:
    text = str(row.get("chunk_text") or "").strip()
    if text:
        return text, False
    return str(row.get("embedding_text") or "").strip(), True


def assemble_evidence_from_payloads(payloads: Iterable[dict]) -> HydratedEvidence:
    rows = [_payload(item) for item in payloads]
    rows = [row for row in rows if row]
    if not rows:
        return HydratedEvidence(text="")

    warnings: list[str] = []
    seen_indexes = set()
    for row in rows:
        index = row.get("chunk_index")
        if index in seen_indexes:
            warnings.append("duplicate_chunk_index")
            break
        seen_indexes.add(index)

    ordered = sorted(
        rows, key=lambda row: (_chunk_index(row), str(row.get("chunk_id") or ""))
    )
    first = ordered[0]
    chunk_texts: list[str] = []
    missing_text = False
    for row in ordered:
        text, used_fallback = _chunk_text(row)
        chunk_texts.append(text)
        missing_text = missing_text or used_fallback

    if missing_text:
        warnings.append("missing_chunk_text")

    text = compose_evidence_text(
        context_header=str(first.get("context_header") or ""),
        colloquial_mappings=[row.get("colloquial_mapping") or {} for row in ordered],
        chunk_texts=chunk_texts,
        term_annotations=[
            annotation
            for row in ordered
            for annotation in (row.get("term_annotations") or [])
            if isinstance(annotation, dict)
        ],
    )
    return HydratedEvidence(
        text=text,
        chunk_ids=[
            str(row.get("chunk_id") or "") for row in ordered if row.get("chunk_id")
        ],
        section_id=str(first.get("section_id") or ""),
        incomplete=bool(warnings),
        warnings=warnings,
    )
