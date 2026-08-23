from __future__ import annotations

import copy
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from corpus_pipeline.evaluation.artifact_contracts import ArtifactContractError

GENERIC_DRUG_FACT_TARGETS = {"tuong-ky", "thong-tin-qui-che", "noi-dung"}
GENERAL_SECTION_SUFFIX = "thong-tin-chung"
MAPPING_PHRASE = "biet duoc chua hoat chat"


@dataclass(frozen=True)
class RejudgingResult:
    rows: list[dict[str, Any]]
    summary: dict[str, int]


def query_projection(rows: list[dict[str, Any]]) -> list[tuple[str, str]]:
    return [(str(row["query_id"]), str(row["query"])) for row in rows]


def validate_immutable_queries(
    before: list[dict[str, Any]], after: list[dict[str, Any]]
) -> None:
    if query_projection(before) != query_projection(after):
        raise ArtifactContractError(
            "Rejudging changed query inputs; embedding and rerank artifacts cannot be reused"
        )


def _normalize(value: object) -> str:
    normalized = unicodedata.normalize("NFD", str(value or "").lower())
    without_marks = "".join(
        char for char in normalized if unicodedata.category(char) != "Mn"
    )
    without_marks = without_marks.replace("đ", "d")
    return re.sub(r"[^a-z0-9]+", " ", without_marks).strip()


def _is_drug_fact(row: dict[str, Any]) -> bool:
    tags = {str(tag) for tag in row.get("eval_tags") or []}
    return row.get("eval_group") == "formulary" and (
        "drug_fact" in tags or row.get("intent_category") == "general_info"
    )


def _general_section_id(section_id: str) -> str | None:
    parts = section_id.split(":")
    if len(parts) < 3 or parts[0] != "drug":
        return None
    if parts[-1] not in GENERIC_DRUG_FACT_TARGETS:
        return None
    return ":".join((*parts[:-1], GENERAL_SECTION_SUFFIX))


def _retarget_drug_fact(
    row: dict[str, Any], known_section_ids: set[str]
) -> dict[str, Any] | None:
    current = str(row.get("expected_section_id") or "")
    target = _general_section_id(current)
    if target is None or target not in known_section_ids:
        return None
    revised = copy.deepcopy(row)
    revised["expected_section_id"] = target
    revised["expected_section_ids"] = [target]
    chunk_id = f"{target}:chunk-001"
    revised["expected_chunk_id"] = chunk_id
    revised["expected_chunk_ids"] = [chunk_id]
    revised["expected_chunk_index"] = 1
    revised["retrieval_granularity"] = "section"
    return revised


def _brand_title(row: dict[str, Any]) -> str:
    title = str(row.get("expected_title") or "")
    return _normalize(title.rsplit(" - ", 1)[0])


def _candidate_text(candidate: dict[str, Any]) -> str:
    payload = candidate.get("payload") or {}
    return str(candidate.get("document_text") or payload.get("document_text") or "")


def _brand_lookup_chunk_ids(
    row: dict[str, Any], candidate_rows: dict[str, list[dict[str, Any]]]
) -> list[str]:
    title = _brand_title(row)
    if not title:
        return []
    accepted: list[str] = []
    for candidate in candidate_rows.get(str(row["query_id"]), []):
        text = _normalize(_candidate_text(candidate))
        chunk_id = str(candidate.get("chunk_id") or "")
        if chunk_id and title in text and MAPPING_PHRASE in text:
            accepted.append(chunk_id)
    return sorted(set(accepted))


def rejudge_rows(
    rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    known_section_ids: set[str],
) -> RejudgingResult:
    candidate_index: dict[str, list[dict[str, Any]]] = {}
    for record in candidate_rows:
        candidate_index.setdefault(str(record["query_id"]), []).extend(
            list(record.get("candidates") or [])
        )

    summary = {
        "changed": 0,
        "unchanged": 0,
        "drug_fact_changed": 0,
        "drug_fact_skipped": 0,
        "brand_lookup_changed": 0,
        "brand_lookup_skipped": 0,
    }
    revised_rows: list[dict[str, Any]] = []
    for row in rows:
        updated = None
        if row.get("answer_mode") != "multi_required" and _is_drug_fact(row):
            updated = _retarget_drug_fact(row, known_section_ids)
            if updated is not None:
                summary["drug_fact_changed"] += 1
            else:
                summary["drug_fact_skipped"] += 1

        if updated is None and row.get("intent_category") == "brand_lookup":
            chunk_ids = _brand_lookup_chunk_ids(row, candidate_index)
            legacy = str(row.get("expected_chunk_id") or "")
            existing = list(row.get("expected_chunk_ids") or [])
            additional = [
                chunk_id
                for chunk_id in chunk_ids
                if chunk_id != legacy and chunk_id not in existing
            ]
            if additional:
                updated = copy.deepcopy(row)
                if legacy and legacy not in existing:
                    existing.insert(0, legacy)
                updated["expected_chunk_ids"] = sorted(set(existing + additional))
                if legacy and updated["expected_chunk_ids"][0] != legacy:
                    updated["expected_chunk_ids"].remove(legacy)
                    updated["expected_chunk_ids"].insert(0, legacy)
                summary["brand_lookup_changed"] += 1
            elif row.get("retrieval_granularity") == "chunk_exact":
                summary["brand_lookup_skipped"] += 1

        if updated is None:
            revised_rows.append(copy.deepcopy(row))
            summary["unchanged"] += 1
        else:
            revised_rows.append(updated)
            summary["changed"] += 1
    validate_immutable_queries(rows, revised_rows)
    return RejudgingResult(revised_rows, summary)
