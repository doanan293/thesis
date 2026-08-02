from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

CONFIDENCE_VALUES = {"high", "medium", "low"}
TERM_ENRICHMENT_LIMIT = 8
ACRONYM_RE = re.compile(
    r"(?<![A-Za-zÀ-ỹ0-9])([A-Z][A-Z0-9]{1,}(?:[-/][A-Z0-9]+)*)(?![A-Za-zÀ-ỹ0-9])"
)
NOISE_ACRONYMS = {
    "I",
    "II",
    "III",
    "IV",
    "V",
    "VI",
    "VII",
    "VIII",
    "IX",
    "X",
    "ML",
    "MG",
    "KG",
    "G",
    "UI",
    "IU",
    "MCG",
    "CM",
    "MM",
    "L",
    "H",
}


def _normalized_list(value) -> list[str]:
    if value in (None, "", [], {}):
        return []
    raw_values = value if isinstance(value, list) else [value]
    output: list[str] = []
    seen = set()
    for raw in raw_values:
        text = re.sub(r"\s+", " ", str(raw or "")).strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        output.append(text)
    return output


def load_term_glossary(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("term glossary must be a JSON array")

    entries: list[dict] = []
    seen_terms = set()
    for index, record in enumerate(data, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"term glossary row {index} must be an object")
        term = str(record.get("term") or "").strip()
        if not term:
            raise ValueError(f"term glossary row {index} missing term")
        if term in seen_terms:
            raise ValueError(f"duplicate glossary term: {term}")
        seen_terms.add(term)

        confidence = str(record.get("confidence") or "").strip()
        if confidence not in CONFIDENCE_VALUES:
            raise ValueError(
                f"term glossary row {index} invalid confidence: {confidence}"
            )
        source = str(record.get("source") or "").strip()
        if not source:
            raise ValueError(f"term glossary row {index} missing source")

        vietnamese_expansions = _normalized_list(record.get("vietnamese_expansions"))
        english_expansions = _normalized_list(record.get("english_expansions"))
        aliases = _normalized_list(record.get("aliases"))
        if not vietnamese_expansions and not english_expansions and not aliases:
            raise ValueError(
                f"term glossary row {index} must define an expansion or alias"
            )

        match_texts = _normalized_list([term, *aliases])
        entries.append(
            {
                "term": term,
                "case_sensitive": bool(record.get("case_sensitive", True)),
                "vietnamese_expansions": vietnamese_expansions,
                "english_expansions": english_expansions,
                "aliases": aliases,
                "category": str(record.get("category") or "").strip(),
                "confidence": confidence,
                "source": source,
                "match_texts": match_texts,
            }
        )
    return entries


def _term_pattern(match_text: str, case_sensitive: bool) -> re.Pattern:
    flags = 0 if case_sensitive else re.IGNORECASE
    escaped = re.escape(match_text)
    return re.compile(rf"(?<![A-Za-zÀ-ỹ0-9]){escaped}(?![A-Za-zÀ-ỹ0-9])", flags)


def detect_term_enrichments(
    text: str, glossary_entries: list[dict], limit: int = TERM_ENRICHMENT_LIMIT
) -> list[dict]:
    haystack = str(text or "")
    matches: list[tuple[int, dict]] = []
    seen_terms = set()
    for entry in glossary_entries:
        term = entry["term"]
        best_match = None
        for match_text in entry.get("match_texts", []):
            match = _term_pattern(
                match_text, bool(entry.get("case_sensitive", True))
            ).search(haystack)
            if match is not None and (
                best_match is None or match.start() < best_match.start()
            ):
                best_match = match
        if best_match is None or term in seen_terms:
            continue
        seen_terms.add(term)
        matches.append(
            (
                best_match.start(),
                {
                    "term": term,
                    "matched_text": best_match.group(0),
                    "vietnamese_expansions": list(
                        entry.get("vietnamese_expansions", [])
                    ),
                    "english_expansions": list(entry.get("english_expansions", [])),
                    "aliases": list(entry.get("aliases", [])),
                    "category": str(entry.get("category") or ""),
                    "confidence": str(entry.get("confidence") or ""),
                    "source": str(entry.get("source") or ""),
                },
            )
        )
    return [
        item
        for _position, item in sorted(
            matches, key=lambda pair: (pair[0], pair[1]["term"])
        )[:limit]
    ]


def _expansion_text(enrichment: dict) -> str:
    values = _normalized_list(
        [
            *enrichment.get("vietnamese_expansions", []),
            *enrichment.get("english_expansions", []),
        ]
    )
    return "; ".join(values)


def _dedupe_enrichments(
    term_enrichments: Iterable[dict], limit: int = TERM_ENRICHMENT_LIMIT
) -> list[dict]:
    output = []
    seen = set()
    for enrichment in term_enrichments or []:
        term = str(enrichment.get("term") or "").strip()
        if not term or term in seen:
            continue
        seen.add(term)
        output.append(enrichment)
        if len(output) >= limit:
            break
    return output


def format_term_search_text(
    term_enrichments: Iterable[dict], limit: int = TERM_ENRICHMENT_LIMIT
) -> str:
    parts = []
    for enrichment in _dedupe_enrichments(term_enrichments, limit):
        expansion = _expansion_text(enrichment)
        if expansion:
            parts.append(f"{enrichment['term']} = {expansion}")
    return "Thuật ngữ: " + " | ".join(parts) if parts else ""


def format_term_annotation(
    term_enrichments: Iterable[dict],
    visible_text: str = "",
    limit: int = TERM_ENRICHMENT_LIMIT,
) -> str:
    lines = []
    visible = str(visible_text or "").casefold()
    for enrichment in _dedupe_enrichments(term_enrichments, limit):
        expansion = _expansion_text(enrichment)
        if not expansion:
            continue
        if enrichment["term"].casefold() in visible and expansion.casefold() in visible:
            continue
        lines.append(f"- {enrichment['term']}: {expansion}")
    return "Thuật ngữ trong đoạn:\n" + "\n".join(lines) if lines else ""


def validate_term_enrichments(term_enrichments, chunk_id: str) -> None:
    if term_enrichments in (None, []):
        return
    if not isinstance(term_enrichments, list):
        raise ValueError(
            f"Canonical chunk {chunk_id} field term_enrichments must be a list"
        )
    if len(term_enrichments) > TERM_ENRICHMENT_LIMIT:
        raise ValueError(f"Canonical chunk {chunk_id} has too many term_enrichments")
    for index, item in enumerate(term_enrichments, start=1):
        if not isinstance(item, dict):
            raise ValueError(
                f"Canonical chunk {chunk_id} term_enrichments[{index}] must be an object"
            )
        term = str(item.get("term") or "").strip()
        if not term:
            raise ValueError(
                f"Canonical chunk {chunk_id} term_enrichments[{index}] missing term"
            )
        confidence = str(item.get("confidence") or "")
        if confidence not in CONFIDENCE_VALUES:
            raise ValueError(
                f"Canonical chunk {chunk_id} term_enrichments[{index}] invalid confidence: {confidence}"
            )


def _audit_terms(chunk: dict) -> list[str]:
    raw_enrichments = chunk.get("term_enrichments") or []
    if raw_enrichments:
        return [
            str(item.get("term") or "")
            for item in raw_enrichments
            if isinstance(item, dict)
        ]
    annotations = chunk.get("term_annotations") or []
    return [
        str(item.get("term") or "") for item in annotations if isinstance(item, dict)
    ]


def build_term_enrichment_audit(
    canonical_chunks: list[dict], glossary_entries: list[dict]
) -> dict:
    glossary_terms = {entry["term"] for entry in glossary_entries}
    mapped_counts = Counter()
    unmapped_counts = Counter()
    unmapped_examples: dict[str, list[str]] = defaultdict(list)
    chunks_with_terms = 0

    for chunk in canonical_chunks:
        chunk_id = str(chunk.get("chunk_id") or chunk.get("id") or "")
        terms = _audit_terms(chunk)
        if terms:
            chunks_with_terms += 1
        mapped_counts.update(term for term in terms if term)
        text = str(chunk.get("chunk_text") or chunk.get("text") or "")
        for match in ACRONYM_RE.finditer(text):
            token = match.group(1)
            if token in glossary_terms or token in NOISE_ACRONYMS:
                continue
            unmapped_counts[token] += 1
            examples = unmapped_examples[token]
            if chunk_id and chunk_id not in examples and len(examples) < 5:
                examples.append(chunk_id)

    chunk_count = len(canonical_chunks)
    return {
        "schema_version": 1,
        "chunk_count": chunk_count,
        "chunks_with_term_enrichments": chunks_with_terms,
        "chunks_with_term_enrichments_percent": round(
            chunks_with_terms / chunk_count * 100, 2
        )
        if chunk_count
        else 0.0,
        "glossary_term_count": len(glossary_entries),
        "glossary_term_counts": dict(sorted(mapped_counts.items())),
        "unmapped_acronym_counts": dict(unmapped_counts.most_common(100)),
        "unmapped_acronym_examples": {
            key: unmapped_examples[key]
            for key, _count in unmapped_counts.most_common(100)
        },
    }
