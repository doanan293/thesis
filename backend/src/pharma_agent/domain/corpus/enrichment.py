"""Glossary terms, colloquial mappings and embedding text (spec C §7.1).

Public API. Ported from pharma-lab ``term_enrichment.py``, ``payload_layers.py``
and ``build_rag_metadata.py``; output must stay identical to the old chunks.jsonl.
"""

import re
from collections.abc import Iterable, Sequence
from functools import cache

from pharma_agent.domain.corpus.bundle import ColloquialMappingRecord, GlossaryEntry
from pharma_agent.domain.retrieval.models import ColloquialMapping, TermAnnotation

TERM_ANNOTATION_LIMIT = 8
COLLOQUIAL_ALIAS_LABEL = "Tên gọi khác"
VISUAL_SIGN_LABEL = "Dấu hiệu nhận biết"
TERM_SEARCH_PREFIX = "Thuật ngữ: "
_WORD_CHARS = "A-Za-zÀ-ỹ0-9"
_WHITESPACE_RE = re.compile(r"\s+")


def _normalized_list(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for raw in values:
        text = _WHITESPACE_RE.sub(" ", raw).strip()
        if not text or text.casefold() in seen:
            continue
        seen.add(text.casefold())
        output.append(text)
    return output


def build_context_header(title: str, context_path: Sequence[str]) -> str:
    """``title\\n> path...`` without blank lines or colloquial label lines."""
    raw = "\n> ".join(line for line in (title, *context_path) if line)
    labels = (
        f"{COLLOQUIAL_ALIAS_LABEL}:".casefold(),
        f"{VISUAL_SIGN_LABEL}:".casefold(),
    )
    lines = [
        line.strip()
        for line in raw.splitlines()
        if line.strip() and not line.strip().casefold().startswith(labels)
    ]
    return "\n".join(lines).strip()


@cache
def _term_pattern(match_text: str, case_sensitive: bool) -> re.Pattern[str]:
    flags = 0 if case_sensitive else re.IGNORECASE
    return re.compile(
        rf"(?<![{_WORD_CHARS}]){re.escape(match_text)}(?![{_WORD_CHARS}])", flags
    )


def detect_terms(text: str, glossary: Sequence[GlossaryEntry]) -> list[TermAnnotation]:
    """Glossary terms found in ``text``, by first occurrence, at most 8 matches."""
    matches: list[tuple[int, str, GlossaryEntry]] = []
    seen_terms: set[str] = set()
    for entry in glossary:
        term = entry.term.strip()
        best: re.Match[str] | None = None
        for match_text in _normalized_list([term, *entry.aliases]):
            match = _term_pattern(match_text, entry.case_sensitive).search(text)
            if match is not None and (best is None or match.start() < best.start()):
                best = match
        if best is None or term in seen_terms:
            continue
        seen_terms.add(term)
        matches.append((best.start(), term, entry))
    matches.sort(key=lambda item: (item[0], item[1]))

    annotations: list[TermAnnotation] = []
    seen_annotations: set[str] = set()
    for _, term, entry in matches[:TERM_ANNOTATION_LIMIT]:
        if not term or term.casefold() in seen_annotations:
            continue
        seen_annotations.add(term.casefold())
        vi = _normalized_list(entry.vietnamese_expansions)
        en = _normalized_list(entry.english_expansions)
        if vi or en:
            annotations.append(TermAnnotation(term=term, vi=vi, en=en))
    return annotations


def mapping_for_section(
    section_key: str, mappings: Sequence[ColloquialMappingRecord]
) -> ColloquialMapping | None:
    for record in mappings:
        if section_key not in record.section_keys:
            continue
        mapping = ColloquialMapping(
            key=record.key.strip(),
            aliases=_normalized_list(record.aliases),
            visual_sign=record.visual_sign.strip(),
            product_names=_normalized_list(record.product_names),
        )
        has_content = (
            mapping.key
            or mapping.aliases
            or mapping.visual_sign
            or mapping.product_names
        )
        return mapping if has_content else None
    return None


def _colloquial_text(mapping: ColloquialMapping, visible_text: str) -> str:
    visible = visible_text.casefold()
    lines: list[str] = []
    aliases = [
        alias
        for alias in _normalized_list(mapping.aliases)
        if alias.casefold() not in visible
    ]
    if aliases:
        lines.append(f"{COLLOQUIAL_ALIAS_LABEL}: {', '.join(aliases)}")
    visual_sign = mapping.visual_sign.strip()
    if visual_sign and visual_sign.casefold() not in visible:
        lines.append(f"{VISUAL_SIGN_LABEL}: {visual_sign}")
    return "\n".join(lines)


def _term_search_text(terms: Sequence[TermAnnotation]) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for annotation in terms:
        term = annotation.term.strip()
        if not term or term in seen:
            continue
        seen.add(term)
        expansion = "; ".join(_normalized_list([*annotation.vi, *annotation.en]))
        if expansion:
            parts.append(f"{term} = {expansion}")
        if len(seen) >= TERM_ANNOTATION_LIMIT:
            break
    return TERM_SEARCH_PREFIX + " | ".join(parts) if parts else ""


def compose_embedding_text(
    *,
    context_header: str,
    chunk_text: str,
    colloquial: ColloquialMapping | None,
    terms: Sequence[TermAnnotation],
) -> str:
    """Header, unseen colloquial names, chunk, then term expansions."""
    visible_text = f"{context_header}\n\n{chunk_text}" if context_header else chunk_text
    parts: list[str] = []
    if context_header:
        parts.append(context_header)
    if colloquial is not None:
        colloquial_text = _colloquial_text(colloquial, visible_text)
        if colloquial_text:
            parts.append(colloquial_text)
    parts.append(chunk_text)
    term_text = _term_search_text(terms)
    if term_text and "thuật ngữ:" not in "\n\n".join(parts).casefold():
        parts.append(term_text)
    return "\n\n".join(parts).strip()
