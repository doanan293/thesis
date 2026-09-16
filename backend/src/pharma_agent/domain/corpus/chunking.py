"""The single corpus chunker (spec C §7.1).

Public API. Splitting is ported from pharma-lab ``build_final_chunk_records``,
``split_lines_without_breaking_entries`` and ``split_long_text``; tables follow the leaflet
``split_table`` header rule, which also reproduces every formulary table. The splitter is
chosen only by ``BlockRecord.kind``.
"""

import re
import uuid
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from pharma_agent.domain.corpus.bundle import (
    BlockKind,
    ColloquialMappingRecord,
    DocumentRecord,
    GlossaryEntry,
    SectionRecord,
)
from pharma_agent.domain.corpus.enrichment import (
    build_context_header,
    compose_embedding_text,
    detect_terms,
    mapping_for_section,
)
from pharma_agent.domain.corpus.identity import chunk_version_id, sha256_hex
from pharma_agent.domain.retrieval.models import ColloquialMapping, TermAnnotation

CHUNKER_VERSION = "chunker-v1"
MAX_CHUNK_CHARS = 3000

_PARAGRAPH_BREAK_RE = re.compile(r"\n\s*\n")
_SENTENCE_END_RE = re.compile(r"[.!?;:…]\s+")
_LAST_WORD_RE = re.compile(r"\s+\S*$")


class ChunkDraft(BaseModel):
    """One chunk of a section, ready to be stored as an immutable chunk version."""

    model_config = ConfigDict(frozen=True)

    chunk_version_id: uuid.UUID
    section_key: str
    ordinal: int
    kind: BlockKind
    chunk_text: str
    context_header: str
    embedding_text: str
    embedding_text_sha256: str
    start_page: int | None
    end_page: int | None
    table_key: str | None
    term_annotations: list[TermAnnotation]
    colloquial: ColloquialMapping | None


def split_oversized_paragraph(paragraph: str, max_chars: int) -> list[str]:
    """Cut at the last sentence end, else newline, else whitespace, else hard."""
    parts: list[str] = []
    remaining = paragraph.strip()
    while len(remaining) > max_chars:
        window = remaining[: max_chars + 1]
        split_at = -1
        sentence_ends = list(_SENTENCE_END_RE.finditer(window))
        if sentence_ends:
            split_at = sentence_ends[-1].end()
        if split_at <= 0:
            newline_at = window.rfind("\n", 0, max_chars + 1)
            if newline_at > 0:
                split_at = newline_at + 1
        if split_at <= 0:
            last_word = _LAST_WORD_RE.search(window)
            if last_word and last_word.start() > 0:
                split_at = last_word.start()
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
    """Pack blank-line separated paragraphs; text that fits is returned as is."""
    if len(text) <= max_chars:
        return [text]
    paragraphs = [
        paragraph.strip()
        for paragraph in _PARAGRAPH_BREAK_RE.split(text)
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


def split_table_markdown(markdown: str, max_chars: int) -> list[str]:
    """Split rows into parts that each repeat the table's first two rows.

    The first two rows are the header whether or not the second one is a separator, because
    leaflet tables often spread their header over several rows. A part is closed before it
    reaches ``max_chars``. Text without two leading table rows is split as prose.
    """
    text = markdown.strip()
    if len(text) <= max_chars:
        return [text]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 3 or not all(line.startswith("|") for line in lines[:2]):
        return split_long_text(text, max_chars)
    header = "\n".join(lines[:2])
    chunks: list[str] = []
    rows: list[str] = []
    length = len(header)
    for row in lines[2:]:
        if rows and length + 1 + len(row) >= max_chars:
            chunks.append("\n".join([header, *rows]))
            rows = []
            length = len(header)
        rows.append(row)
        length += 1 + len(row)
    if rows:
        chunks.append("\n".join([header, *rows]))
    return chunks


def split_lines_without_breaking_entries(text: str, max_chars: int) -> list[str]:
    """Pack lines; a line starting lowercase continues the entry above it."""
    lines = [line.rstrip() for line in text.strip().splitlines() if line.strip()]
    chunks: list[str] = []
    current: list[str] = []
    for line in lines:
        candidate = "\n".join([*current, line]) if current else line
        if current and len(candidate) > max_chars:
            next_lines = [line]
            while current and next_lines[0][:1].islower():
                next_lines.insert(0, current.pop())
            if current:
                chunks.append("\n".join(current))
            current = next_lines
            continue
        current.append(line)
    if current:
        chunks.append("\n".join(current))
    return chunks


def _split_block(kind: BlockKind, text: str, max_chars: int) -> list[str]:
    if kind is BlockKind.TABLE:
        return split_table_markdown(text, max_chars)
    if kind is BlockKind.LIST or kind is BlockKind.INDEX_ENTRIES:
        return split_lines_without_breaking_entries(text, max_chars)
    return split_long_text(text, max_chars)


def _context_header(title: str, section: SectionRecord) -> str:
    header = build_context_header(title, section.context_path)
    if header:
        return header
    title_text = title.strip()
    heading = section.heading.strip()
    if title_text and heading:
        return f"{title_text} > {heading}"
    return title_text or heading


def chunk_section(
    document: DocumentRecord,
    section: SectionRecord,
    glossary: Sequence[GlossaryEntry],
    mappings: Sequence[ColloquialMappingRecord],
    *,
    max_chars: int = MAX_CHUNK_CHARS,
) -> list[ChunkDraft]:
    """Chunk one section; a section whose blocks are all blank yields no chunks."""
    context_header = _context_header(document.title, section)
    colloquial = mapping_for_section(section.key, mappings)
    drafts: list[ChunkDraft] = []
    for block in section.blocks:
        raw_text = block.markdown.strip()
        if not raw_text:
            continue
        for part in _split_block(block.kind, raw_text, max_chars):
            chunk_text = part.strip()
            searchable = (
                f"{context_header}\n\n{chunk_text}" if context_header else chunk_text
            )
            terms = detect_terms(searchable, glossary)
            embedding_text = compose_embedding_text(
                context_header=context_header,
                chunk_text=chunk_text,
                colloquial=colloquial,
                terms=terms,
            )
            drafts.append(
                ChunkDraft(
                    chunk_version_id=chunk_version_id(
                        section.key, chunk_text, embedding_text, CHUNKER_VERSION
                    ),
                    section_key=section.key,
                    ordinal=len(drafts) + 1,
                    kind=block.kind,
                    chunk_text=chunk_text,
                    context_header=context_header,
                    embedding_text=embedding_text,
                    embedding_text_sha256=sha256_hex(embedding_text),
                    start_page=block.start_page
                    if block.start_page is not None
                    else section.start_page,
                    end_page=block.end_page
                    if block.end_page is not None
                    else section.end_page,
                    table_key=block.table_key
                    if block.kind is BlockKind.TABLE
                    else None,
                    term_annotations=terms,
                    colloquial=colloquial,
                )
            )
    return drafts
