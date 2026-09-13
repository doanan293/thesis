"""The single corpus chunker (spec C §7.1).

Public API. Splitting is ported from corpus-pipeline ``build_final_chunk_records``,
``split_table_markdown``, ``split_lines_without_breaking_entries`` and
``split_long_text``; the splitter is chosen only by ``BlockRecord.kind``.
"""

import re

CHUNKER_VERSION = "chunker-v1"
MAX_CHUNK_CHARS = 3000

_PARAGRAPH_BREAK_RE = re.compile(r"\n\s*\n")
_SENTENCE_END_RE = re.compile(r"[.!?;:…]\s+")
_LAST_WORD_RE = re.compile(r"\s+\S*$")


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


def _is_separator_row(line: str) -> bool:
    stripped = line.strip()
    if not stripped.startswith("|") or "-" not in stripped:
        return False
    return stripped.replace("|", "").replace("-", "").replace(":", "").strip() == ""


def split_table_markdown(markdown: str, max_chars: int) -> list[str]:
    """Split rows into parts that each repeat the header and separator rows."""
    text = markdown.strip()
    if len(text) <= max_chars:
        return [text]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 3 or not _is_separator_row(lines[1]):
        return split_long_text(text, max_chars)
    prefix = [lines[0], lines[1]]
    chunks: list[str] = []
    current_rows: list[str] = []
    for row in lines[2:]:
        candidate_rows = [*current_rows, row]
        if current_rows and len("\n".join(prefix + candidate_rows)) > max_chars:
            chunks.append("\n".join(prefix + current_rows))
            current_rows = [row]
            continue
        current_rows = candidate_rows
    if current_rows:
        chunks.append("\n".join(prefix + current_rows))
    return chunks or [text]


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
