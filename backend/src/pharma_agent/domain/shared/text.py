"""Text helpers shared by the corpus and retrieval domains."""

import re
import unicodedata

ELLIPSIS = "…"
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """NFC, unify line endings, drop trailing spaces per line and outer blank lines."""
    unified = (
        unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    )
    lines = [line.rstrip() for line in unified.split("\n")]
    start = 0
    end = len(lines)
    while start < end and not lines[start]:
        start += 1
    while end > start and not lines[end - 1]:
        end -= 1
    return "\n".join(lines[start:end])


def _is_table_rule(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and "-" in stripped and not stripped.strip("|-: ")


def make_snippet(text: str, max_chars: int) -> str:
    """One-line preview: no table pipes or rules, cut at a word boundary with "…"."""
    if max_chars < 1:
        raise ValueError("max_chars must be at least 1")
    kept = [line for line in text.splitlines() if not _is_table_rule(line)]
    flat = _WHITESPACE_RE.sub(" ", " ".join(kept).replace("|", " ")).strip()
    if len(flat) <= max_chars:
        return flat
    head = flat[: max_chars - 1]
    if flat[max_chars - 1] != " ":
        boundary = head.rfind(" ")
        if boundary > 0:
            head = head[:boundary]
    return head.rstrip() + ELLIPSIS
