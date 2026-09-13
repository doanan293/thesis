import re
from collections.abc import Iterable, Sequence

from pharma_agent.domain.conversation.models import Citation
from pharma_agent.domain.retrieval.evidence import Evidence
from pharma_agent.domain.shared.text import make_snippet

SNIPPET_CHARS = 200
"""Snippet length for evidence items and citations (spec A §3.3)."""

_COMPLETE = re.compile(r"\[(\d{1,3})\]")
_PARTIAL = re.compile(r"\[\d{0,3}$")


class CitationSanitizer:
    """Stateful filter over streamed text: keeps valid [n] markers, drops invalid ones, never leaks half markers."""

    def __init__(self, valid_indexes: Iterable[int]) -> None:
        self._valid = set(valid_indexes)
        self._buffer = ""
        self.used: list[int] = []

    def feed(self, delta: str) -> str:
        text = self._buffer + delta
        self._buffer = ""
        out: list[str] = []
        pos = 0
        while True:
            start = text.find("[", pos)
            if start == -1:
                out.append(text[pos:])
                break
            out.append(text[pos:start])
            rest = text[start:]
            match = _COMPLETE.match(rest)
            if match:
                index = int(match.group(1))
                if index in self._valid:
                    out.append(f"[{index}]")
                    if index not in self.used:
                        self.used.append(index)
                pos = start + match.end()
                continue
            if _PARTIAL.match(rest):
                self._buffer = rest
                break
            out.append("[")
            pos = start + 1
        return "".join(out)

    def flush(self) -> str:
        leftover, self._buffer = self._buffer, ""
        return leftover


def citations_from(
    numbered: Sequence[tuple[int, Evidence]], used: Sequence[int]
) -> list[Citation]:
    by_index = dict(numbered)
    citations: list[Citation] = []
    for index in used:
        evidence = by_index.get(index)
        if evidence is None:
            continue
        hit = evidence.hit
        citations.append(
            Citation(
                index=index,
                chunk_version_id=hit.chunk_version_id,
                release_id=hit.release_id,
                strategy=evidence.applied_strategy,
                block_chunk_version_ids=evidence.text_chunk_version_ids(),
                source=hit.source,
                title=hit.title,
                section=hit.section,
                start_page=hit.start_page,
                end_page=hit.end_page,
                snippet=make_snippet(hit.chunk_text, SNIPPET_CHARS),
            )
        )
    return citations
