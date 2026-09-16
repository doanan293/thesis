"""Plain text of every section and document title in a knowledge bundle."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from pharma_agent.domain.corpus.bundle import read_bundle


def normalize_space(text: str) -> str:
    return " ".join(text.split())


@dataclass(frozen=True)
class CorpusText:
    # section key -> section markdown (blocks joined by blank lines)
    sections: dict[str, str]
    # section key -> document key
    section_documents: dict[str, str]
    # document key -> title
    titles: dict[str, str]

    def normalized_section(self, section_id: str) -> str | None:
        text = self.sections.get(section_id)
        return None if text is None else normalize_space(text)

    @cached_property
    def _folded(self) -> str:
        texts = [*self.sections.values(), *self.titles.values()]
        return "\n".join(normalize_space(text).casefold() for text in texts)

    def contains(self, term: str) -> bool:
        """Whether `term` occurs, case-insensitively, in any section or title."""
        needle = normalize_space(term).casefold()
        return bool(needle) and needle in self._folded


def load_corpus_text(bundle_dir: Path) -> CorpusText:
    bundle = read_bundle(bundle_dir)
    return CorpusText(
        sections={
            section.key: "\n\n".join(block.markdown for block in section.blocks)
            for section in bundle.sections
        },
        section_documents={
            section.key: section.document_key for section in bundle.sections
        },
        titles={document.key: document.title for document in bundle.documents},
    )
