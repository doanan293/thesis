"""Hydrate policy for a section revision (spec C §7.2).

Public API: pharma-lab imports ``hydrate_strategy_for``.
"""

from pharma_agent.domain.corpus.bundle import RetrievalMode, SectionRecord
from pharma_agent.domain.retrieval.models import HydrateStrategy

FULL_SECTION_MAX_CHARS = 16000


def section_char_count(section: SectionRecord) -> int:
    return len("\n\n".join(block.markdown for block in section.blocks))


def hydrate_strategy_for(section: SectionRecord) -> HydrateStrategy:
    if section.retrieval is RetrievalMode.INDEX_ONLY:
        return HydrateStrategy.SEARCH_ONLY
    if section_char_count(section) > FULL_SECTION_MAX_CHARS:
        return HydrateStrategy.CHUNK_WINDOW
    return HydrateStrategy.FULL_SECTION
