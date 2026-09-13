from pharma_agent.domain.corpus.bundle import (
    BlockKind,
    BlockRecord,
    RetrievalMode,
    SectionRecord,
)
from pharma_agent.domain.corpus.hydrate import (
    FULL_SECTION_MAX_CHARS,
    hydrate_strategy_for,
    section_char_count,
)
from pharma_agent.domain.retrieval.models import HydrateStrategy

SENTENCE = "Paracetamol 500 mg. "


def make_section(
    *markdowns: str, retrieval: RetrievalMode = RetrievalMode.DEFAULT
) -> SectionRecord:
    return SectionRecord(
        key="drug:paracetamol:thong-tin-chung",
        document_key="drug:paracetamol",
        heading="Thông tin chung",
        context_path=["Thông tin chung"],
        ordinal=1,
        retrieval=retrieval,
        blocks=[
            BlockRecord(kind=BlockKind.PROSE, markdown=markdown)
            for markdown in markdowns
        ],
    )


def test_section_char_count_joins_blocks_with_a_blank_line() -> None:
    section = make_section("Hạ sốt", "Giảm đau")

    assert section_char_count(section) == len("Hạ sốt\n\nGiảm đau")


def test_full_section_up_to_the_limit() -> None:
    half = (SENTENCE * 400)[:7999]
    section = make_section(half, half)

    assert FULL_SECTION_MAX_CHARS == 16000
    assert section_char_count(section) == 16000
    assert hydrate_strategy_for(section) is HydrateStrategy.FULL_SECTION


def test_chunk_window_above_the_limit() -> None:
    section = make_section(SENTENCE * 400, (SENTENCE * 400)[:7999])

    assert section_char_count(section) == 16001
    assert hydrate_strategy_for(section) is HydrateStrategy.CHUNK_WINDOW


def test_index_only_sections_are_search_only_whatever_their_length() -> None:
    short = make_section(
        "- **Efferalgan**: Paracetamol", retrieval=RetrievalMode.INDEX_ONLY
    )
    long = make_section(SENTENCE * 1000, retrieval=RetrievalMode.INDEX_ONLY)

    assert hydrate_strategy_for(short) is HydrateStrategy.SEARCH_ONLY
    assert hydrate_strategy_for(long) is HydrateStrategy.SEARCH_ONLY
