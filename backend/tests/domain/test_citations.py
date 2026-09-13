from pharma_agent.domain.agent.citations import (
    SNIPPET_CHARS,
    CitationSanitizer,
    citations_from,
)
from pharma_agent.domain.retrieval.evidence import EvidenceSet
from pharma_agent.domain.retrieval.models import HydrateStrategy, RetrievedItem
from pharma_agent.domain.shared.text import make_snippet
from tests.domain.factories import RELEASE_ID, SOURCE, chunk_uuid, make_hit, make_item


def run_stream(deltas: list[str], valid: set[int]) -> tuple[str, list[int]]:
    sanitizer = CitationSanitizer(valid)
    out = "".join(sanitizer.feed(d) for d in deltas) + sanitizer.flush()
    return out, sanitizer.used


def test_marker_split_across_deltas_is_reassembled() -> None:
    out, used = run_stream(["Liều 500 mg [", "1", "] mỗi 6 giờ [2]."], {1, 2})
    assert out == "Liều 500 mg [1] mỗi 6 giờ [2]."
    assert used == [1, 2]


def test_invalid_marker_is_dropped_and_used_is_deduped() -> None:
    out, used = run_stream(["A [7] B [1] C [1]"], {1})
    assert out == "A  B [1] C [1]"
    assert used == [1]


def test_non_citation_brackets_pass_through() -> None:
    out, _ = run_stream(["x [ghi chú] y [", "abc] z"], {1})
    assert out == "x [ghi chú] y [abc] z"


def test_unfinished_marker_is_flushed_verbatim() -> None:
    sanitizer = CitationSanitizer({1})
    assert sanitizer.feed("cuối [1") == "cuối "
    assert sanitizer.flush() == "[1"


def test_citations_from_numbered_evidence() -> None:
    evidence = EvidenceSet()
    evidence.merge(
        [
            make_item("c1", rerank=0.9),
            RetrievedItem(
                hit=make_hit(
                    "c2", rerank=0.8, table_key="t1", start_page=None, end_page=None
                )
            ),
        ]
    )
    packed = evidence.pack(10_000)
    _, numbered = evidence.context_view(packed)
    citations = citations_from(numbered, used=[2, 1])
    assert [c.index for c in citations] == [2, 1]
    second, first = citations
    assert (
        second.chunk_version_id,
        second.strategy,
        second.block_chunk_version_ids,
        second.start_page,
        second.end_page,
    ) == (chunk_uuid("c2"), HydrateStrategy.SEARCH_ONLY, [chunk_uuid("c2")], None, None)
    assert (first.chunk_version_id, first.release_id, first.strategy) == (
        chunk_uuid("c1"),
        RELEASE_ID,
        HydrateStrategy.CHUNK_WINDOW,
    )
    assert first.block_chunk_version_ids == [chunk_uuid("c1")]
    assert (first.source, first.title, first.section, first.start_page) == (
        SOURCE,
        "Paracetamol",
        "Liều dùng",
        10,
    )
    assert first.snippet == make_snippet("paracetamol 500 mg", SNIPPET_CHARS)
