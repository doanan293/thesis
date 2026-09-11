from pharma_agent.domain.agent.citations import CitationSanitizer, citations_from
from pharma_agent.domain.retrieval.evidence import EvidenceSet
from pharma_agent.domain.retrieval.models import RetrievedItem
from tests.domain.factories import make_hit


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
            RetrievedItem(hit=make_hit("c1", rerank=0.9)),
            RetrievedItem(hit=make_hit("c2", rerank=0.8, table_id="t1")),
        ]
    )
    packed = evidence.pack(10_000)
    _, numbered = evidence.context_view(packed)
    citations = citations_from(numbered, used=[2, 1])
    assert [c.index for c in citations] == [2, 1]
    assert citations[0].chunk_id == "c2" and citations[0].table_id == "t1"
    assert citations[1].title == "Paracetamol" and citations[1].start_page == 10
