from pharma_agent.domain.retrieval.evidence import EvidenceSet
from pharma_agent.domain.retrieval.models import (
    ColloquialMapping,
    HydrateStrategy,
    Query,
    QueryOrigin,
    RetrievedItem,
    TermAnnotation,
    page_label,
)
from pharma_agent.domain.shared.text import make_snippet
from tests.domain.factories import chunk_uuid, make_chunk, make_hit


def test_query_normalizes_whitespace_and_case() -> None:
    assert (
        Query(text="  Liều   PARACETAMOL ", origin=QueryOrigin.INITIAL).normalized
        == "liều paracetamol"
    )


def test_page_label_handles_missing_pages() -> None:
    assert page_label(10, 11) == "trang 10-11"
    assert page_label(10, 10) == "trang 10"
    assert page_label(None, 12) == "trang 12"
    assert page_label(None, None) == ""
    assert make_hit("c1", start_page=None, end_page=None).page_label == ""


def test_term_hints_collect_aliases_products_and_annotations() -> None:
    hit = make_hit("c1").model_copy(
        update={
            "colloquial_mapping": ColloquialMapping(
                key="paracetamol",
                aliases=["thuốc hạ sốt"],
                product_names=["Panadol", "Efferalgan"],
            ),
            "term_annotations": [
                TermAnnotation(term="APAP", vi=["acetaminophen"], en=["acetaminophen"])
            ],
        }
    )
    assert hit.term_hints() == [
        "thuốc hạ sốt",
        "Panadol",
        "Efferalgan",
        "APAP",
        "acetaminophen",
    ]


def test_merge_assigns_stable_refs_and_keeps_best_score() -> None:
    evidence = EvidenceSet()
    first = evidence.merge(
        [
            RetrievedItem(hit=make_hit("c1", rerank=0.2)),
            RetrievedItem(hit=make_hit("c2", rerank=0.9)),
        ]
    )
    assert [e.ref for e in first] == ["E2", "E1"]  # sorted by score desc, refs stable
    evidence.supersede_all()
    again = evidence.merge(
        [
            RetrievedItem(
                hit=make_hit("c1", rerank=0.95).model_copy(
                    update={"matched_queries": ["q2"]}
                )
            )
        ]
    )
    assert [e.ref for e in again] == ["E1"]
    e1 = next(e for e in evidence.items if e.ref == "E1")
    assert e1.hit.rerank_score == 0.95
    assert e1.hit.matched_queries == ["q1", "q2"]
    assert e1.superseded is False
    assert next(e for e in evidence.items if e.ref == "E2").superseded is True
    assert [e.ref for e in evidence.active()] == ["E1"]


def test_pack_downgrades_strategy_instead_of_truncating() -> None:
    long_chunks = [make_chunk(f"c{i}", ordinal=i, text="x" * 100) for i in range(5)]
    hit = make_hit(
        "c2",
        ordinal=2,
        strategy=HydrateStrategy.FULL_SECTION,
        rerank=0.9,
        text="y" * 50,
    )
    evidence = EvidenceSet()
    evidence.merge([RetrievedItem(hit=hit, chunks=long_chunks)])

    full = evidence.pack(max_chars=1000)
    assert full[0].applied_strategy is HydrateStrategy.FULL_SECTION
    assert len(full[0].text()) > 400

    window = evidence.pack(max_chars=350)
    assert window[0].applied_strategy is HydrateStrategy.CHUNK_WINDOW
    assert [c.ordinal for c in window[0].chunks] == [1, 2, 3]

    search_only = evidence.pack(max_chars=80)
    assert search_only[0].applied_strategy is HydrateStrategy.SEARCH_ONLY
    assert search_only[0].text() == "y" * 50
    assert search_only[0].text_chunk_version_ids() == [chunk_uuid("c2")]

    assert evidence.pack(max_chars=10) == []


def test_context_view_numbers_sources_and_puts_tables_first() -> None:
    table = make_chunk(
        "t1", ordinal=3, text="| liều | mg |", kind="table", table_key="tbl-1"
    )
    body = make_chunk("c1", ordinal=1, text="Người lớn 500 mg.")
    hit = make_hit("c1", ordinal=1, strategy=HydrateStrategy.FULL_SECTION, rerank=0.8)
    evidence = EvidenceSet()
    evidence.merge([RetrievedItem(hit=hit, chunks=[body, table])])
    packed = evidence.pack(max_chars=1000)
    text, numbered = evidence.context_view(packed)
    assert numbered[0][0] == 1 and numbered[0][1].ref == "E1"
    assert text.startswith("[1] Paracetamol > Liều dùng (trang 10-11)")
    assert text.index("| liều | mg |") < text.index("Người lớn 500 mg.")
    assert numbered[0][1].text_chunk_version_ids() == [
        chunk_uuid("t1"),
        chunk_uuid("c1"),
    ]


def test_context_view_omits_page_label_without_pages() -> None:
    evidence = EvidenceSet()
    evidence.merge(
        [RetrievedItem(hit=make_hit("c1", rerank=0.8, start_page=None, end_page=None))]
    )
    text, _ = evidence.context_view(evidence.pack(max_chars=1000))
    assert text.startswith("[1] Paracetamol > Liều dùng\n")


def test_summary_view_lists_refs_snippets_and_hints() -> None:
    text = "Người lớn uống 500 mg mỗi 4 đến 6 giờ, tối đa 4 g mỗi ngày. " * 10
    hit = make_hit("c1", rerank=0.7, text=text).model_copy(
        update={
            "colloquial_mapping": ColloquialMapping(
                key="paracetamol", product_names=["Panadol"]
            )
        }
    )
    evidence = EvidenceSet()
    evidence.merge([RetrievedItem(hit=hit)])
    view = evidence.summary_view()
    assert view.startswith("E1 | Paracetamol > Liều dùng | trang 10-11 | ")
    assert make_snippet(text, 300) in view
    assert "gợi ý thuật ngữ: Panadol" in view
    assert make_snippet(text, 40) in evidence.summary_view(snippet_chars=40)


def test_rerank_scores_include_superseded_evidence() -> None:
    evidence = EvidenceSet()
    evidence.merge(
        [
            RetrievedItem(hit=make_hit("c1", rerank=0.8)),
            RetrievedItem(hit=make_hit("c2")),
        ]
    )
    evidence.supersede_all()
    assert evidence.rerank_scores() == {chunk_uuid("c1"): 0.8}
