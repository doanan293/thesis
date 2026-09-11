from pharma_agent.domain.retrieval.evidence import EvidenceSet
from pharma_agent.domain.retrieval.models import (
    Chunk,
    ColloquialMapping,
    Hit,
    HydrateStrategy,
    Query,
    QueryOrigin,
    RetrievedItem,
    TermAnnotation,
)


def make_hit(
    chunk_id: str,
    *,
    section_id: str = "sec-1",
    chunk_index: int = 0,
    strategy: HydrateStrategy = HydrateStrategy.CHUNK_WINDOW,
    fusion: float = 0.5,
    rerank: float | None = None,
    text: str = "paracetamol 500 mg",
    table_id: str = "",
) -> Hit:
    return Hit(
        chunk_id=chunk_id,
        section_id=section_id,
        chunk_index=chunk_index,
        hydrate_strategy=strategy,
        source="duoc_thu",
        title="Paracetamol",
        section="Liều dùng",
        start_page=10,
        end_page=11,
        context_header="Paracetamol > Liều dùng",
        chunk_text=text,
        embedding_text=f"Paracetamol > Liều dùng\n\n{text}",
        table_id=table_id,
        fusion_score=fusion,
        rerank_score=rerank,
        matched_queries=["q1"],
    )


def test_query_normalizes_whitespace_and_case() -> None:
    assert (
        Query(text="  Liều   PARACETAMOL ", origin=QueryOrigin.INITIAL).normalized
        == "liều paracetamol"
    )


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
    long_chunks = [
        Chunk(chunk_id=f"c{i}", section_id="sec-1", chunk_index=i, text="x" * 100)
        for i in range(5)
    ]
    hit = make_hit(
        "c2",
        chunk_index=2,
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
    assert [c.chunk_index for c in window[0].chunks] == [1, 2, 3]

    search_only = evidence.pack(max_chars=80)
    assert search_only[0].applied_strategy is HydrateStrategy.SEARCH_ONLY
    assert search_only[0].text() == "y" * 50

    assert evidence.pack(max_chars=10) == []


def test_context_view_numbers_sources_and_puts_tables_first() -> None:
    table = Chunk(
        chunk_id="t1",
        section_id="sec-1",
        chunk_index=3,
        text="| liều | mg |",
        content_type="table",
        table_id="tbl-1",
    )
    body = Chunk(
        chunk_id="c1", section_id="sec-1", chunk_index=1, text="Người lớn 500 mg."
    )
    hit = make_hit(
        "c1",
        chunk_index=1,
        strategy=HydrateStrategy.FULL_SECTION,
        rerank=0.8,
        table_id="tbl-1",
    )
    evidence = EvidenceSet()
    evidence.merge([RetrievedItem(hit=hit, chunks=[body, table])])
    packed = evidence.pack(max_chars=1000)
    text, numbered = evidence.context_view(packed)
    assert numbered[0][0] == 1 and numbered[0][1].ref == "E1"
    assert text.startswith("[1] Paracetamol > Liều dùng (trang 10-11)")
    assert text.index("| liều | mg |") < text.index("Người lớn 500 mg.")


def test_summary_view_lists_refs_snippets_and_hints() -> None:
    hit = make_hit("c1", rerank=0.7).model_copy(
        update={
            "colloquial_mapping": ColloquialMapping(
                key="paracetamol", product_names=["Panadol"]
            )
        }
    )
    evidence = EvidenceSet()
    evidence.merge([RetrievedItem(hit=hit)])
    view = evidence.summary_view(snippet_chars=10)
    assert "E1 | Paracetamol > Liều dùng | trang 10-11" in view
    assert "paracetam…" in view  # 9 chars + ellipsis = snippet_chars
    assert "gợi ý thuật ngữ: Panadol" in view
