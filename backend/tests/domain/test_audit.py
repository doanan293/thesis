from pharma_agent.domain.retrieval.audit import audit_from_run
from pharma_agent.domain.retrieval.models import Query, QueryOrigin, RetrievedItem
from pharma_agent.domain.retrieval.service import SearchResult
from pharma_agent.domain.shared.text import make_snippet
from tests.domain.factories import (
    COLLECTION_ID,
    NOW,
    RELEASE_ID,
    SECTION_KEY,
    chunk_uuid,
    make_citation,
    make_hit,
    make_run,
)


def result_for(query: str, *hits: tuple[str, float]) -> SearchResult:
    return SearchResult(
        items=[
            RetrievedItem(
                hit=make_hit(label, rerank=score).model_copy(
                    update={"matched_queries": [query]}
                )
            )
            for label, score in hits
        ]
    )


def test_one_record_per_query_per_round_with_ranked_hits_and_citations() -> None:
    run = make_run()
    q1 = Query(text="paracetamol liều", origin=QueryOrigin.INITIAL)
    run.record_search([q1], result_for(q1.text, ("c1", 0.4), ("c2", 0.9)), now=NOW)
    q2 = Query(text="paracetamol quá liều", origin=QueryOrigin.REFINED)
    q3 = Query(text="paracetamol trẻ em", origin=QueryOrigin.REFINED)
    run.record_search([q2, q3], result_for(q2.text, ("c3", 0.7)), now=NOW)

    records = audit_from_run(run, [make_citation("c2")], snippet_chars=12)

    assert [(r.round, r.query_text) for r in records] == [
        (1, "paracetamol liều"),
        (2, "paracetamol quá liều"),
        (2, "paracetamol trẻ em"),
    ]
    first = records[0]
    assert first.release_ids == {str(COLLECTION_ID): str(RELEASE_ID)}
    assert [(h.rank, h.chunk_version_id, h.cited) for h in first.hits] == [
        (1, chunk_uuid("c2"), True),
        (2, chunk_uuid("c1"), False),
    ]
    top = first.hits[0]
    assert (top.section_key, top.table_key, top.rerank_score) == (
        SECTION_KEY,
        None,
        0.9,
    )
    assert top.snippet == make_snippet("paracetamol 500 mg", 12)
    assert [h.chunk_version_id for h in records[1].hits] == [chunk_uuid("c3")]
    assert records[2].hits == [] and records[2].release_ids == {}


def test_failed_search_still_produces_a_record() -> None:
    run = make_run()
    q1 = Query(text="x", origin=QueryOrigin.INITIAL)
    run.record_search([q1], SearchResult(error="qdrant down"), now=NOW)
    records = audit_from_run(run, [])
    assert len(records) == 1 and records[0].hits == []
