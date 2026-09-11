from pharma_agent.domain.conversation.models import Citation
from pharma_agent.domain.retrieval.audit import audit_from_run
from pharma_agent.domain.retrieval.models import Query, QueryOrigin, RetrievedItem
from pharma_agent.domain.retrieval.service import SearchResult
from tests.domain.factories import NOW, make_hit, make_run


def result_for(query: str, *hits: tuple[str, float]) -> SearchResult:
    return SearchResult(
        items=[
            RetrievedItem(
                hit=make_hit(chunk_id, rerank=score).model_copy(
                    update={"matched_queries": [query]}
                )
            )
            for chunk_id, score in hits
        ]
    )


def test_one_record_per_query_per_round_with_ranked_hits_and_citations() -> None:
    run = make_run()
    q1 = Query(text="paracetamol liều", origin=QueryOrigin.INITIAL)
    run.record_search([q1], result_for(q1.text, ("c1", 0.4), ("c2", 0.9)), now=NOW)
    q2 = Query(text="paracetamol quá liều", origin=QueryOrigin.REFINED)
    q3 = Query(text="paracetamol trẻ em", origin=QueryOrigin.REFINED)
    run.record_search([q2, q3], result_for(q2.text, ("c3", 0.7)), now=NOW)

    citations = [
        Citation(
            index=1,
            chunk_id="c2",
            section_id="sec-1",
            title="t",
            section="s",
            start_page=1,
            end_page=1,
        )
    ]
    records = audit_from_run(run, citations, snippet_chars=5)

    assert [(r.round, r.query_text) for r in records] == [
        (1, "paracetamol liều"),
        (2, "paracetamol quá liều"),
        (2, "paracetamol trẻ em"),
    ]
    first = records[0]
    assert [(h.rank, h.chunk_id, h.cited) for h in first.hits] == [
        (1, "c2", True),
        (2, "c1", False),
    ]
    assert first.hits[0].rerank_score == 0.9 and first.hits[0].snippet == "parac"
    assert [h.chunk_id for h in records[1].hits] == ["c3"]
    assert records[2].hits == []


def test_failed_search_still_produces_a_record() -> None:
    run = make_run()
    q1 = Query(text="x", origin=QueryOrigin.INITIAL)
    run.record_search([q1], SearchResult(error="qdrant down"), now=NOW)
    records = audit_from_run(run, [])
    assert len(records) == 1 and records[0].hits == []
