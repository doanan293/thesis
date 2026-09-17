import math

import pytest

from pharma_lab.evaluation.retrieval_metrics import is_hit, score_ranked_payloads

LEAFLET_CHUNK = "leaflet:thuoc-tri-tieu-duong:acarbose-a:chunk-001"


def test_chunk_exact_accepts_any_declared_expected_chunk_id():
    row = {
        "retrieval_granularity": "chunk_exact",
        "expected_section_id": "drug:x:ten-thuong-mai",
        "expected_chunk_id": "drug:x:ten-thuong-mai:chunk-001",
        "expected_chunk_ids": [
            "drug:x:ten-thuong-mai:chunk-001",
            "general:index:chunk-007",
        ],
    }

    assert is_hit(
        {"section_id": "general:index", "chunk_id": "general:index:chunk-007"},
        row,
    )


def test_chunk_exact_keeps_legacy_single_expected_chunk_behavior():
    row = {
        "retrieval_granularity": "chunk_exact",
        "expected_section_id": "drug:x:ten-thuong-mai",
        "expected_chunk_id": "drug:x:ten-thuong-mai:chunk-001",
    }

    assert is_hit(
        {
            "section_id": "drug:x:ten-thuong-mai",
            "chunk_id": "drug:x:ten-thuong-mai:chunk-001",
        },
        row,
    )


def _payload(chunk_id: str) -> dict:
    section_id, _, _ordinal = chunk_id.rpartition(":")
    return {"chunk_id": chunk_id, "section_id": section_id}


def test_an_accepted_leaflet_chunk_counts_as_a_hit_for_its_section():
    row = {
        "answer_mode": "single",
        "retrieval_granularity": "section",
        "expected_section_id": "drug:acarbose:lieu-luong-va-cach-dung",
    }
    payloads = [
        _payload(LEAFLET_CHUNK),
        _payload("drug:acarbose:lieu-luong-va-cach-dung:chunk-001"),
    ]
    judgments = {"drug:acarbose:lieu-luong-va-cach-dung": frozenset({LEAFLET_CHUNK})}

    judged = score_ranked_payloads(payloads, row, top_k=3, judgments=judgments)
    unjudged = score_ranked_payloads(payloads, row, top_k=3)

    assert (judged["hit@3"], judged["mrr"]) == (1, 1.0)
    assert (unjudged["hit@3"], unjudged["mrr"]) == (1, 0.5)


def test_an_accepted_leaflet_chunk_satisfies_its_intent_of_a_multi_query():
    row = {
        "answer_mode": "multi_required",
        "retrieval_granularity": "multi_section",
        "expected_section_id": "drug:acarbose:chi-dinh",
        "expected_section_ids": [
            "drug:acarbose:chi-dinh",
            "drug:acarbose:lieu-luong-va-cach-dung",
        ],
    }
    payloads = [
        _payload(LEAFLET_CHUNK),
        _payload("drug:acarbose:lieu-luong-va-cach-dung:chunk-001"),
        _payload("drug:metformin:chi-dinh:chunk-001"),
    ]
    judgments = {"drug:acarbose:chi-dinh": frozenset({LEAFLET_CHUNK})}

    judged = score_ranked_payloads(payloads, row, top_k=3, judgments=judgments)
    unjudged = score_ranked_payloads(payloads, row, top_k=3)

    assert judged["multi_section_recall@3"] == 1.0
    assert judged["multi_all_hit@3"] == 1
    assert judged["mrr"] == 1.0
    assert unjudged["multi_section_recall@3"] == 0.5
    assert unjudged["multi_all_hit@3"] == 0
    assert unjudged["mrr"] == 0.5


def _section_row(section: str = "drug:a:lieu") -> dict:
    return {
        "answer_mode": "single",
        "retrieval_granularity": "section",
        "expected_section_id": section,
        "expected_section_ids": [section],
    }


def _ranked(*sections: str) -> list[dict]:
    return [
        {"chunk_id": f"{section}:chunk-{index:03d}", "section_id": section}
        for index, section in enumerate(sections, start=1)
    ]


def test_ndcg_and_mrr_at_10_for_a_single_answer_query():
    payloads = _ranked("x", "y", "drug:a:lieu", "drug:a:lieu", *["z"] * 10)

    scores = score_ranked_payloads(payloads, _section_row(), top_k=30)

    # First hit at rank 3; the second chunk of the same section adds no gain.
    assert scores["ndcg@10"] == pytest.approx(1 / math.log2(4))
    assert scores["mrr@10"] == pytest.approx(1 / 3)
    assert scores["mrr"] == pytest.approx(1 / 3)


def test_hits_after_rank_10_count_for_mrr_at_30_only():
    payloads = _ranked(*["z"] * 11, "drug:a:lieu")

    scores = score_ranked_payloads(payloads, _section_row(), top_k=30)

    assert scores["ndcg@10"] == 0.0
    assert scores["mrr@10"] == 0.0
    assert scores["mrr"] == pytest.approx(1 / 12)


def test_ndcg_of_a_multi_part_query_needs_every_section():
    row = {
        "answer_mode": "multi_required",
        "retrieval_granularity": "multi_section",
        "expected_section_ids": ["drug:a:lieu", "drug:a:chong-chi-dinh"],
    }
    ideal = 1 + 1 / math.log2(3)

    one = score_ranked_payloads(_ranked("drug:a:lieu", "x"), row, top_k=10)
    both = score_ranked_payloads(
        _ranked("drug:a:lieu", "drug:a:chong-chi-dinh"), row, top_k=10
    )

    assert one["ndcg@10"] == pytest.approx(1 / ideal)
    assert both["ndcg@10"] == pytest.approx(1.0)


def test_an_accepted_chunk_satisfies_its_multi_part_unit():
    row = {
        "answer_mode": "multi_required",
        "retrieval_granularity": "multi_section",
        "expected_section_ids": ["drug:a:lieu", "drug:a:chong-chi-dinh"],
    }
    judgments = {"drug:a:chong-chi-dinh": frozenset({LEAFLET_CHUNK})}
    payloads = [_payload(LEAFLET_CHUNK), *_ranked("drug:a:lieu")]

    scores = score_ranked_payloads(payloads, row, top_k=10, judgments=judgments)

    assert scores["ndcg@10"] == pytest.approx(1.0)


def test_ranking_scores_need_a_cutoff_of_at_least_10():
    assert "ndcg@10" not in score_ranked_payloads(_ranked("x"), _section_row(), top_k=5)
