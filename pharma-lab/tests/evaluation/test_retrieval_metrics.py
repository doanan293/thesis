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
