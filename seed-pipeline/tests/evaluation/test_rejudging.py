import copy

import pytest

from seed_pipeline.evaluation.artifact_contracts import ArtifactContractError
from seed_pipeline.evaluation.rejudging import (
    rejudge_rows,
    validate_immutable_queries,
)


@pytest.mark.parametrize(
    "changed",
    [
        [{"query_id": "q2", "query": "same"}],
        [{"query_id": "q1", "query": "changed"}],
    ],
)
def test_rejudging_rejects_query_identity_or_text_changes(changed):
    original = [{"query_id": "q1", "query": "same"}]

    with pytest.raises(ArtifactContractError, match="query inputs"):
        validate_immutable_queries(original, changed)


def test_rejudging_rejects_row_reordering():
    original = [
        {"query_id": "q1", "query": "one"},
        {"query_id": "q2", "query": "two"},
    ]

    with pytest.raises(ArtifactContractError, match="query inputs"):
        validate_immutable_queries(original, list(reversed(original)))


@pytest.mark.parametrize("suffix", ["tuong-ky", "thong-tin-qui-che", "noi-dung"])
def test_generic_drug_fact_retargets_known_non_general_section(suffix):
    row = {
        "query_id": f"q-{suffix}",
        "query": "cho tôi thông tin về LEVOFLOXACIN",
        "eval_group": "formulary",
        "eval_tags": ["drug_fact", "formulary"],
        "expected_section_id": f"drug:levofloxacin:{suffix}",
        "expected_section_ids": [f"drug:levofloxacin:{suffix}"],
        "expected_chunk_id": f"drug:levofloxacin:{suffix}:chunk-001",
        "expected_chunk_index": 1,
        "retrieval_granularity": "section",
    }

    result = rejudge_rows(
        [row],
        candidate_rows=[],
        known_section_ids={
            "drug:levofloxacin:tuong-thuong-mai",
            "drug:levofloxacin:thong-tin-chung",
        },
    )

    revised = result.rows[0]
    assert revised["expected_section_id"] == "drug:levofloxacin:thong-tin-chung"
    assert revised["expected_section_ids"] == ["drug:levofloxacin:thong-tin-chung"]
    assert revised["expected_chunk_id"] == (
        "drug:levofloxacin:thong-tin-chung:chunk-001"
    )
    assert revised["query"] == row["query"]
    assert result.summary["drug_fact_changed"] == 1


def test_generic_drug_fact_without_general_section_is_skipped():
    row = {
        "query_id": "q-missing",
        "query": "cho tôi thông tin về X",
        "eval_group": "formulary",
        "eval_tags": ["drug_fact"],
        "expected_section_id": "drug:x:tuong-ky",
        "expected_section_ids": ["drug:x:tuong-ky"],
    }

    result = rejudge_rows([row], [], known_section_ids=set())

    assert result.rows == [row]
    assert result.summary["drug_fact_skipped"] == 1


def test_brand_lookup_appends_only_deterministic_mapping_candidates():
    row = {
        "query_id": "q-brand",
        "query": "Người bệnh đưa tên biệt dược, cần tra hoạt chất tương ứng ở đâu? (SPIRAMYCIN - Tên thương mại)",
        "eval_group": "formulary",
        "intent_category": "brand_lookup",
        "expected_section_id": "drug:spiramycin:ten-thuong-mai",
        "expected_section_ids": ["drug:spiramycin:ten-thuong-mai"],
        "expected_title": "SPIRAMYCIN - Tên thương mại",
        "expected_chunk_id": "drug:spiramycin:ten-thuong-mai:chunk-001",
        "retrieval_granularity": "chunk_exact",
    }
    candidate_rows = [
        {
            "query_id": "q-brand",
            "candidates": [
                {
                    "chunk_id": "general:index:chunk-007",
                    "document_text": ("Spibiotic: biệt dược chứa hoạt chất Spiramycin"),
                },
                {
                    "chunk_id": "general:index:chunk-008",
                    "document_text": "Thông tin khác về Spiramycin",
                },
            ],
        }
    ]

    result = rejudge_rows([row], candidate_rows, known_section_ids=set())

    revised = result.rows[0]
    assert revised["expected_chunk_ids"] == [
        "drug:spiramycin:ten-thuong-mai:chunk-001",
        "general:index:chunk-007",
    ]
    assert "general:index:chunk-008" not in revised["expected_chunk_ids"]
    assert result.summary["brand_lookup_changed"] == 1


def test_rejudging_is_idempotent_for_already_added_brand_chunks():
    row = {
        "query_id": "q-brand",
        "query": "tra hoạt chất SPIRAMYCIN",
        "eval_group": "formulary",
        "intent_category": "brand_lookup",
        "expected_title": "SPIRAMYCIN - Tên thương mại",
        "expected_chunk_id": "drug:spiramycin:ten-thuong-mai:chunk-001",
        "expected_chunk_ids": [
            "drug:spiramycin:ten-thuong-mai:chunk-001",
            "general:index:chunk-007",
        ],
        "retrieval_granularity": "chunk_exact",
    }
    candidates = [
        {
            "query_id": "q-brand",
            "candidates": [
                {
                    "chunk_id": "general:index:chunk-007",
                    "document_text": "Spibiotic: biệt dược chứa hoạt chất Spiramycin",
                }
            ],
        }
    ]

    result = rejudge_rows([row], candidates, set())

    assert result.rows == [row]
    assert result.summary["changed"] == 0
    assert result.summary["unchanged"] == 1


def test_unrelated_rows_and_multi_intent_rows_are_unchanged():
    rows = [
        {
            "query_id": "q-exact",
            "query": "Tìm chống chỉ định",
            "eval_group": "formulary",
            "intent_category": "contraindication",
            "expected_section_id": "drug:x:chong-chi-dinh",
            "expected_chunk_id": "drug:x:chong-chi-dinh:chunk-001",
        },
        {
            "query_id": "q-multi",
            "query": "X chỉ định và liều dùng?",
            "eval_group": "multi_intent",
            "answer_mode": "multi_required",
            "expected_section_ids": ["drug:x:chi-dinh", "drug:x:lieu-luong"],
        },
    ]
    original = copy.deepcopy(rows)

    result = rejudge_rows(
        rows,
        candidate_rows=[{"query_id": "q-exact", "candidates": []}],
        known_section_ids=set(),
    )

    assert result.rows == original
    assert result.summary["unchanged"] == 2
