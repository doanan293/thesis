import pytest

from corpus_pipeline.evaluation.section_eval_schema import (
    validate_query_rows,
)


def test_validate_query_rows_accepts_valid_jsonl_row():
    valid_row = {
        "query_id": "q1",
        "query": "Sample query",
        "eval_group": "formulary",
        "source_family": "drug_formulary",
        "query_form": "natural",
        "intent_category": "indication",
        "source_subcategory": "formulary_indication",
        "expected_section_id": "sec1",
        "expected_section_ids": ["sec1"],
        "query_intent_count": 1,
        "answer_mode": "single",
        "expected_title": "Title",
        "expected_chunk_id": "chk1",
        "expected_chunk_index": 0,
        "expected_chunk_role": "primary",
        "retrieval_granularity": "section",
        "difficulty": "easy",
        "eval_tags": ["tag1"],
        "notes": "",
    }
    validate_query_rows([valid_row])  # should not raise


def test_validate_query_rows_rejects_string_instead_of_list_for_expected_section_ids():
    invalid_row = {
        "query_id": "q1",
        "query": "Sample query",
        "eval_group": "formulary",
        "source_family": "drug_formulary",
        "query_form": "natural",
        "intent_category": "indication",
        "source_subcategory": "formulary_indication",
        "expected_section_id": "sec1",
        "expected_section_ids": "sec1;sec2",  # String instead of list
        "query_intent_count": 1,
        "answer_mode": "single",
        "expected_title": "Title",
        "expected_chunk_id": "chk1",
        "expected_chunk_index": 0,
        "expected_chunk_role": "primary",
        "retrieval_granularity": "section",
        "difficulty": "easy",
        "eval_tags": ["tag1"],
        "notes": "",
    }
    with pytest.raises(ValueError, match="expected_section_ids must be a list"):
        validate_query_rows([invalid_row])


def test_validate_query_rows_rejects_non_list_eval_tags():
    invalid_row = {
        "query_id": "q1",
        "query": "Sample query",
        "eval_group": "formulary",
        "source_family": "drug_formulary",
        "query_form": "natural",
        "intent_category": "indication",
        "source_subcategory": "formulary_indication",
        "expected_section_id": "sec1",
        "expected_section_ids": ["sec1"],
        "query_intent_count": 1,
        "answer_mode": "single",
        "expected_title": "Title",
        "expected_chunk_id": "chk1",
        "expected_chunk_index": 0,
        "expected_chunk_role": "primary",
        "retrieval_granularity": "section",
        "difficulty": "easy",
        "eval_tags": "tag1|tag2",  # String instead of list
        "notes": "",
    }
    with pytest.raises(ValueError, match="eval_tags must be a list"):
        validate_query_rows([invalid_row])


def test_validate_query_rows_rejects_non_integer_query_intent_count():
    invalid_row = {
        "query_id": "q1",
        "query": "Sample query",
        "eval_group": "formulary",
        "source_family": "drug_formulary",
        "query_form": "natural",
        "intent_category": "indication",
        "source_subcategory": "formulary_indication",
        "expected_section_id": "sec1",
        "expected_section_ids": ["sec1"],
        "query_intent_count": "1",  # String instead of int
        "answer_mode": "single",
        "expected_title": "Title",
        "expected_chunk_id": "chk1",
        "expected_chunk_index": 0,
        "expected_chunk_role": "primary",
        "retrieval_granularity": "section",
        "difficulty": "easy",
        "eval_tags": ["tag1"],
        "notes": "",
    }
    with pytest.raises(ValueError, match="query_intent_count must be an integer"):
        validate_query_rows([invalid_row])


def test_validate_query_rows_rejects_non_integer_expected_chunk_index():
    invalid_row = {
        "query_id": "q1",
        "query": "Sample query",
        "eval_group": "formulary",
        "source_family": "drug_formulary",
        "query_form": "natural",
        "intent_category": "indication",
        "source_subcategory": "formulary_indication",
        "expected_section_id": "sec1",
        "expected_section_ids": ["sec1"],
        "query_intent_count": 1,
        "answer_mode": "single",
        "expected_title": "Title",
        "expected_chunk_id": "chk1",
        "expected_chunk_index": "0",  # String instead of int/None
        "expected_chunk_role": "primary",
        "retrieval_granularity": "section",
        "difficulty": "easy",
        "eval_tags": ["tag1"],
        "notes": "",
    }
    with pytest.raises(
        ValueError, match="expected_chunk_index must be an integer or None"
    ):
        validate_query_rows([invalid_row])


def test_validate_query_rows_validates_intent_count_matching_expected_ids_len():
    invalid_row = {
        "query_id": "q1",
        "query": "Sample query",
        "eval_group": "formulary",
        "source_family": "drug_formulary",
        "query_form": "natural",
        "intent_category": "indication",
        "source_subcategory": "formulary_indication",
        "expected_section_id": "sec1",
        "expected_section_ids": ["sec1"],
        "query_intent_count": 2,  # Mismatch
        "answer_mode": "single",
        "expected_title": "Title",
        "expected_chunk_id": "chk1",
        "expected_chunk_index": 0,
        "expected_chunk_role": "primary",
        "retrieval_granularity": "section",
        "difficulty": "easy",
        "eval_tags": ["tag1"],
        "notes": "",
    }
    with pytest.raises(
        ValueError, match="query_intent_count 2 does not match 1 expected sections"
    ):
        validate_query_rows([invalid_row])


def test_retrieval_affinity_tags_and_validation():
    from corpus_pipeline.evaluation.section_eval_schema import (
        ALLOWED_QUERY_FORMS,
        RETRIEVAL_AFFINITY_TAGS,
        validate_query_rows,
    )

    assert "natural" in ALLOWED_QUERY_FORMS
    assert {
        "hybrid_favored",
        "bm25_favored",
        "dense_favored",
    } == RETRIEVAL_AFFINITY_TAGS

    sample_row = {
        "query_id": "test_001",
        "query": "Tác dụng phụ của Klacid MR 500mg trên hệ tiêu hóa",
        "eval_group": "ankhang",
        "source_family": "ankhang_brand",
        "query_form": "natural",
        "intent_category": "adr",
        "source_subcategory": "ankhang_adr",
        "expected_section_id": "brand:ankhang:1",
        "expected_section_ids": ["brand:ankhang:1"],
        "query_intent_count": 1,
        "answer_mode": "single",
        "expected_title": "Klacid MR 500mg",
        "expected_chunk_id": "",
        "expected_chunk_index": None,
        "expected_chunk_role": "prose",
        "retrieval_granularity": "section",
        "difficulty": "medium",
        "eval_tags": [
            "ankhang",
            "ankhang_brand",
            "natural",
            "adr",
            "ankhang_adr",
            "hybrid_favored",
        ],
        "notes": "test hybrid row",
    }
    validate_query_rows([sample_row])
