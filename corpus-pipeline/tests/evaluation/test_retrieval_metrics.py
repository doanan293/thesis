from corpus_pipeline.evaluation.retrieval_metrics import is_hit


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
