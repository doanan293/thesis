import pytest

from corpus_pipeline.corpus.metadata.qdrant_payload_contract import (
    compact_validated_runtime_payload,
    validate_runtime_payload,
)


def rich_unified_chunk() -> dict:
    return {
        "chunk_id": "section-1:chunk-001",
        "section_id": "section-1",
        "chunk_index": 1,
        "hydrate_strategy": "full_section",
        "source": "Dược thư",
        "title": "Thuốc A",
        "section": "Liều dùng",
        "start_page": 1,
        "end_page": 1,
        "context_header": "Thuốc A > Liều dùng",
        "chunk_text": "Nội dung",
        "embedding_text": "Thuốc A\n\nNội dung",
        "chunk_content_type": "prose",
        "chunk_role": "prose",
        "context_path": ["Thuốc A", "Liều dùng"],
        "section_char_count": 8,
        "source_block_id": "block-1",
        "warnings": [],
    }


def test_compact_validated_runtime_payload_drops_unified_metadata():
    payload = compact_validated_runtime_payload(rich_unified_chunk())

    assert payload["chunk_id"] == "section-1:chunk-001"
    assert "chunk_content_type" not in payload
    assert "chunk_role" not in payload
    assert "context_path" not in payload
    assert "section_char_count" not in payload
    assert "source_block_id" not in payload
    assert "warnings" not in payload


def test_validate_runtime_payload_remains_strict_for_uncompacted_input():
    with pytest.raises(ValueError, match="contains unexpected field"):
        validate_runtime_payload(rich_unified_chunk())
