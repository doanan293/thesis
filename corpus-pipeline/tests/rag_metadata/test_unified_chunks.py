from corpus_pipeline.corpus.metadata.build_rag_metadata import build_unified_chunk


def test_unified_chunk_keeps_source_and_runtime_fields() -> None:
    source = {
        "id": "section-1:chunk-001",
        "section_id": "section-1",
        "chunk_index": 0,
        "chunk_role": "table",
        "chunk_content_type": "table",
        "source_block_id": "block-1",
        "table_id": "table-1",
        "text": "| A | B |\n|---|---|\n| 1 | 2 |",
        "source": "Dược thư",
        "title": "Thuốc A",
        "section": "Liều dùng",
        "start_page": 100,
        "end_page": 100,
        "hydrate_strategy": "full_section",
    }
    section = {"id": "section-1", "text": "section text"}

    row = build_unified_chunk(source, section, glossary_entries=[])

    assert row["chunk_id"] == source["id"]
    assert row["chunk_text"] == source["text"]
    assert row["chunk_role"] == "table"
    assert row["chunk_content_type"] == "table"
    assert row["source_block_id"] == "block-1"
    assert row["table_id"] == "table-1"
    assert row["embedding_text"].endswith(source["text"])
    assert "id" not in row
    assert "text" not in row
