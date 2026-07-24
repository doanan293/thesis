import json
from pathlib import Path

from validation.validate_final_rag import (
    combine_validation_reports,
    validate_unified_chunks,
)


def test_validate_unified_chunks_accepts_new_schema(tmp_path: Path) -> None:
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text(
        json.dumps(
            {
                "chunk_id": "c1",
                "section_id": "s1",
                "chunk_index": 0,
                "chunk_text": "Nội dung",
                "embedding_text": "Thuốc A\n\nNội dung",
                "hydrate_strategy": "full_section",
                "source": "Dược thư",
                "title": "Thuốc A",
                "section": "Liều dùng",
                "start_page": 1,
                "end_page": 1,
                "context_header": "Thuốc A",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    report = validate_unified_chunks(chunks)
    assert report.ok


def test_combined_report_is_blocking_when_deep_audit_fails() -> None:
    report = combine_validation_reports(
        source_report={"ok": True, "errors": [], "warnings": [], "findings": []},
        deep_report={
            "ok": False,
            "errors": ["missing table"],
            "warnings": [],
            "findings": [],
        },
        unified_report={"ok": True, "errors": [], "warnings": [], "findings": []},
    )
    assert report["ok"] is False
    assert report["errors"] == ["missing table"]
