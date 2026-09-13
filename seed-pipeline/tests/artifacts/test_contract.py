import json
from pathlib import Path

import pytest

from seed_pipeline.artifacts.contract import (
    ContractError,
    build_manifest,
    validate_contract_directory,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _published(tmp_path: Path, *, block_section: str = "drug:a:b") -> Path:
    final_dir = tmp_path / "rag-final"
    final_dir.mkdir()
    _write_jsonl(final_dir / "sections.jsonl", [{"id": "drug:a:b", "text": "Văn bản"}])
    _write_jsonl(
        final_dir / "blocks.jsonl",
        [{"block_id": "block-000001", "section_id": block_section, "text": "Văn bản"}],
    )
    _write_jsonl(
        final_dir / "chunks.jsonl",
        [
            {
                "chunk_id": "drug:a:b:chunk-001",
                "section_id": "drug:a:b",
                "chunk_text": "Văn bản",
                "embedding_text": "Văn bản",
            }
        ],
    )
    (final_dir / "validation_report.json").write_text(
        '{"ok": true}\n', encoding="utf-8"
    )
    manifest = build_manifest(
        final_dir,
        build_id="build",
        source_pdf_sha256="pdf",
        snapshot_id="snapshot",
        snapshot_sha256="archive",
        curated_input_digests={"glossary": "g"},
        config_digest="config",
    )
    (final_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return final_dir


def test_manifest_tracks_blocks_and_contract_validates(tmp_path: Path) -> None:
    final_dir = _published(tmp_path)

    manifest = validate_contract_directory(final_dir)

    assert manifest["block_count"] == 1
    assert set(manifest["files"]) == {
        "sections.jsonl",
        "blocks.jsonl",
        "chunks.jsonl",
        "validation_report.json",
    }


def test_contract_requires_blocks_file(tmp_path: Path) -> None:
    final_dir = _published(tmp_path)
    (final_dir / "blocks.jsonl").unlink()

    with pytest.raises(ContractError, match="must contain exactly"):
        validate_contract_directory(final_dir)


def test_contract_rejects_blocks_of_unknown_sections(tmp_path: Path) -> None:
    final_dir = _published(tmp_path, block_section="drug:missing:section")

    with pytest.raises(ContractError, match="unknown sections"):
        validate_contract_directory(final_dir)
