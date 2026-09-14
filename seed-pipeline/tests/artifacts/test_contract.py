import json
from pathlib import Path

import pytest

from seed_pipeline.artifacts.contract import (
    CONTRACT_SCHEMA_VERSION,
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
    (final_dir / "validation_report.json").write_text(
        '{"ok": true}\n', encoding="utf-8"
    )
    manifest = build_manifest(
        final_dir,
        build_id="build",
        source_pdf_sha256="pdf",
        leaflet_source={"manifest_sha256": "leaflets", "file_count": 2},
        curated_input_digests={"glossary": "g"},
        config_digest="config",
    )
    (final_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return final_dir


def test_manifest_tracks_sections_and_blocks(tmp_path: Path) -> None:
    manifest = validate_contract_directory(_published(tmp_path))

    assert manifest["schema_version"] == CONTRACT_SCHEMA_VERSION == "rag-final-v3"
    assert (manifest["section_count"], manifest["block_count"]) == (1, 1)
    assert "chunk_count" not in manifest
    assert manifest["leaflet_source"] == {
        "manifest_sha256": "leaflets",
        "file_count": 2,
    }
    assert "snapshot_id" not in manifest
    assert set(manifest["files"]) == {
        "sections.jsonl",
        "blocks.jsonl",
        "validation_report.json",
    }


def test_contract_requires_blocks_and_rejects_leftover_chunks(tmp_path: Path) -> None:
    final_dir = _published(tmp_path)
    (final_dir / "chunks.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ContractError, match="must contain exactly"):
        validate_contract_directory(final_dir)
    (final_dir / "chunks.jsonl").unlink()
    (final_dir / "blocks.jsonl").unlink()
    with pytest.raises(ContractError, match="must contain exactly"):
        validate_contract_directory(final_dir)


def test_contract_rejects_blocks_of_unknown_sections(tmp_path: Path) -> None:
    final_dir = _published(tmp_path, block_section="drug:missing:section")

    with pytest.raises(ContractError, match="unknown sections"):
        validate_contract_directory(final_dir)
