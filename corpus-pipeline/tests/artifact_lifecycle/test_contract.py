import json
from pathlib import Path

import pytest

from corpus_pipeline.artifacts.contract import (
    ContractError,
    build_manifest,
    require_materialized_pdf,
    require_workspace_capacity,
    validate_contract_directory,
)


def test_pdf_preflight_rejects_git_lfs_pointer(tmp_path: Path) -> None:
    pdf = tmp_path / "source.pdf"
    pdf.write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 38795771\n",
        encoding="utf-8",
    )
    with pytest.raises(ContractError, match="git lfs pull"):
        require_materialized_pdf(pdf)


def test_four_file_contract_validates_checksums(tmp_path: Path) -> None:
    final = tmp_path / "rag-final"
    final.mkdir()
    (final / "sections.jsonl").write_text('{"id":"s1","text":"section"}\n')
    (final / "chunks.jsonl").write_text(
        '{"chunk_id":"c1","section_id":"s1","chunk_text":"chunk",'
        '"embedding_text":"chunk"}\n'
    )
    (final / "validation_report.json").write_text('{"ok":true}\n')
    manifest = build_manifest(
        final,
        build_id="content-abc",
        source_pdf_sha256="pdf-sha",
        snapshot_id="ankhang-2026-07-24-deadbeefcafe",
        snapshot_sha256="snapshot-sha",
        curated_input_digests={"docling_tables.jsonl": "table-sha"},
        config_digest="config-sha",
    )
    (final / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )

    validate_contract_directory(final)


def test_workspace_preflight_rejects_insufficient_space(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    usage = type("Usage", (), {"total": 1000, "used": 950, "free": 50})()
    monkeypatch.setattr(
        "corpus_pipeline.artifacts.contract.shutil.disk_usage", lambda _: usage
    )
    with pytest.raises(ContractError, match="free disk space"):
        require_workspace_capacity(tmp_path, required_bytes=100)
