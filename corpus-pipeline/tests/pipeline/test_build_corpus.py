import json
from pathlib import Path

import pytest

from artifact_lifecycle.contract import build_manifest
from artifact_lifecycle.paths import ArtifactPaths
from pipeline.build_corpus import BuildConfig, BuildHooks, run_build


def _config(tmp_path: Path) -> BuildConfig:
    inputs = {
        "pdf_path": tmp_path / "source.pdf",
        "snapshot_archive": tmp_path / "snapshot.tar.zst",
        "snapshot_manifest": tmp_path / "snapshot.manifest.json",
        "curated_tables_path": tmp_path / "docling_tables.jsonl",
        "table_overrides_path": tmp_path / "table_duplicate_overrides.json",
        "mappings_path": tmp_path / "colloquial_mappings.json",
        "glossary_path": tmp_path / "term_glossary.json",
    }
    inputs["pdf_path"].write_bytes(b"%PDF-test")
    inputs["snapshot_archive"].write_bytes(b"snapshot")
    for path in inputs.values():
        if not path.exists():
            path.write_text("{}\n", encoding="utf-8")
    return BuildConfig(
        **inputs,
        work_root=tmp_path / ".work",
        final_dir=tmp_path / "processed" / "rag-final",
        max_chars=3000,
    )


def _write_candidate(_config: BuildConfig, paths: ArtifactPaths) -> None:
    final = paths.candidate_final_dir
    (final / "sections.jsonl").write_text('{"id":"s1","text":"section"}\n')
    (final / "chunks.jsonl").write_text(
        '{"chunk_id":"c1","section_id":"s1","chunk_text":"chunk",'
        '"embedding_text":"chunk"}\n'
    )
    (final / "validation_report.json").write_text('{"ok":true}\n')
    manifest = build_manifest(
        final,
        build_id="test-build",
        source_pdf_sha256="pdf",
        snapshot_id="snapshot",
        snapshot_sha256="archive",
        curated_input_digests={},
        config_digest="config",
    )
    (final / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_success_publishes_and_removes_workspace(tmp_path: Path) -> None:
    config = _config(tmp_path)
    result = run_build(config, hooks=BuildHooks(build_candidate=_write_candidate))

    assert result.final_dir == config.final_dir
    assert sorted(path.name for path in result.final_dir.iterdir()) == [
        "chunks.jsonl",
        "manifest.json",
        "sections.jsonl",
        "validation_report.json",
    ]
    assert not any(config.work_root.glob("build-*"))


def test_failure_preserves_final_and_retains_latest_workspace(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.final_dir.mkdir(parents=True)
    (config.final_dir / "marker").write_text("old", encoding="utf-8")

    def fail(_config: BuildConfig, _paths: ArtifactPaths) -> None:
        raise RuntimeError("canonical")

    with pytest.raises(RuntimeError, match="canonical"):
        run_build(config, hooks=BuildHooks(build_candidate=fail))

    assert (config.final_dir / "marker").read_text() == "old"
    assert (config.work_root / "failed" / "latest").is_dir()
