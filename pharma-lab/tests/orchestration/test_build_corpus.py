from pathlib import Path

import pytest

from pharma_lab.orchestration import build_corpus
from pharma_lab.orchestration.build_corpus import (
    BuildConfig,
    _digest_payload,
    build_id_for,
)


def _config(root: Path) -> BuildConfig:
    names = ("formulary.pdf", "tables.jsonl", "overrides.json", "mappings.json")
    for name in (*names, "glossary.json"):
        (root / name).write_text(name, encoding="utf-8")
    leaflets = root / "leaflets"
    leaflets.mkdir()
    (leaflets / "manifest.json").write_text("{}", encoding="utf-8")
    return BuildConfig(
        pdf_path=root / "formulary.pdf",
        leaflets_dir=leaflets,
        curated_tables_path=root / "tables.jsonl",
        table_overrides_path=root / "overrides.json",
        mappings_path=root / "mappings.json",
        glossary_path=root / "glossary.json",
    )


def test_build_id_changes_with_the_pipeline_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(build_corpus, "PIPELINE_VERSION", "0.1.0")
    old_build_id = build_id_for(_digest_payload(config))

    monkeypatch.setattr(build_corpus, "PIPELINE_VERSION", "0.2.0")

    assert build_id_for(_digest_payload(config)) != old_build_id
