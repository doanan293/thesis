from pathlib import Path

import pytest

from seed_pipeline.corpus.sources.leaflet_source import (
    LeafletFile,
    LeafletManifest,
    LeafletSourceError,
    read_leaflet_manifest,
    write_leaflet_manifest,
)


def test_manifest_round_trips(tmp_path: Path) -> None:
    manifest = LeafletManifest(
        sitemap_url="https://example.test/sitemap.xml",
        crawled_on="2026-09-14",
        url_list_sha256="a" * 64,
        files=(
            LeafletFile(
                "thuoc-a/one.html", 3, "b" * 64, "https://example.test/thuoc-a/one"
            ),
        ),
    )
    path = tmp_path / "manifest.json"

    write_leaflet_manifest(path, manifest)

    assert read_leaflet_manifest(path) == manifest


def test_manifest_with_another_schema_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text('{"schema_version": "other", "files": []}', encoding="utf-8")

    with pytest.raises(LeafletSourceError, match="leaflet-source-v1"):
        read_leaflet_manifest(path)
