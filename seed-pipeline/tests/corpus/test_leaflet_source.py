from pathlib import Path

import pytest

from seed_pipeline.corpus.sources.leaflet_source import (
    LeafletFile,
    LeafletManifest,
    LeafletSourceError,
    file_sha256,
    read_leaflet_manifest,
    verify_leaflet_source,
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


def write_tree(root: Path, files: dict[str, bytes]) -> LeafletManifest:
    entries = []
    for relative, payload in sorted(files.items()):
        path = root / "html" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        entries.append(LeafletFile(relative, len(payload), file_sha256(path), None))
    manifest = LeafletManifest(
        "https://example.test/sitemap.xml", "2026-09-14", "c" * 64, tuple(entries)
    )
    write_leaflet_manifest(root / "manifest.json", manifest)
    return manifest


def test_verify_returns_the_manifest_digest_and_count(tmp_path: Path) -> None:
    write_tree(tmp_path, {"thuoc-a/one.html": b"one", "thuoc-b/two.html": b"two"})

    source = verify_leaflet_source(tmp_path)

    assert source.html_dir == tmp_path / "html"
    assert source.file_count == 2
    assert source.manifest_sha256 == file_sha256(tmp_path / "manifest.json")


def test_verify_rejects_a_changed_file(tmp_path: Path) -> None:
    write_tree(tmp_path, {"thuoc-a/one.html": b"one"})
    (tmp_path / "html" / "thuoc-a" / "one.html").write_bytes(b"ONE")

    with pytest.raises(LeafletSourceError, match=r"thuoc-a/one\.html"):
        verify_leaflet_source(tmp_path)


def test_verify_rejects_extra_or_missing_files(tmp_path: Path) -> None:
    write_tree(tmp_path, {"thuoc-a/one.html": b"one"})
    (tmp_path / "html" / "thuoc-a" / "extra.html").write_bytes(b"extra")

    with pytest.raises(LeafletSourceError, match="1 unexpected"):
        verify_leaflet_source(tmp_path)
