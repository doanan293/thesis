from pathlib import Path

import pytest
from tests.data_archive.archive_project import FILES, make_project

from seed_pipeline.data_archive.manifest import ArchiveError, ArchiveManifest
from seed_pipeline.data_archive.pack import git_head, list_archive_files, pack_archive
from seed_pipeline.data_archive.unpack import UnpackResult, unpack_archive


def packed(tmp_path: Path) -> tuple[Path, ArchiveManifest]:
    root = make_project(tmp_path / "project")
    archive = tmp_path / "archive"
    manifest = pack_archive(
        root / "data",
        list_archive_files(root),
        archive,
        part_size=1024,
        created_at="2026-09-14T00:00:00+00:00",
        git_commit=git_head(root),
    )
    return archive, manifest


def test_unpack_restores_every_file_and_skips_identical_ones(tmp_path: Path) -> None:
    archive, manifest = packed(tmp_path)
    target = tmp_path / "restore" / "data"

    first = unpack_archive(archive, manifest, target, force=False)
    second = unpack_archive(archive, manifest, target, force=False)

    assert first == UnpackResult(written=2, unchanged=0)
    assert second == UnpackResult(written=0, unchanged=2)
    for relative, payload in FILES.items():
        assert (target / relative).read_bytes() == payload


def test_a_corrupt_part_is_rejected_before_writing(tmp_path: Path) -> None:
    archive, manifest = packed(tmp_path)
    broken = archive / manifest.parts[1].name
    data = bytearray(broken.read_bytes())
    data[0] ^= 0xFF
    broken.write_bytes(bytes(data))
    target = tmp_path / "restore" / "data"

    with pytest.raises(ArchiveError, match=manifest.parts[1].name):
        unpack_archive(archive, manifest, target, force=False)
    assert not target.exists()


def test_a_different_existing_file_stops_unless_forced(tmp_path: Path) -> None:
    archive, manifest = packed(tmp_path)
    target = tmp_path / "restore" / "data"
    changed = target / "cache" / "nested" / "scores.bin"
    changed.parent.mkdir(parents=True)
    changed.write_bytes(b"local edit")

    with pytest.raises(ArchiveError, match="--force"):
        unpack_archive(archive, manifest, target, force=False)
    assert changed.read_bytes() == b"local edit"

    result = unpack_archive(archive, manifest, target, force=True)

    assert result == UnpackResult(written=2, unchanged=0)
    assert changed.read_bytes() == FILES["cache/nested/scores.bin"]
