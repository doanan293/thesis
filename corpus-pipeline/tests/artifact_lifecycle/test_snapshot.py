import io
import json
import tarfile
from pathlib import Path

import pytest
import zstandard

from artifact_lifecycle.snapshot import (
    SnapshotError,
    extract_snapshot,
    pack_snapshot,
    verify_snapshot,
)


def _source(root: Path) -> Path:
    source = root / "html"
    (source / "tim-mach").mkdir(parents=True)
    (source / "tim-mach" / "a.html").write_text("<html>A</html>", encoding="utf-8")
    (source / "b.html").write_text("<html>B</html>", encoding="utf-8")
    return source


def test_snapshot_is_deterministic_and_round_trips(tmp_path: Path) -> None:
    source = _source(tmp_path)
    urls = tmp_path / "drug_urls.txt"
    urls.write_text(
        "https://www.nhathuocankhang.com/tim-mach/a.html\n"
        "https://www.nhathuocankhang.com/b.html\n",
        encoding="utf-8",
    )
    first = pack_snapshot(source, tmp_path / "first", urls_path=urls)
    second = pack_snapshot(source, tmp_path / "second", urls_path=urls)

    assert first.archive_sha256 == second.archive_sha256
    assert first.snapshot_id == second.snapshot_id
    verify_snapshot(first.archive_path, first.manifest_path)

    restored = tmp_path / "restored"
    extract_snapshot(first.archive_path, first.manifest_path, restored)
    assert (restored / "tim-mach" / "a.html").read_text() == "<html>A</html>"
    assert (restored / "b.html").read_text() == "<html>B</html>"


def test_extract_snapshot_rejects_parent_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.tar.zst"
    compressor = zstandard.ZstdCompressor(level=3)
    with archive.open("wb") as raw, compressor.stream_writer(raw) as compressed:
        with tarfile.open(fileobj=compressed, mode="w|") as tar:
            info = tarfile.TarInfo("../escape.html")
            payload = b"unsafe"
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    manifest = tmp_path / "unsafe.manifest.json"
    manifest.write_text(json.dumps({"archive_sha256": "unused", "files": []}))

    with pytest.raises(SnapshotError, match="Unsafe archive member"):
        extract_snapshot(archive, manifest, tmp_path / "output", verify=False)
