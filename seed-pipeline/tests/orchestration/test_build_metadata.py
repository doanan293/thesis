import json

from seed_pipeline.orchestration import build_corpus
from seed_pipeline.orchestration.build_corpus import latest_snapshot_pair


def test_latest_snapshot_pair_reads_manifest_from_tracked_root(tmp_path, monkeypatch):
    archive_dir = tmp_path / "heavy" / "raw" / "ankhang" / "snapshots"
    manifest_dir = tmp_path / "manifests" / "source"
    archive_dir.mkdir(parents=True)
    manifest_dir.mkdir(parents=True)
    archive = archive_dir / "snapshot.tar.zst"
    archive.write_bytes(b"archive")
    manifest = manifest_dir / "snapshot.manifest.json"
    manifest.write_text(json.dumps({"archive_name": archive.name}), encoding="utf-8")
    monkeypatch.setattr(build_corpus, "verify_snapshot", lambda *_args: None)

    resolved_archive, resolved_manifest = latest_snapshot_pair(
        archive_dir, manifest_dir
    )

    assert resolved_archive == archive
    assert resolved_manifest == manifest
