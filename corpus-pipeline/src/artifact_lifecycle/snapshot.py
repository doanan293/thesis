from __future__ import annotations

import hashlib
import json
import tarfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

import zstandard


class SnapshotError(RuntimeError):
    """Raised when a snapshot is invalid or unsafe to materialize."""


@dataclass(frozen=True)
class SnapshotResult:
    snapshot_id: str
    archive_path: Path
    manifest_path: Path
    archive_sha256: str
    file_count: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_records(source_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(Path(source_dir).rglob("*.html")):
        if not path.is_file():
            continue
        relative = path.relative_to(source_dir).as_posix()
        records.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def _url_records(urls_path: Path | None) -> tuple[str | None, dict[str, str]]:
    if urls_path is None or not Path(urls_path).exists():
        return None, {}
    path = Path(urls_path)
    urls = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    urls = [url for url in urls if url]
    mapping: dict[str, str] = {}
    for url in urls:
        parsed = urlparse(url)
        name = Path(parsed.path).name
        if name:
            mapping[name.casefold()] = url
    return sha256_file(path), mapping


def _normalized_tar_info(path: Path, relative: Path) -> tarfile.TarInfo:
    info = tarfile.TarInfo(relative.as_posix())
    info.size = path.stat().st_size
    info.mtime = 0
    info.mode = 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    return info


def pack_snapshot(
    source_dir: Path,
    output_dir: Path,
    *,
    urls_path: Path | None = None,
    crawler_version: str = "snapshot-v1",
) -> SnapshotResult:
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    if not source_dir.is_dir():
        raise SnapshotError(f"Snapshot source directory is missing: {source_dir}")
    records = _file_records(source_dir)
    if not records:
        raise SnapshotError(f"No HTML files found under {source_dir}")

    content_digest = hashlib.sha256()
    for record in records:
        content_digest.update(record["path"].encode("utf-8"))
        content_digest.update(record["sha256"].encode("ascii"))
    snapshot_id = (
        f"ankhang-{date.today().isoformat()}-{content_digest.hexdigest()[:12]}"
    )
    archive_path = output_dir / f"{snapshot_id}.tar.zst"
    manifest_path = output_dir / f"{snapshot_id}.manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    url_list_sha256, url_mapping = _url_records(urls_path)
    for record in records:
        record["source_url"] = url_mapping.get(Path(record["path"]).name.casefold())

    compressor = zstandard.ZstdCompressor(level=19)
    with (
        archive_path.open("wb") as raw,
        compressor.stream_writer(raw) as compressed,
        tarfile.open(fileobj=compressed, mode="w|") as archive,
    ):
        for record in records:
            relative = Path(record["path"])
            source_path = source_dir / relative
            with source_path.open("rb") as payload:
                archive.addfile(_normalized_tar_info(source_path, relative), payload)

    manifest = {
        "schema_version": "ankhang-snapshot-v1",
        "crawler_version": crawler_version,
        "snapshot_id": snapshot_id,
        "archive_name": archive_path.name,
        "archive_sha256": sha256_file(archive_path),
        "url_list_sha256": url_list_sha256,
        "file_count": len(records),
        "files": records,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return SnapshotResult(
        snapshot_id=snapshot_id,
        archive_path=archive_path,
        manifest_path=manifest_path,
        archive_sha256=manifest["archive_sha256"],
        file_count=len(records),
    )


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SnapshotError(f"Invalid snapshot manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), list):
        raise SnapshotError(f"Snapshot manifest has invalid shape: {manifest_path}")
    return manifest


def _safe_member(member: tarfile.TarInfo, seen: set[str]) -> PurePosixPath:
    name = PurePosixPath(member.name)
    if name.is_absolute() or ".." in name.parts:
        raise SnapshotError(f"Unsafe archive member: {member.name}")
    if not member.isfile() or member.name in seen:
        raise SnapshotError(f"Unsupported or duplicate archive member: {member.name}")
    seen.add(member.name)
    return name


def _manifest_index(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(record["path"]): record for record in manifest["files"]}


def verify_snapshot(archive_path: Path, manifest_path: Path) -> dict[str, Any]:
    archive_path = Path(archive_path)
    manifest = _load_manifest(Path(manifest_path))
    expected_archive = str(manifest.get("archive_sha256") or "")
    if expected_archive and sha256_file(archive_path) != expected_archive:
        raise SnapshotError(f"Snapshot archive checksum mismatch: {archive_path}")
    expected = _manifest_index(manifest)
    seen: set[str] = set()
    decompressor = zstandard.ZstdDecompressor()
    with (
        archive_path.open("rb") as raw,
        decompressor.stream_reader(raw) as stream,
        tarfile.open(fileobj=stream, mode="r|") as archive,
    ):
        for member in archive:
            name = _safe_member(member, seen)
            record = expected.get(name.as_posix())
            if record is None:
                raise SnapshotError(f"Archive member is absent from manifest: {name}")
            payload = archive.extractfile(member)
            if payload is None:
                raise SnapshotError(f"Cannot read archive member: {name}")
            digest = hashlib.sha256()
            size = 0
            while block := payload.read(1024 * 1024):
                digest.update(block)
                size += len(block)
            if size != int(record["size"]) or digest.hexdigest() != record["sha256"]:
                raise SnapshotError(f"Snapshot member checksum mismatch: {name}")
    if seen != set(expected):
        missing = sorted(set(expected) - seen)
        raise SnapshotError(f"Snapshot is missing members: {missing[:3]}")
    if int(manifest.get("file_count", -1)) != len(seen):
        raise SnapshotError("Snapshot file count does not match manifest")
    return manifest


def extract_snapshot(
    archive_path: Path,
    manifest_path: Path,
    output_dir: Path,
    *,
    verify: bool = True,
) -> None:
    archive_path = Path(archive_path)
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    manifest = (
        verify_snapshot(archive_path, manifest_path)
        if verify
        else _load_manifest(manifest_path)
    )
    expected = _manifest_index(manifest)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SnapshotError(f"Extraction destination is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    decompressor = zstandard.ZstdDecompressor()
    with (
        archive_path.open("rb") as raw,
        decompressor.stream_reader(raw) as stream,
        tarfile.open(fileobj=stream, mode="r|") as archive,
    ):
        for member in archive:
            name = _safe_member(member, seen)
            record = expected.get(name.as_posix())
            if record is None:
                raise SnapshotError(f"Archive member is absent from manifest: {name}")
            destination = (output_dir / Path(*name.parts)).resolve()
            if output_dir.resolve() not in destination.parents:
                raise SnapshotError(f"Unsafe extraction destination: {name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            payload = archive.extractfile(member)
            if payload is None:
                raise SnapshotError(f"Cannot read archive member: {name}")
            destination.write_bytes(payload.read())
    if seen != set(expected):
        raise SnapshotError("Extracted archive does not contain every manifest file")
