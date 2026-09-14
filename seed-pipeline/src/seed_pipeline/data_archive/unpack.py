from __future__ import annotations

import hashlib
import io
import os
import tarfile
from collections.abc import Buffer, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, BinaryIO

import zstandard

from seed_pipeline.data_archive.manifest import (
    ArchiveError,
    ArchiveFile,
    ArchiveManifest,
    ArchivePart,
)
from seed_pipeline.evaluation.artifact_contracts import sha256_file


@dataclass(frozen=True)
class UnpackResult:
    written: int
    unchanged: int


def part_matches(path: Path, part: ArchivePart) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == part.size
        and sha256_file(path) == part.sha256
    )


def verify_parts(archive_dir: Path, manifest: ArchiveManifest) -> None:
    for part in manifest.parts:
        if not part_matches(Path(archive_dir) / part.name, part):
            raise ArchiveError(
                f"Archive part {part.name} is missing or does not match its sha256"
            )


def _file_matches(path: Path, record: ArchiveFile) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == record.size
        and sha256_file(path) == record.sha256
    )


class _PartReader(io.RawIOBase):
    """Reads the part files one after another as a single stream."""

    def __init__(self, paths: Sequence[Path]) -> None:
        super().__init__()
        self._pending = list(paths)
        self._handle: BinaryIO | None = None

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Buffer, /) -> int:
        view = memoryview(buffer).cast("B")
        while True:
            if self._handle is None:
                if not self._pending:
                    return 0
                self._handle = self._pending.pop(0).open("rb")
            data = self._handle.read(len(view))
            if data:
                view[: len(data)] = data
                return len(data)
            self._handle.close()
            self._handle = None

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        super().close()


def _write_verified(target: Path, payload: IO[bytes], record: ArchiveFile) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.pulling")
    digest = hashlib.sha256()
    size = 0
    with temporary.open("wb") as handle:
        while block := payload.read(1024 * 1024):
            handle.write(block)
            digest.update(block)
            size += len(block)
    if size != record.size or digest.hexdigest() != record.sha256:
        temporary.unlink()
        raise ArchiveError(f"{record.path} does not match the archive manifest")
    os.replace(temporary, target)


def unpack_archive(
    archive_dir: Path, manifest: ArchiveManifest, data_dir: Path, *, force: bool
) -> UnpackResult:
    verify_parts(archive_dir, manifest)
    data_dir = Path(data_dir)
    expected = {item.path: item for item in manifest.files}
    unchanged: set[str] = set()
    conflicts: list[str] = []
    for item in manifest.files:
        target = data_dir / item.path
        if not target.exists():
            continue
        if _file_matches(target, item):
            unchanged.add(item.path)
        else:
            conflicts.append(item.path)
    if conflicts and not force:
        raise ArchiveError(
            f"{len(conflicts)} file(s) in {data_dir} differ from the archive "
            f"(first: {', '.join(conflicts[:5])}); use --force to overwrite them"
        )
    seen: set[str] = set()
    written = 0
    parts = [Path(archive_dir) / part.name for part in manifest.parts]
    with (
        io.BufferedReader(_PartReader(parts)) as source,
        zstandard.ZstdDecompressor().stream_reader(source) as stream,
        tarfile.open(fileobj=stream, mode="r|") as archive,
    ):
        for member in archive:
            record = expected.get(member.name)
            if record is None or member.name in seen or not member.isfile():
                raise ArchiveError(f"Unexpected archive member: {member.name!r}")
            seen.add(member.name)
            if member.name in unchanged:
                continue
            payload = archive.extractfile(member)
            if payload is None:
                raise ArchiveError(f"Cannot read archive member: {member.name}")
            _write_verified(data_dir / record.path, payload, record)
            written += 1
    missing = set(expected) - seen
    if missing:
        raise ArchiveError(
            f"The archive lacks {len(missing)} file(s) that its manifest lists"
        )
    return UnpackResult(written, len(unchanged))
