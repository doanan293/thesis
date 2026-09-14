from __future__ import annotations

import hashlib
import io
import subprocess
import tarfile
from collections.abc import Buffer, Sequence
from pathlib import Path
from typing import BinaryIO

import zstandard

from seed_pipeline.data_archive.manifest import (
    PART_PREFIX,
    ArchiveError,
    ArchiveFile,
    ArchiveManifest,
    ArchivePart,
    safe_data_path,
)

PART_SIZE_BYTES = 1900 * 1024 * 1024
ZSTD_LEVEL = 10


def list_archive_files(project_root: Path) -> list[str]:
    """Git-ignored files under data/, relative to data/, without data/work/."""
    output = subprocess.run(
        [
            "git",
            "ls-files",
            "--others",
            "--ignored",
            "--exclude-standard",
            "-z",
            "--",
            "data",
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
    ).stdout
    files: list[str] = []
    for item in output.split(b"\0"):
        if not item:
            continue
        relative = item.decode("utf-8").removeprefix("data/")
        if not relative.startswith("work/"):
            files.append(safe_data_path(relative))
    return sorted(files)


def git_head(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ArchiveError(
            f"Cannot read the Git commit of {project_root}: {result.stderr.strip()}"
        )
    return result.stdout.strip()


class PartWriter(io.RawIOBase):
    """Binary sink that splits a stream into numbered part files while hashing them."""

    def __init__(self, directory: Path, *, part_size: int) -> None:
        super().__init__()
        if part_size < 1:
            raise ValueError("part_size must be >= 1")
        self._directory = Path(directory)
        self._part_size = part_size
        self._handle: BinaryIO | None = None
        self._name = ""
        self._digest = hashlib.sha256()
        self._written = 0
        self.parts: list[ArchivePart] = []

    def writable(self) -> bool:
        return True

    def write(self, data: Buffer, /) -> int:
        view = memoryview(data).cast("B")
        offset = 0
        while offset < len(view):
            handle = self._current_handle()
            take = min(len(view) - offset, self._part_size - self._written)
            chunk = view[offset : offset + take]
            handle.write(chunk)
            self._digest.update(chunk)
            self._written += take
            offset += take
        return len(view)

    def flush(self) -> None:
        if self._handle is not None:
            self._handle.flush()

    def close(self) -> None:
        self._finish_part()
        super().close()

    def _current_handle(self) -> BinaryIO:
        if self._handle is not None and self._written < self._part_size:
            return self._handle
        self._finish_part()
        self._name = f"{PART_PREFIX}{len(self.parts) + 1:04d}"
        handle = (self._directory / self._name).open("wb")
        self._handle = handle
        return handle

    def _finish_part(self) -> None:
        if self._handle is None:
            return
        self._handle.close()
        self.parts.append(
            ArchivePart(self._name, self._written, self._digest.hexdigest())
        )
        self._handle = None
        self._digest = hashlib.sha256()
        self._written = 0


class _HashingReader(io.RawIOBase):
    def __init__(self, handle: BinaryIO) -> None:
        super().__init__()
        self._handle = handle
        self._digest = hashlib.sha256()
        self.size = 0

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Buffer, /) -> int:
        view = memoryview(buffer).cast("B")
        data = self._handle.read(len(view))
        view[: len(data)] = data
        self._digest.update(data)
        self.size += len(data)
        return len(data)

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def pack_archive(
    data_dir: Path,
    files: Sequence[str],
    output_dir: Path,
    *,
    part_size: int,
    created_at: str,
    git_commit: str,
) -> ArchiveManifest:
    if not files:
        raise ArchiveError(f"No Git-ignored files under {data_dir} to archive")
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = PartWriter(output_dir, part_size=part_size)
    records: list[ArchiveFile] = []
    compressor = zstandard.ZstdCompressor(level=ZSTD_LEVEL, threads=-1)
    with (
        compressor.stream_writer(io.BufferedWriter(writer)) as compressed,
        tarfile.open(fileobj=compressed, mode="w|") as archive,
    ):
        for relative in files:
            path = Path(data_dir) / relative
            info = tarfile.TarInfo(relative)
            info.size = path.stat().st_size
            info.mode = 0o644
            with path.open("rb") as handle:
                reader = _HashingReader(handle)
                archive.addfile(info, io.BufferedReader(reader))
            if reader.size != info.size:
                raise ArchiveError(f"{relative} changed while it was being archived")
            records.append(ArchiveFile(relative, info.size, reader.hexdigest()))
    return ArchiveManifest(created_at, git_commit, tuple(records), tuple(writer.parts))
