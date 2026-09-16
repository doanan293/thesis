from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

ARCHIVE_SCHEMA = "seed-data-archive-v1"
ARCHIVE_MANIFEST_NAME = "archive-manifest.json"
PART_PREFIX = "seed-pipeline-data.tar.zst.part-"
_PART_NAME = re.compile(rf"{re.escape(PART_PREFIX)}\d{{4}}")


class ArchiveError(RuntimeError):
    """Raised when the data archive is incomplete, corrupt or unsafe to restore."""


@dataclass(frozen=True)
class ArchiveFile:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class ArchivePart:
    name: str
    size: int
    sha256: str


@dataclass(frozen=True)
class ArchiveManifest:
    created_at: str
    git_commit: str
    files: tuple[ArchiveFile, ...]
    parts: tuple[ArchivePart, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ARCHIVE_SCHEMA,
            "created_at": self.created_at,
            "git_commit": self.git_commit,
            "file_count": len(self.files),
            "files": [asdict(item) for item in self.files],
            "parts": [asdict(item) for item in self.parts],
        }


def safe_data_path(value: str) -> str:
    """A normalized relative path inside data/ that is not scratch space."""
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != value
        or path.parts[0] == "work"
    ):
        raise ArchiveError(f"Unsafe archive path: {value!r}")
    return value


def read_archive_manifest(path: Path) -> ArchiveManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != ARCHIVE_SCHEMA:
        raise ArchiveError(f"{path} is not a {ARCHIVE_SCHEMA} manifest")
    files = tuple(
        ArchiveFile(
            safe_data_path(str(item["path"])), int(item["size"]), str(item["sha256"])
        )
        for item in payload["files"]
    )
    parts = tuple(
        ArchivePart(str(item["name"]), int(item["size"]), str(item["sha256"]))
        for item in payload["parts"]
    )
    if int(payload["file_count"]) != len(files):
        raise ArchiveError(f"{path} file_count does not match its file list")
    if len({item.path for item in files}) != len(files):
        raise ArchiveError(f"{path} lists a file more than once")
    for part in parts:
        if _PART_NAME.fullmatch(part.name) is None:
            raise ArchiveError(f"Unexpected archive part name: {part.name!r}")
    return ArchiveManifest(
        str(payload["created_at"]), str(payload["git_commit"]), files, parts
    )


def write_archive_manifest(path: Path, manifest: ArchiveManifest) -> None:
    Path(path).write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
