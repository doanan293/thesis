from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

LEAFLET_SOURCE_SCHEMA = "leaflet-source-v1"


class LeafletSourceError(RuntimeError):
    """Raised when the leaflet HTML tree or its manifest is invalid."""


@dataclass(frozen=True)
class LeafletFile:
    path: str
    size: int
    sha256: str
    source_url: str | None


@dataclass(frozen=True)
class LeafletManifest:
    sitemap_url: str
    crawled_on: str
    url_list_sha256: str
    files: tuple[LeafletFile, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": LEAFLET_SOURCE_SCHEMA,
            "sitemap_url": self.sitemap_url,
            "crawled_on": self.crawled_on,
            "url_list_sha256": self.url_list_sha256,
            "file_count": len(self.files),
            "files": [asdict(item) for item in self.files],
        }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_leaflet_manifest(path: Path) -> LeafletManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != LEAFLET_SOURCE_SCHEMA:
        raise LeafletSourceError(
            f"{path} is not a {LEAFLET_SOURCE_SCHEMA} manifest "
            f"(schema_version {payload.get('schema_version')!r})"
        )
    files = tuple(
        LeafletFile(
            str(item["path"]),
            int(item["size"]),
            str(item["sha256"]),
            None if item.get("source_url") is None else str(item["source_url"]),
        )
        for item in payload["files"]
    )
    if int(payload["file_count"]) != len(files):
        raise LeafletSourceError(f"{path} file_count does not match its file list")
    return LeafletManifest(
        str(payload["sitemap_url"]),
        str(payload["crawled_on"]),
        str(payload["url_list_sha256"]),
        files,
    )


def write_leaflet_manifest(path: Path, manifest: LeafletManifest) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
