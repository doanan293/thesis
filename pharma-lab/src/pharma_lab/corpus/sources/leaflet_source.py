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


@dataclass(frozen=True)
class LeafletSource:
    html_dir: Path
    manifest_sha256: str
    file_count: int


def verify_leaflet_source(leaflets_dir: Path) -> LeafletSource:
    leaflets_dir = Path(leaflets_dir)
    manifest_path = leaflets_dir / "manifest.json"
    if not manifest_path.is_file():
        raise LeafletSourceError(f"Leaflet manifest is missing: {manifest_path}")
    manifest = read_leaflet_manifest(manifest_path)
    html_dir = leaflets_dir / "html"
    present = (
        {path.relative_to(html_dir).as_posix() for path in html_dir.rglob("*.html")}
        if html_dir.is_dir()
        else set()
    )
    expected = {item.path for item in manifest.files}
    if present != expected:
        raise LeafletSourceError(
            f"Leaflet HTML under {html_dir} does not match {manifest_path}: "
            f"{len(expected - present)} missing, {len(present - expected)} unexpected"
        )
    for item in manifest.files:
        path = html_dir / item.path
        if path.stat().st_size != item.size or file_sha256(path) != item.sha256:
            raise LeafletSourceError(
                f"Leaflet file changed since the crawl: {item.path}"
            )
    return LeafletSource(html_dir, file_sha256(manifest_path), len(manifest.files))
