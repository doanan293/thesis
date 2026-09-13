from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


class CrawlConfigurationError(ValueError):
    """Raised when source crawl options are invalid."""


class CrawlFetchError(RuntimeError):
    """Raised when a source page cannot be fetched."""


@dataclass(frozen=True)
class CrawlRequest:
    sitemap_url: str
    output_dir: Path
    workers: int = 8
    request_timeout_seconds: float = 10.0
    force: bool = False
    dry_run: bool = False


@dataclass(frozen=True)
class CrawlResult:
    total: int
    downloaded: int
    skipped: int
    failed: int
    output_dir: Path


def parse_locations(payload: bytes) -> tuple[str, ...]:
    text = payload.decode("utf-8", errors="replace")
    return tuple(sorted(set(re.findall(r"<loc>\s*(.*?)\s*</loc>", text))))


def filter_drug_urls(urls: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    selected = []
    for url in urls:
        path = urlparse(url).path.lower()
        if "thuoc" in path or "dieu-tri" in path or "duoc" in path:
            selected.append(url)
    return tuple(sorted(set(selected)))


def _filename(url: str) -> str:
    path = urlparse(url).path.strip("/") or "index"
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", path)
    return name if name.endswith(".html") else name + ".html"


def materialize_pages(
    request: CrawlRequest,
    urls: tuple[str, ...],
    fetch,
) -> CrawlResult:
    request.output_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    skipped = 0
    for url in urls:
        target = request.output_dir / _filename(url)
        if target.is_file() and not request.force:
            skipped += 1
        else:
            pending.append((url, target))
    downloaded = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=request.workers) as pool:
        futures = {
            pool.submit(fetch, url, request.request_timeout_seconds): (url, target)
            for url, target in pending
        }
        for future in as_completed(futures):
            _url, target = futures[future]
            try:
                target.write_bytes(future.result())
            except Exception as exc:
                failed += 1
                if isinstance(exc, CrawlFetchError):
                    continue
            else:
                downloaded += 1
    return CrawlResult(len(urls), downloaded, skipped, failed, request.output_dir)


def crawl_source(request: CrawlRequest, fetch) -> CrawlResult:
    if request.workers < 1:
        raise CrawlConfigurationError("workers must be >= 1")
    if request.request_timeout_seconds <= 0:
        raise CrawlConfigurationError("request timeout must be positive")
    if request.dry_run:
        return CrawlResult(0, 0, 0, 0, request.output_dir)
    index = fetch(request.sitemap_url, request.request_timeout_seconds)
    urls = filter_drug_urls(parse_locations(index))
    return materialize_pages(request, urls, fetch)
