from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

from seed_pipeline.corpus.sources.leaflet_source import (
    LeafletFile,
    LeafletManifest,
    file_sha256,
    read_leaflet_manifest,
    write_leaflet_manifest,
)

Fetch = Callable[[str, float], bytes]

DRUG_CATEGORY_PREFIX = "thuoc-"
DRUG_CATEGORIES = frozenset(
    {
        "tim-mach-huyet-ap",
        "duong-tieu-hoa",
        "khang-sinh-khang-nam",
        "tri-ho-hen-phe-quan",
        "thuoc-nho-mat-tai-mui-hong",
        "tri-giun-san",
        "thuoc-dieu-tri-ung-thu",
        "chong-di-ung",
        "thuoc-ke-don",
        "khang-nam-khang-virus",
        "thuoc-dung-ngoai-da",
        "giam-dau-ha-sot",
        "vitamin-va-khoang-chat",
    }
)
NON_DRUG_CATEGORIES = frozenset(
    {
        "sua-rua-mat",
        "kem-chong-nang",
        "kem-duong-da",
        "tinh-chat-duong-da",
        "mat-na-cham-soc-da",
        "dau-goi-dau",
        "kem-danh-rang",
        "ban-chai-danh-rang",
        "nuoc-suc-mieng",
        "bang-ve-sinh",
        "bao-cao-su",
        "ta-cho-be",
        "sua-bot-cong-thuc",
        "khan-uot",
        "son-duong-moi",
        "sua-tam",
        "kem-tri-mun",
        "xit-khoang",
        "nuoc-hoa-hong",
        "tay-trang",
        "kem-duong-the",
        "kem-chong-muoi",
        "dau-xa",
        "gel-rua-tay",
        "khan-giay",
        "dung-dich-ve-sinh",
        "mieng-dan-mun",
        "mat-na",
        "sua-mat",
        "tay-te-bao-chet",
        "bong-tay-trang",
    }
)


class CrawlConfigurationError(ValueError):
    """Raised when crawl options are invalid."""


class CrawlFetchError(RuntimeError):
    """Raised when a page cannot be fetched."""


@dataclass(frozen=True)
class CrawlRequest:
    leaflets_dir: Path
    sitemap_url: str | None = None
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
    manifest_path: Path


def parse_locations(payload: bytes) -> tuple[str, ...]:
    text = payload.decode("utf-8", errors="replace")
    return tuple(sorted(set(re.findall(r"<loc>\s*(.*?)\s*</loc>", text))))


def leaflet_page(url: str) -> tuple[str, str] | None:
    parts = [part for part in urlparse(url).path.split("/") if part]
    if len(parts) != 2:
        return None
    category, slug = parts
    if category in NON_DRUG_CATEGORIES:
        return None
    if not (category.startswith(DRUG_CATEGORY_PREFIX) or category in DRUG_CATEGORIES):
        return None
    return category, slug


def collect_page_urls(
    sitemap_url: str, fetch: Fetch, *, timeout: float, workers: int
) -> tuple[str, ...]:
    sub_sitemaps = parse_locations(fetch(sitemap_url, timeout))
    urls: set[str] = set()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for payload in pool.map(lambda url: fetch(url, timeout), sub_sitemaps):
            urls.update(parse_locations(payload))
    return tuple(sorted(urls))


def _write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def crawl_leaflets(request: CrawlRequest, fetch: Fetch, *, today: date) -> CrawlResult:
    if request.workers < 1:
        raise CrawlConfigurationError("workers must be >= 1")
    if request.request_timeout_seconds <= 0:
        raise CrawlConfigurationError("request timeout must be positive")
    manifest_path = request.leaflets_dir / "manifest.json"
    previous = read_leaflet_manifest(manifest_path) if manifest_path.is_file() else None
    sitemap_url = request.sitemap_url or (previous.sitemap_url if previous else None)
    if not sitemap_url:
        raise CrawlConfigurationError(
            "The first crawl needs --sitemap-url; later crawls read it from manifest.json"
        )
    if request.dry_run:
        return CrawlResult(0, 0, 0, 0, manifest_path)
    all_urls = collect_page_urls(
        sitemap_url,
        fetch,
        timeout=request.request_timeout_seconds,
        workers=request.workers,
    )
    pages = {url: page for url in all_urls if (page := leaflet_page(url)) is not None}
    urls_dir = request.leaflets_dir / "urls"
    _write_lines(urls_dir / "all_urls.txt", all_urls)
    drug_urls = sorted(pages)
    _write_lines(urls_dir / "drug_urls.txt", drug_urls)
    html_dir = request.leaflets_dir / "html"

    def target_for(url: str) -> Path:
        category, slug = pages[url]
        return html_dir / category / f"{slug}.html"

    pending: list[str] = []
    skipped = 0
    for url in drug_urls:
        if target_for(url).is_file() and not request.force:
            skipped += 1
        else:
            pending.append(url)
    downloaded = failed = 0
    with ThreadPoolExecutor(max_workers=request.workers) as pool:
        futures = {
            pool.submit(fetch, url, request.request_timeout_seconds): url
            for url in pending
        }
        for future in as_completed(futures):
            url = futures[future]
            try:
                payload = future.result()
            except CrawlFetchError:
                failed += 1
                continue
            target = target_for(url)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            downloaded += 1
    # Every HTML file on disk is listed, including pages whose URL has left the sitemap,
    # so `verify_leaflet_source` never meets a file the manifest does not know.
    url_by_path = {
        target_for(url).relative_to(html_dir).as_posix(): url for url in drug_urls
    }
    files = tuple(
        LeafletFile(
            path.relative_to(html_dir).as_posix(),
            path.stat().st_size,
            file_sha256(path),
            url_by_path.get(path.relative_to(html_dir).as_posix()),
        )
        for path in sorted(html_dir.rglob("*.html"))
    )
    write_leaflet_manifest(
        manifest_path,
        LeafletManifest(
            sitemap_url,
            today.isoformat(),
            file_sha256(urls_dir / "drug_urls.txt"),
            files,
        ),
    )
    return CrawlResult(len(drug_urls), downloaded, skipped, failed, manifest_path)
