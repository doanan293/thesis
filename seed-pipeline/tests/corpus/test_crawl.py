import json
from datetime import date
from pathlib import Path

import pytest

from seed_pipeline.corpus.sources.crawl import (
    CrawlConfigurationError,
    CrawlFetchError,
    CrawlRequest,
    crawl_leaflets,
    leaflet_page,
)

SITEMAP = "https://example.test/sitemap.xml"
PAGES = {
    SITEMAP: b"<sitemapindex><loc>https://example.test/sub-1.xml</loc>"
    b"<loc>https://example.test/sub-2.xml</loc></sitemapindex>",
    "https://example.test/sub-1.xml": b"<urlset>"
    b"<loc>https://example.test/thuoc-giam-dau/panadol</loc>"
    b"<loc>https://example.test/sua-tam/xa-bong</loc></urlset>",
    "https://example.test/sub-2.xml": b"<urlset>"
    b"<loc>https://example.test/tim-mach-huyet-ap/amlor</loc>"
    b"<loc>https://example.test/thuoc-giam-dau</loc>"
    b"<loc>https://example.test/thuoc-giam-dau/hong/1</loc></urlset>",
    "https://example.test/thuoc-giam-dau/panadol": b"<html>panadol</html>",
    "https://example.test/tim-mach-huyet-ap/amlor": b"<html>amlor</html>",
}


def fetch(url: str, timeout: float) -> bytes:
    del timeout
    return PAGES[url]


@pytest.mark.parametrize(
    ("url", "page"),
    [
        ("https://example.test/thuoc-giam-dau/panadol", ("thuoc-giam-dau", "panadol")),
        (
            "https://example.test/tim-mach-huyet-ap/amlor",
            ("tim-mach-huyet-ap", "amlor"),
        ),
        ("https://example.test/sua-tam/xa-bong", None),
        ("https://example.test/thuoc-giam-dau", None),
        ("https://example.test/thuoc-giam-dau/hong/1", None),
    ],
)
def test_leaflet_pages_are_two_segment_drug_urls(url: str, page: object) -> None:
    assert leaflet_page(url) == page


def test_crawl_writes_html_tree_url_lists_and_manifest(tmp_path: Path) -> None:
    leaflets = tmp_path / "leaflets"

    result = crawl_leaflets(
        CrawlRequest(leaflets, sitemap_url=SITEMAP, workers=2),
        fetch,
        today=date(2026, 9, 14),
    )

    assert (result.total, result.downloaded, result.skipped, result.failed) == (
        2,
        2,
        0,
        0,
    )
    assert (
        leaflets / "html" / "thuoc-giam-dau" / "panadol.html"
    ).read_bytes() == PAGES["https://example.test/thuoc-giam-dau/panadol"]
    assert (leaflets / "urls" / "drug_urls.txt").read_text(
        encoding="utf-8"
    ).splitlines() == [
        "https://example.test/thuoc-giam-dau/panadol",
        "https://example.test/tim-mach-huyet-ap/amlor",
    ]
    assert (
        len(
            (leaflets / "urls" / "all_urls.txt")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        == 5
    )
    manifest = json.loads((leaflets / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "leaflet-source-v1"
    assert manifest["sitemap_url"] == SITEMAP
    assert manifest["crawled_on"] == "2026-09-14"
    assert [item["path"] for item in manifest["files"]] == [
        "thuoc-giam-dau/panadol.html",
        "tim-mach-huyet-ap/amlor.html",
    ]
    assert (
        manifest["files"][0]["source_url"]
        == "https://example.test/thuoc-giam-dau/panadol"
    )


def test_later_crawls_read_the_sitemap_from_the_manifest_and_skip_files(
    tmp_path: Path,
) -> None:
    leaflets = tmp_path / "leaflets"
    crawl_leaflets(
        CrawlRequest(leaflets, sitemap_url=SITEMAP), fetch, today=date(2026, 9, 14)
    )

    again = crawl_leaflets(CrawlRequest(leaflets), fetch, today=date(2026, 9, 15))

    assert (again.downloaded, again.skipped) == (0, 2)


def test_first_crawl_needs_a_sitemap_url(tmp_path: Path) -> None:
    with pytest.raises(CrawlConfigurationError, match="--sitemap-url"):
        crawl_leaflets(
            CrawlRequest(tmp_path / "leaflets"), fetch, today=date(2026, 9, 14)
        )


def test_failed_pages_are_counted_and_left_out_of_the_manifest(tmp_path: Path) -> None:
    def flaky(url: str, timeout: float) -> bytes:
        if url.endswith("/amlor"):
            raise CrawlFetchError(url)
        return fetch(url, timeout)

    leaflets = tmp_path / "leaflets"
    result = crawl_leaflets(
        CrawlRequest(leaflets, sitemap_url=SITEMAP), flaky, today=date(2026, 9, 14)
    )

    assert result.failed == 1
    manifest = json.loads((leaflets / "manifest.json").read_text(encoding="utf-8"))
    assert [item["path"] for item in manifest["files"]] == [
        "thuoc-giam-dau/panadol.html"
    ]
