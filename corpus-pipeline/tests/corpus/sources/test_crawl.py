from pathlib import Path

import pytest

from corpus_pipeline.corpus.sources.crawl import (
    CrawlConfigurationError,
    CrawlRequest,
    crawl_source,
)


def test_crawl_filters_sorts_and_resumes(tmp_path: Path):
    calls = []

    def fetch(url, timeout):
        calls.append((url, timeout))
        if url.startswith("http://sitemap"):
            return b"<loc>https://x/thuoc-b.html</loc><loc>https://x/other.html</loc><loc>https://x/thuoc-a.html</loc>"
        return url.encode()

    (tmp_path / "thuoc-b.html").write_bytes(b"old")
    result = crawl_source(
        CrawlRequest("http://sitemap", tmp_path, workers=2, request_timeout_seconds=3),
        fetch,
    )
    assert result.total == 2
    assert result.skipped == 1
    assert result.downloaded == 1
    assert calls[0] == ("http://sitemap", 3)


def test_crawl_force_replaces_existing_file(tmp_path: Path):
    target = tmp_path / "thuoc-a.html"
    target.write_bytes(b"old")

    def fetch(url, _timeout):
        return b"<loc>https://x/thuoc-a.html</loc>" if "sitemap" in url else b"new"

    result = crawl_source(CrawlRequest("http://sitemap", tmp_path, force=True), fetch)
    assert result.downloaded == 1
    assert target.read_bytes() == b"new"


def test_crawl_dry_run_has_no_fetch_or_output(tmp_path: Path):
    called = False

    def fetch(_url, _timeout):
        nonlocal called
        called = True
        return b""

    result = crawl_source(CrawlRequest("http://sitemap", tmp_path, dry_run=True), fetch)
    assert result == result.__class__(0, 0, 0, 0, tmp_path)
    assert not called
    assert list(tmp_path.iterdir()) == []


def test_crawl_rejects_invalid_workers(tmp_path: Path):
    with pytest.raises(CrawlConfigurationError, match="workers"):
        crawl_source(
            CrawlRequest("http://sitemap", tmp_path, workers=0), lambda *_: b""
        )
