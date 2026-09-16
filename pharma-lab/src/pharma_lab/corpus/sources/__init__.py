"""Typed, deterministic source acquisition services."""

from pharma_lab.corpus.sources.crawl import (
    CrawlConfigurationError,
    CrawlFetchError,
    CrawlRequest,
    CrawlResult,
    crawl_leaflets,
)

__all__ = [
    "CrawlConfigurationError",
    "CrawlFetchError",
    "CrawlRequest",
    "CrawlResult",
    "crawl_leaflets",
]
