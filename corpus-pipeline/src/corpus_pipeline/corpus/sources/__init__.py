"""Typed, deterministic source acquisition services."""

from corpus_pipeline.corpus.sources.crawl import (
    CrawlConfigurationError,
    CrawlRequest,
    CrawlResult,
    crawl_source,
)

__all__ = ["CrawlConfigurationError", "CrawlRequest", "CrawlResult", "crawl_source"]
