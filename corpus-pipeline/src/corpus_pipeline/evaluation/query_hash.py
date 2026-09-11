from __future__ import annotations

import hashlib


def normalize_query_for_hash(query: str) -> str:
    return query.strip()


def query_hash(query: str) -> str:
    normalized = normalize_query_for_hash(query)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def model_slug(model_name: str) -> str:
    return (
        model_name.replace(":", "_")
        .replace("-", "_")
        .replace(".", "_")
        .replace("/", "_")
    )
