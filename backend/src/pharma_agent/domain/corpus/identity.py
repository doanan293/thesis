"""Deterministic identities for section revisions and chunk versions (spec C §6.1).

Public API: seed-pipeline uses ``sha256_hex`` to key precomputed embeddings.
"""

import hashlib
import json
import uuid
from collections.abc import Mapping, Sequence

from pharma_agent.domain.corpus.bundle import BlockRecord
from pharma_agent.domain.shared.text import normalize_text

CORPUS_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "pharma-agent:corpus:v1")
_SEPARATOR = "\x1f"


def _normalized(value: object) -> object:
    if isinstance(value, str):
        return normalize_text(value)
    if isinstance(value, Mapping):
        return {str(key): _normalized(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_normalized(item) for item in value]
    return value


def canonical_json(value: object) -> str:
    """Sorted keys, no insignificant whitespace, every string normalized."""
    return json.dumps(
        _normalized(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def section_revision_id(section_key: str, blocks: Sequence[BlockRecord]) -> uuid.UUID:
    blocks_digest = sha256_hex(
        canonical_json([block.model_dump(mode="json") for block in blocks])
    )
    return uuid.uuid5(
        CORPUS_NAMESPACE,
        _SEPARATOR.join(("section-revision", section_key, blocks_digest)),
    )


def chunk_version_id(
    section_key: str, chunk_text: str, embedding_text: str, chunker_version: str
) -> uuid.UUID:
    digest = sha256_hex(
        _SEPARATOR.join(
            (
                section_key,
                normalize_text(chunk_text),
                normalize_text(embedding_text),
                chunker_version,
            )
        )
    )
    return uuid.uuid5(CORPUS_NAMESPACE, "chunk" + _SEPARATOR + digest)
