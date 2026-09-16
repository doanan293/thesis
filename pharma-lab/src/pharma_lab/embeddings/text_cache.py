"""Checksummed JSONL cache of text embeddings keyed by (model, embedding_text_sha256, vector_dim)."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
from array import array
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from pharma_lab.cache.jsonl_records import (
    CacheRecordError,
    append_record,
    iter_records,
)
from pharma_lab.integrations.kaggle.workers.scheduling import stream_map_ordered

CACHE_SCHEMA = "text-embedding-v1"


class TextEmbeddingError(RuntimeError):
    """Raised when embedding inputs or the text embedding cache are invalid."""


@dataclass(frozen=True)
class EmbeddingInput:
    embedding_text_sha256: str
    embedding_text: str


@dataclass(frozen=True)
class EmbedRun:
    total: int
    cached: int
    embedded: int
    stopped_early: bool


class EmbeddingClient(Protocol):
    def embed(
        self, texts: list[str], model: str, expected_dimension: int
    ) -> list[list[float]]: ...


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_embedding_inputs(inputs: Iterable[EmbeddingInput], path: Path) -> int:
    unique = {item.embedding_text_sha256: item for item in inputs}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for digest in sorted(unique):
            row = {
                "embedding_text_sha256": digest,
                "embedding_text": unique[digest].embedding_text,
            }
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return len(unique)


def read_embedding_inputs(path: Path) -> list[EmbeddingInput]:
    inputs: list[EmbeddingInput] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            text = str(row["embedding_text"])
            digest = str(row["embedding_text_sha256"])
            if text_sha256(text) != digest:
                raise TextEmbeddingError(
                    f"embedding_text_sha256 mismatch at {path}:{line_number}"
                )
            inputs.append(EmbeddingInput(digest, text))
    return inputs


def cache_record_key(record: Mapping[str, Any]) -> tuple[str, str, int]:
    return (
        str(record["model"]),
        str(record["embedding_text_sha256"]),
        int(record["vector_dim"]),
    )


def _vector(value: object, vector_dim: int) -> list[float]:
    if not isinstance(value, list) or len(value) != vector_dim:
        raise TextEmbeddingError(f"embedding must have dimension {vector_dim}")
    vector = [float(item) for item in value]
    if not all(math.isfinite(item) for item in vector):
        raise TextEmbeddingError("embedding values must be finite numbers")
    return vector


class TextEmbeddingCache:
    """The vectors of one model in a checksummed JSONL cache file.

    The file is read line by line and each vector is kept as a float64 array, eight bytes
    per value, so a 4096-dimension cache of 25,000 texts takes about 1 GB of memory.
    """

    def __init__(
        self, path: Path, *, model: str, model_sha256: str, vector_dim: int
    ) -> None:
        self.path = Path(path)
        self.model = model
        self.model_sha256 = model_sha256
        self.vector_dim = vector_dim
        self._vectors: dict[str, array[float]] = {}
        for record in self._own_records():
            if record.get("model_sha256") != model_sha256:
                raise TextEmbeddingError(
                    f"Model digest mismatch in {self.path} for "
                    f"{record.get('embedding_text_sha256')}"
                )
            self._vectors[str(record["embedding_text_sha256"])] = array(
                "d", _vector(record.get("embedding"), vector_dim)
            )

    def _own_records(self) -> Iterator[dict[str, Any]]:
        try:
            for record in iter_records(self.path):
                if cache_record_key(record)[0::2] == (self.model, self.vector_dim):
                    yield record
        except CacheRecordError as exc:
            raise TextEmbeddingError(str(exc)) from exc

    def __len__(self) -> int:
        return len(self._vectors)

    def digests(self) -> list[str]:
        return sorted(self._vectors)

    def get(self, digest: str) -> list[float] | None:
        vector = self._vectors.get(digest)
        return None if vector is None else vector.tolist()

    def set(self, digest: str, embedding: Sequence[float]) -> None:
        vector = _vector(list(embedding), self.vector_dim)
        append_record(
            self.path,
            {
                "model": self.model,
                "model_sha256": self.model_sha256,
                "embedding_text_sha256": digest,
                "vector_dim": self.vector_dim,
                "embedding": vector,
                "created_at": datetime.now(UTC).isoformat(),
            },
            schema=CACHE_SCHEMA,
        )
        self._vectors[digest] = array("d", vector)

    def missing(self, inputs: Iterable[EmbeddingInput]) -> list[EmbeddingInput]:
        seen: set[str] = set()
        missing: list[EmbeddingInput] = []
        for item in inputs:
            digest = item.embedding_text_sha256
            if digest in seen or digest in self._vectors:
                continue
            seen.add(digest)
            missing.append(item)
        return missing

    def records_for(self, digests: Iterable[str]) -> list[dict[str, Any]]:
        """Full cache records of the cached `digests`, read again from the file."""
        requested = [digest for digest in digests if digest in self._vectors]
        wanted = set(requested)
        found: dict[str, dict[str, Any]] = {}
        for record in self._own_records():
            digest = str(record["embedding_text_sha256"])
            if digest in wanted:
                found[digest] = record
        return [found[digest] for digest in requested if digest in found]


def embed_missing(
    inputs: Sequence[EmbeddingInput],
    cache: TextEmbeddingCache,
    clients: Sequence[EmbeddingClient],
    *,
    batch_size: int,
    concurrency: int = 1,
    deadline: float = math.inf,
    clock: Callable[[], float] = time.monotonic,
) -> EmbedRun:
    total = len({item.embedding_text_sha256 for item in inputs})
    missing = cache.missing(inputs)
    cached = total - len(missing)
    if not missing:
        return EmbedRun(total=total, cached=cached, embedded=0, stopped_early=False)
    if not clients:
        raise TextEmbeddingError("embedding requires at least one model server")
    size = max(1, batch_size)
    batches = [missing[start : start + size] for start in range(0, len(missing), size)]
    embedded = 0

    async def operation(
        client: EmbeddingClient, _index: int, batch: list[EmbeddingInput]
    ) -> tuple[list[EmbeddingInput], list[list[float]]]:
        vectors = await asyncio.to_thread(
            client.embed,
            [item.embedding_text for item in batch],
            cache.model,
            cache.vector_dim,
        )
        return batch, vectors

    def on_completed(
        completed: list[tuple[int, tuple[list[EmbeddingInput], list[list[float]]]]],
    ) -> None:
        nonlocal embedded
        for _index, (batch, vectors) in completed:
            for item, vector in zip(batch, vectors, strict=True):
                cache.set(item.embedding_text_sha256, vector)
            embedded += len(batch)
        print(
            f"Embedding progress: {cached + embedded}/{total}, "
            f"cached={cached}, embedded={embedded}",
            flush=True,
        )

    scheduled = asyncio.run(
        stream_map_ordered(
            batches,
            list(clients),
            max(1, concurrency),
            operation,
            deadline,
            on_completed=on_completed,
            clock=clock,
        )
    )
    return EmbedRun(
        total=total,
        cached=cached,
        embedded=embedded,
        stopped_early=scheduled.stopped_early,
    )
