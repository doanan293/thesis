from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


NATIVE_RERANK_PROTOCOL = "native_rerank"


@dataclass(frozen=True)
class RerankContract:
    """How a reranker is called: POST /v1/rerank with the template inside the GGUF.

    The model sha256 already covers that template, so the contract names only the
    protocol.
    """

    protocol: str = NATIVE_RERANK_PROTOCOL

    def __post_init__(self) -> None:
        if self.protocol != NATIVE_RERANK_PROTOCOL:
            raise ValueError(f"unsupported rerank protocol: {self.protocol}")

    def canonical_payload(self) -> dict[str, object]:
        # Score caches and rerank variant identities hash this payload, so it keeps the
        # shape it had when contracts also described completion prompts.
        return {
            "protocol": self.protocol,
            "template_id": None,
            "template_version": None,
            "instruction": "",
            "scoring": None,
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.canonical_payload())


@dataclass(frozen=True)
class RerankRuntimeProfile:
    server_slots_per_gpu: int
    concurrency_per_gpu: int
    context_per_slot: int
    logical_batch_size: int
    physical_batch_size: int
    benchmark_concurrency: tuple[int, ...]

    def __post_init__(self) -> None:
        values = (
            self.server_slots_per_gpu,
            self.concurrency_per_gpu,
            self.context_per_slot,
            self.logical_batch_size,
            self.physical_batch_size,
        )
        if any(value < 1 for value in values):
            raise ValueError("rerank runtime values must be positive")
        if self.concurrency_per_gpu > self.server_slots_per_gpu:
            raise ValueError("rerank concurrency cannot exceed server slots")
        if not self.benchmark_concurrency or any(
            value < 1 for value in self.benchmark_concurrency
        ):
            raise ValueError("rerank benchmark concurrency must be positive")


@dataclass(frozen=True)
class EmbeddingWorkloadProfile:
    production_batch_size: int
    production_concurrency: int
    benchmark_batch_sizes: tuple[int, ...]
    benchmark_concurrency: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.production_batch_size < 1 or self.production_concurrency < 1:
            raise ValueError("embedding production values must be positive")
        if not self.benchmark_batch_sizes or any(
            value < 1 for value in self.benchmark_batch_sizes
        ):
            raise ValueError("embedding benchmark batch sizes must be positive")
        if not self.benchmark_concurrency or any(
            value < 1 for value in self.benchmark_concurrency
        ):
            raise ValueError("embedding benchmark concurrency must be positive")


@dataclass(frozen=True)
class EmbeddingRuntimeProfile:
    query: EmbeddingWorkloadProfile
    corpus: EmbeddingWorkloadProfile
    context_per_slot: int
    logical_batch_size: int
    physical_batch_size: int

    def __post_init__(self) -> None:
        if (
            min(
                self.context_per_slot, self.logical_batch_size, self.physical_batch_size
            )
            < 1
        ):
            raise ValueError("embedding runtime values must be positive")


def native_rerank_contract() -> RerankContract:
    return RerankContract()
