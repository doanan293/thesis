from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from seed_pipeline.runtime.catalog import ModelKind, ModelSpec


@dataclass(frozen=True)
class InferenceCachePolicy:
    host_cache_ram_mib: int
    cache_idle_slots: bool
    workload_locality: str

    def __post_init__(self) -> None:
        if self.host_cache_ram_mib < 0:
            raise ValueError("host_cache_ram_mib must be non-negative")
        if not self.workload_locality.strip():
            raise ValueError("workload_locality must not be empty")

    def arguments(self) -> tuple[str, ...]:
        return (
            "--cache-ram",
            str(self.host_cache_ram_mib),
            "--cache-idle-slots" if self.cache_idle_slots else "--no-cache-idle-slots",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "host_cache_ram_mib": self.host_cache_ram_mib,
            "cache_idle_slots": self.cache_idle_slots,
            "workload_locality": self.workload_locality,
        }

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


def inference_cache_policy(spec: ModelSpec) -> InferenceCachePolicy:
    if spec.kind is ModelKind.RERANKER:
        return InferenceCachePolicy(0, False, "query-group-request-v1")
    return InferenceCachePolicy(0, False, "batch-independent-v1")
