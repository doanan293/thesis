from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from corpus_pipeline.runtime.catalog import ModelKind, ModelSpec


@dataclass(frozen=True)
class InferenceCachePolicy:
    host_cache_ram_mib: int
    cache_idle_slots: bool
    completion_cache_prompt: bool | None
    slot_prompt_similarity: float | None
    workload_locality: str

    def __post_init__(self) -> None:
        if self.host_cache_ram_mib < 0:
            raise ValueError("host_cache_ram_mib must be non-negative")
        if (
            self.slot_prompt_similarity is not None
            and not 0.0 <= self.slot_prompt_similarity <= 1.0
        ):
            raise ValueError("slot_prompt_similarity must be between 0 and 1")
        if not self.workload_locality.strip():
            raise ValueError("workload_locality must not be empty")

    def arguments(self) -> tuple[str, ...]:
        arguments = ["--cache-ram", str(self.host_cache_ram_mib)]
        arguments.append(
            "--cache-idle-slots" if self.cache_idle_slots else "--no-cache-idle-slots"
        )
        if self.slot_prompt_similarity is not None:
            arguments.extend(
                ("--slot-prompt-similarity", str(self.slot_prompt_similarity))
            )
        return tuple(arguments)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "host_cache_ram_mib": self.host_cache_ram_mib,
            "cache_idle_slots": self.cache_idle_slots,
            "completion_cache_prompt": self.completion_cache_prompt,
            "slot_prompt_similarity": self.slot_prompt_similarity,
            "workload_locality": self.workload_locality,
        }

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


def inference_cache_policy(spec: ModelSpec) -> InferenceCachePolicy:
    if (
        spec.kind is ModelKind.RERANKER
        and spec.reranker_protocol == "completion_logprobs"
    ):
        return InferenceCachePolicy(0, False, True, 0.1, "query-adjacent-v1")
    if spec.kind is ModelKind.RERANKER:
        return InferenceCachePolicy(0, False, None, None, "query-group-request-v1")
    return InferenceCachePolicy(0, False, None, None, "batch-independent-v1")
