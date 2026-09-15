"""Deterministic, side-effect-free runtime benchmark primitives."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any

from seed_pipeline.runtime.model_profiles import (
    EmbeddingWorkloadProfile,
    RerankRuntimeProfile,
)


@dataclass(frozen=True, order=True)
class BenchmarkLevel:
    batch_size: int = 1
    concurrency: int = 1


@dataclass(frozen=True)
class BenchmarkWorkload:
    stage: str
    model: str
    sample_count: int
    levels: tuple[BenchmarkLevel, ...]
    sample_identity: str = ""


@dataclass(frozen=True)
class BenchmarkMeasurement:
    level: BenchmarkLevel
    items: int
    input_characters: int
    elapsed_seconds: float
    status: str = "ok"
    error_category: str | None = None
    warmup: bool = False
    latency_p95_seconds: float | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("items", self.items),
            ("input_characters", self.input_characters),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be finite and non-negative")
        if self.latency_p95_seconds is not None and (
            not math.isfinite(self.latency_p95_seconds) or self.latency_p95_seconds < 0
        ):
            raise ValueError("latency_p95_seconds must be finite and non-negative")
        if self.level.batch_size < 1 or self.level.concurrency < 1:
            raise ValueError("benchmark level values must be positive")

    @property
    def characters_per_second(self) -> float:
        return (
            self.input_characters / self.elapsed_seconds
            if self.elapsed_seconds
            else math.inf
        )


@dataclass(frozen=True)
class BenchmarkReport:
    workload: BenchmarkWorkload
    measurements: tuple[BenchmarkMeasurement, ...]
    recommendation: BenchmarkLevel | None
    schema_version: int = 1

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "workload": asdict(self.workload),
            "measurements": [asdict(item) for item in self.measurements],
            "recommendation": asdict(self.recommendation)
            if self.recommendation
            else None,
        }

    def to_json(self) -> str:
        return (
            json.dumps(self.to_payload(), ensure_ascii=False, sort_keys=True, indent=2)
            + "\n"
        )


def stratified_sample(items, count: int, key, length) -> tuple:
    if count < 1:
        raise ValueError("sample count must be positive")
    ordered = sorted(items, key=key)
    if not ordered:
        return ()
    strata: list[list] = [[] for _ in range(8)]
    lengths = [length(item) for item in ordered]
    low, high = min(lengths), max(lengths)
    span = max(high - low + 1, 1)
    for item, item_length in zip(ordered, lengths, strict=True):
        index = min(7, max(0, int((item_length - low) * 8 / span)))
        strata[index].append(item)
    nonempty = [bucket for bucket in strata if bucket]
    target = min(count, len(ordered))
    quotient, remainder = divmod(target, len(nonempty))
    selected = []
    for index, bucket in enumerate(nonempty):
        take = quotient + (index < remainder)
        if take >= len(bucket):
            selected.extend(bucket)
            continue
        for offset in range(take):
            selected.append(bucket[(offset * len(bucket)) // take])
    return tuple(sorted(selected, key=key))


def embedding_levels(profile: EmbeddingWorkloadProfile) -> tuple[BenchmarkLevel, ...]:
    return tuple(
        BenchmarkLevel(batch_size=batch, concurrency=concurrency)
        for batch in profile.benchmark_batch_sizes
        for concurrency in profile.benchmark_concurrency
    )


def rerank_levels(profile: RerankRuntimeProfile) -> tuple[BenchmarkLevel, ...]:
    return tuple(
        BenchmarkLevel(concurrency=value) for value in profile.benchmark_concurrency
    )


def recommend(
    measurements: list[BenchmarkMeasurement] | tuple[BenchmarkMeasurement, ...],
) -> BenchmarkLevel | None:
    valid = [
        item
        for item in measurements
        if item.status == "ok"
        and not item.warmup
        and math.isfinite(item.characters_per_second)
    ]
    if not valid:
        return None
    best_rate = max(item.characters_per_second for item in valid)
    tied = [item for item in valid if item.characters_per_second >= best_rate * 0.98]
    return min(
        tied, key=lambda item: (item.level.concurrency, item.level.batch_size)
    ).level
