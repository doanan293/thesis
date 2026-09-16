"""Deterministic, side-effect-free runtime benchmark primitives."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

from pharma_lab.runtime.runtime_profiles import RuntimeCandidate

KAGGLE_RERANK_BENCHMARK_GROUPS = 32
LOCAL_RERANK_BENCHMARK_GROUPS = 6
LOG_TAIL_CHARACTERS = 16_000

BenchmarkObjective = Literal["throughput", "latency"]


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    raise ValueError("benchmark measurement number is invalid")


def _count(value: object) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise ValueError("benchmark measurement count is invalid")


def _optional_text(value: object) -> str | None:
    return None if value is None else str(value)


@dataclass(frozen=True)
class BenchmarkWorkload:
    stage: str
    model: str
    sample_count: int
    levels: tuple[RuntimeCandidate, ...]


@dataclass(frozen=True)
class BenchmarkMeasurement:
    candidate: RuntimeCandidate
    items: int
    input_characters: int
    elapsed_seconds: float
    status: str = "ok"
    error_category: str | None = None
    latency_p50_seconds: float | None = None
    latency_p95_seconds: float | None = None
    max_abs_score_delta: float | None = None
    log_tail: str | None = None

    def __post_init__(self) -> None:
        if self.items < 0 or self.input_characters < 0:
            raise ValueError("benchmark counts must be non-negative")
        for name, value in (
            ("elapsed_seconds", self.elapsed_seconds),
            ("latency_p50_seconds", self.latency_p50_seconds),
            ("latency_p95_seconds", self.latency_p95_seconds),
            ("max_abs_score_delta", self.max_abs_score_delta),
        ):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and non-negative")

    @classmethod
    def invalid(
        cls,
        candidate: RuntimeCandidate,
        error_category: str,
        log_tail: str | None = None,
    ) -> BenchmarkMeasurement:
        return cls(
            candidate,
            0,
            0,
            0.0,
            status="invalid",
            error_category=error_category,
            log_tail=log_tail[-LOG_TAIL_CHARACTERS:] if log_tail else None,
        )

    @property
    def items_per_second(self) -> float:
        return self.items / self.elapsed_seconds if self.elapsed_seconds else math.inf

    @property
    def characters_per_second(self) -> float:
        return (
            self.input_characters / self.elapsed_seconds
            if self.elapsed_seconds
            else math.inf
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate": self.candidate.to_dict(),
            "items": self.items,
            "input_characters": self.input_characters,
            "elapsed_seconds": self.elapsed_seconds,
            "status": self.status,
            "error_category": self.error_category,
            "latency_p50_seconds": self.latency_p50_seconds,
            "latency_p95_seconds": self.latency_p95_seconds,
            "max_abs_score_delta": self.max_abs_score_delta,
            "log_tail": self.log_tail,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> BenchmarkMeasurement:
        candidate = payload.get("candidate")
        if not isinstance(candidate, Mapping):
            raise ValueError("benchmark measurement has no candidate")
        elapsed = _optional_float(payload.get("elapsed_seconds"))
        if elapsed is None:
            raise ValueError("benchmark measurement has no elapsed time")
        return cls(
            RuntimeCandidate.from_dict(candidate),
            items=_count(payload.get("items")),
            input_characters=_count(payload.get("input_characters")),
            elapsed_seconds=elapsed,
            status=str(payload.get("status", "")),
            error_category=_optional_text(payload.get("error_category")),
            latency_p50_seconds=_optional_float(payload.get("latency_p50_seconds")),
            latency_p95_seconds=_optional_float(payload.get("latency_p95_seconds")),
            max_abs_score_delta=_optional_float(payload.get("max_abs_score_delta")),
            log_tail=_optional_text(payload.get("log_tail")),
        )


@dataclass(frozen=True)
class BenchmarkReport:
    workload: BenchmarkWorkload
    measurements: tuple[BenchmarkMeasurement, ...]
    recommendation: RuntimeCandidate | None
    schema_version: int = 2


@dataclass(frozen=True)
class RerankGroup:
    """One /v1/rerank request: a query with its first candidates."""

    query_id: str
    query: str
    chunk_ids: tuple[str, ...]
    documents: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.documents or len(self.chunk_ids) != len(self.documents):
            raise ValueError("rerank group needs one chunk id per document")

    @property
    def characters(self) -> int:
        return sum(len(self.query) + len(document) for document in self.documents)

    def score_keys(self) -> tuple[str, ...]:
        return tuple(f"{self.query_id}\x1f{chunk_id}" for chunk_id in self.chunk_ids)


def rerank_groups_from_rows(
    rows: Iterable[Mapping[str, Any]], *, documents_per_group: int
) -> list[RerankGroup]:
    """Full query groups (exactly `documents_per_group` candidates) from candidate rows."""
    if documents_per_group < 1:
        raise ValueError("documents_per_group must be positive")
    groups: list[RerankGroup] = []
    for row in rows:
        candidates = list(row["candidates"])[:documents_per_group]
        if len(candidates) < documents_per_group:
            continue
        groups.append(
            RerankGroup(
                str(row["query_id"]),
                str(row.get("query", "")),
                tuple(str(item["chunk_id"]) for item in candidates),
                tuple(str(item.get("document_text", "")) for item in candidates),
            )
        )
    return groups


def sample_rerank_groups(
    groups: Sequence[RerankGroup], count: int
) -> tuple[RerankGroup, ...]:
    """The middle group of `count` equal strata ordered by total characters."""
    if count < 1:
        raise ValueError("sample count must be positive")
    ordered = sorted(groups, key=lambda group: (group.characters, group.query_id))
    if len(ordered) <= count:
        return tuple(ordered)
    return tuple(
        ordered[(2 * index + 1) * len(ordered) // (2 * count)] for index in range(count)
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


def percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


@dataclass(frozen=True)
class LevelResult:
    measurement: BenchmarkMeasurement
    # "<query_id>\x1f<chunk_id>" -> score; empty for embeddings and failed levels.
    scores: Mapping[str, float]


def _max_abs_delta(
    baseline: Mapping[str, float], scores: Mapping[str, float]
) -> float | None:
    if set(baseline) != set(scores):
        return None
    return max((abs(scores[key] - baseline[key]) for key in baseline), default=0.0)


def _ranking(scores: Mapping[str, float]) -> dict[str, tuple[str, ...]]:
    """Candidate order per query: best score first, ties broken by chunk id."""
    grouped: dict[str, list[tuple[float, str]]] = {}
    for key, score in scores.items():
        query_id, _, chunk_id = key.partition("\x1f")
        grouped.setdefault(query_id, []).append((score, chunk_id))
    return {
        query_id: tuple(
            chunk_id
            for _score, chunk_id in sorted(items, key=lambda item: (-item[0], item[1]))
        )
        for query_id, items in grouped.items()
    }


def check_score_consistency(
    results: Sequence[LevelResult],
) -> tuple[BenchmarkMeasurement, ...]:
    """Mark levels that rank the benchmark candidates differently from the first level.

    A score depends on how llama.cpp groups documents into a batch, so two levels of one
    model differ by far more than float noise (0.02 to 0.07 measured on a T4). The
    pipeline only uses the scores to order the candidates of a query, so a level is
    rejected when that order changes, and the largest score difference is recorded either
    way. A level that scored other pairs than the first level is rejected as well.
    """
    baseline: Mapping[str, float] | None = None
    baseline_ranking: dict[str, tuple[str, ...]] = {}
    checked: list[BenchmarkMeasurement] = []
    for result in results:
        measurement = result.measurement
        if measurement.status != "ok" or not result.scores:
            checked.append(measurement)
            continue
        if baseline is None:
            baseline = result.scores
            baseline_ranking = _ranking(baseline)
            checked.append(replace(measurement, max_abs_score_delta=0.0))
            continue
        delta = _max_abs_delta(baseline, result.scores)
        if delta is None:
            checked.append(
                replace(
                    measurement,
                    status="invalid",
                    error_category="score_mismatch",
                    max_abs_score_delta=None,
                )
            )
        elif _ranking(result.scores) != baseline_ranking:
            checked.append(
                replace(
                    measurement,
                    status="invalid",
                    error_category="rank_mismatch",
                    max_abs_score_delta=delta,
                )
            )
        else:
            checked.append(replace(measurement, max_abs_score_delta=delta))
    return tuple(checked)


def recommend(
    measurements: Sequence[BenchmarkMeasurement], *, objective: BenchmarkObjective
) -> RuntimeCandidate | None:
    valid = [
        item
        for item in measurements
        if item.status == "ok"
        and item.items > 0
        and math.isfinite(item.items_per_second)
    ]
    if objective == "throughput":
        if not valid:
            return None
        return max(valid, key=lambda item: item.items_per_second).candidate
    timed = [
        (item.latency_p95_seconds, item)
        for item in valid
        if item.latency_p95_seconds is not None
    ]
    if not timed:
        return None
    return min(timed, key=lambda pair: (pair[0], -pair[1].items_per_second))[
        1
    ].candidate


def describe_levels(measurements: Sequence[BenchmarkMeasurement]) -> str:
    lines: list[str] = []
    for item in measurements:
        lines.append(
            f"- {json.dumps(item.candidate.to_dict(), sort_keys=True)} "
            f"status={item.status} error={item.error_category}"
        )
        if item.log_tail:
            lines.append(item.log_tail)
    return "\n".join(lines)
