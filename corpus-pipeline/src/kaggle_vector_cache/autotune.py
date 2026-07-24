from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

AUTOTUNE_SCHEMA_VERSION = 2
DEFAULT_BUDGET_SECONDS = 600


class AutotuneError(RuntimeError):
    pass


@dataclass(frozen=True)
class AutotuneIdentity:
    model: str
    corpus_sha256: str
    gpu_names: tuple[str, ...]
    runtime_revision: str
    topology: str
    candidates: tuple[int, ...]


@dataclass(frozen=True)
class CandidateMeasurement:
    parallel: int
    request_batch_size: int
    success: bool
    chunks_per_second: float
    elapsed_seconds: float = 0.0
    successful_chunks: int = 0
    latency_p50_seconds: float | None = None
    latency_p95_seconds: float | None = None
    gpu_utilization_average: tuple[float | None, ...] = ()
    gpu_utilization_p95: tuple[float | None, ...] = ()
    peak_memory_mib: tuple[float | None, ...] = ()
    peak_memory_percent: tuple[float | None, ...] = ()
    error: str | None = None


def _json_value(value):
    return json.loads(json.dumps(value))


def identity_record(identity: AutotuneIdentity) -> dict:
    return _json_value(asdict(identity))


def profile_is_compatible(profile: dict, identity: AutotuneIdentity) -> bool:
    if not isinstance(profile, dict):
        return False
    if profile.get("schema_version") != AUTOTUNE_SCHEMA_VERSION:
        return False
    if profile.get("identity") != identity_record(identity):
        return False
    selected = profile.get("selected_parallel")
    return isinstance(selected, int) and selected in identity.candidates


def select_best_candidate(
    measurements: list[CandidateMeasurement],
) -> CandidateMeasurement:
    successful = [item for item in measurements if item.success]
    if not successful:
        raise AutotuneError("No autotune candidate succeeded")
    maximum = max(item.chunks_per_second for item in successful)
    near_best = [
        item for item in successful if item.chunks_per_second >= maximum * 0.97
    ]
    return min(near_best, key=lambda item: item.parallel)


def autotune(
    identity: AutotuneIdentity,
    sample_size: int,
    candidates: tuple[int, ...],
    budget_seconds: float,
    measure: Callable[[int, float], CandidateMeasurement],
    clock: Callable[[], float] = time.monotonic,
) -> dict:
    if not candidates or tuple(sorted(candidates)) != candidates:
        raise AutotuneError("Autotune candidates must be non-empty and ordered")
    if any(value < 1 for value in candidates):
        raise AutotuneError("Autotune candidates must be positive")
    if tuple(candidates) != identity.candidates:
        raise AutotuneError("Autotune candidates do not match profile identity")

    started_at = clock()
    measurements: list[CandidateMeasurement] = []
    previous: CandidateMeasurement | None = None
    for parallel in candidates:
        elapsed = clock() - started_at
        remaining = budget_seconds - elapsed
        if remaining <= 0:
            break
        result = measure(parallel, remaining)
        measurements.append(result)
        if not result.success:
            break
        if any(
            value is not None and value >= 90.0 for value in result.peak_memory_percent
        ):
            break
        if previous is not None and previous.chunks_per_second > 0:
            improvement = (
                result.chunks_per_second - previous.chunks_per_second
            ) / previous.chunks_per_second
            if improvement < 0.05:
                break
        previous = result

    selected = select_best_candidate(measurements)
    return {
        "schema_version": AUTOTUNE_SCHEMA_VERSION,
        "identity": identity_record(identity),
        "selected_parallel": selected.parallel,
        "selected_batch_size": selected.request_batch_size,
        "sample_size": int(sample_size),
        "budget_seconds": float(budget_seconds),
        "benchmark_duration_seconds": max(0.0, clock() - started_at),
        "measurements": [_json_value(asdict(item)) for item in measurements],
        "created_at": datetime.now(UTC).isoformat(),
    }
