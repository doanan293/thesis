from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from pathlib import Path

from pharma_lab.integrations.kaggle.models import CloudArtifact
from pharma_lab.integrations.kaggle.workers.runtime import (
    artifact_from_output,
    identity_from_config,
    managed_model_servers,
    resolve_input_file,
)
from pharma_lab.integrations.kaggle.workers.scheduling import stream_map_ordered
from pharma_lab.integrations.kaggle.workers.telemetry import RuntimeTelemetry
from pharma_lab.runtime.benchmarking import (
    BenchmarkMeasurement,
    BenchmarkReport,
    BenchmarkWorkload,
    LevelResult,
    RerankGroup,
    check_score_consistency,
    percentile,
    recommend,
    rerank_groups_from_rows,
    sample_rerank_groups,
    stratified_sample,
)
from pharma_lab.runtime.catalog import ModelKind, ModelSpec, require_model
from pharma_lab.runtime.client import LlamaCppClient
from pharma_lab.runtime.runtime_profiles import RuntimeCandidate

MeasureLevel = Callable[[int, RuntimeCandidate], LevelResult]
Clients = list[tuple[int, LlamaCppClient]]
SERVER_LOG_TAIL_CHARACTERS = 4000


def run_benchmark_worker(
    config: dict,
    *,
    measure_level: MeasureLevel | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> CloudArtifact:
    """Measure every runtime level end to end; never creates a production cache."""
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    identity = identity_from_config(config)
    levels = tuple(
        RuntimeCandidate.from_dict(item) for item in config.get("benchmark_levels", ())
    )
    if not levels:
        raise ValueError("benchmark_levels must not be empty")
    workload = BenchmarkWorkload(
        stage=str(config.get("stage", "benchmark")),
        model=str(config["model"]),
        sample_count=int(config.get("benchmark_items", 0)),
        levels=levels,
    )
    measure = measure_level or _default_measure(config, levels, output_dir, clock)
    results: list[LevelResult] = []
    for index, candidate in enumerate(levels):
        try:
            result = measure(index, candidate)
            if result.measurement.candidate != candidate:
                raise ValueError("benchmark measurement level mismatch")
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            detail = f"{exc}\n{_server_log_tail(output_dir / f'level-{index}')}"
            result = LevelResult(
                BenchmarkMeasurement.invalid(
                    candidate, type(exc).__name__, detail.strip()
                ),
                {},
            )
        results.append(result)
    measurements = check_score_consistency(results)
    report = BenchmarkReport(
        workload, measurements, recommend(measurements, objective="throughput")
    )
    data_path = output_dir / "benchmark_results.jsonl"
    with data_path.open("w", encoding="utf-8") as handle:
        for measurement in measurements:
            row = {
                "identity": identity.sha256,
                "schema_version": report.schema_version,
                "measurement": measurement.to_dict(),
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    (output_dir / "benchmark_report.md").write_text(_markdown(report), encoding="utf-8")
    return artifact_from_output(
        data_path,
        artifact_type="runtime_benchmark",
        identity=identity,
        total=len(measurements),
        complete=len(measurements),
        runtime_summary={
            "recommendation": (
                report.recommendation.to_dict() if report.recommendation else None
            ),
            "sample_count": workload.sample_count,
        },
    )


def _default_measure(
    config: dict,
    levels: tuple[RuntimeCandidate, ...],
    output_dir: Path,
    clock: Callable[[], float],
) -> MeasureLevel:
    spec = require_model(str(config["model"]))
    if spec.kind is ModelKind.RERANKER:
        groups = sample_rerank_groups(
            rerank_groups_from_rows(
                _candidate_rows(config),
                documents_per_group=levels[0].request_batch_size,
            ),
            int(config["benchmark_groups"]),
        )
        if not groups:
            raise ValueError("candidates contain no full query group to benchmark")

        def measure_rerank(index: int, candidate: RuntimeCandidate) -> LevelResult:
            with _level_clients(config, spec, index, candidate, output_dir) as clients:
                return _measure_rerank(spec, candidate, groups, clients, clock)

        return measure_rerank
    texts, characters = _sample_texts(config, int(config.get("benchmark_items", 0)))

    def measure_embedding(index: int, candidate: RuntimeCandidate) -> LevelResult:
        with _level_clients(config, spec, index, candidate, output_dir) as clients:
            return _measure_embedding(
                spec, candidate, texts, characters, clients, clock
            )

    return measure_embedding


@contextmanager
def _level_clients(
    config: dict,
    spec: ModelSpec,
    index: int,
    candidate: RuntimeCandidate,
    output_dir: Path,
) -> Generator[Clients, None, None]:
    level_dir = output_dir / f"level-{index}"
    level_config = {
        **config,
        "runtime_overrides": candidate.to_dict(),
        "output_dir": str(level_dir),
    }
    telemetry = RuntimeTelemetry("benchmark", spec.name, level_dir)
    try:
        # A fresh server per level, so no slot or KV state carries over.
        with managed_model_servers(level_config, telemetry=telemetry) as servers:
            yield [
                (server_index, LlamaCppClient(server.base_url))
                for server_index, server in enumerate(servers)
            ]
    finally:
        telemetry.close()
        telemetry.write_report()


def _measure_rerank(
    spec: ModelSpec,
    candidate: RuntimeCandidate,
    groups: tuple[RerankGroup, ...],
    clients: Clients,
    clock: Callable[[], float],
) -> LevelResult:
    latencies: list[float] = []
    scores: dict[str, float] = {}

    async def operation(
        resource: tuple[int, LlamaCppClient], _index: int, group: RerankGroup
    ) -> None:
        _server_index, client = resource
        started = clock()
        values = await asyncio.to_thread(
            client.rerank_native, group.query, list(group.documents), spec.name
        )
        latencies.append(max(0.0, clock() - started))
        scores.update(zip(group.score_keys(), values, strict=True))

    asyncio.run(stream_map_ordered(groups[:1], clients, 1, operation, float("inf")))
    latencies.clear()
    scores.clear()
    started = clock()
    asyncio.run(
        stream_map_ordered(
            groups, clients, candidate.concurrency, operation, float("inf")
        )
    )
    elapsed = max(0.0, clock() - started)
    return LevelResult(
        BenchmarkMeasurement(
            candidate,
            sum(len(group.documents) for group in groups),
            sum(group.characters for group in groups),
            elapsed,
            latency_p50_seconds=percentile(latencies, 0.50),
            latency_p95_seconds=percentile(latencies, 0.95),
        ),
        scores,
    )


def _measure_embedding(
    spec: ModelSpec,
    candidate: RuntimeCandidate,
    texts: list[str],
    characters: int,
    clients: Clients,
    clock: Callable[[], float],
) -> LevelResult:
    size = candidate.request_batch_size
    batches = [texts[start : start + size] for start in range(0, len(texts), size)]
    if not batches:
        raise ValueError("benchmark input contains no text")
    latencies: list[float] = []

    async def operation(
        resource: tuple[int, LlamaCppClient], _index: int, batch: list[str]
    ) -> None:
        _server_index, client = resource
        started = clock()
        await asyncio.to_thread(
            client.embed, batch, spec.name, spec.vector_dimension or 0
        )
        latencies.append(max(0.0, clock() - started))

    asyncio.run(stream_map_ordered(batches[:1], clients, 1, operation, float("inf")))
    latencies.clear()
    started = clock()
    asyncio.run(
        stream_map_ordered(
            batches, clients, candidate.concurrency, operation, float("inf")
        )
    )
    return LevelResult(
        BenchmarkMeasurement(
            candidate,
            len(texts),
            characters,
            max(0.0, clock() - started),
            latency_p50_seconds=percentile(latencies, 0.50),
            latency_p95_seconds=percentile(latencies, 0.95),
        ),
        {},
    )


def _candidate_rows(config: dict) -> Iterator[dict]:
    path = resolve_input_file(config, "candidates")
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _sample_texts(config: dict, count: int) -> tuple[list[str], int]:
    stage = str(config.get("stage", ""))
    path = resolve_input_file(config, "input")
    rows: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(
                str(
                    row.get("query", "")
                    if "query" in stage
                    else row.get("embedding_text", row.get("text", ""))
                )
            )
    sample = stratified_sample(
        rows, count, key=lambda item: (len(item), item), length=len
    )
    return list(sample), sum(len(item) for item in sample)


def _server_log_tail(level_dir: Path) -> str:
    return "\n".join(
        f"{path.name}:\n"
        + path.read_text(encoding="utf-8", errors="replace")[
            -SERVER_LOG_TAIL_CHARACTERS:
        ]
        for path in sorted(level_dir.glob("server-*.log"))
    )


def _seconds(value: float | None) -> str:
    return "" if value is None else f"{value:.3f}"


def _markdown(report: BenchmarkReport) -> str:
    recommendation = (
        json.dumps(report.recommendation.to_dict(), sort_keys=True)
        if report.recommendation
        else "none"
    )
    rows = [
        f"| {item.candidate.server_slots} | {item.candidate.physical_batch_size} "
        f"| {item.candidate.request_batch_size} | {item.candidate.concurrency} "
        f"| {item.items} | {item.items_per_second:.2f} "
        f"| {_seconds(item.latency_p50_seconds)} | {_seconds(item.latency_p95_seconds)} "
        f"| {item.status} | {item.error_category or ''} |"
        for item in report.measurements
    ]
    return (
        "# Runtime benchmark\n\n"
        f"- stage: `{report.workload.stage}`\n"
        f"- model: `{report.workload.model}`\n"
        f"- recommendation: `{recommendation}`\n\n"
        "| slots | ubatch | batch | concurrency | items | items/s | p50 s | p95 s "
        "| status | error |\n"
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|---|\n" + "\n".join(rows) + "\n"
    )


def main() -> int:
    config = json.loads(
        Path(os.environ["KAGGLE_PIPELINE_CONFIG"]).read_text(encoding="utf-8")
    )
    run_benchmark_worker(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
