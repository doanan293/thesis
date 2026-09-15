from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Callable
from pathlib import Path

from seed_pipeline.integrations.kaggle.workers.runtime import (
    artifact_from_output,
    identity_from_config,
    managed_model_servers,
    resolve_input_file,
)
from seed_pipeline.integrations.kaggle.workers.scheduling import stream_map_ordered
from seed_pipeline.integrations.kaggle.workers.telemetry import RuntimeTelemetry
from seed_pipeline.runtime.benchmarking import (
    BenchmarkLevel,
    BenchmarkMeasurement,
    BenchmarkReport,
    BenchmarkWorkload,
    recommend,
)
from seed_pipeline.runtime.catalog import require_model


def run_benchmark_worker(
    config: dict,
    *,
    measure_level: Callable[[BenchmarkLevel], BenchmarkMeasurement] | None = None,
    clock: Callable[[], float] = time.monotonic,
):
    """Run isolated diagnostic levels; never creates a production cache."""
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    identity = identity_from_config(config)
    levels = tuple(
        BenchmarkLevel(
            batch_size=int(item.get("batch_size", 1)),
            concurrency=int(item.get("concurrency", 1)),
        )
        for item in config.get("benchmark_levels", ())
    )
    if not levels:
        raise ValueError("benchmark_levels must not be empty")
    workload = BenchmarkWorkload(
        stage=str(config.get("stage", "benchmark")),
        model=str(config["model"]),
        sample_count=int(
            config.get("benchmark_items", config.get("benchmark_pairs", 0))
        ),
        levels=levels,
        sample_identity=str(config.get("sample_identity", "")),
    )
    measurements: list[BenchmarkMeasurement] = []
    for level in levels:
        try:
            measurement = (
                measure_level(level)
                if measure_level is not None
                else _measure_level(
                    config, level, workload.sample_count, output_dir, clock
                )
            )
            if measurement.level != level:
                raise ValueError("benchmark measurement level mismatch")
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            measurement = BenchmarkMeasurement(
                level, 0, 0, 0.0, status="invalid", error_category=type(exc).__name__
            )
        measurements.append(measurement)
    report = BenchmarkReport(workload, tuple(measurements), recommend(measurements))
    data_path = output_dir / "benchmark_results.jsonl"
    with data_path.open("w", encoding="utf-8") as handle:
        for measurement in measurements:
            handle.write(
                json.dumps(
                    {
                        "identity": identity.sha256,
                        **{
                            key: value
                            for key, value in report.to_payload().items()
                            if key == "schema_version"
                        },
                        "measurement": {
                            "level": {
                                "batch_size": measurement.level.batch_size,
                                "concurrency": measurement.level.concurrency,
                            },
                            "items": measurement.items,
                            "input_characters": measurement.input_characters,
                            "elapsed_seconds": measurement.elapsed_seconds,
                            "status": measurement.status,
                            "error_category": measurement.error_category,
                        },
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    (output_dir / "benchmark_report.md").write_text(
        "# Runtime benchmark\n\n"
        f"- stage: `{workload.stage}`\n- model: `{workload.model}`\n"
        f"- recommendation: `{report.recommendation}`\n\n"
        + "| batch | concurrency | items | chars/s | status | error |\n"
        + "|---:|---:|---:|---:|---|---|\n"
        + "\n".join(
            f"| {item.level.batch_size} | {item.level.concurrency} | {item.items} | "
            f"{item.characters_per_second:.2f} | {item.status} | {item.error_category or ''} |"
            for item in measurements
        )
        + "\n",
        encoding="utf-8",
    )
    return artifact_from_output(
        data_path,
        artifact_type="runtime_benchmark",
        identity=identity,
        total=len(measurements),
        complete=len(measurements),
        runtime_summary={
            "recommendation": report.recommendation
            and {
                "batch_size": report.recommendation.batch_size,
                "concurrency": report.recommendation.concurrency,
            }
        },
    )


def _sample_workload(config: dict, count: int) -> tuple[list, int]:
    from seed_pipeline.runtime.benchmarking import stratified_sample

    stage = str(config.get("stage", ""))
    if "rerank" in stage:
        path = resolve_input_file(config, "candidates")
        pairs = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                for candidate in row["candidates"]:
                    pairs.append(
                        (
                            str(row.get("query", "")),
                            str(candidate.get("document_text", "")),
                        )
                    )
        sample = stratified_sample(
            pairs,
            count,
            key=lambda item: (item[0], item[1]),
            length=lambda item: len(item[0]) + len(item[1]),
        )
        return list(sample), sum(
            len(query) + len(document) for query, document in sample
        )
    path = resolve_input_file(config, "input")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            text = str(
                row.get("query", "")
                if "query" in stage
                else row.get("embedding_text", row.get("text", ""))
            )
            rows.append(text)
    sample = stratified_sample(
        rows, count, key=lambda item: (len(item), item), length=len
    )
    return list(sample), sum(len(item) for item in sample)


def _measure_level(
    config: dict,
    level: BenchmarkLevel,
    sample_count: int,
    output_dir: Path,
    clock,
) -> BenchmarkMeasurement:
    spec = require_model(str(config["model"]))
    candidates = [dict(item) for item in config.get("benchmark_candidates", ())]
    matching = [
        item
        for item in candidates
        if int(item.get("request_batch_size", 0)) == level.batch_size
        and int(item.get("concurrency", 0)) == level.concurrency
    ]
    if len(matching) != 1:
        raise ValueError("benchmark level does not identify one runtime candidate")
    candidate = matching[0]
    samples, input_characters = _sample_workload(config, sample_count)
    level_config = dict(config)
    level_config["runtime_overrides"] = candidate
    telemetry = RuntimeTelemetry("benchmark", str(config["model"]), output_dir)
    try:
        with managed_model_servers(level_config, telemetry=telemetry) as servers:
            resources = [
                (
                    index,
                    __import__(
                        "seed_pipeline.runtime.client", fromlist=["LlamaCppClient"]
                    ).LlamaCppClient(server.base_url),
                )
                for index, server in enumerate(servers)
            ]
            recording = False
            if "rerank" in str(config.get("stage", "")):
                items = samples

                async def operation(resource, _index, item):
                    server_index, client = resource
                    started = time.monotonic()
                    query, document = item
                    await asyncio.to_thread(
                        client.rerank_native,
                        query,
                        [document],
                        str(config["model"]),
                    )
                    if recording:
                        telemetry.record_operation(
                            server_index,
                            1,
                            len(query) + len(document),
                            time.monotonic() - started,
                            "ok",
                            0,
                        )
                    return None
            else:
                batches = [
                    samples[start : start + level.batch_size]
                    for start in range(0, len(samples), level.batch_size)
                ]

                async def operation(resource, _index, batch):
                    server_index, client = resource
                    started = time.monotonic()
                    await asyncio.to_thread(
                        client.embed,
                        batch,
                        str(config["model"]),
                        spec.vector_dimension or 0,
                    )
                    if recording:
                        telemetry.record_operation(
                            server_index,
                            len(batch),
                            sum(len(item) for item in batch),
                            time.monotonic() - started,
                            "ok",
                            0,
                        )
                    return None

                items = batches
            if items:
                asyncio.run(
                    stream_map_ordered(items[:1], resources, 1, operation, float("inf"))
                )
            recording = True
            started_total = clock()
            if items:
                asyncio.run(
                    stream_map_ordered(
                        items, resources, level.concurrency, operation, float("inf")
                    )
                )
            summary = telemetry.summary()
            return BenchmarkMeasurement(
                level,
                len(samples),
                input_characters,
                max(clock() - started_total, 0.0),
                status="ok",
                latency_p95_seconds=summary["operations"].get("latency_p95_seconds"),
            )
    finally:
        telemetry.close()
        telemetry.write_report()


def main() -> int:
    config = json.loads(
        Path(os.environ["KAGGLE_PIPELINE_CONFIG"]).read_text(encoding="utf-8")
    )
    run_benchmark_worker(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
