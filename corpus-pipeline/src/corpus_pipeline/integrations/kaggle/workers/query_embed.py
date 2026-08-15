from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

from corpus_pipeline.artifacts.jsonl import iter_jsonl_objects
from corpus_pipeline.cache.jsonl_records import seal_record
from corpus_pipeline.evaluation.query_hash import query_hash
from corpus_pipeline.integrations.kaggle.models import CloudArtifact
from corpus_pipeline.integrations.kaggle.parsers import format_elapsed
from corpus_pipeline.integrations.kaggle.workers.checkpointing import (
    AppendOnlyJournal,
)
from corpus_pipeline.integrations.kaggle.workers.runtime import (
    artifact_from_output,
    identity_from_config,
    managed_model_servers,
    resolve_input_file,
    resolve_optional_input_file,
    worker_deadline,
)
from corpus_pipeline.integrations.kaggle.workers.scheduling import stream_map_ordered
from corpus_pipeline.integrations.kaggle.workers.telemetry import RuntimeTelemetry
from corpus_pipeline.runtime.client import LlamaCppClient


def _emit_progress(message: str) -> None:
    print(f"[query-embed] {message}", flush=True)


def _query_key(record: dict) -> str:
    return str(record["query_id"])


def _query_fingerprint(record: dict) -> str:
    return f"{record.get('model', '')}:{record.get('query_hash', '')}"


def _query_input_record(row: dict, model: str) -> dict:
    query = str(row["query"])
    return {
        "model": model,
        "query_id": str(row["query_id"]),
        "query_hash": query_hash(query),
        "query": query,
    }


def _legacy_query_records(data_path: Path, rows: list[dict], model: str) -> list[dict]:
    if not data_path.is_file():
        return []
    expected = {str(row["query_id"]): _query_input_record(row, model) for row in rows}
    records = []
    with data_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            key = str(item.get("query_id", ""))
            current = expected.get(key)
            if (
                current is not None
                and item.get("model") == model
                and item.get("query_hash") == current["query_hash"]
                and isinstance(item.get("embedding"), list)
            ):
                records.append(item)
    return records


def run_query_embed_worker(
    config: dict,
    *,
    embed_batch: Callable[[list[str], str], list[list[float]]] | None = None,
    emit: Callable[[str], None] = _emit_progress,
    clock: Callable[[], float] = time.monotonic,
) -> CloudArtifact:
    started = clock()
    identity = identity_from_config(config)
    input_path = resolve_input_file(config, "input")
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "query_embeddings.jsonl"
    rows = list(iter_jsonl_objects(input_path))
    model = str(config["model"])
    runtime_overrides = config.get("runtime_overrides")
    if not isinstance(runtime_overrides, dict):
        raise ValueError("runtime_overrides is required")
    telemetry = RuntimeTelemetry("query_embed", model, output_dir)
    journal_path = output_dir / "query_embeddings.journal.jsonl"
    seed_path = resolve_optional_input_file(config, "checkpoint_filename")
    journal = AppendOnlyJournal.open(
        journal_path,
        {"reuse_sha256": identity.reuse_sha256},
        _query_key,
        _query_fingerprint,
        seed_path=seed_path,
    )
    if not journal.records:
        legacy = _legacy_query_records(data_path, rows, model)
        if legacy:
            journal.append_batch(legacy)
    current = [_query_input_record(row, model) for row in rows]
    if len({_query_key(item) for item in current}) != len(current):
        raise ValueError("query input contains duplicate query_id values")
    existing = journal.reusable(current)
    missing = [row for row in rows if str(row["query_id"]) not in existing]
    emit(f"input total={len(rows)} cached={len(existing)} missing={len(missing)}")
    emit(
        f"input total={len(rows)} reusable={len(existing)} "
        f"changed_or_new={len(missing)} deleted="
        f"{len(journal.records) - len(existing)}"
    )
    if missing:
        deadline = worker_deadline(config, clock)
        batch_size = max(1, int(runtime_overrides["request_batch_size"]))
        next_milestone = ((len(existing) // 500) + 1) * 500
        newly_processed = 0
        last_reported_count = len(existing)
        last_reported_at = started

        def commit_results(results: list[tuple[dict, list[float]]]) -> None:
            nonlocal \
                next_milestone, \
                newly_processed, \
                last_reported_count, \
                last_reported_at
            records = []
            for row, vector in results:
                records.append(
                    seal_record(
                        {
                            **_query_input_record(row, model),
                            "model_sha256": str(
                                identity.payload.get("model_sha256") or ""
                            ),
                            "vector_dim": len(vector),
                            "embedding": vector,
                        },
                        "query-embedding-v2",
                    )
                )
            journal.append_batch(records)
            existing.update({_query_key(record): record for record in records})
            newly_processed += len(records)
            while len(existing) >= next_milestone:
                now = clock()
                recent_rate = (len(existing) - last_reported_count) / max(
                    now - last_reported_at, 1.0
                )
                average_rate = newly_processed / max(now - started, 1.0)
                emit(
                    f"milestone={next_milestone} "
                    f"processed={len(existing)}/{len(rows)} "
                    f"progress={len(existing) / max(len(rows), 1) * 100:.2f}% "
                    f"elapsed={format_elapsed(max(0.0, now - started))} "
                    f"recent_rate={recent_rate:.2f} "
                    f"average_rate={average_rate:.2f} queries/s"
                )
                last_reported_count = len(existing)
                last_reported_at = now
                next_milestone += 500

        def process_local() -> None:
            assert embed_batch is not None
            for start in range(0, len(missing), batch_size):
                if clock() >= deadline:
                    break
                batch = missing[start : start + batch_size]
                vectors = embed_batch([str(row["query"]) for row in batch], model)
                commit_results(list(zip(batch, vectors, strict=True)))

        def process_servers(servers) -> None:
            clients = [LlamaCppClient(server.base_url) for server in servers]
            resources = list(enumerate(clients))
            batches = [missing[start : start + batch_size] for start in range(0, len(missing), batch_size)]

            async def operation(resource, _index, rows):
                server_index, client = resource
                operation_started = time.monotonic()
                vectors = await asyncio.to_thread(
                    client.embed,
                    [str(row["query"]) for row in rows],
                    model,
                    int(config["vector_dimension"]),
                )
                telemetry.record_operation(server_index, len(rows), sum(len(str(row["query"])) for row in rows), max(0.0, time.monotonic() - operation_started), "ok", 0)
                return list(zip(rows, vectors, strict=True))

            import asyncio

            scheduled = asyncio.run(
                stream_map_ordered(
                    batches,
                    resources,
                    max(1, int(runtime_overrides["concurrency"])),
                    operation,
                    deadline,
                    clock=clock,
                )
            )
            for result in scheduled.results:
                commit_results(result)

        if embed_batch is None:
            emit("model-server starting")
            server_config = dict(config)
            server_config["runtime_overrides"] = runtime_overrides
            with managed_model_servers(server_config) as servers:
                emit(f"model-server ready replicas={len(servers)}")
                process_servers(servers)
        else:
            process_local()
    if len(existing) < len(rows):
        emit(
            f"budget exhausted processed={len(existing)}/{len(rows)} "
            f"remaining={len(rows) - len(existing)}"
        )
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    journal_path.touch(exist_ok=True)
    journal.compact(current, data_path)
    telemetry.close()
    runtime_summary = telemetry.summary()
    telemetry.write_report()
    artifact = artifact_from_output(
        data_path,
        artifact_type="query_embedding_cache",
        identity=identity,
        total=len(rows),
        complete=len(existing),
        checkpoint_path=journal_path,
        runtime_summary=runtime_summary,
    )
    emit(
        f"artifact processed={len(existing)}/{len(rows)} "
        f"complete={'true' if len(existing) == len(rows) else 'false'}"
    )
    return artifact


def main() -> int:
    config = json.loads(
        Path(__import__("os").environ["KAGGLE_PIPELINE_CONFIG"]).read_text(
            encoding="utf-8"
        )
    )
    run_query_embed_worker(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
