from __future__ import annotations

import asyncio
import json
import math
import time
from collections.abc import Callable
from pathlib import Path

from seed_pipeline.artifacts.jsonl import iter_jsonl_objects
from seed_pipeline.cache.jsonl_records import seal_record
from seed_pipeline.evaluation.query_hash import query_hash
from seed_pipeline.evaluation.rerank_contract import (
    document_hash,
)
from seed_pipeline.integrations.kaggle.models import CloudArtifact
from seed_pipeline.integrations.kaggle.parsers import format_elapsed
from seed_pipeline.integrations.kaggle.workers.checkpointing import (
    AppendOnlyJournal,
)
from seed_pipeline.integrations.kaggle.workers.recovery import (
    recoverable_termination,
)
from seed_pipeline.integrations.kaggle.workers.runtime import (
    artifact_from_output,
    identity_from_config,
    managed_model_servers,
    resolve_input_file,
    resolve_optional_input_file,
    worker_deadline,
)
from seed_pipeline.integrations.kaggle.workers.scheduling import (
    stream_map_ordered,
)
from seed_pipeline.integrations.kaggle.workers.telemetry import RuntimeTelemetry
from seed_pipeline.runtime.catalog import require_model
from seed_pipeline.runtime.model_profiles import NATIVE_RERANK_PROTOCOL

Pair = dict[str, str]


def _pair_key(record: dict) -> str:
    return f"{record['query_id']}\x1f{record['chunk_id']}"


def _pair_fingerprint(record: dict) -> str:
    return ":".join(
        (
            str(record.get("reranker", "")),
            str(record.get("protocol", "")),
            str(record.get("query_hash", "")),
            str(record.get("document_hash", "")),
        )
    )


def _pair_input(
    model: str, protocol: str, query_id: str, query: str, chunk_id: str, text: str
) -> Pair:
    return {
        "reranker": model,
        "query_id": query_id,
        "query_hash": query_hash(query),
        "chunk_id": chunk_id,
        "document_hash": document_hash(text),
        "protocol": protocol,
    }


def _score_record(pair: Pair, score: float) -> dict:
    if not math.isfinite(float(score)):
        raise ValueError(f"reranker returned a non-finite score for {_pair_key(pair)}")
    return {**pair, "score": float(score)}


def _load_pairs(candidate_path: Path, model: str, protocol: str) -> list[Pair]:
    pairs: list[Pair] = []
    for record in iter_jsonl_objects(candidate_path):
        query_id = str(record["query_id"])
        query = str(record.get("query", ""))
        for candidate in record["candidates"]:
            pairs.append(
                _pair_input(
                    model,
                    protocol,
                    query_id,
                    query,
                    str(candidate["chunk_id"]),
                    str(candidate.get("document_text", "")),
                )
            )
    keys = [_pair_key(pair) for pair in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError("candidate input contains duplicate query_id/chunk_id pairs")
    return pairs


def _pair_details(candidate_path: Path) -> dict[str, tuple[str, str]]:
    details: dict[str, tuple[str, str]] = {}
    for record in iter_jsonl_objects(candidate_path):
        query_id = str(record["query_id"])
        query = str(record.get("query", ""))
        for candidate in record["candidates"]:
            chunk_id = str(candidate["chunk_id"])
            details[f"{query_id}\x1f{chunk_id}"] = (
                query,
                str(candidate.get("document_text", "")),
            )
    return details


def _emit_progress(message: str) -> None:
    print(f"[rerank] {message}", flush=True)


def run_rerank_worker(
    config: dict,
    *,
    score_pair: Callable[[str, str, str], float] | None = None,
    emit: Callable[[str], None] = _emit_progress,
    clock: Callable[[], float] = time.monotonic,
    server_manager=managed_model_servers,
    client_factory=None,
) -> CloudArtifact:
    started = clock()
    identity = identity_from_config(config)
    candidate_path = resolve_input_file(config, "candidates")
    resolve_input_file(config, "candidate_manifest")
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "rerank_scores.jsonl"
    journal_path = output_dir / "rerank_scores.journal.jsonl"
    model = str(config["model"])
    protocol = str(config.get("protocol", NATIVE_RERANK_PROTOCOL))
    if protocol != NATIVE_RERANK_PROTOCOL:
        raise ValueError(f"unsupported rerank protocol: {protocol}")
    seed_path = resolve_optional_input_file(config, "checkpoint_filename")
    journal = AppendOnlyJournal.open(
        journal_path,
        {"reuse_sha256": identity.reuse_sha256},
        _pair_key,
        _pair_fingerprint,
        seed_path=seed_path,
    )
    pairs = _load_pairs(candidate_path, model, protocol)
    details = _pair_details(candidate_path)
    existing = journal.reusable(pairs)
    missing = [pair for pair in pairs if _pair_key(pair) not in existing]
    deadline = worker_deadline(config)
    telemetry = RuntimeTelemetry("rerank", model, output_dir)

    expected_by_query: dict[str, int] = {}
    for pair in pairs:
        query_id = pair["query_id"]
        expected_by_query[query_id] = expected_by_query.get(query_id, 0) + 1
    remaining_by_query = dict(expected_by_query)
    for record in existing.values():
        remaining_by_query[record["query_id"]] -= 1
    completed_queries = sum(remaining == 0 for remaining in remaining_by_query.values())
    next_milestone = ((completed_queries // 500) + 1) * 500
    last_reported_pairs = len(existing)
    last_reported_at = started
    newly_processed_pairs = 0
    emit(
        f"input queries total={len(expected_by_query)} "
        f"reusable={completed_queries} "
        f"missing={len(expected_by_query) - completed_queries} "
        f"pairs total={len(pairs)} reusable={len(existing)} "
        f"missing={len(missing)}"
    )

    def commit(records: list[dict]) -> None:
        nonlocal completed_queries, last_reported_at, last_reported_pairs
        nonlocal newly_processed_pairs, next_milestone
        if not records:
            return
        spec = require_model(model)
        if spec.rerank_contract is None:
            raise ValueError(f"Reranker {model} has no scoring contract")
        contract = spec.rerank_contract.sha256
        records = [
            seal_record(
                {
                    **record,
                    "model_sha256": str(identity.payload.get("model_sha256") or ""),
                    "request_contract_sha256": contract,
                },
                "rerank-score-v2",
            )
            for record in records
        ]
        journal.append_batch(records)
        existing.update({_pair_key(record): record for record in records})
        newly_processed_pairs += len(records)
        for record in records:
            query_id = record["query_id"]
            remaining_by_query[query_id] -= 1
            if remaining_by_query[query_id] == 0:
                completed_queries += 1
        while completed_queries >= next_milestone:
            now = clock()
            recent_rate = (len(existing) - last_reported_pairs) / max(
                now - last_reported_at, 1.0
            )
            average_rate = newly_processed_pairs / max(now - started, 1.0)
            emit(
                f"milestone={next_milestone} "
                f"processed={completed_queries}/{len(expected_by_query)} queries "
                f"pairs={len(existing)}/{len(pairs)} "
                f"progress={completed_queries / max(len(expected_by_query), 1) * 100:.2f}% "
                f"elapsed={format_elapsed(max(0.0, now - started))} "
                f"recent_rate={recent_rate:.2f} "
                f"average_rate={average_rate:.2f} pairs/s"
            )
            next_milestone += 500
            last_reported_pairs = len(existing)
            last_reported_at = now

    if score_pair is not None:
        batch_size = max(1, int(config.get("batch_size", 32)))
        for start in range(0, len(missing), batch_size):
            batch = missing[start : start + batch_size]
            batch_started = time.monotonic()
            records = []
            for pair in batch:
                records.append(
                    _score_record(
                        pair,
                        score_pair(
                            details[_pair_key(pair)][0],
                            details[_pair_key(pair)][1],
                            model,
                        ),
                    )
                )
            telemetry.record_operation(
                0,
                len(batch),
                sum(
                    len(details[_pair_key(p)][0]) + len(details[_pair_key(p)][1])
                    for p in batch
                ),
                max(0.0, time.monotonic() - batch_started),
                "ok",
                0,
            )
            commit(records)
    termination = None
    if score_pair is None and missing:
        server_config = dict(config)
        runtime_overrides = server_config.get("runtime_overrides")
        if not isinstance(runtime_overrides, dict):
            raise ValueError("runtime_overrides is required")
        server_config["runtime_overrides"] = runtime_overrides
        try:
            if client_factory is None:
                from seed_pipeline.runtime.client import LlamaCppClient

                client_factory = LlamaCppClient
            with server_manager(server_config, telemetry=telemetry) as servers:
                clients = [client_factory(server.base_url) for server in servers]
                resources = list(enumerate(clients))
                per_client = max(1, int(runtime_overrides["concurrency"]))
                request_batch_size = max(
                    1, int(runtime_overrides["request_batch_size"])
                )
                grouped: dict[tuple[str, str], list[Pair]] = {}
                for pair in missing:
                    query, _text = details[_pair_key(pair)]
                    grouped.setdefault((pair["query_id"], query), []).append(pair)
                # One /v1/rerank request per query; llama-server packs the documents of
                # all busy slots into shared physical batches.
                groups = [
                    (query_id, query, group[start : start + request_batch_size])
                    for (query_id, query), group in grouped.items()
                    for start in range(0, len(group), request_batch_size)
                ]

                async def native_operation(resource, _index, item):
                    server_index, client = resource
                    query_id, query, group = item
                    started_operation = clock()
                    scores = await asyncio.to_thread(
                        client.rerank_native,
                        query,
                        [details[_pair_key(pair)][1] for pair in group],
                        model,
                    )
                    if len(scores) != len(group):
                        raise ValueError("reranker returned an unexpected score count")
                    telemetry.record_operation(
                        server_index,
                        len(group),
                        sum(
                            len(query) + len(details[_pair_key(pair)][1])
                            for pair in group
                        ),
                        max(0.0, clock() - started_operation),
                        "ok",
                        0,
                    )
                    return (query_id, group, scores)

                async def run_native():
                    return await stream_map_ordered(
                        groups,
                        resources,
                        per_client,
                        native_operation,
                        deadline,
                        on_completed=lambda batch: commit(
                            [
                                _score_record(pair, score)
                                for _index, value in batch
                                for pair, score in zip(value[1], value[2], strict=True)
                            ]
                        ),
                    )

                scheduled = asyncio.run(run_native())
                if scheduled.stopped_early:
                    emit(f"budget exhausted pairs={len(existing)}/{len(pairs)}")
        except Exception as error:
            termination = recoverable_termination(error)
            if termination is None:
                raise

    journal_path.parent.mkdir(parents=True, exist_ok=True)
    journal_path.touch(exist_ok=True)
    journal.compact(pairs, data_path)
    telemetry.close()
    runtime_summary = telemetry.summary()
    if termination is not None:
        runtime_summary["termination"] = termination
        emit(
            "recoverable model-server termination; "
            f"checkpointed pairs={len(existing)}/{len(pairs)}"
        )
    telemetry.write_report()
    artifact = artifact_from_output(
        data_path,
        artifact_type="rerank_scores",
        identity=identity,
        total=len(pairs),
        complete=len(existing),
        checkpoint_path=journal_path,
        runtime_summary=runtime_summary,
    )
    emit(
        f"artifact queries={completed_queries}/{len(expected_by_query)} "
        f"pairs={len(existing)}/{len(pairs)} "
        f"complete={'true' if len(existing) == len(pairs) else 'false'}"
    )
    return artifact


def main() -> int:
    config = json.loads(
        Path(__import__("os").environ["KAGGLE_PIPELINE_CONFIG"]).read_text(
            encoding="utf-8"
        )
    )
    run_rerank_worker(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
