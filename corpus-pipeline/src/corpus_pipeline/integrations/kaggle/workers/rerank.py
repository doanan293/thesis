from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Callable
from pathlib import Path

from corpus_pipeline.artifacts.jsonl import iter_jsonl_objects
from corpus_pipeline.cache.jsonl_records import seal_record
from corpus_pipeline.evaluation.query_embedding_cache import query_hash
from corpus_pipeline.evaluation.rerank_score_cache import prompt_contract_hash
from corpus_pipeline.evaluation.retrieval_candidate_artifact import document_hash
from corpus_pipeline.integrations.kaggle.models import CloudArtifact
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
from corpus_pipeline.integrations.kaggle.workers.scheduling import (
    map_batches_ordered,
)

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


def run_rerank_worker(
    config: dict, *, score_pair: Callable[[str, str, str], float] | None = None
) -> CloudArtifact:
    identity = identity_from_config(config)
    candidate_path = resolve_input_file(config, "candidate_path")
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "rerank_scores.jsonl"
    journal_path = output_dir / "rerank_scores.journal.jsonl"
    model = str(config["model"])
    protocol = str(config.get("protocol", "native_rerank"))
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

    def commit(records: list[dict]) -> None:
        if not records:
            return
        contract = prompt_contract_hash(protocol=protocol)
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

    if score_pair is not None:
        batch_size = max(1, int(config.get("batch_size", 32)))
        for start in range(0, len(missing), batch_size):
            batch = missing[start : start + batch_size]
            commit(
                [
                    _score_record(
                        pair,
                        score_pair(
                            details[_pair_key(pair)][0],
                            details[_pair_key(pair)][1],
                            model,
                        ),
                    )
                    for pair in batch
                ]
            )
    elif missing:
        from corpus_pipeline.runtime.client import LlamaCppClient

        with managed_model_servers(config) as servers:
            clients = [LlamaCppClient(server.base_url) for server in servers]
            per_client = max(1, int(config.get("parallelism", 1)))
            if protocol == "native_rerank":
                groups: list[tuple[str, str, list[Pair]]] = []
                grouped: dict[tuple[str, str], list[Pair]] = {}
                for pair in missing:
                    query, _text = details[_pair_key(pair)]
                    grouped.setdefault((pair["query_id"], query), []).append(pair)
                groups = [
                    (query_id, query, group)
                    for (query_id, query), group in grouped.items()
                ]

                def native_operation(client, indexed_groups):
                    returned = []
                    for index, (_query_id, query, group) in indexed_groups:
                        scores = client.rerank_native(
                            query,
                            [details[_pair_key(pair)][1] for pair in group],
                            model,
                        )
                        if len(scores) != len(group):
                            raise ValueError(
                                "reranker returned an unexpected score count"
                            )
                        returned.append((index, (group, scores)))
                    return returned

                for start in range(0, len(groups), per_client * len(clients)):
                    scheduled = map_batches_ordered(
                        groups[start : start + per_client * len(clients)],
                        clients,
                        per_client,
                        native_operation,
                        deadline,
                    )
                    records = []
                    for group, scores in scheduled.results:
                        records.extend(
                            _score_record(pair, score)
                            for pair, score in zip(group, scores, strict=True)
                        )
                    commit(records)
                    if scheduled.stopped_early:
                        break
            else:

                def completion_operation(client, indexed_pairs):
                    prompts = [
                        f"Query: {details[_pair_key(pair)][0]}\n"
                        f"Document: {details[_pair_key(pair)][1]}\nRelevant:"
                        for _index, pair in indexed_pairs
                    ]
                    scores = asyncio.run(
                        client.rerank_completions_async(
                            prompts, model, concurrency=per_client
                        )
                    )
                    if len(scores) != len(indexed_pairs):
                        raise ValueError("reranker returned an unexpected score count")
                    return list(
                        zip(
                            [index for index, _pair in indexed_pairs],
                            scores,
                            strict=True,
                        )
                    )

                for start in range(0, len(missing), per_client * len(clients)):
                    scheduled = map_batches_ordered(
                        missing[start : start + per_client * len(clients)],
                        clients,
                        per_client,
                        completion_operation,
                        deadline,
                    )
                    commit(
                        [
                            _score_record(pair, score)
                            for pair, score in zip(
                                missing[start : start + per_client * len(clients)],
                                scheduled.results,
                                strict=True,
                            )
                        ]
                    )
                    if scheduled.stopped_early:
                        break

    journal_path.parent.mkdir(parents=True, exist_ok=True)
    journal_path.touch(exist_ok=True)
    journal.compact(pairs, data_path)
    return artifact_from_output(
        data_path,
        artifact_type="rerank_scores",
        identity=identity,
        total=len(pairs),
        complete=len(existing),
        checkpoint_path=journal_path,
    )


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
