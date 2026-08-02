from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from corpus_pipeline.artifacts.jsonl import iter_jsonl_objects
from corpus_pipeline.evaluation.query_embedding_cache import query_hash
from corpus_pipeline.evaluation.retrieval_candidate_artifact import document_hash
from corpus_pipeline.integrations.kaggle.models import CloudArtifact
from corpus_pipeline.integrations.kaggle.workers.checkpointing import CheckpointStore
from corpus_pipeline.integrations.kaggle.workers.runtime import (
    artifact_from_output,
    identity_from_config,
    managed_model_servers,
)


def run_rerank_worker(
    config: dict, *, score_pair: Callable[[str, str, str], float] | None = None
) -> CloudArtifact:
    identity = identity_from_config(config)
    candidate_path = Path(config["candidate_path"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "rerank_scores.jsonl"
    store = CheckpointStore.open(
        data_path,
        identity={"job_sha256": identity.sha256},
        key_fn=lambda item: (str(item["query_id"]), str(item["chunk_id"])),
    )
    existing = {
        (str(item["query_id"]), str(item["chunk_id"])): item
        for item in store.records.values()
    }
    pairs = []
    for record in iter_jsonl_objects(candidate_path):
        for candidate in record["candidates"]:
            pairs.append(
                (
                    str(record["query_id"]),
                    str(record.get("query", "")),
                    str(candidate["chunk_id"]),
                    str(candidate.get("document_text", "")),
                )
            )
    protocol = str(config.get("protocol", "native_rerank"))
    existing = {
        (query_id, chunk_id): item
        for query_id, query, chunk_id, text in pairs
        if (item := existing.get((query_id, chunk_id))) is not None
        and item.get("reranker") == str(config["model"])
        and item.get("query_hash") == query_hash(query)
        and item.get("document_hash") == document_hash(text)
        and item.get("protocol") == protocol
    }
    if score_pair is not None:
        scorer = score_pair
        for query_id, query, chunk_id, text in pairs:
            key = (query_id, chunk_id)
            if key not in existing:
                item = {
                    "reranker": str(config["model"]),
                    "query_id": query_id,
                    "query_hash": query_hash(query),
                    "chunk_id": chunk_id,
                    "document_hash": document_hash(text),
                    "score": float(scorer(query, text, str(config["model"]))),
                    "protocol": protocol,
                }
                existing[key] = item
                store.append(item)
    else:
        from corpus_pipeline.runtime.client import LlamaCppClient

        with managed_model_servers(config) as servers:
            client = LlamaCppClient(servers[0].base_url)
            if protocol == "native_rerank":
                grouped: dict[tuple[str, str], list[tuple[str, str]]] = {}
                for query_id, query, chunk_id, text in pairs:
                    if (query_id, chunk_id) not in existing:
                        grouped.setdefault((query_id, query), []).append(
                            (chunk_id, text)
                        )
                for (query_id, query), group in grouped.items():
                    scores = client.rerank_native(
                        query, [text for _, text in group], str(config["model"])
                    )
                    for (chunk_id, text), score in zip(group, scores, strict=True):
                        item = {
                            "reranker": str(config["model"]),
                            "query_id": query_id,
                            "query_hash": query_hash(query),
                            "chunk_id": chunk_id,
                            "document_hash": document_hash(text),
                            "score": float(score),
                            "protocol": protocol,
                        }
                        existing[(query_id, chunk_id)] = item
                        store.append(item)
            else:
                for query_id, query, chunk_id, text in pairs:
                    key = (query_id, chunk_id)
                    if key in existing:
                        continue
                    prompt = f"Query: {query}\nDocument: {text}\nRelevant:"
                    score = client.rerank_completion(prompt, str(config["model"]))
                    item = {
                        "reranker": str(config["model"]),
                        "query_id": query_id,
                        "query_hash": query_hash(query),
                        "chunk_id": chunk_id,
                        "document_hash": document_hash(text),
                        "score": float(score),
                        "protocol": protocol,
                    }
                    existing[key] = item
                    store.append(item)
    return artifact_from_output(
        data_path,
        artifact_type="rerank_scores",
        identity=identity,
        total=len(pairs),
        complete=len(existing),
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
