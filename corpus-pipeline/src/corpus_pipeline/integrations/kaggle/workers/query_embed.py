from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

from corpus_pipeline.artifacts.jsonl import iter_jsonl_objects
from corpus_pipeline.evaluation.query_embedding_cache import query_hash
from corpus_pipeline.integrations.kaggle.models import CloudArtifact
from corpus_pipeline.integrations.kaggle.workers.checkpointing import CheckpointStore
from corpus_pipeline.integrations.kaggle.workers.runtime import (
    artifact_from_output,
    identity_from_config,
    managed_model_servers,
    worker_deadline,
)
from corpus_pipeline.runtime.client import LlamaCppClient


def run_query_embed_worker(
    config: dict,
    *,
    embed_batch: Callable[[list[str], str], list[list[float]]] | None = None,
) -> CloudArtifact:
    identity = identity_from_config(config)
    input_path = Path(config["input_path"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "query_embeddings.jsonl"
    rows = list(iter_jsonl_objects(input_path))
    checkpoint = CheckpointStore.open(
        data_path, {"job_sha256": identity.sha256}, lambda item: str(item["query_id"])
    )
    existing = {
        key: item
        for key, item in checkpoint.records.items()
        if item.get("model") == str(config["model"])
        and any(
            str(row["query_id"]) == key
            and item.get("query_hash") == query_hash(str(row["query"]))
            for row in rows
        )
    }
    missing = [row for row in rows if str(row["query_id"]) not in existing]
    if missing:
        deadline = worker_deadline(config)
        batch_size = max(1, int(config.get("batch_size", 32)))

        def process_batches(client: LlamaCppClient | None) -> None:
            for start in range(0, len(missing), batch_size):
                if time.monotonic() >= deadline:
                    break
                batch = missing[start : start + batch_size]
                texts = [str(row["query"]) for row in batch]
                if embed_batch is None:
                    assert client is not None
                    vectors = client.embed(
                        texts, str(config["model"]), int(config["vector_dimension"])
                    )
                else:
                    vectors = embed_batch(texts, str(config["model"]))
                for row, vector in zip(batch, vectors, strict=True):
                    existing[str(row["query_id"])] = {
                        "model": str(config["model"]),
                        "query_id": str(row["query_id"]),
                        "query_hash": query_hash(str(row["query"])),
                        "query": str(row["query"]),
                        "embedding": vector,
                    }
                checkpoint.append_batch(existing[str(row["query_id"])] for row in batch)

        if embed_batch is None:
            with managed_model_servers(config) as servers:
                process_batches(LlamaCppClient(servers[0].base_url))
        else:
            process_batches(None)
    with data_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    existing[str(row["query_id"])], ensure_ascii=False, sort_keys=True
                )
                + "\n"
            )
    return artifact_from_output(
        data_path,
        artifact_type="query_embedding_cache",
        identity=identity,
        total=len(rows),
        complete=len(existing),
    )


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
