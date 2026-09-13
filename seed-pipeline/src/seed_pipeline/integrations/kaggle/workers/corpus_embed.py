from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

from seed_pipeline.artifacts.jsonl import iter_jsonl_objects
from seed_pipeline.integrations.kaggle.models import CloudArtifact
from seed_pipeline.integrations.kaggle.workers.runtime import (
    artifact_from_output,
    identity_from_config,
    managed_model_servers,
    resolve_input_file,
    resolve_optional_input_file,
    worker_deadline,
)
from seed_pipeline.integrations.kaggle.workers.telemetry import RuntimeTelemetry
from seed_pipeline.runtime.client import LlamaCppClient
from seed_pipeline.vector_store.embedding_core import (
    ChunkEmbeddingCache,
    collect_or_create_embeddings,
    read_normalized_input_points,
)


def run_corpus_embed_worker(
    config: dict,
    *,
    command_executor: Callable[[dict], int] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> CloudArtifact:
    identity = identity_from_config(config)
    input_path = resolve_input_file(config, "input")
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(
        config.get("output_path", output_dir / "vector_embeddings.jsonl")
    )
    model_name = str(config["model"])
    runtime_overrides = config.get("runtime_overrides")
    if not isinstance(runtime_overrides, dict):
        raise ValueError("runtime_overrides is required")
    telemetry = RuntimeTelemetry("corpus_embed", model_name, output_dir)
    checkpoint_path = resolve_optional_input_file(config, "checkpoint_filename")
    if (
        checkpoint_path is not None
        and checkpoint_path.is_file()
        and not output_path.exists()
    ):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(checkpoint_path.read_bytes())
    if command_executor is None:
        points = read_normalized_input_points(input_path)
        vector_dimension = int(config["vector_dimension"])
        cache = ChunkEmbeddingCache(
            output_path,
            vector_dimension,
            allow_truncated_final_record=True,
            model_sha256=str(identity.payload.get("model_sha256") or ""),
        )
        deadline = worker_deadline(config, clock)
        server_config = dict(config)
        server_config["runtime_overrides"] = runtime_overrides
        with managed_model_servers(server_config, telemetry=telemetry) as servers:
            args = SimpleNamespace(
                model=model_name,
                input_batch_size=int(runtime_overrides["request_batch_size"]),
                mock=False,
                llama_clients=[LlamaCppClient(server.base_url) for server in servers],
                embedding_concurrency=int(runtime_overrides["concurrency"]),
                runtime_stop_deadline=deadline,
            )
            collect_or_create_embeddings(
                points,
                args,
                vector_dimension,
                cache,
                retain_embeddings=False,
            )
        return_code = 0
    else:
        return_code = command_executor(config)
    if return_code != 0 or not output_path.is_file():
        raise RuntimeError(f"Corpus embedding worker did not produce {output_path}")
    telemetry.close()
    runtime_summary = telemetry.summary()
    telemetry.write_report()
    total = sum(1 for _ in iter_jsonl_objects(input_path))
    complete = sum(1 for _ in iter_jsonl_objects(output_path))
    return artifact_from_output(
        output_path,
        artifact_type="vector_embedding_cache",
        identity=identity,
        total=total,
        complete=complete,
        runtime_summary=runtime_summary,
    )


def main() -> int:
    config = json.loads(
        Path(os.environ["KAGGLE_PIPELINE_CONFIG"]).read_text(encoding="utf-8")
    )
    run_corpus_embed_worker(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
