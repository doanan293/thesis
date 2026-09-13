from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from pathlib import Path

from seed_pipeline.embeddings.text_cache import (
    TextEmbeddingCache,
    embed_missing,
    read_embedding_inputs,
)
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

OUTPUT_FILENAME = "text_embeddings.jsonl"


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
    output_path = Path(config.get("output_path", output_dir / OUTPUT_FILENAME))
    model_name = str(config["model"])
    vector_dimension = int(config["vector_dimension"])
    model_sha256 = str(identity.payload.get("model_sha256") or "")
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
        output_path.write_bytes(checkpoint_path.read_bytes())
    inputs = read_embedding_inputs(input_path)
    if command_executor is None:
        cache = TextEmbeddingCache(
            output_path,
            model=model_name,
            model_sha256=model_sha256,
            vector_dim=vector_dimension,
        )
        server_config = dict(config)
        server_config["runtime_overrides"] = runtime_overrides
        with managed_model_servers(server_config, telemetry=telemetry) as servers:
            embed_missing(
                inputs,
                cache,
                [LlamaCppClient(server.base_url) for server in servers],
                batch_size=int(runtime_overrides["request_batch_size"]),
                concurrency=int(runtime_overrides["concurrency"]),
                deadline=worker_deadline(config, clock),
                clock=clock,
            )
        return_code = 0
    else:
        return_code = command_executor(config)
    if return_code != 0 or not output_path.is_file():
        raise RuntimeError(f"Corpus embedding worker did not produce {output_path}")
    telemetry.close()
    runtime_summary = telemetry.summary()
    telemetry.write_report()
    complete = len(
        TextEmbeddingCache(
            output_path,
            model=model_name,
            model_sha256=model_sha256,
            vector_dim=vector_dimension,
        ).records_for(item.embedding_text_sha256 for item in inputs)
    )
    return artifact_from_output(
        output_path,
        artifact_type="text_embedding_cache",
        identity=identity,
        total=len(inputs),
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
