from dataclasses import replace
from pathlib import Path

import pytest

from seed_pipeline.runtime.catalog import require_model
from seed_pipeline.runtime.compose import LlamaCppComposeManager, file_sha256


class RecordingRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, str] | None]] = []

    def run(self, args, env=None, capture_output=False):
        self.calls.append((list(args), env))
        return ""


class HealthyClient:
    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint

    def health(self) -> None:
        return None


@pytest.mark.parametrize(
    ("model", "reranking"),
    [("bge-reranker-v2-m3:f16", "true"), ("qwen3-reranker:0.6b-fp16", "false")],
)
def test_reranker_service_serves_rerank_endpoint_only_for_native_protocol(
    tmp_path: Path, model: str, reranking: str
) -> None:
    artifact = tmp_path / "reranker.gguf"
    artifact.write_bytes(b"gguf")
    spec = replace(
        require_model(model),
        canonical_filename=artifact.name,
        byte_size=artifact.stat().st_size,
        sha256=file_sha256(artifact),
    )
    runner = RecordingRunner()
    manager = LlamaCppComposeManager(
        tmp_path / "compose.yaml",
        runner=runner,
        client_factory=HealthyClient,
        environment={},
    )

    endpoint = manager.ensure("reranker", spec, tmp_path)

    command, environment = runner.calls[0]
    assert command[-1] == "llama-reranker"
    assert environment is not None
    assert environment["LLAMA_RERANKER_MODEL"] == "reranker.gguf"
    assert environment["LLAMA_RERANKER_RERANKING"] == reranking
    assert endpoint == "http://127.0.0.1:11435"
