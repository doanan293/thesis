from dataclasses import replace
from pathlib import Path

import pytest

from seed_pipeline.config.paths import COMPOSE_FILE
from seed_pipeline.runtime.catalog import LOCAL_RERANK_SEARCH_SPACE, require_model
from seed_pipeline.runtime.compose import (
    LlamaCppComposeManager,
    compose_llama_cpp_image,
    file_sha256,
    reranker_environment,
)
from seed_pipeline.runtime.runtime_profiles import RuntimeCandidate


class RecordingRunner:
    def __init__(self, logs: str = "") -> None:
        self.calls: list[tuple[list[str], dict[str, str] | None]] = []
        self.logs = logs

    def run(self, args, env=None, capture_output=False):
        self.calls.append((list(args), env))
        return self.logs if capture_output else ""


class HealthyClient:
    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint

    def health(self) -> None:
        return None


def _manager(tmp_path: Path, runner: RecordingRunner) -> LlamaCppComposeManager:
    return LlamaCppComposeManager(
        tmp_path / "compose.yaml",
        runner=runner,
        client_factory=HealthyClient,
        environment={},
    )


def _spec(tmp_path: Path, model: str):
    artifact = tmp_path / "model.gguf"
    artifact.write_bytes(b"gguf")
    return replace(
        require_model(model),
        canonical_filename=artifact.name,
        byte_size=artifact.stat().st_size,
        sha256=file_sha256(artifact),
    )


def test_reranker_without_runtime_keeps_the_compose_defaults(tmp_path: Path) -> None:
    runner = RecordingRunner()

    endpoint = _manager(tmp_path, runner).ensure(
        "reranker", _spec(tmp_path, "qwen3-reranker:4b-fp16"), tmp_path
    )

    command, environment = runner.calls[0]
    assert command[-3:] == ["-d", "--force-recreate", "llama-reranker"]
    assert environment is not None
    assert environment["LLAMA_RERANKER_MODEL"] == "model.gguf"
    for name in (
        "LLAMA_RERANKER_PARALLEL",
        "LLAMA_RERANKER_UBATCH_SIZE",
        "LLAMA_RERANKER_CONTEXT_SIZE",
        "LLAMA_RERANKER_THREADS",
        "LLAMA_RERANKER_RERANKING",
    ):
        assert name not in environment
    assert endpoint == "http://127.0.0.1:11435"


def test_reranker_runtime_recreates_the_service_with_its_level(tmp_path: Path) -> None:
    runner = RecordingRunner()
    level = LOCAL_RERANK_SEARCH_SPACE.candidates[3]

    _manager(tmp_path, runner).ensure(
        "reranker", _spec(tmp_path, "qwen3-reranker:4b-fp16"), tmp_path, runtime=level
    )

    _command, environment = runner.calls[0]
    assert environment is not None
    assert (
        environment["LLAMA_RERANKER_PARALLEL"],
        environment["LLAMA_RERANKER_UBATCH_SIZE"],
        environment["LLAMA_RERANKER_CONTEXT_SIZE"],
        environment["LLAMA_RERANKER_THREADS"],
    ) == ("8", "4096", "20480", "12")


def test_runtime_is_rejected_for_the_embedding_service(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="llama-reranker"):
        _manager(tmp_path, RecordingRunner()).ensure(
            "embedding",
            _spec(tmp_path, "qwen3-embedding:4b-fp16"),
            tmp_path,
            runtime=LOCAL_RERANK_SEARCH_SPACE.candidates[0],
        )


def test_reranker_environment_needs_one_size_for_batch_and_ubatch() -> None:
    with pytest.raises(ValueError, match="one size for batch and ubatch"):
        reranker_environment(RuntimeCandidate(16, 1, 15, 2048, 8192, 4096))


def test_logs_reads_the_service_tail(tmp_path: Path) -> None:
    runner = RecordingRunner(logs="model loaded")

    assert _manager(tmp_path, runner).logs("reranker") == "model loaded"
    command, _environment = runner.calls[0]
    assert command[-4:] == ["logs", "--tail", "80", "llama-reranker"]


def test_compose_pins_the_llama_cpp_server_image() -> None:
    assert (
        compose_llama_cpp_image(COMPOSE_FILE)
        == "ghcr.io/ggml-org/llama.cpp:server-b10920"
    )


def test_compose_reranker_serves_batched_unified_kv() -> None:
    text = COMPOSE_FILE.read_text(encoding="utf-8")
    block = text[text.index("  llama-reranker:") : text.index("  postgres:")]

    for line in (
        'LLAMA_ARG_RERANKING: "true"',
        'LLAMA_ARG_KV_UNIFIED: "true"',
        'LLAMA_ARG_N_PARALLEL: "${LLAMA_RERANKER_PARALLEL:-4}"',
        'LLAMA_ARG_CTX_SIZE: "${LLAMA_RERANKER_CONTEXT_SIZE:-10240}"',
        'LLAMA_ARG_BATCH: "${LLAMA_RERANKER_UBATCH_SIZE:-2048}"',
        'LLAMA_ARG_UBATCH: "${LLAMA_RERANKER_UBATCH_SIZE:-2048}"',
        'LLAMA_ARG_THREADS: "${LLAMA_RERANKER_THREADS:-12}"',
    ):
        assert line in block
    assert "KV_UNIFIED_PER_SLOT" not in block


def test_compose_backend_sends_one_rerank_request_at_a_time() -> None:
    text = COMPOSE_FILE.read_text(encoding="utf-8")

    assert 'PHARMA_RETRIEVAL__RERANK__MAX_CONCURRENT: "1"' in text
    assert (
        'PHARMA_RETRIEVAL__RERANK__MAX_CANDIDATES: "${RERANK_MAX_CANDIDATES:-15}"'
        in text
    )
    assert (
        'PHARMA_RETRIEVAL__RERANK__TIMEOUT_SECONDS: "${RERANK_TIMEOUT_SECONDS:-898}"'
        in text
    )
