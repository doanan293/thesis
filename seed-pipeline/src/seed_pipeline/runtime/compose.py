from __future__ import annotations

import hashlib
import os
import re
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path

from seed_pipeline.runtime.catalog import ModelKind, ModelSpec
from seed_pipeline.runtime.client import LlamaCppClient
from seed_pipeline.runtime.runtime_profiles import (
    RuntimeCandidate,
    reranker_context_size,
)

_LLAMA_CPP_IMAGE = re.compile(
    r"^\s*image:\s*(ghcr\.io/ggml-org/llama\.cpp:\S+)\s*$", re.MULTILINE
)


class ModelArtifactError(RuntimeError):
    pass


class ComposeStartupError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model_artifact(root: Path, spec: ModelSpec) -> Path:
    model = Path(root) / spec.canonical_filename
    if not model.is_file() or model.is_symlink():
        raise ModelArtifactError(f"Model file is missing or invalid: {model}")
    actual_size = model.stat().st_size
    if actual_size != spec.byte_size:
        raise ModelArtifactError(
            f"Model byte size mismatch for {spec.name}: {actual_size} != {spec.byte_size}"
        )
    actual_hash = file_sha256(model)
    if actual_hash != spec.sha256:
        raise ModelArtifactError(
            f"Model SHA-256 mismatch for {spec.name}: {actual_hash} != {spec.sha256}"
        )
    return model


def compose_llama_cpp_image(compose_file: Path) -> str:
    """The llama.cpp server image tag that compose.yaml pins."""
    match = _LLAMA_CPP_IMAGE.search(Path(compose_file).read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"{compose_file} declares no llama.cpp server image")
    return match.group(1)


def reranker_environment(runtime: RuntimeCandidate) -> dict[str, str]:
    """Root .env variables that compose.yaml maps to llama-reranker's LLAMA_ARG_*."""
    if runtime.logical_batch_size != runtime.physical_batch_size:
        raise ValueError("reranker runtime must use one size for batch and ubatch")
    context = reranker_context_size(
        server_slots=runtime.server_slots,
        ubatch=runtime.physical_batch_size,
        context_per_slot=runtime.context_per_slot,
    )
    environment = {
        "LLAMA_RERANKER_PARALLEL": str(runtime.server_slots),
        "LLAMA_RERANKER_UBATCH_SIZE": str(runtime.physical_batch_size),
        "LLAMA_RERANKER_CONTEXT_SIZE": str(context),
    }
    if runtime.threads is not None:
        environment["LLAMA_RERANKER_THREADS"] = str(runtime.threads)
    return environment


class SubprocessRunner:
    def run(self, args, env=None, capture_output=False):
        completed = subprocess.run(
            list(args),
            env=env,
            check=True,
            text=True,
            stdout=subprocess.PIPE if capture_output else None,
            stderr=subprocess.STDOUT if capture_output else None,
        )
        return completed.stdout or ""


class LlamaCppComposeManager:
    def __init__(
        self,
        compose_file: Path,
        runner=None,
        client_factory=LlamaCppClient,
        environment=None,
        startup_timeout: float = 180.0,
        poll_interval: float = 2.0,
    ) -> None:
        self.compose_file = Path(compose_file)
        self.runner = runner or SubprocessRunner()
        self.client_factory = client_factory
        self.environment = dict(os.environ if environment is None else environment)
        self.startup_timeout = float(startup_timeout)
        self.poll_interval = float(poll_interval)

    def ensure(
        self,
        role: str,
        spec: ModelSpec,
        gguf_root: Path,
        runtime: RuntimeCandidate | None = None,
    ) -> str:
        if role not in {"embedding", "reranker"}:
            raise ValueError(f"Unsupported llama.cpp service role: {role}")
        if role == "embedding" and spec.kind is not ModelKind.EMBEDDING:
            raise ValueError(f"Embedding service cannot load {spec.kind.value} model")
        if role == "reranker" and spec.kind is not ModelKind.RERANKER:
            raise ValueError(f"Reranker service cannot load {spec.kind.value} model")
        if runtime is not None and role != "reranker":
            raise ValueError(
                "runtime settings apply to the llama-reranker service only"
            )
        verify_model_artifact(gguf_root, spec)
        service = f"llama-{role}"
        prefix = f"LLAMA_{role.upper()}"
        environment = dict(self.environment)
        environment["GGUF_DIR"] = str(Path(gguf_root).resolve())
        environment[f"{prefix}_MODEL"] = spec.canonical_filename
        if runtime is not None:
            environment.update(reranker_environment(runtime))
        command = [
            "docker",
            "compose",
            "-f",
            str(self.compose_file),
            "up",
            "-d",
            "--force-recreate",
            service,
        ]
        self.runner.run(command, env=environment)
        port = environment.get(
            f"{prefix}_PORT", "11434" if role == "embedding" else "11435"
        )
        endpoint = f"http://127.0.0.1:{port}"
        client = self.client_factory(endpoint)
        deadline = time.monotonic() + self.startup_timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                client.health()
                return endpoint
            except Exception as exc:
                last_error = exc
                time.sleep(self.poll_interval)
        logs = self.logs(role, environment=environment)
        raise ComposeStartupError(
            f"{service} did not become healthy: {last_error}\n{logs[-8000:]}"
        )

    def logs(
        self,
        role: str,
        *,
        tail: int = 80,
        environment: Mapping[str, str] | None = None,
    ) -> str:
        return self.runner.run(
            [
                "docker",
                "compose",
                "-f",
                str(self.compose_file),
                "logs",
                "--tail",
                str(tail),
                f"llama-{role}",
            ],
            env=dict(self.environment if environment is None else environment),
            capture_output=True,
        )


def resolve_server(
    mode: str,
    external_urls: list[str],
    spec: ModelSpec,
    manager: LlamaCppComposeManager,
    gguf_root: Path,
) -> list[str]:
    if mode == "external":
        endpoints = [value.rstrip("/") for value in external_urls if value.strip()]
        if not endpoints:
            raise ValueError("External server mode requires --llama-server-url")
        return endpoints
    if mode != "compose":
        raise ValueError(f"Unsupported server mode: {mode}")
    role = "embedding" if spec.kind is ModelKind.EMBEDDING else "reranker"
    return [manager.ensure(role, spec, gguf_root)]
