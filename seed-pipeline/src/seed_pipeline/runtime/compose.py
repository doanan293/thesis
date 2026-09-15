from __future__ import annotations

import hashlib
import os
import subprocess
import time
from pathlib import Path

from seed_pipeline.runtime.catalog import ModelKind, ModelSpec
from seed_pipeline.runtime.client import LlamaCppClient


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

    def ensure(self, role: str, spec: ModelSpec, gguf_root: Path) -> str:
        if role not in {"embedding", "reranker"}:
            raise ValueError(f"Unsupported llama.cpp service role: {role}")
        if role == "embedding" and spec.kind is not ModelKind.EMBEDDING:
            raise ValueError(f"Embedding service cannot load {spec.kind.value} model")
        if role == "reranker" and spec.kind is not ModelKind.RERANKER:
            raise ValueError(f"Reranker service cannot load {spec.kind.value} model")
        verify_model_artifact(gguf_root, spec)
        service = f"llama-{role}"
        prefix = f"LLAMA_{role.upper()}"
        environment = dict(self.environment)
        environment["GGUF_DIR"] = str(Path(gguf_root).resolve())
        environment[f"{prefix}_MODEL"] = spec.canonical_filename
        environment[f"{prefix}_PARALLEL"] = str(spec.local_parallel)
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
        logs = self.runner.run(
            [
                "docker",
                "compose",
                "-f",
                str(self.compose_file),
                "logs",
                "--tail",
                "80",
                service,
            ],
            env=environment,
            capture_output=True,
        )
        raise ComposeStartupError(
            f"{service} did not become healthy: {last_error}\n{logs[-8000:]}"
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
