from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from corpus_pipeline.integrations.kaggle.artifacts import sha256_file
from corpus_pipeline.integrations.kaggle.models import (
    CloudArtifact,
    Completion,
    JobIdentity,
)
from corpus_pipeline.integrations.kaggle.workers.topology import server_layout
from corpus_pipeline.runtime.catalog import (
    ModelKind,
    ModelSpec,
    ModelTopology,
    require_model,
)


def find_unique(root: Path, filename: str, required: bool = True) -> Path | None:
    matches = [path for path in Path(root).rglob(filename) if path.is_file()]
    if not matches and not required:
        return None
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {filename}, found {len(matches)}")
    return matches[0]


def resolve_input_file(
    config: dict, key: str, *, input_root: Path = Path("/kaggle/input")
) -> Path:
    configured = Path(str(config[key]))
    if configured.is_file():
        return configured
    identity = config.get("identity")
    expected_sha256 = (
        str(identity.get("input_sha256") or "")
        if isinstance(identity, dict)
        else ""
    )
    candidates = [
        path
        for path in Path(input_root).rglob(configured.name)
        if path.is_file()
        and (not expected_sha256 or sha256_file(path) == expected_sha256)
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one mounted input matching {configured.name} "
            f"and job SHA-256, found {len(candidates)} under {input_root}"
        )
    return candidates[0]


def resolve_optional_input_file(
    config: dict,
    key: str,
    *,
    input_root: Path = Path("/kaggle/input"),
) -> Path | None:
    configured_value = config.get(key)
    if configured_value is None:
        return None
    configured = Path(str(configured_value))
    if configured.is_file():
        return configured
    candidates = [
        path
        for path in Path(input_root).rglob(configured.name)
        if path.is_file()
    ]
    if not candidates:
        return None
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected at most one mounted input matching {configured.name}, "
            f"found {len(candidates)} under {input_root}"
        )
    return candidates[0]


def identity_from_config(config: dict) -> JobIdentity:
    identity = config.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("Worker config is missing identity")
    return JobIdentity(dict(identity), str(config["job_sha256"]))


def write_artifact_manifest(
    data_path: Path,
    *,
    artifact_type: str,
    identity: JobIdentity,
    completion: Completion,
    checkpoint_path: Path | None = None,
) -> Path:
    checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
    manifest_path = Path(data_path).with_name("manifest.json")
    payload = {
        "schema_version": 1,
        "artifact_type": artifact_type,
        "created_at": datetime.now(UTC).isoformat(),
        "data_filename": Path(data_path).name,
        "data_sha256": sha256_file(data_path),
        "record_count": completion.complete,
        "identity": identity.payload | {"job_sha256": identity.sha256},
        "reuse_sha256": identity.reuse_sha256,
        "total": completion.total,
        "complete": completion.complete,
        "missing": completion.missing,
    }
    if checkpoint_path is not None:
        payload["checkpoint_filename"] = checkpoint_path.name
        payload["checkpoint_sha256"] = sha256_file(checkpoint_path)
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def artifact_from_output(
    data_path: Path,
    *,
    artifact_type: str,
    identity: JobIdentity,
    total: int,
    complete: int,
    checkpoint_path: Path | None = None,
) -> CloudArtifact:
    missing = max(0, total - complete)
    completion = Completion(total, complete, missing)
    manifest_path = write_artifact_manifest(
        data_path,
        artifact_type=artifact_type,
        identity=identity,
        completion=completion,
        checkpoint_path=checkpoint_path,
    )
    return CloudArtifact(
        Path(data_path),
        manifest_path,
        identity,
        completion,
        checkpoint_path=checkpoint_path,
    )


@dataclass(frozen=True)
class WorkerServer:
    base_url: str
    visible_devices: str


def worker_deadline(config: dict, clock=time.monotonic) -> float:
    """Return a safety-margin deadline for bounded worker execution."""
    budget = float(
        config.get("budget_seconds", config.get("total_budget_seconds", 21_600))
    )
    margin = float(config.get("budget_margin_seconds", min(120.0, budget * 0.05)))
    return clock() + max(0.0, budget - margin)


def build_server_environment(
    base: dict[str, str], library: Path, visible_devices: str
) -> dict[str, str]:
    environment = dict(base)
    existing = environment.get("LD_LIBRARY_PATH", "").strip()
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(
        part for part in (str(library), existing) if part
    )
    environment["CUDA_VISIBLE_DEVICES"] = visible_devices
    return environment


def build_server_command(
    *,
    binary: str | Path,
    model: str | Path,
    port: int,
    visible_devices: str,
    spec: ModelSpec,
) -> list[str]:
    command = [
        str(binary),
        "--model",
        str(model),
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
        "--offline",
        "--no-webui",
        "--n-gpu-layers",
        "99",
        "-np",
        str(spec.kaggle_parallel),
        "-c",
        str(spec.kaggle_context_per_slot * spec.kaggle_parallel),
        "-b",
        str(spec.kaggle_logical_batch_size),
        "-ub",
        str(spec.kaggle_physical_batch_size),
    ]
    if spec.kind is ModelKind.EMBEDDING:
        command.append("--embedding")
    elif spec.reranker_protocol == "native_rerank":
        command.append("--reranking")
    if spec.topology is ModelTopology.SHARDED_1X2:
        command.extend(["--tensor-split", "1,1"])
    return command


def materialize_runtime(source: Path, destination: Path) -> Path:
    manifest = json.loads(
        (Path(source) / "runtime_manifest.json").read_text(encoding="utf-8")
    )
    target = Path(destination)
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)
    for logical, record in manifest["files"].items():
        source_file = Path(source) / record["stored_name"]
        if sha256_file(source_file) != record["sha256"]:
            raise RuntimeError(f"Runtime hash mismatch: {logical}")
        destination_file = target / logical
        destination_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, destination_file)
        if logical == "bin/llama-server":
            destination_file.chmod(0o755)
    return target


@contextmanager
def managed_model_servers(config: dict) -> Generator[list[WorkerServer], None, None]:
    """Start one local llama.cpp server per configured replica on Kaggle."""
    spec = require_model(str(config["model"]))
    input_root = Path(config.get("kaggle_input_root", "/kaggle/input"))
    model_path = find_unique(input_root, "*.gguf")
    runtime_manifest = find_unique(input_root, "runtime_manifest.json")
    if model_path is None or runtime_manifest is None:
        raise RuntimeError("Kaggle input is missing GGUF model or runtime manifest")
    runtime_root = materialize_runtime(
        runtime_manifest.parent,
        Path(config.get("runtime_work_dir", "/tmp/llama-cpp-runtime")),
    )
    binary = runtime_root / "bin/llama-server"
    library = runtime_root / "lib"
    processes = []
    servers: list[WorkerServer] = []
    base_port = int(config.get("server_port", 11434))
    for index, layout in enumerate(server_layout(spec)):
        port = base_port + index
        command = build_server_command(
            binary=binary,
            model=model_path,
            port=port,
            visible_devices=layout.visible_devices,
            spec=spec,
        )
        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            env=build_server_environment(os.environ, library, layout.visible_devices),
        )
        processes.append(process)
        servers.append(WorkerServer(f"http://127.0.0.1:{port}", layout.visible_devices))
    try:
        from corpus_pipeline.runtime.client import LlamaCppClient

        deadline = time.monotonic() + float(config.get("server_start_timeout", 120))
        for server in servers:
            client = LlamaCppClient(server.base_url, timeout=5)
            while True:
                try:
                    client.health()
                    break
                except Exception:
                    if time.monotonic() >= deadline:
                        raise RuntimeError(
                            f"llama.cpp server did not start: {server.base_url}"
                        )
                    time.sleep(1)
        yield servers
    finally:
        for process in processes:
            process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
