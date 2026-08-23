from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager, suppress
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
    descriptor = config.get("input_files")
    if not isinstance(descriptor, dict) or key not in descriptor:
        raise RuntimeError(f"Worker config is missing input descriptor for {key!r}")
    value = descriptor[key]
    if not isinstance(value, dict):
        raise RuntimeError(f"Input descriptor for {key!r} must be an object")
    configured_value = value.get("path")
    filename = value.get("filename")
    expected_sha256 = value.get("sha256")
    if not all(
        isinstance(item, str) and item.strip()
        for item in (configured_value, filename, expected_sha256)
    ):
        raise RuntimeError(
            f"Input descriptor for {key!r} requires path, filename, and sha256"
        )
    if Path(filename).name != filename:
        raise RuntimeError(f"Input descriptor for {key!r} has invalid filename")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise RuntimeError(f"Input descriptor for {key!r} has invalid sha256")

    configured = Path(configured_value)
    if (
        configured.name == filename
        and configured.is_file()
        and sha256_file(configured) == expected_sha256
    ):
        return configured

    named = [path for path in Path(input_root).rglob(filename) if path.is_file()]
    matches = [path for path in named if sha256_file(path) == expected_sha256]
    if len(matches) == 1:
        return matches[0]
    digest_prefix = expected_sha256[:12]
    if not named:
        raise RuntimeError(
            f"Expected input {key!r} with filename {filename!r}; "
            f"no mounted file found under {input_root}"
        )
    if not matches:
        raise RuntimeError(
            f"Expected input {key!r} with filename {filename!r} and "
            f"checksum {digest_prefix}...; checksum mismatch among "
            f"{len(named)} candidate(s)"
        )
    raise RuntimeError(
        f"Expected one input {key!r} with filename {filename!r} and "
        f"checksum {digest_prefix}...; found {len(matches)} matches"
    )


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
        path for path in Path(input_root).rglob(configured.name) if path.is_file()
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
    runtime_summary: dict | None = None,
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
    if runtime_summary is not None:
        payload["runtime"] = dict(runtime_summary)
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
    runtime_summary: dict | None = None,
) -> CloudArtifact:
    missing = max(0, total - complete)
    completion = Completion(total, complete, missing)
    manifest_path = write_artifact_manifest(
        data_path,
        artifact_type=artifact_type,
        identity=identity,
        completion=completion,
        checkpoint_path=checkpoint_path,
        runtime_summary=runtime_summary,
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


class BoundedLogCollector:
    def __init__(self, max_bytes: int = 1024 * 1024):
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        self.max_bytes = int(max_bytes)
        self._buffer = bytearray()
        self._lock = threading.Lock()

    def feed(self, chunk: bytes | str) -> None:
        raw = (
            chunk.encode("utf-8", errors="replace")
            if isinstance(chunk, str)
            else bytes(chunk)
        )
        with self._lock:
            self._buffer.extend(raw)
            if len(self._buffer) > self.max_bytes:
                del self._buffer[: len(self._buffer) - self.max_bytes]

    def write(self, path: Path) -> None:
        with self._lock:
            data = bytes(self._buffer)
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    def tail_text(self, max_bytes: int = 16 * 1024) -> str:
        with self._lock:
            data = bytes(self._buffer[-max_bytes:])
        return data.decode("utf-8", errors="replace").strip()


def worker_deadline(config: dict, clock=time.monotonic) -> float:
    """Return a safety-margin deadline for bounded worker execution."""
    budget = float(
        config.get("budget_seconds", config.get("total_budget_seconds", 21_600))
    )
    margin = float(config.get("budget_margin_seconds", min(120.0, budget * 0.05)))
    return clock() + max(0.0, budget - margin)


def _discover_cuda_library_paths() -> list[str]:
    discovered: set[str] = set()
    for path in (
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/targets/x86_64-linux/lib",
        "/usr/local/cuda-12/lib64",
        "/usr/local/cuda-12.2/lib64",
        "/usr/local/cuda-12.1/lib64",
        "/usr/local/nvidia/lib64",
        "/usr/local/nvidia/lib",
        "/usr/lib/x86_64-linux-gnu",
    ):
        if os.path.isdir(path):
            discovered.add(path)
    try:
        import site
        import sys

        roots = set(sys.path)
        if hasattr(site, "getsitepackages"):
            roots.update(site.getsitepackages())
        for root in roots:
            p = Path(root)
            if not p.is_dir():
                continue
            for n_lib in (p / "nvidia").glob("*/lib"):
                if n_lib.is_dir():
                    discovered.add(str(n_lib))
            torch_lib = p / "torch" / "lib"
            if torch_lib.is_dir():
                discovered.add(str(torch_lib))
    except Exception:
        pass
    for root in ("/usr/local", "/opt/conda"):
        if not os.path.isdir(root):
            continue
        try:
            for match in Path(root).rglob("libcudart.so*"):
                if match.is_file() or match.is_symlink():
                    discovered.add(str(match.parent))
        except OSError:
            pass
    return sorted(discovered)


def _link_cuda_dependencies(runtime_lib_dir: Path) -> None:
    target_dir = Path(runtime_lib_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    cuda_dirs = _discover_cuda_library_paths()
    for directory in cuda_dirs:
        try:
            for so_file in Path(directory).glob("*.so*"):
                if so_file.is_file() or so_file.is_symlink():
                    dest = target_dir / so_file.name
                    if not dest.exists():
                        with suppress(OSError):
                            dest.symlink_to(so_file.resolve())
        except OSError:
            pass


def build_server_environment(
    base: dict[str, str], library: Path, visible_devices: str
) -> dict[str, str]:
    environment = dict(base)
    existing = environment.get("LD_LIBRARY_PATH", "").strip()
    paths = [str(library), *_discover_cuda_library_paths()]
    if existing:
        paths.append(existing)
    seen = set()
    unique_paths = []
    for p in paths:
        if p and p not in seen:
            seen.add(p)
            unique_paths.append(p)
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(unique_paths)
    environment["CUDA_VISIBLE_DEVICES"] = visible_devices
    return environment


def build_server_command(
    *,
    binary: str | Path,
    model: str | Path,
    port: int,
    visible_devices: str,
    spec: ModelSpec,
    runtime_overrides: dict[str, int] | None = None,
) -> list[str]:
    if runtime_overrides is None:
        raise ValueError("runtime_overrides is required")
    overrides = dict(runtime_overrides)
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
        str(int(overrides["server_slots"])),
        "-c",
        str(int(overrides["context_per_slot"]) * int(overrides["server_slots"])),
        "-b",
        str(int(overrides["logical_batch_size"])),
        "-ub",
        str(int(overrides["physical_batch_size"])),
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
    _link_cuda_dependencies(target / "lib")
    return target


@contextmanager
def managed_model_servers(
    config: dict,
    *,
    telemetry: object | None = None,
) -> Generator[list[WorkerServer], None, None]:
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
    commands: list[list[str]] = []
    collectors: list[BoundedLogCollector] = []
    readers: list[threading.Thread] = []
    servers: list[WorkerServer] = []
    base_port = int(config.get("server_port", 11434))
    runtime_overrides = config.get("runtime_overrides")
    if not isinstance(runtime_overrides, dict):
        raise ValueError("runtime_overrides is required")
    for index, layout in enumerate(server_layout(spec)):
        port = base_port + index
        command = build_server_command(
            binary=binary,
            model=model_path,
            port=port,
            visible_devices=layout.visible_devices,
            spec=spec,
            runtime_overrides=runtime_overrides,
        )
        commands.append(command)
        collector = BoundedLogCollector(
            int(config.get("server_log_max_bytes", 1024 * 1024))
        )
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=build_server_environment(os.environ, library, layout.visible_devices),
        )
        processes.append(process)
        collectors.append(collector)
        if process.stdout is not None:
            reader = threading.Thread(
                target=_drain_process_output,
                args=(process.stdout, collector),
                daemon=True,
            )
            reader.start()
            readers.append(reader)
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
                    failed = [
                        index
                        for index, process in enumerate(processes)
                        if process.poll() is not None
                    ]
                    if failed:
                        for index in failed:
                            readers[index].join(timeout=2)
                        replicas = ",".join(str(index) for index in failed)
                        raise RuntimeError(
                            _server_startup_diagnostic(
                                f"replica(s) {replicas} exited before becoming healthy",
                                servers,
                                processes,
                                collectors,
                                commands,
                            )
                        ) from None
                    if time.monotonic() >= deadline:
                        raise RuntimeError(
                            _server_startup_diagnostic(
                                "health check timed out",
                                servers,
                                processes,
                                collectors,
                                commands,
                            )
                        ) from None
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
        for reader in readers:
            reader.join(timeout=2)
        output_dir = Path(config.get("output_dir", "/kaggle/working/artifact"))
        for index, collector in enumerate(collectors):
            collector.write(output_dir / f"server-{index}.log")
        if telemetry is not None:
            close = getattr(telemetry, "close", None)
            write_report = getattr(telemetry, "write_report", None)
            if callable(close):
                close()
            if callable(write_report):
                write_report()


def _drain_process_output(stream, collector: BoundedLogCollector) -> None:
    while True:
        chunk = stream.readline()
        if not chunk:
            return
        collector.feed(chunk)


def _server_startup_diagnostic(
    reason: str,
    servers: list[WorkerServer],
    processes: list[subprocess.Popen],
    collectors: list[BoundedLogCollector],
    commands: list[list[str]],
) -> str:
    lines = [f"llama.cpp server startup failed: {reason}"]
    for index, (server, process, collector, command) in enumerate(
        zip(servers, processes, collectors, commands, strict=True)
    ):
        lines.append(
            f"replica={index} url={server.base_url} "
            f"devices={server.visible_devices} exit_code={process.poll()}"
        )
        lines.append(f"command={shlex.join(command)}")
        log_tail = collector.tail_text()
        lines.append(f"log_tail:\n{log_tail or '<empty>'}")
    return "\n".join(lines)
