from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

EMBEDDED_SOURCE_B64 = ""
EMBEDDED_CONFIG_JSON = ""
SOURCE_BUNDLE = Path("/tmp/source_bundle.zip")
if EMBEDDED_SOURCE_B64:
    SOURCE_BUNDLE.parent.mkdir(parents=True, exist_ok=True)
    SOURCE_BUNDLE.write_bytes(base64.b64decode(EMBEDDED_SOURCE_B64))
    sys.path.insert(0, str(SOURCE_BUNDLE))
else:
    bundled = Path(__file__).resolve().parent / "src"
    if not bundled.is_dir():
        bundled = Path(__file__).resolve().parent
    if bundled.is_dir() and str(bundled) not in sys.path:
        sys.path.insert(0, str(bundled))

from kaggle_vector_cache.manifests import (
    build_checkpoint_manifest,
    file_sha256,
    load_json,
    write_json,
)
from kaggle_vector_cache.models import ModelTopology, require_model
from vector_store.ingest_vectors import (
    detect_cache_vector_dimension,
    inspect_embedding_cache_completion,
)

INPUT_ROOT = Path("/kaggle/input")
WORKING_ROOT = Path("/kaggle/working")
TEMP_ROOT = Path("/tmp")


@dataclass(frozen=True)
class Server:
    port: int
    visible_devices: str

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


@dataclass
class Runtime:
    process: subprocess.Popen
    log_path: Path
    log_handle: object


def worker_milestone(event: str, **fields) -> str:
    suffix = "".join(f" | {key}={value}" for key, value in fields.items())
    return f"Worker {event}{suffix}"


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


def servers_for_topology(topology: ModelTopology) -> list[Server]:
    if topology is ModelTopology.REPLICATED_2X1:
        return [Server(11434, "0"), Server(11435, "1")]
    if topology is ModelTopology.SHARDED_1X2:
        return [Server(11434, "0,1")]
    raise RuntimeError(f"Unsupported topology: {topology}")


def build_server_command(
    binary: Path,
    model: Path,
    server: Server,
    parallel: int,
    topology: ModelTopology,
    context_per_slot: int = 2048,
    logical_batch: int = 2048,
    physical_batch: int = 2048,
) -> list[str]:
    context = context_per_slot * parallel
    command = [
        str(binary),
        "--model",
        str(model),
        "--port",
        str(server.port),
        "--host",
        "127.0.0.1",
        "--embedding",
        "--offline",
        "--no-webui",
        "--n-gpu-layers",
        "99",
        "-np",
        str(parallel),
        "-c",
        str(context),
        "-b",
        str(logical_batch),
        "-ub",
        str(physical_batch),
    ]
    if topology is ModelTopology.SHARDED_1X2:
        command.extend(["--tensor-split", "1,1"])
    return command


def build_ingest_command(
    python: Path,
    model: str,
    corpus: Path,
    cache: Path,
    budget: int,
    reserve: int,
    servers: list[Server],
    batch: int,
) -> list[str]:
    command = [
        str(python),
        "-m",
        "cli.ingest_vectors",
        "--model",
        model,
        "--input",
        str(corpus),
        "--embedding-cache",
        str(cache),
        "--cache-only",
        "--max-runtime-seconds",
        str(budget),
        "--stop-margin-seconds",
        str(reserve),
        "--input-batch-size",
        str(batch),
        "--server-mode",
        "external",
    ]
    for server in servers:
        command.extend(["--llama-server-url", server.base_url])
    return command


def find_unique(root: Path, filename: str, required: bool = True) -> Path | None:
    matches = [path for path in Path(root).rglob(filename) if path.is_file()]
    if not matches and not required:
        return None
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {filename}, found {len(matches)}")
    return matches[0]


def materialize_runtime(source: Path) -> Path:
    manifest = load_json(Path(source) / "runtime_manifest.json")
    target = TEMP_ROOT / "llama-cpp-runtime"
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)
    for logical, record in manifest["files"].items():
        source_file = Path(source) / record["stored_name"]
        if file_sha256(source_file) != record["sha256"]:
            raise RuntimeError(f"Runtime hash mismatch: {logical}")
        destination = target / logical
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, destination)
        if logical == "bin/llama-server":
            destination.chmod(0o755)
    return target


def start_servers(
    servers: list[Server],
    binary: Path,
    model: Path,
    library: Path,
    parallel: int,
    topology: ModelTopology,
    context_per_slot: int = 2048,
    logical_batch: int = 2048,
    physical_batch: int = 2048,
) -> list[Runtime]:
    runtimes = []
    logs = TEMP_ROOT / "llama-server-logs"
    logs.mkdir(parents=True, exist_ok=True)
    for index, server in enumerate(servers):
        environment = build_server_environment(
            dict(os.environ), library, server.visible_devices
        )
        log_path = logs / f"server-{index}.log"
        handle = log_path.open("wb")
        process = subprocess.Popen(
            build_server_command(
                binary,
                model,
                server,
                parallel,
                topology,
                context_per_slot,
                logical_batch,
                physical_batch,
            ),
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        runtimes.append(Runtime(process, log_path, handle))
    return runtimes


def stop_servers(runtimes: list[Runtime]) -> None:
    for runtime in runtimes:
        runtime.process.terminate()
    for runtime in runtimes:
        try:
            runtime.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            runtime.process.kill()
            runtime.process.wait()
        runtime.log_handle.close()


def server_log_tail(path: Path, max_bytes: int = 8000) -> str:
    try:
        with Path(path).open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max(1, int(max_bytes))))
            return handle.read().decode("utf-8", errors="replace").strip()
    except OSError as exc:
        return f"server log unavailable: {exc}"


def wait_and_probe(
    server: Server,
    model: str,
    dimension: int,
    runtime: Runtime,
    timeout: int = 300,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        return_code = runtime.process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"Server exited with exit code {return_code}: {server.base_url}\n"
                f"{server_log_tail(runtime.log_path)}"
            )
        try:
            with urllib.request.urlopen(
                f"{server.base_url}/health", timeout=5
            ) as response:
                if response.status == 200:
                    break
        except Exception:
            time.sleep(2)
    else:
        raise RuntimeError(
            f"Server did not become ready: {server.base_url}\n"
            f"{server_log_tail(runtime.log_path)}"
        )
    request = urllib.request.Request(
        f"{server.base_url}/v1/embeddings",
        data=json.dumps({"model": model, "input": ["warmup"]}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        payload = json.loads(response.read())
    if len(payload["data"][0]["embedding"]) != dimension:
        raise RuntimeError("Embedding dimension mismatch")


def finalize_checkpoint(
    corpus: Path,
    cache: Path,
    spec,
    output: Path,
    config: dict,
    interruption: str | None = None,
):
    cache.touch(exist_ok=True)
    completion = inspect_embedding_cache_completion(corpus, cache, spec.name)
    vector_dimension = (
        detect_cache_vector_dimension(cache, spec.name)
        if completion.complete
        else int(spec.vector_dimension)
    )
    profile_path = output / "autotune_profile.json"
    corpus_manifest_path = corpus.parent / "manifest.json"
    if corpus_manifest_path.is_file():
        try:
            corpus_hash = load_json(corpus_manifest_path).get(
                "sha256", file_sha256(corpus)
            )
        except Exception:
            corpus_hash = file_sha256(corpus)
    else:
        corpus_hash = file_sha256(corpus)

    checkpoint = build_checkpoint_manifest(
        model=spec.name,
        vector_dim=vector_dimension,
        corpus_sha256=corpus_hash,
        cache_path=cache,
        total=completion.total,
        complete=completion.complete,
        missing=completion.missing,
        kernel_version=str(config.get("kernel_version", "unknown")),
        autotune_profile_path=profile_path if profile_path.is_file() else None,
        interruption=interruption,
    )
    manifest_path = output / "checkpoint_manifest.json"
    write_json(manifest_path, checkpoint)
    try:
        import zipfile

        zip_dir = WORKING_ROOT if WORKING_ROOT.exists() else output.parent
        zip_path = zip_dir / "checkpoint.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(manifest_path, "manifest.json")
            if cache.is_file():
                zf.write(cache, "checkpoint.jsonl")
            if profile_path.is_file():
                zf.write(profile_path, "autotune_profile.json")
    except Exception as zip_err:
        print(f"Warning: Failed to create output zip: {zip_err}", flush=True)

    print(
        worker_milestone(
            "checkpoint",
            complete=completion.complete,
            total=completion.total,
            missing=completion.missing,
        ),
        flush=True,
    )
    return completion


def resolve_runtime_budget(
    config: dict[str, object], started: float, now: float
) -> tuple[int, int]:
    total = int(config.get("total_budget_seconds", 21_600))
    reserve = int(config.get("export_reserve_seconds", 900))
    return int(total - (now - started)), reserve


def main() -> int:
    started = time.monotonic()
    config = (
        json.loads(EMBEDDED_CONFIG_JSON)
        if EMBEDDED_CONFIG_JSON
        else load_json(Path("vector_cache_config.json"))
    )
    spec = require_model(str(config["model"]))
    gpu_output = subprocess.run(
        ["nvidia-smi", "-L"], check=True, text=True, stdout=subprocess.PIPE
    ).stdout
    gpu_lines = [line for line in gpu_output.splitlines() if line.startswith("GPU ")]
    if len(gpu_lines) != 2 or any("T4" not in line.upper() for line in gpu_lines):
        raise RuntimeError("Exactly two Tesla T4 GPUs are required")
    print(worker_milestone("gpu_ready", devices="2xT4"), flush=True)
    runtime_manifest = find_unique(INPUT_ROOT, "runtime_manifest.json")
    runtime = materialize_runtime(runtime_manifest.parent)
    model_file = find_unique(INPUT_ROOT, spec.canonical_filename)
    if (
        model_file.stat().st_size != spec.byte_size
        or file_sha256(model_file) != spec.sha256
    ):
        raise RuntimeError(f"GGUF identity mismatch for {spec.name}")
    corpus = find_unique(INPUT_ROOT, "chunks.jsonl")
    print(
        worker_milestone(
            "assets_verified", model=spec.name, runtime="llama.cpp", corpus=corpus.name
        ),
        flush=True,
    )
    output = WORKING_ROOT / "data/cache/vector_embeddings"
    output.mkdir(parents=True, exist_ok=True)
    cache = output / f"{spec.slug}.jsonl"
    previous = find_unique(INPUT_ROOT, cache.name, required=False)
    if previous:
        shutil.copy2(previous, cache)
    print(
        worker_milestone("resume_cache", found=str(previous is not None).lower()),
        flush=True,
    )
    servers = servers_for_topology(spec.topology)
    runtimes = start_servers(
        servers,
        runtime / "bin/llama-server",
        model_file,
        runtime / "lib",
        spec.kaggle_parallel,
        spec.topology,
        spec.kaggle_context_per_slot,
        spec.kaggle_logical_batch_size,
        spec.kaggle_physical_batch_size,
    )
    interruption = None
    try:
        for server, server_runtime in zip(servers, runtimes, strict=True):
            wait_and_probe(
                server, spec.name, int(spec.vector_dimension), server_runtime
            )
        print(
            worker_milestone(
                "server_ready",
                topology=spec.topology.value,
                parallel=spec.kaggle_parallel,
                batch=spec.kaggle_request_batch_size,
                servers=len(servers),
                context_per_slot=spec.kaggle_context_per_slot,
                logical_batch=spec.kaggle_logical_batch_size,
                physical_batch=spec.kaggle_physical_batch_size,
            ),
            flush=True,
        )
        profile = {
            "schema_version": 1,
            "runtime": "llama.cpp",
            "model": spec.name,
            "gguf_sha256": spec.sha256,
            "topology": spec.topology.value,
            "selected_parallel": spec.kaggle_parallel,
            "selected_batch_size": spec.kaggle_request_batch_size,
        }
        profile_path = output / "autotune_profile.json"
        write_json(profile_path, profile)
        if config.get("benchmark_only"):
            write_json(output / "benchmark_summary.json", profile)
            return 0
        budget, reserve = resolve_runtime_budget(
            config, started=started, now=time.monotonic()
        )
        if budget <= reserve:
            raise RuntimeError("Insufficient runtime budget")
        print(
            worker_milestone(
                "ingest_start", budget_seconds=budget, reserve_seconds=reserve
            ),
            flush=True,
        )
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(
            SOURCE_BUNDLE if SOURCE_BUNDLE.is_file() else Path.cwd() / "src"
        )
        try:
            subprocess.run(
                build_ingest_command(
                    Path(sys.executable),
                    spec.name,
                    corpus,
                    cache,
                    budget,
                    reserve,
                    servers,
                    spec.kaggle_request_batch_size,
                ),
                check=True,
                env=environment,
            )
        except subprocess.CalledProcessError as exc:
            interruption = str(exc)
            print(
                worker_milestone("ingest_interrupted", return_code=exc.returncode),
                flush=True,
            )
    finally:
        stop_servers(runtimes)
    try:
        finalize_checkpoint(
            corpus, cache, spec, output, config, interruption=interruption
        )
    except Exception as checkpoint_error:
        if interruption:
            raise RuntimeError(
                f"Ingestion failed ({interruption}) and checkpoint finalization "
                f"also failed: {checkpoint_error}"
            ) from checkpoint_error
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
