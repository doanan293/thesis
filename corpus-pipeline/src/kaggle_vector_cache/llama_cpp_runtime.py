from __future__ import annotations

import hashlib
import json
from pathlib import Path

LLAMA_CPP_TAG = "b9637"
LLAMA_CPP_COMMIT = "aedb2a5e9ca3d4064148bbb919e0ddc0c1b70ab3"
LLAMA_CPP_ARCHIVE_SHA256 = (
    "3857876e4a2461f7041166bd74b5d39e3db51b8639353d55f87d6f904b3b75bd"
)
LLAMA_CPP_DATASET_SLUG = "vector-cache-llama-cpp-cuda-t4"
LLAMA_CPP_BUILDER_KERNEL_SLUG = "vector-cache-build-llama-cpp-cuda-t4"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_runtime_artifact(root: Path) -> dict:
    root = Path(root)
    manifest_path = root / "runtime_manifest.json"
    report_path = root / "gpu_smoke_report.json"
    if not manifest_path.is_file() or not report_path.is_file():
        raise ValueError(
            "llama.cpp runtime output is missing its manifest or GPU report"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if manifest.get("build_commit") != LLAMA_CPP_COMMIT:
        raise ValueError("llama.cpp runtime commit does not match the pinned build")
    required_report = {
        "status": "passed",
        "cuda_backend": True,
        "gpu_layers_offloaded": True,
        "embedding_dimension": 1024,
        "finite_embedding": True,
    }
    for key, expected in required_report.items():
        if report.get(key) != expected:
            raise ValueError(
                f"CUDA smoke report has invalid {key}: {report.get(key)!r}"
            )
    if "T4" not in str(report.get("gpu_name", "")).upper():
        raise ValueError("CUDA smoke report did not identify a Tesla T4")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("llama.cpp runtime manifest contains no files")
    if "bin/llama-server" not in files or "lib/libggml-cuda.so" not in files:
        raise ValueError("llama.cpp runtime manifest is missing server or CUDA backend")
    for logical_path, entry in files.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid runtime manifest entry: {logical_path}")
        stored = root / str(entry.get("stored_name", ""))
        if not stored.is_file() or _sha256(stored) != entry.get("sha256"):
            raise ValueError(f"Runtime file checksum mismatch: {logical_path}")
    return manifest


def build_kernel_source(model_dataset_slug: str) -> str:
    config = {
        "commit": LLAMA_CPP_COMMIT,
        "archive_sha256": LLAMA_CPP_ARCHIVE_SHA256,
        "model_dataset_slug": model_dataset_slug,
    }
    return f'''from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tarfile
import time
import urllib.request
from pathlib import Path

CONFIG = {json.dumps(config)!r}
config = json.loads(CONFIG)
work = Path("/kaggle/working/llama-cpp-build")
source_archive = work / "llama.cpp.tar.gz"
source_root = work / "source"
build_root = work / "build"
output = Path("/kaggle/working/llama-cpp-runtime")
smoke_model = "bge-m3:567m-fp16"
expected_embedding_dimension = 1024


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def run(command, **kwargs):
    print("[llama.cpp-builder] " + " ".join(map(str, command)), flush=True)
    return subprocess.run(command, check=True, **kwargs)


shutil.rmtree(work, ignore_errors=True)
shutil.rmtree(output, ignore_errors=True)
work.mkdir(parents=True)
output.mkdir(parents=True)
archive_url = "https://github.com/ggml-org/llama.cpp/archive/" + config["commit"] + ".tar.gz"
print("[llama.cpp-builder] downloading pinned upstream source", flush=True)
urllib.request.urlretrieve(archive_url, source_archive)
if sha256(source_archive) != config["archive_sha256"]:
    raise RuntimeError("upstream source archive checksum mismatch")
source_root.mkdir()
with tarfile.open(source_archive, "r:gz") as archive:
    archive.extractall(source_root, filter="data")
checkout = next(source_root.iterdir())
driver_candidates = [
    Path("/usr/lib/x86_64-linux-gnu/libcuda.so.1"),
    Path("/usr/local/nvidia/lib64/libcuda.so.1"),
]
driver_source = next((path for path in driver_candidates if path.is_file()), None)
if driver_source is None:
    raise RuntimeError("NVIDIA driver library libcuda.so.1 was not found")
driver_link = work / "libcuda.so"
driver_link.symlink_to(driver_source)

run([
    "cmake", "-S", str(checkout), "-B", str(build_root),
    "-DGGML_CUDA=ON",
    "-DCMAKE_CUDA_ARCHITECTURES=75",
    "-DCUDA_cuda_driver_LIBRARY=" + str(driver_link),
    "-DGGML_NATIVE=OFF",
    "-DLLAMA_BUILD_TESTS=OFF",
    "-DLLAMA_BUILD_EXAMPLES=OFF",
    "-DLLAMA_BUILD_SERVER=ON",
    "-DLLAMA_BUILD_UI=OFF",
    "-DCMAKE_BUILD_TYPE=Release",
])
run(["cmake", "--build", str(build_root), "--config", "Release", "-j4", "--target", "llama-server"])
server = build_root / "bin/llama-server"
if not server.is_file():
    raise RuntimeError("llama-server was not produced")
runtime_driver_link = build_root / "bin/libcuda.so.1"
runtime_driver_link.symlink_to(driver_source)

gpu_query = run([
    "nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"
], text=True, stdout=subprocess.PIPE).stdout.strip().splitlines()
if not gpu_query or any("T4" not in line.upper() for line in gpu_query):
    raise RuntimeError("builder requires Tesla T4 GPU")

model_mappings = []
for mapping_path in Path("/kaggle/input").rglob("flat_mapping.json"):
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    if mapping.get("model") == smoke_model:
        model_mappings.append((mapping_path, mapping))
if len(model_mappings) != 1:
    raise RuntimeError("expected one BGE-M3 flat mapping")
mapping_path, mapping = model_mappings[0]
blob_entries = [entry for entry in mapping["files"] if entry.get("kind") == "model_blob"]
if not blob_entries:
    raise RuntimeError("BGE-M3 mapping contains no model blob")
model_entry = max(blob_entries, key=lambda entry: int(entry.get("size", 0)))
model_blob = mapping_path.parent / model_entry["flat_name"]

server_log = work / "server.log"
environment = dict(os.environ)
environment["LD_LIBRARY_PATH"] = str(build_root / "bin")
with server_log.open("wb") as log_handle:
    process = subprocess.Popen([
        str(server), "--model", str(model_blob), "--host", "127.0.0.1",
        "--port", "18080", "--embedding", "--no-webui", "--offline",
        "--n-gpu-layers", "99", "-np", "2", "-c", "4096", "-b", "4096",
        "-ub", "2048",
    ], env=environment, stdout=log_handle, stderr=subprocess.STDOUT)
try:
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if process.poll() is not None:
            break
        try:
            with urllib.request.urlopen("http://127.0.0.1:18080/health", timeout=5) as response:
                if response.status == 200:
                    break
        except Exception:
            time.sleep(2)
    else:
        raise RuntimeError("llama-server health check timed out")
    if process.poll() is not None:
        raise RuntimeError(
            "llama-server exited during startup\\n"
            + server_log.read_text(encoding="utf-8", errors="replace")[-12000:]
        )
    request = urllib.request.Request(
        "http://127.0.0.1:18080/v1/embeddings",
        data=json.dumps({{"model": smoke_model, "input": ["smoke test"]}}).encode(),
        headers={{"Content-Type": "application/json"}}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = json.loads(response.read().decode())
    compute_apps = run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
    ).stdout
    gpu_process_visible = any(
        line.split(",", 1)[0].strip() == str(process.pid)
        for line in compute_apps.splitlines()
    )
finally:
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()

log_text = server_log.read_text(encoding="utf-8", errors="replace")
embedding = payload["data"][0]["embedding"]
cuda_backend = "CUDA" in log_text
gpu_layers_offloaded = gpu_process_visible and cuda_backend
report = {{
    "status": "passed",
    "build_tag": "{LLAMA_CPP_TAG}",
    "build_commit": config["commit"],
    "gpu_name": "; ".join(gpu_query),
    "cuda_backend": cuda_backend,
    "gpu_layers_offloaded": gpu_layers_offloaded,
    "compute_apps": compute_apps.strip().splitlines(),
    "embedding_dimension": len(embedding),
    "finite_embedding": all(math.isfinite(float(value)) for value in embedding),
}}
if not cuda_backend or not gpu_layers_offloaded:
    raise RuntimeError("upstream llama-server did not use CUDA GPU layers\\n" + log_text[-6000:])
if report["embedding_dimension"] != expected_embedding_dimension or not report["finite_embedding"]:
    raise RuntimeError("BGE-M3 smoke response is invalid")

candidates = [
    server,
    *sorted(
        path
        for path in (build_root / "bin").glob("*.so*")
        if not path.name.startswith("libcuda")
    ),
]
logical_files = {{}}
for index, source in enumerate(dict.fromkeys(candidates), start=1):
    logical = "bin/llama-server" if source == server else "lib/" + source.name
    stored_name = "artifact-%04d.bin" % index
    shutil.copy2(source, output / stored_name)
    logical_files[logical] = {{
        "stored_name": stored_name,
        "size": source.stat().st_size,
        "sha256": sha256(source),
    }}
manifest = {{
    "schema_version": 1,
    "build_tag": "{LLAMA_CPP_TAG}",
    "build_commit": config["commit"],
    "cuda_architectures": [75],
    "files": logical_files,
}}
(output / "runtime_manifest.json").write_text(json.dumps(manifest, indent=2) + "\\n")
(output / "gpu_smoke_report.json").write_text(json.dumps(report, indent=2) + "\\n")
shutil.rmtree(work)
print("[llama.cpp-builder] CUDA build and BGE-M3 smoke passed", flush=True)
'''
