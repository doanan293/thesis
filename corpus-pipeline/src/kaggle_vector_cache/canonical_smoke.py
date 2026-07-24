from __future__ import annotations

import json

from kaggle_vector_cache.models import ModelSpec, ModelTopology


def canonical_smoke_kernel_slug(spec: ModelSpec) -> str:
    return "vector-cache-smoke-" + spec.gguf_dataset_slug.removeprefix(
        "vector-cache-gguf-"
    )


def validate_canonical_smoke_report(
    report: dict,
    spec: ModelSpec,
    model_sha256: str,
) -> None:
    expected = {
        "status": "passed",
        "model": spec.name,
        "model_sha256": model_sha256,
        "embedding_dimension": spec.vector_dimension,
        "finite_embedding": True,
        "cuda_backend": True,
        "gpu_process_visible": True,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"Canonical smoke report has invalid {key}")
    gpu_names = report.get("gpu_names")
    if (
        not isinstance(gpu_names, list)
        or len(gpu_names) != 2
        or any("T4" not in str(name).upper() for name in gpu_names)
    ):
        raise ValueError("Canonical smoke report did not identify two Tesla T4 GPUs")


def build_canonical_smoke_kernel(spec: ModelSpec) -> str:
    config = {
        "model": spec.name,
        "filename": spec.canonical_filename,
        "dimension": spec.vector_dimension,
        "topology": spec.topology.value,
    }
    marker = (
        "SHARDED_1X2"
        if spec.topology == ModelTopology.SHARDED_1X2
        else "REPLICATED_2X1"
    )
    return (
        "import json\nCONFIG = json.loads("
        + repr(json.dumps(config))
        + ")\n"
        + f"""# {marker}

import hashlib
import json
import math
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

output = Path("/kaggle/working/canonical-smoke")
runtime = Path("/kaggle/working/canonical-smoke-runtime")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def wait_ready(process, port, log_path):
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                "llama-server exited during smoke startup\\n"
                + log_path.read_text(encoding="utf-8", errors="replace")[-6000:]
            )
        try:
            with urllib.request.urlopen(
                "http://127.0.0.1:" + str(port) + "/health", timeout=5
            ) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(2)
    raise RuntimeError("llama-server smoke health check timed out")


def embed(port, texts):
    request = urllib.request.Request(
        "http://127.0.0.1:" + str(port) + "/v1/embeddings",
        data=json.dumps({{"model": CONFIG["model"], "input": texts}}).encode(),
        headers={{"Content-Type": "application/json"}},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=240) as response:
        payload = json.loads(response.read().decode())
    return [item["embedding"] for item in payload["data"]]


shutil.rmtree(output, ignore_errors=True)
shutil.rmtree(runtime, ignore_errors=True)
output.mkdir(parents=True)
runtime.mkdir(parents=True)
runtime_manifests = list(Path("/kaggle/input").rglob("runtime_manifest.json"))
if len(runtime_manifests) != 1:
    raise RuntimeError("expected one packaged llama.cpp runtime")
runtime_manifest_path = runtime_manifests[0]
runtime_manifest = json.loads(runtime_manifest_path.read_text(encoding="utf-8"))
for logical, entry in runtime_manifest["files"].items():
    source = runtime_manifest_path.parent / entry["stored_name"]
    target = runtime / logical
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
server = runtime / "bin/llama-server"
server.chmod(server.stat().st_mode | 0o111)
driver_candidates = [
    Path("/usr/lib/x86_64-linux-gnu/libcuda.so.1"),
    Path("/usr/local/nvidia/lib64/libcuda.so.1"),
]
driver_source = next((path for path in driver_candidates if path.is_file()), None)
if driver_source is None:
    raise RuntimeError("NVIDIA driver library libcuda.so.1 was not found")
driver_link = runtime / "lib/libcuda.so.1"
driver_link.symlink_to(driver_source)

model_manifests = []
for path in Path("/kaggle/input").rglob("model_manifest.json"):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("model") == CONFIG["model"]:
        model_manifests.append((path, manifest))
if len(model_manifests) != 1:
    raise RuntimeError("expected one canonical model manifest")
manifest_path, manifest = model_manifests[0]
if manifest.get("filename") != CONFIG["filename"]:
    raise RuntimeError("canonical filename mismatch")
model = manifest_path.parent / CONFIG["filename"]
if not model.is_file() or sha256(model) != manifest.get("sha256"):
    raise RuntimeError("canonical model checksum mismatch")

gpu_names = subprocess.run(
    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
    check=True, text=True, stdout=subprocess.PIPE,
).stdout.strip().splitlines()
if len(gpu_names) != 2 or any("T4" not in name.upper() for name in gpu_names):
    raise RuntimeError("canonical smoke requires two Tesla T4 GPUs")

base_environment = dict(os.environ)
base_environment["LD_LIBRARY_PATH"] = str(runtime / "lib")
processes = []
logs = []
vectors = []
try:
    if CONFIG["topology"] == "replicated_2x1":
        devices = [0, 1]
        for device in devices:
            port = 18080 + device
            environment = dict(base_environment)
            environment["CUDA_VISIBLE_DEVICES"] = str(device)
            log_path = output / ("server-" + str(device) + ".log")
            log_handle = log_path.open("wb")
            process = subprocess.Popen(
                [
                    str(server), "--model", str(model), "--host", "127.0.0.1",
                    "--port", str(port), "--embedding", "--no-webui", "--offline",
                    "--n-gpu-layers", "99", "-np", "1", "-c", "4096",
                    "-b", "4096", "-ub", "2048",
                ],
                env=environment, stdout=log_handle, stderr=subprocess.STDOUT,
            )
            log_handle.close()
            processes.append(process)
            logs.append(log_path)
            wait_ready(process, port, log_path)
            vectors.extend(embed(port, ["smoke-" + str(device)]))
    else:
        port = 18080
        log_path = output / "server-sharded.log"
        log_handle = log_path.open("wb")
        process = subprocess.Popen(
            [
                str(server), "--model", str(model), "--host", "127.0.0.1",
                "--port", str(port), "--embedding", "--no-webui", "--offline",
                "--n-gpu-layers", "99", "--tensor-split", "1,1", "-np", "1",
                "-c", "4096", "-b", "4096", "-ub", "2048",
            ],
            env=base_environment, stdout=log_handle, stderr=subprocess.STDOUT,
        )
        log_handle.close()
        processes.append(process)
        logs.append(log_path)
        wait_ready(process, port, log_path)
        vectors.extend(embed(port, ["smoke-a", "smoke-b"]))
    compute_apps = subprocess.run(
        [
            "nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True, text=True, stdout=subprocess.PIPE,
    ).stdout
finally:
    for process in processes:
        process.terminate()
    for process in processes:
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

pids = {{str(process.pid) for process in processes}}
visible_rows = [
    row for row in compute_apps.splitlines()
    if row.split(",", 1)[0].strip() in pids
]
if CONFIG["topology"] == "replicated_2x1":
    gpu_process_visible = len({{row.split(",")[1].strip() for row in visible_rows}}) == 2
else:
    gpu_process_visible = len({{row.split(",")[1].strip() for row in visible_rows}}) == 2
log_text = "\\n".join(
    path.read_text(encoding="utf-8", errors="replace") for path in logs
)
finite = (
    len(vectors) == 2
    and all(len(vector) == CONFIG["dimension"] for vector in vectors)
    and all(math.isfinite(float(value)) for vector in vectors for value in vector)
)
cuda_backend = "CUDA" in log_text
report = {{
    "status": "passed" if finite and cuda_backend and gpu_process_visible else "failed",
    "model": CONFIG["model"],
    "model_sha256": manifest["sha256"],
    "embedding_dimension": len(vectors[0]) if vectors else 0,
    "finite_embedding": finite,
    "cuda_backend": cuda_backend,
    "gpu_process_visible": gpu_process_visible,
    "gpu_names": gpu_names,
    "topology": CONFIG["topology"],
}}
if report["status"] != "passed":
    raise RuntimeError("canonical GGUF smoke failed\\n" + log_text[-8000:])
(output / "canonical_smoke_report.json").write_text(
    json.dumps(report, indent=2) + "\\n", encoding="utf-8"
)
for path in logs:
    path.unlink(missing_ok=True)
shutil.rmtree(runtime, ignore_errors=True)
print(
    "[canonical-smoke] " + CONFIG["model"] + " passed on two T4 GPUs",
    flush=True,
)
"""
    )
