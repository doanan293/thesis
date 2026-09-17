"""Serve a reranker from a Kaggle GPU session through a cloudflared quick tunnel.

The local E2E harness points its reranker at the printed URL. llama-server checks the
API key from the request file; a small round-robin proxy spreads connections over the
replicas, and the tunnel exposes only that proxy.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import stat
import subprocess
import threading
import time
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from pharma_lab.integrations.kaggle.models import CloudArtifact
from pharma_lab.integrations.kaggle.workers.runtime import (
    artifact_from_output,
    identity_from_config,
    managed_model_servers,
    resolve_input_file,
    worker_deadline,
)

CLOUDFLARED_URL = (
    "https://github.com/cloudflare/cloudflared/releases/latest/download/"
    "cloudflared-linux-amd64"
)
TUNNEL_URL = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
PROXY_PORT = 18080
HEARTBEAT_SECONDS = 300


@dataclass
class ProxyStats:
    connections: int = 0
    active: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


def upstream_ports(base_urls: Sequence[str]) -> list[int]:
    return [int(url.rsplit(":", 1)[1]) for url in base_urls]


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while chunk := await reader.read(65536):
            writer.write(chunk)
            await writer.drain()
    finally:
        writer.close()


async def serve_round_robin(
    ports: Sequence[int], listen_port: int, stats: ProxyStats, ready: threading.Event
) -> None:
    """Accept on 127.0.0.1:listen_port and hand each connection to the next replica."""
    turn = 0

    async def handle(
        client_reader: asyncio.StreamReader, client_writer: asyncio.StreamWriter
    ) -> None:
        nonlocal turn
        port = ports[turn % len(ports)]
        turn += 1
        with stats.lock:
            stats.connections += 1
            stats.active += 1
        try:
            upstream_reader, upstream_writer = await asyncio.open_connection(
                "127.0.0.1", port
            )
            await asyncio.gather(
                _pipe(client_reader, upstream_writer),
                _pipe(upstream_reader, client_writer),
                return_exceptions=True,
            )
        except OSError:
            client_writer.close()
        finally:
            with stats.lock:
                stats.active -= 1

    server = await asyncio.start_server(handle, "127.0.0.1", listen_port)
    ready.set()
    async with server:
        await server.serve_forever()


def start_proxy(ports: Sequence[int], listen_port: int, stats: ProxyStats) -> None:
    ready = threading.Event()
    thread = threading.Thread(
        target=lambda: asyncio.run(serve_round_robin(ports, listen_port, stats, ready)),
        daemon=True,
    )
    thread.start()
    if not ready.wait(timeout=30):
        raise RuntimeError("round-robin proxy did not start")


def download_cloudflared(target: Path) -> Path:
    if not target.is_file():
        urllib.request.urlretrieve(CLOUDFLARED_URL, target)
        target.chmod(target.stat().st_mode | stat.S_IXUSR)
    return target


def _discard(stream: TextIO) -> None:
    """Keep reading the tunnel's output so its pipe never fills."""
    for _ in iter(stream.readline, ""):
        pass


def start_tunnel(
    binary: Path, port: int, *, timeout: float = 120
) -> tuple[subprocess.Popen[str], str]:
    process = subprocess.Popen(
        [
            str(binary),
            "tunnel",
            "--no-autoupdate",
            "--url",
            f"http://127.0.0.1:{port}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert process.stdout is not None
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        line = process.stdout.readline()
        if not line:
            if process.poll() is not None:
                break
            continue
        match = TUNNEL_URL.search(line)
        if match:
            threading.Thread(
                target=_discard, args=(process.stdout,), daemon=True
            ).start()
            return process, match.group(0)
    process.kill()
    process.wait()
    process.stdout.close()
    raise RuntimeError("cloudflared did not report a tunnel URL")


def serve_until(
    config: dict,
    *,
    hours: float,
    api_key: str,
    clock=time.monotonic,
) -> dict:
    os.environ["LLAMA_ARG_API_KEY"] = api_key
    deadline = min(worker_deadline(config, clock), clock() + hours * 3600)
    stats = ProxyStats()
    report: dict = {"urls": []}
    with managed_model_servers(config) as servers:
        start_proxy(upstream_ports([s.base_url for s in servers]), PROXY_PORT, stats)
        binary = download_cloudflared(Path("/tmp/cloudflared"))
        tunnel, url = start_tunnel(binary, PROXY_PORT)
        report["urls"].append(url)
        print(f"[serve] url={url} replicas={len(servers)}", flush=True)
        last_beat = clock()
        try:
            while clock() < deadline:
                time.sleep(10)
                if tunnel.poll() is not None:
                    tunnel, url = start_tunnel(binary, PROXY_PORT)
                    report["urls"].append(url)
                    print(f"[serve] url={url} (tunnel restarted)", flush=True)
                if clock() - last_beat >= HEARTBEAT_SECONDS:
                    last_beat = clock()
                    with stats.lock:
                        print(
                            f"[serve] alive connections={stats.connections} "
                            f"active={stats.active} "
                            f"remaining_seconds={int(deadline - clock())}",
                            flush=True,
                        )
        finally:
            tunnel.terminate()
    report["connections"] = stats.connections
    print("[serve] stopped", flush=True)
    return report


def run_serve_worker(config: dict) -> CloudArtifact:
    identity = identity_from_config(config)
    request_path = resolve_input_file(config, "serve_request")
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    report = serve_until(
        config, hours=float(request["hours"]), api_key=str(request["api_key"])
    )
    output_dir = Path(config.get("output_dir", "/kaggle/working/artifact"))
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "serve_report.jsonl"
    data_path.write_text(json.dumps(report) + "\n", encoding="utf-8")
    return artifact_from_output(
        data_path,
        artifact_type="serve_report",
        identity=identity,
        total=1,
        complete=1,
    )


def main() -> int:
    config = json.loads(
        Path(os.environ["KAGGLE_PIPELINE_CONFIG"]).read_text(encoding="utf-8")
    )
    run_serve_worker(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
