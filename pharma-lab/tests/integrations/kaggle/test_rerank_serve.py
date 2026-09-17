import asyncio
import json
import sys
import threading
from pathlib import Path

import pytest
from tests.integrations.kaggle.factories import (
    UnusedKernelService,
    rerank_runtime_profile,
    stage_request,
)

from pharma_lab.integrations.kaggle.kernels import PipelineKernelService
from pharma_lab.integrations.kaggle.models import StageName
from pharma_lab.integrations.kaggle.stages import (
    SERVE_REPORT_FILENAME,
    get_stage_adapter,
)
from pharma_lab.integrations.kaggle.workers.serve import (
    ProxyStats,
    start_proxy,
    start_tunnel,
    upstream_ports,
)

MODEL = "qwen3-reranker:0.6b-fp16"


def request_file(tmp_path: Path, nonce: str) -> Path:
    path = tmp_path / f"request-{nonce}.json"
    path.write_text(
        json.dumps({"hours": 1, "api_key": "k", "nonce": nonce}), encoding="utf-8"
    )
    return path


def build(tmp_path: Path, nonce: str):
    return get_stage_adapter(StageName.RERANK_SERVE).build_job(
        stage_request(
            StageName.RERANK_SERVE,
            MODEL,
            request_file(tmp_path, nonce),
            runtime_profile=rerank_runtime_profile(MODEL),
        )
    )


def test_each_serve_request_is_a_new_one_record_job(tmp_path: Path) -> None:
    first = build(tmp_path, "a")
    second = build(tmp_path, "b")

    assert first.identity.sha256 != second.identity.sha256
    assert first.worker_module == "pharma_lab.integrations.kaggle.workers.serve"
    assert first.data_filename == SERVE_REPORT_FILENAME
    assert first.worker_config["enable_internet"] is True
    assert first.worker_config["runtime_overrides"] == (
        rerank_runtime_profile(MODEL).to_dict()
    )


def test_serve_requires_a_reranker_and_a_profile(tmp_path: Path) -> None:
    adapter = get_stage_adapter(StageName.RERANK_SERVE)
    with pytest.raises(ValueError, match="requires a reranker"):
        adapter.build_job(
            stage_request(
                StageName.RERANK_SERVE,
                "qwen3-embedding:0.6b-fp16",
                request_file(tmp_path, "x"),
            )
        )


def test_only_serve_kernels_enable_internet(tmp_path: Path) -> None:
    service = PipelineKernelService(
        service=UnusedKernelService(), owner="owner", source_root=Path("src")
    )
    bundle = service.prepare_bundle(
        build(tmp_path, "a"),
        root=tmp_path / "bundle",
        dataset_references=["owner/runtime"],
        checkpoint_reference=None,
        total_budget_seconds=60,
    )
    metadata = json.loads((bundle / "kernel-metadata.json").read_text("utf-8"))
    assert metadata["enable_internet"] == "true"


def test_upstream_ports_come_from_server_urls() -> None:
    assert upstream_ports(["http://127.0.0.1:11434", "http://127.0.0.1:11435"]) == [
        11434,
        11435,
    ]


async def _tagged_server(tag: bytes) -> tuple[asyncio.AbstractServer, int]:
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        data = await reader.read(100)
        writer.write(tag + data)
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    return server, server.sockets[0].getsockname()[1]


def test_round_robin_proxy_alternates_replicas() -> None:
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        servers = [
            asyncio.run_coroutine_threadsafe(_tagged_server(tag), loop).result()
            for tag in (b"A:", b"B:")
        ]
        stats = ProxyStats()
        listen = 18999
        start_proxy([port for _, port in servers], listen, stats)

        async def ask() -> bytes:
            reader, writer = await asyncio.open_connection("127.0.0.1", listen)
            writer.write(b"ping")
            await writer.drain()
            reply = await reader.read(100)
            writer.close()
            await writer.wait_closed()
            return reply

        replies = [asyncio.run(ask()) for _ in range(3)]

        assert replies == [b"A:ping", b"B:ping", b"A:ping"]
        assert stats.connections == 3
        for server, _ in servers:
            server.close()
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()


def fake_cloudflared(tmp_path: Path, output: str) -> Path:
    script = tmp_path / "cloudflared"
    script.write_text(
        f"#!{sys.executable}\n"
        "import sys, time\n"
        f"sys.stdout.write({output!r})\n"
        "sys.stdout.flush()\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def test_tunnel_url_is_read_from_cloudflared_output(tmp_path: Path) -> None:
    binary = fake_cloudflared(
        tmp_path,
        "INF Requesting new quick Tunnel\n"
        "INF |  https://brave-cat-12.trycloudflare.com  |\n",
    )
    process, url = start_tunnel(binary, 18080, timeout=10)
    try:
        assert url == "https://brave-cat-12.trycloudflare.com"
        assert process.poll() is None
    finally:
        process.kill()
        process.wait()
        if process.stdout is not None:
            process.stdout.close()


def test_tunnel_without_url_fails(tmp_path: Path) -> None:
    binary = tmp_path / "cloudflared"
    binary.write_text(f"#!{sys.executable}\nprint('no url')\n", encoding="utf-8")
    binary.chmod(0o755)
    with pytest.raises(RuntimeError, match="did not report"):
        start_tunnel(binary, 18080, timeout=5)
