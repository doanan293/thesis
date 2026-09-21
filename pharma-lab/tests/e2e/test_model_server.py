import json
import shlex
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest
from pharma_agent.domain.llm.models import LlmRole

from pharma_lab.e2e.model_server import (
    UrlWatcher,
    api_key_from_runner,
    api_key_path,
    env_lines,
    find_url,
    serve_stage,
    write_request,
)
from pharma_lab.integrations.kaggle.models import StageName

RERANKER = "qwen3-reranker:4b-fp16"


def test_each_request_has_a_new_key_and_private_permissions(tmp_path: Path) -> None:
    first, key_a = write_request(tmp_path, hours=2)
    payload = json.loads(first.read_text("utf-8"))
    assert payload["hours"] == 2 and "api_key" not in payload
    key_file = api_key_path(first)
    assert key_file.read_text("utf-8") == key_a
    assert stat.S_IMODE(key_file.stat().st_mode) == 0o600
    _, key_b = write_request(tmp_path, hours=2)
    assert key_a != key_b
    with pytest.raises(ValueError, match="positive"):
        write_request(tmp_path, hours=0)


def test_env_lines_configure_the_backend_reranker() -> None:
    text = env_lines(
        model="qwen3-reranker:4b-fp16", url="https://x.trycloudflare.com", api_key="k"
    )
    assert "PHARMA_RETRIEVAL__RERANK__BASE_URL=https://x.trycloudflare.com\n" in text
    assert "PHARMA_RETRIEVAL__RERANK__API_KEY=k\n" in text
    assert "PHARMA_RETRIEVAL__RERANK__MODEL=qwen3-reranker:4b-fp16\n" in text


def test_env_lines_route_every_llm_role_to_the_served_chat_model() -> None:
    text = env_lines(
        model="qwen3.5:9b-f16", url="https://y.trycloudflare.com", api_key="k"
    )
    values = dict(shlex.split(line)[0].split("=", 1) for line in text.splitlines())
    for role in LlmRole:
        prefix = f"PHARMA_LLM__ROLES__{role.name}__"
        assert values[prefix + "BASE_URL"] == "https://y.trycloudflare.com/v1"
        assert values[prefix + "API_KEY"] == "k"
        assert values[prefix + "MODEL"] == "qwen3.5:9b-f16"
        assert json.loads(values[prefix + "EXTRA_BODY"]) == {
            "chat_template_kwargs": {"enable_thinking": False}
        }
    assert "PHARMA_RETRIEVAL__RERANK__BASE_URL" not in values
    assert values["PHARMA_LLM__TIMEOUT_SECONDS"] == "600"


def test_serve_stage_follows_the_model_kind() -> None:
    assert serve_stage("qwen3-reranker:4b-fp16") is StageName.RERANK_SERVE
    assert serve_stage("qwen3.5:9b-f16") is StageName.LLM_SERVE
    with pytest.raises(ValueError, match="reranker or a chat model"):
        serve_stage("qwen3-embedding:4b-fp16")


def test_find_url_takes_the_latest_tunnel() -> None:
    log = (
        "[serve] url=https://a.trycloudflare.com replicas=2\n"
        "[serve] url=https://b.trycloudflare.com (tunnel restarted)\n"
    )
    assert find_url(log) == "https://b.trycloudflare.com"
    assert find_url("[rerank] milestone") is None


def follow(lines: list[str]):
    code = "import sys\n" + "".join(f"print({line!r}, flush=True)\n" for line in lines)
    return lambda: subprocess.Popen(
        [sys.executable, "-c", code],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_watcher_writes_the_env_file_once_per_url(tmp_path: Path) -> None:
    messages: list[str] = []
    target = tmp_path / "serve" / "m.env"
    watcher = UrlWatcher(
        follow(
            [
                "[00:00:00] Kaggle kernel running",
                "[serve] url=https://a.trycloudflare.com replicas=2",
                "[serve] alive connections=0",
            ]
        ),
        target,
        RERANKER,
        "k",
        messages.append,
        interval=0.05,
    )

    watcher.start()
    deadline = time.monotonic() + 10
    while watcher.url is None and time.monotonic() < deadline:
        time.sleep(0.05)
    watcher.stop()

    assert watcher.url == "https://a.trycloudflare.com"
    assert "BASE_URL=https://a.trycloudflare.com" in target.read_text("utf-8")
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert len(messages) == 1


def test_watcher_follows_a_restarted_tunnel(tmp_path: Path) -> None:
    watcher = UrlWatcher(
        lambda: None, tmp_path / "m.env", RERANKER, "k", lambda _: None
    )
    watcher.feed("[serve] url=https://a.trycloudflare.com replicas=2")
    watcher.feed("[serve] url=https://b.trycloudflare.com (tunnel restarted)")
    assert "https://b.trycloudflare.com" in (tmp_path / "m.env").read_text("utf-8")


def test_api_key_is_read_back_from_a_pushed_runner() -> None:
    config = {"model": "m", "api_key": "secret-key", "enable_internet": True}
    runner = (
        "import base64, json, os, runpy, sys\n"
        "from pathlib import Path\n"
        "config = Path('/tmp/stage_config.json')\n"
        f"config.write_text({json.dumps(json.dumps(config))}, encoding='utf-8')\n"
        "runpy.run_module('pharma_lab.integrations.kaggle.workers.serve', run_name='__main__')\n"
    )
    assert api_key_from_runner(runner) == "secret-key"
    with pytest.raises(ValueError, match="no serve api_key"):
        api_key_from_runner("config.write_text('{}', encoding='utf-8')\n")


def test_watcher_reopens_a_log_stream_that_stays_silent(tmp_path: Path) -> None:
    # `kaggle kernels logs -f` can hang without output or exit while the Kaggle
    # log API is degraded. The watcher must not wait on it forever.
    def silent() -> subprocess.Popen[str]:
        return subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    streams = iter(
        [silent, follow(["[serve] url=https://a.trycloudflare.com replicas=1"])]
    )
    watcher = UrlWatcher(
        lambda: next(streams, lambda: None)(),
        tmp_path / "m.env",
        RERANKER,
        "k",
        lambda _: None,
        interval=0.05,
        idle_seconds=0.5,
    )

    watcher.start()
    deadline = time.monotonic() + 10
    while watcher.url is None and time.monotonic() < deadline:
        time.sleep(0.05)
    watcher.stop()

    assert watcher.url == "https://a.trycloudflare.com"
