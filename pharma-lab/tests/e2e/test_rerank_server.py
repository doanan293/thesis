import json
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from pharma_lab.e2e.rerank_server import (
    UrlWatcher,
    api_key_from_runner,
    api_key_path,
    env_lines,
    find_url,
    write_request,
)


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
    text = env_lines(model="m", url="https://x.trycloudflare.com", api_key="k")
    assert "PHARMA_RETRIEVAL__RERANK__BASE_URL=https://x.trycloudflare.com\n" in text
    assert "PHARMA_RETRIEVAL__RERANK__API_KEY=k\n" in text
    assert "PHARMA_RETRIEVAL__RERANK__MODEL=m\n" in text


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
        "m",
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
    watcher = UrlWatcher(lambda: None, tmp_path / "m.env", "m", "k", lambda _: None)
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
    with pytest.raises(ValueError, match="no rerank-serve api_key"):
        api_key_from_runner("config.write_text('{}', encoding='utf-8')\n")
