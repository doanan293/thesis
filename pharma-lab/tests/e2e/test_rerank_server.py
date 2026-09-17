import json
import stat
from pathlib import Path

import pytest

from pharma_lab.e2e.rerank_server import (
    UrlWatcher,
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


def test_watcher_writes_the_env_file_once_per_url(tmp_path: Path) -> None:
    logs = iter(
        [
            "starting",
            "[serve] url=https://a.trycloudflare.com",
            "[serve] url=https://a.trycloudflare.com",
        ]
    )
    messages: list[str] = []
    target = tmp_path / "serve" / "m.env"
    watcher = UrlWatcher(lambda: next(logs), target, "m", "k", messages.append)

    for _ in range(3):
        watcher.check()

    assert watcher.url == "https://a.trycloudflare.com"
    assert "BASE_URL=https://a.trycloudflare.com" in target.read_text("utf-8")
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert len(messages) == 1
