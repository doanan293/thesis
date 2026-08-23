import io
from pathlib import Path

import pytest

from corpus_pipeline.integrations.kaggle.artifacts import sha256_file
from corpus_pipeline.integrations.kaggle.workers.runtime import (
    BoundedLogCollector,
    build_server_command,
    managed_model_servers,
    resolve_input_file,
)
from corpus_pipeline.runtime.catalog import require_model


def _config(key: str, expected_path: Path, source: Path) -> dict:
    return {
        "input_files": {
            key: {
                "path": str(expected_path),
                "filename": source.name,
                "sha256": sha256_file(source),
            }
        }
    }


def test_resolve_input_file_accepts_exact_path_with_file_checksum(tmp_path):
    mounted = tmp_path / "mounted" / "candidates.jsonl"
    mounted.parent.mkdir()
    mounted.write_text("{}\n", encoding="utf-8")

    assert (
        resolve_input_file(
            _config("candidates", mounted, mounted),
            "candidates",
            input_root=tmp_path,
        )
        == mounted
    )


def test_resolve_input_file_fallback_selects_checksum_match(tmp_path):
    input_root = tmp_path / "input"
    expected = tmp_path / "source" / "candidates.jsonl"
    expected.parent.mkdir()
    expected.write_text("correct\n", encoding="utf-8")
    wrong = input_root / "first" / "candidates.jsonl"
    right = input_root / "second" / "candidates.jsonl"
    wrong.parent.mkdir(parents=True)
    right.parent.mkdir(parents=True)
    wrong.write_text("wrong\n", encoding="utf-8")
    right.write_text("correct\n", encoding="utf-8")
    config = _config("candidates", input_root / "missing" / expected.name, expected)

    assert resolve_input_file(config, "candidates", input_root=input_root) == right


def test_resolve_input_file_reports_checksum_mismatch(tmp_path):
    input_root = tmp_path / "input"
    mounted = input_root / "mounted" / "candidates.jsonl"
    mounted.parent.mkdir(parents=True)
    mounted.write_text("wrong\n", encoding="utf-8")
    expected = tmp_path / "expected" / "candidates.jsonl"
    expected.parent.mkdir()
    expected.write_text("correct\n", encoding="utf-8")
    config = _config("candidates", mounted, expected)

    with pytest.raises(RuntimeError, match=r"input 'candidates'.*checksum mismatch"):
        resolve_input_file(config, "candidates", input_root=input_root)


def test_resolve_input_file_ignores_composite_identity_hash(tmp_path):
    mounted = tmp_path / "mounted" / "candidates.jsonl"
    mounted.parent.mkdir()
    mounted.write_text("{}\n", encoding="utf-8")
    config = _config("candidates", mounted, mounted)
    config["identity"] = {"input_sha256": "f" * 64}

    assert resolve_input_file(config, "candidates", input_root=tmp_path) == mounted


def test_resolve_input_file_does_not_accept_exact_path_with_wrong_filename(tmp_path):
    mounted = tmp_path / "mounted" / "wrong-name.jsonl"
    mounted.parent.mkdir()
    mounted.write_text("{}\n", encoding="utf-8")
    expected = tmp_path / "expected" / "candidates.jsonl"
    expected.parent.mkdir()
    expected.write_text("{}\n", encoding="utf-8")
    config = _config("candidates", mounted, expected)

    with pytest.raises(RuntimeError, match="no mounted file found"):
        resolve_input_file(config, "candidates", input_root=tmp_path / "mounted")


def test_build_server_command_accepts_benchmark_runtime_overrides():
    command = build_server_command(
        binary="llama-server",
        model="model.gguf",
        port=11434,
        visible_devices="0",
        spec=require_model("qwen3-reranker:0.6b-fp16"),
        runtime_overrides={
            "server_slots": 8,
            "context_per_slot": 4096,
            "logical_batch_size": 8192,
            "physical_batch_size": 4096,
        },
    )

    assert command[command.index("-np") + 1] == "8"
    assert command[command.index("-c") + 1] == "32768"
    assert command[command.index("-b") + 1] == "8192"
    assert command[command.index("-ub") + 1] == "4096"


def test_build_server_command_requires_runtime_overrides():
    with pytest.raises(ValueError, match="runtime_overrides is required"):
        build_server_command(
            binary="llama-server",
            model="model.gguf",
            port=11434,
            visible_devices="0",
            spec=require_model("qwen3-reranker:0.6b-fp16"),
            runtime_overrides=None,
        )


def test_bounded_log_collector_keeps_only_tail(tmp_path):
    collector = BoundedLogCollector(max_bytes=8)
    collector.feed(b"0123456789")
    target = tmp_path / "server-0.log"

    collector.write(target)

    assert target.read_bytes() == b"23456789"


def test_bounded_log_collector_limits_diagnostic_tail():
    collector = BoundedLogCollector(max_bytes=32 * 1024)
    collector.feed(b"x" * (20 * 1024))

    diagnostic = collector.tail_text()

    assert len(diagnostic.encode("utf-8")) == 16 * 1024


def test_build_server_environment(tmp_path):
    from corpus_pipeline.integrations.kaggle.workers.runtime import (
        build_server_environment,
    )

    env = build_server_environment(
        {"BASE": "val", "LD_LIBRARY_PATH": "/custom/lib"},
        tmp_path / "runtime_lib",
        "0",
    )
    assert env["BASE"] == "val"
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert str(tmp_path / "runtime_lib") in env["LD_LIBRARY_PATH"]
    assert "/custom/lib" in env["LD_LIBRARY_PATH"]


def test_managed_model_servers_reports_early_process_failure(monkeypatch, tmp_path):
    from corpus_pipeline.integrations.kaggle.workers import runtime
    from corpus_pipeline.runtime import client

    model_path = tmp_path / "model.gguf"
    model_path.touch()
    manifest_path = tmp_path / "runtime_manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    runtime_root = tmp_path / "runtime"

    class RunningProcess:
        def __init__(self):
            self.stdout = io.BytesIO(b"loading model\n")

        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, timeout=None):
            return 0

    class FailedProcess:
        def __init__(self):
            self.stdout = io.BytesIO(b"cuda exploded\n")

        def poll(self):
            return 23

        def terminate(self):
            pass

        def wait(self, timeout=None):
            return 23

    monkeypatch.setattr(
        runtime,
        "find_unique",
        lambda _root, pattern: model_path if pattern == "*.gguf" else manifest_path,
    )
    monkeypatch.setattr(runtime, "materialize_runtime", lambda *_args: runtime_root)
    pending_processes = [RunningProcess(), FailedProcess()]
    monkeypatch.setattr(
        runtime.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pending_processes.pop(0),
    )
    monkeypatch.setattr(
        client.LlamaCppClient,
        "health",
        lambda _self: (_ for _ in ()).throw(RuntimeError("not healthy")),
    )

    config = {
        "model": "qwen3-reranker:0.6b-fp16",
        "runtime_overrides": {
            "server_slots": 4,
            "context_per_slot": 4096,
            "logical_batch_size": 4096,
            "physical_batch_size": 2048,
        },
        "server_start_timeout": 0,
        "output_dir": str(tmp_path / "output"),
    }

    with pytest.raises(RuntimeError) as exc_info:
        with managed_model_servers(config):
            pass

    message = str(exc_info.value)
    assert "exited before becoming healthy" in message
    assert "replica=0" in message
    assert "replica=1" in message
    assert "exit_code=23" in message
    assert "cuda exploded" in message


def test_managed_model_servers_reports_all_replicas_on_timeout(monkeypatch, tmp_path):
    from corpus_pipeline.integrations.kaggle.workers import runtime
    from corpus_pipeline.runtime import client

    model_path = tmp_path / "model.gguf"
    model_path.touch()
    manifest_path = tmp_path / "runtime_manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    runtime_root = tmp_path / "runtime"

    class RunningProcess:
        def __init__(self):
            self.stdout = io.BytesIO(b"loading model\n")

        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(
        runtime,
        "find_unique",
        lambda _root, pattern: model_path if pattern == "*.gguf" else manifest_path,
    )
    monkeypatch.setattr(runtime, "materialize_runtime", lambda *_args: runtime_root)
    monkeypatch.setattr(
        runtime.subprocess, "Popen", lambda *_args, **_kwargs: RunningProcess()
    )
    monkeypatch.setattr(
        client.LlamaCppClient,
        "health",
        lambda _self: (_ for _ in ()).throw(RuntimeError("not healthy")),
    )

    config = {
        "model": "qwen3-reranker:0.6b-fp16",
        "runtime_overrides": {
            "server_slots": 4,
            "context_per_slot": 4096,
            "logical_batch_size": 4096,
            "physical_batch_size": 2048,
        },
        "server_start_timeout": 0,
        "output_dir": str(tmp_path / "output"),
    }

    with pytest.raises(RuntimeError) as exc_info:
        with managed_model_servers(config):
            pass

    message = str(exc_info.value)
    assert "health check timed out" in message
    assert "replica=0" in message
    assert "replica=1" in message
    assert "exit_code=None" in message
    assert "command=" in message
