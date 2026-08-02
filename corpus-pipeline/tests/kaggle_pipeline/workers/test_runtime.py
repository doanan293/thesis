from pathlib import Path

from corpus_pipeline.integrations.kaggle.workers.runtime import (
    build_server_command,
    find_unique,
)
from corpus_pipeline.runtime.catalog import require_model


def test_find_unique_locates_nested_file(tmp_path: Path):
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "runtime_manifest.json").write_text("{}", encoding="utf-8")
    assert (
        find_unique(tmp_path, "runtime_manifest.json")
        == nested / "runtime_manifest.json"
    )


def test_build_server_command_uses_model_topology(tmp_path: Path):
    command = build_server_command(
        binary=tmp_path / "llama-server",
        model=tmp_path / "model.gguf",
        port=11434,
        visible_devices="0,1",
        spec=require_model("qwen3-embedding:8b-fp16"),
    )
    assert "--embedding" in command
    assert "--tensor-split" in command
    assert "1,1" in command
