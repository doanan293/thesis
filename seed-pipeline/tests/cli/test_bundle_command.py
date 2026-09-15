import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from seed_pipeline.cli.app import app

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "rag_final_small"
runner = CliRunner()


def test_bundle_export_command_writes_a_valid_bundle(tmp_path: Path) -> None:
    output = tmp_path / "bundle"

    result = runner.invoke(
        app,
        [
            "--json",
            "bundle",
            "export",
            "--output",
            str(output),
            "--rag-final-dir",
            str(FIXTURE),
            "--glossary",
            str(FIXTURE / "term_glossary.json"),
            "--mappings",
            str(FIXTURE / "colloquial_mappings.json"),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "complete"
    assert payload["details"] == {"documents": 4, "sections": 5, "skipped_sections": []}
    assert (output / "manifest.json").is_file()


def _export(output: Path) -> None:
    result = runner.invoke(
        app,
        [
            "bundle",
            "export",
            "--output",
            str(output),
            "--rag-final-dir",
            str(FIXTURE),
            "--glossary",
            str(FIXTURE / "term_glossary.json"),
            "--mappings",
            str(FIXTURE / "colloquial_mappings.json"),
        ],
    )
    assert result.exit_code == 0, result.output


def test_bundle_embed_command_uses_the_selected_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.bundle.test_embed import CacheFillingBackend

    import seed_pipeline.cli.commands.bundle as bundle_command

    _export(tmp_path / "bundle")
    created: list[CacheFillingBackend] = []

    def kaggle_backend() -> CacheFillingBackend:
        backend = CacheFillingBackend()
        created.append(backend)
        return backend

    monkeypatch.setattr(bundle_command, "KaggleTextEmbeddingBackend", kaggle_backend)

    result = runner.invoke(
        app,
        [
            "--json",
            "bundle",
            "embed",
            "--bundle",
            str(tmp_path / "bundle"),
            "--backend",
            "kaggle",
            "--model",
            "qwen3-embedding:4b-fp16",
            "--cache",
            str(tmp_path / "cache.jsonl"),
            "--work-dir",
            str(tmp_path / "work"),
            "--kaggle-account",
            "acc2",
        ],
    )

    assert result.exit_code == 0, result.output
    details = json.loads(result.output)["details"]
    assert (details["inputs"], details["vectors"]) == (6, 6)
    assert created[0].requests[0].kaggle_account == "acc2"
