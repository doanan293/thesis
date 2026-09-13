import json
from pathlib import Path

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


def test_bundle_parity_command_passes_on_the_fixture(tmp_path: Path) -> None:
    _export(tmp_path / "bundle")
    report = tmp_path / "parity.json"

    result = runner.invoke(
        app,
        [
            "--json",
            "bundle",
            "parity",
            "--bundle",
            str(tmp_path / "bundle"),
            "--old-chunks",
            str(FIXTURE / "chunks.jsonl"),
            "--report",
            str(report),
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["details"]["mismatches"] == 0
    assert json.loads(report.read_text("utf-8"))["ok"] is True


def test_bundle_parity_command_fails_on_a_changed_chunk(tmp_path: Path) -> None:
    _export(tmp_path / "bundle")
    lines = (FIXTURE / "chunks.jsonl").read_text("utf-8").splitlines()
    changed = json.loads(lines[0])
    changed["chunk_text"] = "Nội dung khác"
    old_chunks = tmp_path / "chunks.jsonl"
    old_chunks.write_text(
        "\n".join([json.dumps(changed, ensure_ascii=False), *lines[1:]]) + "\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "bundle",
            "parity",
            "--bundle",
            str(tmp_path / "bundle"),
            "--old-chunks",
            str(old_chunks),
        ],
    )

    assert result.exit_code == 1
    assert "mismatches=" in result.output
