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
