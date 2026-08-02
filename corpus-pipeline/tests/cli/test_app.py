from importlib.metadata import entry_points

from typer.testing import CliRunner

from corpus_pipeline.cli.app import app

runner = CliRunner()


def test_console_script_targets_single_app():
    scripts = {item.name: item.value for item in entry_points(group="console_scripts")}
    assert scripts["corpus"] == "corpus_pipeline.cli.app:main"


def test_root_exposes_doctor_and_no_legacy_pipeline_command():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "doctor" in result.stdout
    assert "pipeline" not in result.stdout
