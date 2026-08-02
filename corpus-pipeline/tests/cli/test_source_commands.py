from typer.testing import CliRunner

from corpus_pipeline.cli.app import app


def test_source_commands_are_documented():
    result = CliRunner().invoke(app, ["source", "--help"])
    assert result.exit_code == 0
    assert "crawl" in result.stdout
    assert "extract-tables" in result.stdout
    assert "curate-tables" in result.stdout


def test_source_crawl_dry_run_does_not_require_network():
    result = CliRunner().invoke(app, ["source", "crawl", "--dry-run"])
    assert result.exit_code == 0
    assert "status=complete" in result.stdout
