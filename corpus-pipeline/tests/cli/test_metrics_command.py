from typer.testing import CliRunner

from corpus_pipeline.cli.app import app


def test_metrics_help_only_exposes_report_options():
    result = CliRunner().invoke(app, ["metrics", "--help"])
    assert result.exit_code == 0
    assert "--run" in result.stdout
    assert "--top-k" in result.stdout
    assert "--backend" not in result.stdout
