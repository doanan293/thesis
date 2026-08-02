from typer.testing import CliRunner

from corpus_pipeline.cli.app import app


def test_rerank_help_has_shared_backend_flags():
    result = CliRunner().invoke(app, ["rerank", "--help"])
    assert result.exit_code == 0
    assert "--backend" in result.stdout
    assert "--run" in result.stdout
