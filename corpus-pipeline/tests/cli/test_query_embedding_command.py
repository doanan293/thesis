from typer.testing import CliRunner

from corpus_pipeline.cli.app import app


def test_query_embedding_command_has_shared_backend_and_evaluation_flags():
    result = CliRunner().invoke(app, ["embed", "queries", "--help"])

    assert result.exit_code == 0
    assert "--backend" in result.stdout
    assert "--evaluation" in result.stdout
    assert "--query-embedding-cache" not in result.stdout
