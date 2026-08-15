from typer.testing import CliRunner

from corpus_pipeline.cli.app import app

runner = CliRunner()


def test_query_embedding_benchmark_rejects_local_backend():
    result = runner.invoke(
        app,
        ["embed", "queries", "--backend", "local", "--benchmark"],
    )

    assert result.exit_code != 0
    assert result.exit_code == 2


def test_corpus_embedding_benchmark_rejects_local_backend():
    result = runner.invoke(
        app,
        ["embed", "chunks", "--backend", "local", "--benchmark"],
    )

    assert result.exit_code != 0
    assert result.exit_code == 2
