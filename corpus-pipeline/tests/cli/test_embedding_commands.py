from typer.testing import CliRunner

from corpus_pipeline.cli.app import app


def test_chunk_embedding_command_has_backend_and_no_legacy_cache_flags():
    result = CliRunner().invoke(app, ["embed", "chunks", "--help"])

    assert result.exit_code == 0
    assert "--backend" in result.stdout
    assert "--cache-only" not in result.stdout
    assert "--upload-from-cache" not in result.stdout
