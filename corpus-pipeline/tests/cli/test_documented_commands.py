import pytest
from typer.testing import CliRunner

from corpus_pipeline.cli.app import app


@pytest.mark.parametrize(
    "command",
    [
        ["doctor"],
        ["build"],
        ["validate"],
        ["evaluation", "build"],
        ["embed", "chunks"],
        ["vectors", "upload"],
        ["embed", "queries"],
        ["retrieve"],
        ["rerank"],
        ["metrics"],
    ],
)
def test_canonical_command_has_help(command):
    result = CliRunner().invoke(app, [*command, "--help"])
    assert result.exit_code == 0
