import shlex

import pytest
from typer.testing import CliRunner

from seed_pipeline.cli.app import app
from seed_pipeline.config.paths import PROJECT_ROOT

GUIDES = PROJECT_ROOT / "docs" / "guides"
DOCS = (PROJECT_ROOT / "README.md", *sorted(GUIDES.glob("*.md")))


def _code_lines(text: str) -> list[str]:
    lines: list[str] = []
    inside = False
    pending = ""
    for raw in text.splitlines():
        if raw.lstrip().startswith("```"):
            inside = not inside
            continue
        if not inside:
            continue
        if raw.rstrip().endswith("\\"):
            pending += raw.rstrip()[:-1] + " "
            continue
        lines.append((pending + raw).strip())
        pending = ""
    return lines


def _seed_commands() -> list[tuple[str, list[str]]]:
    commands: list[tuple[str, list[str]]] = []
    for doc in DOCS:
        for line in _code_lines(doc.read_text(encoding="utf-8")):
            if line.startswith("uv run seed "):
                commands.append((doc.name, shlex.split(line, comments=True)[3:]))
    return commands


@pytest.mark.parametrize(("doc", "args"), _seed_commands())
def test_documented_seed_commands_parse(doc: str, args: list[str]) -> None:
    result = CliRunner().invoke(app, [*args, "--help"])

    assert result.exit_code == 0, (doc, args, result.output)


def test_migration_runbook_covers_every_spec_step() -> None:
    text = (GUIDES / "migration-2026-09.md").read_text(encoding="utf-8")

    for step in range(1, 8):
        assert f"\n## {step}. " in text
    for required in (
        "uv run seed bundle parity",
        "uv run seed bundle embed --bundle data/heavy/bundles/formulary --backend kaggle",
        "pharma-agent corpus import",
        "hybrid-qwen4b-p50-k30-rrf2",
        "95.98%",
        "0.7960",
    ):
        assert required in text, required
