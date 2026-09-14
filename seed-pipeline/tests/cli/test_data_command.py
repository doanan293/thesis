import json
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import seed_pipeline.cli.commands.data as data_command
from seed_pipeline.cli.app import app
from seed_pipeline.data_archive.service import PullResult, PushResult

runner = CliRunner()


@pytest.fixture
def accounts(monkeypatch: pytest.MonkeyPatch) -> list[str | None]:
    seen: list[str | None] = []

    def resolve(kaggle_account: str | None) -> SimpleNamespace:
        seen.append(kaggle_account)
        return SimpleNamespace(
            runner=object(), owners=SimpleNamespace(execution="owner")
        )

    monkeypatch.setattr(data_command, "resolve_execution_context", resolve)
    return seen


def test_push_forwards_account_and_message(
    monkeypatch: pytest.MonkeyPatch, accounts: list[str | None]
) -> None:
    captured: dict = {}

    def fake_push(**kwargs):
        captured.update(kwargs)
        return PushResult("owner/seed-pipeline-data", 2, 10, 1)

    monkeypatch.setattr(data_command, "push_data", fake_push)

    result = runner.invoke(
        app,
        ["--json", "data", "push", "--kaggle-account", "acc2", "--message", "Refresh"],
    )

    assert result.exit_code == 0, result.output
    assert accounts == ["acc2"]
    assert captured["message"] == "Refresh"
    assert captured["dataset"].owner == "owner"
    assert json.loads(result.stdout)["details"]["dataset"] == "owner/seed-pipeline-data"


def test_pull_forwards_force(
    monkeypatch: pytest.MonkeyPatch, accounts: list[str | None]
) -> None:
    captured: dict = {}

    def fake_pull(**kwargs):
        captured.update(kwargs)
        return PullResult("owner/seed-pipeline-data", 10, 3, 7, 1)

    monkeypatch.setattr(data_command, "pull_data", fake_pull)

    result = runner.invoke(app, ["--json", "data", "pull", "--force"])

    assert result.exit_code == 0, result.output
    assert accounts == [None]
    assert captured["force"] is True
    assert json.loads(result.stdout)["details"]["written"] == 3
