from typer.testing import CliRunner

import seed_pipeline.cli.commands.doctor as doctor_module
from seed_pipeline.cli.app import app
from seed_pipeline.cli.runtime import CommandResult, CommandStatus

runner = CliRunner()


def test_doctor_passes_selected_account(monkeypatch):
    captured = {}

    def fake_doctor_command(backend, env_file, json_output, debug, kaggle_account=None):
        captured["account"] = kaggle_account
        return CommandResult("doctor", CommandStatus.COMPLETE)

    monkeypatch.setattr(doctor_module, "doctor_command", fake_doctor_command)
    result = runner.invoke(
        app,
        ["doctor", "--backend", "kaggle", "--kaggle-account", "acc2"],
    )

    assert result.exit_code == 0
    assert captured["account"] == "acc2"
