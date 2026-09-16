from __future__ import annotations

import contextlib
import os
import subprocess
import sys

import pytest

from pharma_lab.config import environment
from pharma_lab.config.environment import PROJECT_ENV_FILE
from pharma_lab.integrations.kaggle.api import KaggleCommandRunner, quota_command
from pharma_lab.integrations.kaggle.service import (
    load_kaggle_env,
    resolve_profile_execution_contexts,
)
from pharma_lab.orchestration import preflight

GUARD_MESSAGE = "must not launch the real Kaggle CLI"


def test_runner_cannot_launch_the_kaggle_cli(no_real_kaggle):
    runner = KaggleCommandRunner()

    with pytest.raises(RuntimeError, match=GUARD_MESSAGE):
        runner.run_result(quota_command())
    with pytest.raises(RuntimeError, match=GUARD_MESSAGE):
        runner.start(quota_command(), capture_output=True)

    assert no_real_kaggle.blocked == ["kaggle", "kaggle"]
    no_real_kaggle.blocked.clear()


@pytest.mark.parametrize(
    "command",
    [["kaggle", "quota"], "kaggle quota", ["/usr/local/bin/kaggle", "quota"]],
)
def test_direct_subprocess_launches_are_blocked(no_real_kaggle, command):
    with pytest.raises(RuntimeError, match=GUARD_MESSAGE):
        subprocess.run(command, shell=isinstance(command, str), check=False)

    assert no_real_kaggle.blocked == ["kaggle"]
    no_real_kaggle.blocked.clear()


def test_swallowed_launch_attempts_are_still_recorded(no_real_kaggle):
    # Mirrors callers such as owner_configuration that swallow CLI failures.
    with contextlib.suppress(Exception):
        KaggleCommandRunner().run(quota_command())

    assert no_real_kaggle.blocked == ["kaggle"]
    no_real_kaggle.blocked.clear()


def test_other_programs_still_run():
    runner = KaggleCommandRunner(executable=sys.executable)

    result = runner.run_result(["kaggle", "-c", "print('ok')"])

    assert result.stdout == "ok\n"


def test_project_env_credentials_are_invisible():
    load_kaggle_env()

    # Compare key names only, so a failure can never print a credential value.
    assert sorted(environment.parse_env_file(PROJECT_ENV_FILE)) == []
    assert sorted(preflight.parse_env_file(PROJECT_ENV_FILE)) == []
    assert [key for key in os.environ if key.startswith("KAGGLE_")] == []
    assert len(resolve_profile_execution_contexts()) == 0


def test_explicit_env_files_are_still_parsed(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("KAGGLE_ACC1_USERNAME=someone\n", encoding="utf-8")

    assert environment.parse_env_file(env_file) == {"KAGGLE_ACC1_USERNAME": "someone"}
