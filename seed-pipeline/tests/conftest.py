from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Generator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from seed_pipeline.config import environment, paths
from seed_pipeline.integrations.kaggle import job_lock

KAGGLE_ENVIRONMENT_PREFIX = "KAGGLE_"


@pytest.fixture(autouse=True)
def isolated_lock_dir(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Locks taken by tests never land in the real data/work/locks."""
    root = tmp_path_factory.mktemp("locks")
    monkeypatch.setattr(job_lock, "LOCK_DIR", root)
    return root


@dataclass(frozen=True)
class IsolatedLogs:
    root: Path
    original: Path


@pytest.fixture(autouse=True)
def isolated_logs_dir(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> IsolatedLogs:
    """Command logs written by tests never land in the real data/work/logs."""
    logs = IsolatedLogs(tmp_path_factory.mktemp("logs"), paths.LOGS_DIR)
    monkeypatch.setattr(paths, "LOGS_DIR", logs.root)
    return logs


@dataclass
class KaggleGuard:
    """Kaggle CLI launches a unit test attempted; each one was refused."""

    blocked: list[str] = field(default_factory=list)


@pytest.fixture(autouse=True)
def no_real_kaggle(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Generator[KaggleGuard, None, None]:
    """Unit tests never see real Kaggle credentials and never launch the Kaggle CLI.

    Tests marked ``integration`` are opted in explicitly and are left alone.
    """
    guard = KaggleGuard()
    if request.node.get_closest_marker("integration") is not None:
        yield guard
        return
    for key in [key for key in os.environ if key.startswith(KAGGLE_ENVIRONMENT_PREFIX)]:
        monkeypatch.delenv(key)
    _hide_project_env_file(monkeypatch)
    original_init = subprocess.Popen.__init__

    def guarded_init(self, args, *positional, **keywords) -> None:
        program = _program_name(args, keywords.get("executable"))
        if program == "kaggle":
            guard.blocked.append(program)
            raise RuntimeError(
                "Unit tests must not launch the real Kaggle CLI; inject a fake "
                "runner or mark the test as integration."
            )
        original_init(self, args, *positional, **keywords)

    monkeypatch.setattr(subprocess.Popen, "__init__", guarded_init)
    yield guard
    if guard.blocked:
        pytest.fail(
            f"Test attempted {len(guard.blocked)} real Kaggle CLI launch(es)",
            pytrace=False,
        )


def _hide_project_env_file(monkeypatch: pytest.MonkeyPatch) -> None:
    original = environment.parse_env_file
    project_env_file = environment.PROJECT_ENV_FILE.resolve()

    def parse_without_project_env(path: Path) -> dict[str, str]:
        if Path(path).resolve() == project_env_file:
            return {}
        return original(path)

    # Modules that imported the parser by name hold their own reference.
    for module in list(sys.modules.values()):
        if (
            getattr(module, "__name__", "").startswith("seed_pipeline")
            and getattr(module, "parse_env_file", None) is original
        ):
            monkeypatch.setattr(module, "parse_env_file", parse_without_project_env)


def _program_name(args: object, executable: object) -> str | None:
    """Name of the program a Popen call would run, without any extension."""
    if isinstance(executable, str | bytes | os.PathLike):
        return Path(os.fsdecode(executable)).stem
    if isinstance(args, str | bytes | os.PathLike):
        parts = os.fsdecode(args).split()
        return Path(parts[0]).stem if parts else None
    if isinstance(args, Sequence) and args:
        first = args[0]
        if isinstance(first, str | bytes | os.PathLike):
            return Path(os.fsdecode(first)).stem
    return None
