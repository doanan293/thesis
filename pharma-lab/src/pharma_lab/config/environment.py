from __future__ import annotations

import os
from collections.abc import MutableMapping
from pathlib import Path

from pharma_lab.config.paths import PROJECT_ROOT

PROJECT_ENV_FILE = PROJECT_ROOT / ".env"


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    path = Path(path)
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if not key or not key.replace("_", "a").isalnum():
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key] = value
    return values


def load_project_env(
    path: Path = PROJECT_ENV_FILE,
    environ: MutableMapping[str, str] | None = None,
) -> None:
    target = os.environ if environ is None else environ
    for key, value in parse_env_file(path).items():
        target.setdefault(key, value)
