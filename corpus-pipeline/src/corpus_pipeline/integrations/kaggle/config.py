from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


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


@dataclass(frozen=True)
class OwnerConfiguration:
    execution: str
    runtime: str
    corpus: str
    checkpoint: str


def resolve_owner_configuration(
    args,
    environ: Mapping[str, str],
    authenticated_owner: str | None = None,
) -> OwnerConfiguration:
    execution = args.owner or environ.get("KAGGLE_USERNAME") or authenticated_owner
    if not execution:
        raise ValueError(
            "Kaggle owner is unavailable: set --owner/KAGGLE_USERNAME "
            "or authenticate with 'kaggle auth login'"
        )
    runtime = args.runtime_owner or environ.get("KAGGLE_RUNTIME_OWNER") or execution
    corpus = args.corpus_owner or environ.get("KAGGLE_CORPUS_OWNER") or runtime
    checkpoint = (
        args.checkpoint_owner or environ.get("KAGGLE_CHECKPOINT_OWNER") or execution
    )
    return OwnerConfiguration(execution, runtime, corpus, checkpoint)
