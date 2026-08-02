from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from corpus_pipeline.artifacts.manifest import ArtifactContractError


def copy_hash_and_count(source: Path, destination: Path) -> tuple[str, int]:
    """Copy JSONL once while producing its checksum and validated row count."""
    digest = hashlib.sha256()
    count = 0
    source = Path(source)
    destination = Path(destination)
    try:
        with source.open("rb") as src, destination.open("wb") as dst:
            for raw in src:
                digest.update(raw)
                dst.write(raw)
                if raw.strip():
                    try:
                        value: Any = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        raise ArtifactContractError(
                            f"Invalid JSON at {source}:{count + 1}"
                        ) from exc
                    if not isinstance(value, dict):
                        raise ArtifactContractError(
                            f"JSONL row is not an object at {source}:{count + 1}"
                        )
                    count += 1
    except OSError as exc:
        raise ArtifactContractError(f"Cannot copy artifact file: {source}") from exc
    return digest.hexdigest(), count
