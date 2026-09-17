"""Identity of one run/config directory: a resumed run must not change it."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pharma_agent.domain.llm.models import LlmRole
from pharma_agent.infrastructure.settings import Settings

from pharma_lab.evaluation.artifact_contracts import write_json

RUN_FILE = "run.json"
INFORMATIONAL = ("git_commit", "created_at")


@dataclass(frozen=True)
class RunIdentity:
    golden_sha256: str
    config: str
    pipeline: dict[str, bool]
    release_id: str
    retrieval: dict[str, Any]
    rerank: dict[str, Any]
    role_models: dict[str, dict[str, Any]]
    budget: dict[str, Any]


def identity_from_settings(
    settings: Settings,
    *,
    golden_sha256: str,
    config: str,
    pipeline: dict[str, bool],
    release_id: str,
) -> RunIdentity:
    retrieval = settings.retrieval
    rerank = retrieval.rerank
    roles: dict[str, dict[str, Any]] = {}
    for role in LlmRole:
        endpoint = settings.llm.resolve(role)
        roles[role.value] = {
            "model": endpoint.model,
            "reasoning_effort": endpoint.reasoning_effort,
        }
        # Recorded only when set, so identities of earlier runs stay unchanged.
        if endpoint.extra_body:
            roles[role.value]["extra_body"] = endpoint.extra_body
    return RunIdentity(
        golden_sha256=golden_sha256,
        config=config,
        pipeline=pipeline,
        release_id=release_id,
        retrieval={
            "mode": retrieval.mode,
            "collections": list(retrieval.collections),
            "qdrant_collection": retrieval.qdrant_collection,
            "prefetch_k": retrieval.prefetch_k,
            "rrf_k": retrieval.rrf_k,
            "candidate_k": retrieval.candidate_k,
            "hydrate_window": retrieval.hydrate_window,
            "embedding_model": retrieval.embedding.model,
        },
        rerank={
            "protocol": rerank.protocol,
            "model": rerank.model if rerank.protocol != "none" else None,
            "top_n": rerank.top_n,
            "max_candidates": rerank.max_candidates,
        },
        role_models=roles,
        budget=settings.budget.model_dump(),
    )


def git_commit(directory: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=directory,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None


def open_run(directory: Path, identity: RunIdentity, *, commit: str | None) -> None:
    """Write run.json on first use; refuse to resume under a different identity."""
    path = Path(directory) / RUN_FILE
    current = asdict(identity)
    if path.is_file():
        stored = json.loads(path.read_text(encoding="utf-8"))
        recorded = {k: v for k, v in stored.items() if k not in INFORMATIONAL}
        # JSON has no tuples; compare through a JSON round trip.
        if recorded != json.loads(json.dumps(current)):
            changed = sorted(
                key
                for key in set(recorded) | set(current)
                if recorded.get(key) != json.loads(json.dumps(current.get(key)))
            )
            raise ValueError(
                f"{path} was written for a different setup ({', '.join(changed)}); "
                "start a new --run"
            )
        return
    write_json(
        path,
        {
            **current,
            "git_commit": commit,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        },
    )
