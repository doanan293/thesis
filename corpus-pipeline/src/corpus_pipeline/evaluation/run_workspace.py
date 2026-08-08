from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class RunConflictError(RuntimeError):
    """Raised when a named run is reused with different semantics."""


@dataclass(frozen=True)
class RunIdentity:
    evaluation_path: str
    evaluation_sha256: str
    collection_name: str
    embedding_model: str
    query_embeddings_sha256: str | None
    retriever: str
    candidate_k: int
    rrf_k: int
    limit: int | None
    prefetch_k: int | None = None


@dataclass(frozen=True)
class RunRecord:
    schema_version: int
    identity: RunIdentity
    status: str
    candidates_dir: str | None = None
    rerank_scores_dir: str | None = None
    reports_dir: str | None = None
    reranker: str | None = None


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_run_record(path: Path) -> RunRecord:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_identity = dict(payload["identity"])
    if "prefetch_k" not in raw_identity:
        raw_identity["prefetch_k"] = (
            int(raw_identity["candidate_k"])
            if raw_identity.get("retriever") == "hybrid"
            else None
        )
    return RunRecord(
        int(payload["schema_version"]),
        RunIdentity(**raw_identity),
        str(payload["status"]),
        payload.get("candidates_dir"),
        payload.get("rerank_scores_dir"),
        payload.get("reports_dir"),
        payload.get("reranker"),
    )


@dataclass(frozen=True)
class RunWorkspace:
    root: Path
    identity: RunIdentity

    @property
    def candidates_dir(self) -> Path:
        return self.root / "candidates"

    @property
    def rerank_scores_dir(self) -> Path:
        return self.root / "rerank-scores"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @classmethod
    def open_or_create(
        cls, root: Path, identity: RunIdentity, *, force: bool = False
    ) -> RunWorkspace:
        root = Path(root)
        record_path = root / "run.json"
        if record_path.exists():
            current = load_run_record(record_path)
            if current.identity == identity:
                return cls(root, identity)
            if current.identity != identity and not force:
                raise RunConflictError(
                    "Run identity changed; use another --run name or --force"
                )
            if current.identity != identity and force:
                previous = root / "run.previous.json"
                os.replace(record_path, previous)
        workspace = cls(root, identity)
        root.mkdir(parents=True, exist_ok=True)
        workspace.write_record(RunRecord(1, identity, "created"))
        return workspace

    def write_record(self, record: RunRecord) -> None:
        atomic_write_json(self.root / "run.json", asdict(record))

    def record_candidates(self, artifact: Any) -> None:
        self.write_record(
            RunRecord(
                1,
                self.identity,
                "retrieved",
                candidates_dir=str(Path(artifact.data_path).parent),
            )
        )

    def record_rerank_scores(
        self, cache_path: Path, reranker: str | None = None
    ) -> None:
        current = load_run_record(self.root / "run.json")
        self.write_record(
            RunRecord(
                current.schema_version,
                current.identity,
                "reranked",
                current.candidates_dir,
                str(Path(cache_path)),
                current.reports_dir,
                reranker or current.reranker,
            )
        )

    def record_reports(self, output_dir: Path) -> None:
        current = load_run_record(self.root / "run.json")
        self.write_record(
            RunRecord(
                current.schema_version,
                current.identity,
                "metrics",
                current.candidates_dir,
                current.rerank_scores_dir,
                str(Path(output_dir)),
                current.reranker,
            )
        )
