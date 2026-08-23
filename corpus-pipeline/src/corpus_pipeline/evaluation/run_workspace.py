from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


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
class RerankVariantRecord:
    model: str
    artifact_dir: str
    status: Literal["complete"] = "complete"


@dataclass(frozen=True)
class LegacyRerankReference:
    model: str
    cache_path: str


@dataclass(frozen=True)
class RunRecord:
    schema_version: int
    identity: RunIdentity
    status: str | None = None
    candidates_dir: str | None = None
    rerank_scores_dir: str | None = None
    reports_dir: str | None = None
    reranker: str | None = None
    rerank_variants: dict[str, RerankVariantRecord] = field(default_factory=dict)
    legacy_rerank: LegacyRerankReference | None = None
    legacy_status: str | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "identity": asdict(self.identity),
        }
        if self.schema_version == 1:
            payload.update(
                status=self.legacy_status or self.status or "created",
                candidates_dir=self.candidates_dir,
                rerank_scores_dir=(
                    self.legacy_rerank.cache_path
                    if self.legacy_rerank is not None
                    else self.rerank_scores_dir
                ),
                reports_dir=self.reports_dir,
                reranker=(
                    self.legacy_rerank.model
                    if self.legacy_rerank is not None
                    else self.reranker
                ),
            )
        elif self.schema_version == 2:
            payload.update(
                candidates_dir=self.candidates_dir,
                reports_dir=self.reports_dir,
                rerank_variants={
                    digest: asdict(variant)
                    for digest, variant in sorted(self.rerank_variants.items())
                },
            )
        else:
            raise RunConflictError(
                f"Unsupported run registry schema: {self.schema_version}"
            )
        return payload


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
    schema_version = int(payload["schema_version"])
    if schema_version not in {1, 2}:
        raise RunConflictError(f"Unsupported run registry schema: {schema_version}")
    raw_identity = dict(payload["identity"])
    if "prefetch_k" not in raw_identity:
        raw_identity["prefetch_k"] = (
            int(raw_identity["candidate_k"])
            if raw_identity.get("retriever") == "hybrid"
            else None
        )
    identity = RunIdentity(**raw_identity)
    if schema_version == 1:
        reranker = payload.get("reranker")
        cache_path = payload.get("rerank_scores_dir")
        legacy = (
            LegacyRerankReference(str(reranker), str(cache_path))
            if reranker and cache_path
            else None
        )
        return RunRecord(
            schema_version=1,
            identity=identity,
            status=str(payload.get("status") or "created"),
            candidates_dir=payload.get("candidates_dir"),
            rerank_scores_dir=cache_path,
            reports_dir=payload.get("reports_dir"),
            reranker=reranker,
            legacy_rerank=legacy,
            legacy_status=str(payload.get("status") or "created"),
        )
    raw_variants = payload.get("rerank_variants") or {}
    if not isinstance(raw_variants, dict):
        raise RunConflictError("Run rerank_variants must be an object")
    variants: dict[str, RerankVariantRecord] = {}
    for digest, raw_variant in raw_variants.items():
        if not isinstance(raw_variant, dict):
            raise RunConflictError(f"Invalid rerank variant record: {digest}")
        if raw_variant.get("status", "complete") != "complete":
            raise RunConflictError(
                f"Unsupported incomplete rerank variant status: {digest}"
            )
        variants[str(digest)] = RerankVariantRecord(
            str(raw_variant["model"]),
            str(raw_variant["artifact_dir"]),
            "complete",
        )
    return RunRecord(
        schema_version=2,
        identity=identity,
        candidates_dir=payload.get("candidates_dir"),
        reports_dir=payload.get("reports_dir"),
        rerank_variants=variants,
    )


@dataclass(frozen=True)
class RunWorkspace:
    root: Path
    identity: RunIdentity
    artifact_root: Path | None = None

    @property
    def artifact_base(self) -> Path:
        return self.artifact_root or self.root

    @property
    def candidates_dir(self) -> Path:
        return self.artifact_base / "candidates"

    @property
    def rerank_scores_dir(self) -> Path:
        return self.artifact_base / "rerank-scores"

    @property
    def reports_dir(self) -> Path:
        return self.artifact_base / "reports"

    @classmethod
    def open_or_create(
        cls,
        root: Path,
        identity: RunIdentity,
        *,
        artifact_root: Path | None = None,
        force: bool = False,
    ) -> RunWorkspace:
        root = Path(root)
        record_path = root / "run.json"
        if record_path.exists():
            current = load_run_record(record_path)
            if current.identity == identity:
                return cls(root, identity, artifact_root)
            if current.identity != identity and not force:
                raise RunConflictError(
                    "Run identity changed; use another --run name or --force"
                )
            if current.identity != identity and force:
                previous = root / "run.previous.json"
                os.replace(record_path, previous)
        workspace = cls(root, identity, artifact_root)
        root.mkdir(parents=True, exist_ok=True)
        workspace.write_record(RunRecord(schema_version=2, identity=identity))
        return workspace

    def write_record(self, record: RunRecord) -> None:
        atomic_write_json(self.root / "run.json", record.to_payload())

    def record_candidates(self, artifact: Any) -> None:
        current = load_run_record(self.root / "run.json")
        candidate_dir = Path(artifact.data_path).parent
        try:
            candidate_value = candidate_dir.relative_to(self.artifact_base).as_posix()
        except ValueError:
            candidate_value = str(candidate_dir)
        snapshot = self.root / "candidates-manifest.json"
        temporary = snapshot.with_name(f".{snapshot.name}.tmp")
        shutil.copy2(Path(artifact.manifest_path), temporary)
        os.replace(temporary, snapshot)
        self.write_record(
            RunRecord(
                schema_version=current.schema_version,
                identity=self.identity,
                status="retrieved",
                candidates_dir=candidate_value,
                rerank_scores_dir=current.rerank_scores_dir,
                reports_dir=current.reports_dir,
                reranker=current.reranker,
                rerank_variants=current.rerank_variants,
                legacy_rerank=current.legacy_rerank,
                legacy_status=current.legacy_status,
            )
        )

    def resolve_relative_path(self, value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else self.artifact_base / candidate

    def register_rerank_variant(
        self, variant_sha256: str, variant: RerankVariantRecord
    ) -> None:
        if Path(variant.artifact_dir).is_absolute():
            raise RunConflictError("Run artifact paths must be relative")
        current = load_run_record(self.root / "run.json")
        existing = current.rerank_variants.get(variant_sha256)
        if existing is not None and existing != variant:
            raise RunConflictError(
                f"Rerank variant {variant_sha256} is already registered differently"
            )
        variants = {**current.rerank_variants, variant_sha256: variant}
        self.write_record(
            RunRecord(
                schema_version=2,
                identity=current.identity,
                candidates_dir=current.candidates_dir,
                reports_dir=current.reports_dir,
                rerank_variants=variants,
            )
        )

    def variant_records(
        self, model: str | None = None
    ) -> dict[str, RerankVariantRecord]:
        variants = load_run_record(self.root / "run.json").rerank_variants
        return {
            digest: item
            for digest, item in variants.items()
            if model is None or item.model == model
        }

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
