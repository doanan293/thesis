from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

RUN_SCHEMA_VERSION = 3
RunOrigin = Literal["backend", "imported"]


class RunConflictError(RuntimeError):
    """Raised when a named run or one of its artifacts is reused with different inputs."""


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
    release_id: str | None = None
    chunker_version: str | None = None
    # Stratified sample of the evaluation file (seed retrieve --sample); None for runs
    # over the whole file or its first `limit` rows, including run.json files written
    # before sampling existed.
    sample: int | None = None
    sample_seed: int | None = None


@dataclass(frozen=True)
class RerankVariantRecord:
    model: str
    variant_sha256: str
    artifact_dir: str


@dataclass(frozen=True)
class RunRecord:
    identity: RunIdentity
    origin: RunOrigin
    candidates_dir: str | None = None
    rerank_variants: dict[str, RerankVariantRecord] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": RUN_SCHEMA_VERSION,
            "origin": self.origin,
            "identity": asdict(self.identity),
            "candidates_dir": self.candidates_dir,
            "rerank_variants": {
                slug: asdict(variant)
                for slug, variant in sorted(self.rerank_variants.items())
            },
        }


def identity_differences(current: RunIdentity, expected: RunIdentity) -> list[str]:
    return [
        item.name
        for item in fields(RunIdentity)
        if getattr(current, item.name) != getattr(expected, item.name)
    ]


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _remove_tree(path: Path) -> None:
    """Rename a tree aside first so a crash never leaves a half-deleted target."""
    stale = path.with_name(f".{path.name}.replaced")
    if stale.exists():
        shutil.rmtree(stale)
    os.replace(path, stale)
    shutil.rmtree(stale)


def replace_directory(source: Path, target: Path) -> None:
    source, target = Path(source), Path(target)
    if target.exists():
        _remove_tree(target)
    os.replace(source, target)


def evaluation_reference(path: Path, data_dir: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path(data_dir).resolve()).as_posix()
    except ValueError:
        return str(resolved)


def resolve_evaluation_reference(reference: str, data_dir: Path) -> Path:
    path = Path(reference)
    return path if path.is_absolute() else Path(data_dir) / path


def _origin(value: object, path: Path) -> RunOrigin:
    if value == "backend":
        return "backend"
    if value == "imported":
        return "imported"
    raise RunConflictError(f"Unknown run origin {value!r} in {path}")


def load_run_record(path: Path) -> RunRecord:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema_version = payload.get("schema_version")
    if schema_version != RUN_SCHEMA_VERSION:
        raise RunConflictError(
            f"Unsupported run schema {schema_version} in {path}; rebuild the run"
        )
    raw_variants = payload.get("rerank_variants") or {}
    if not isinstance(raw_variants, dict):
        raise RunConflictError(f"Run rerank_variants must be an object in {path}")
    variants = {
        str(slug): RerankVariantRecord(
            str(item["model"]), str(item["variant_sha256"]), str(item["artifact_dir"])
        )
        for slug, item in raw_variants.items()
    }
    return RunRecord(
        identity=RunIdentity(**payload["identity"]),
        origin=_origin(payload.get("origin"), path),
        candidates_dir=payload.get("candidates_dir"),
        rerank_variants=variants,
    )


@dataclass(frozen=True)
class RunWorkspace:
    root: Path
    identity: RunIdentity
    origin: RunOrigin = "backend"

    @property
    def record_path(self) -> Path:
        return self.root / "run.json"

    @property
    def candidates_dir(self) -> Path:
        return self.root / "candidates"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    def rerank_dir(self, model_slug: str) -> Path:
        return self.root / "rerank" / model_slug

    @classmethod
    def open_or_create(
        cls,
        root: Path,
        identity: RunIdentity,
        *,
        origin: RunOrigin = "backend",
        force: bool = False,
    ) -> RunWorkspace:
        workspace = cls(Path(root), identity, origin)
        if workspace.record_path.exists():
            current = workspace.read_record()
            differences = identity_differences(current.identity, identity)
            if current.origin != origin:
                differences.append("origin")
            if not differences:
                return workspace
            if not force:
                raise RunConflictError(
                    f"run {workspace.root} already exists with different "
                    f"{', '.join(differences)}; use another --run name or --force"
                )
            _remove_tree(workspace.root)
        workspace.root.mkdir(parents=True, exist_ok=True)
        workspace.write_record(RunRecord(identity, origin))
        return workspace

    def read_record(self) -> RunRecord:
        return load_run_record(self.record_path)

    def write_record(self, record: RunRecord) -> None:
        atomic_write_json(self.record_path, record.to_payload())

    def record_candidates(self, artifact: Any) -> None:
        candidate_dir = Path(artifact.data_path).parent
        try:
            relative = candidate_dir.relative_to(self.root).as_posix()
        except ValueError as exc:
            raise RunConflictError(
                f"Candidates {candidate_dir} are outside run {self.root}"
            ) from exc
        current = self.read_record()
        self.write_record(
            RunRecord(
                current.identity, current.origin, relative, current.rerank_variants
            )
        )

    def resolve(self, relative: str) -> Path:
        path = Path(relative)
        if path.is_absolute():
            raise RunConflictError(f"Run artifact paths must be relative: {relative}")
        return self.root / path

    def register_rerank_variant(
        self, model_slug: str, variant: RerankVariantRecord
    ) -> None:
        current = self.read_record()
        variants = {**current.rerank_variants, model_slug: variant}
        self.write_record(
            RunRecord(
                current.identity, current.origin, current.candidates_dir, variants
            )
        )

    def unregister_rerank_variant(self, model_slug: str) -> RerankVariantRecord:
        current = self.read_record()
        if model_slug not in current.rerank_variants:
            raise RunConflictError(
                f"Run {self.root} has no rerank variant {model_slug}"
            )
        variants = dict(current.rerank_variants)
        removed = variants.pop(model_slug)
        self.write_record(
            RunRecord(
                current.identity, current.origin, current.candidates_dir, variants
            )
        )
        return removed

    def variant_records(self) -> dict[str, RerankVariantRecord]:
        return dict(self.read_record().rerank_variants)
