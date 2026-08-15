from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from corpus_pipeline.config.enums import Backend
from corpus_pipeline.config.environment import PROJECT_ENV_FILE, parse_env_file
from corpus_pipeline.config.paths import (
    HEAVY_RAW_DIR,
    MANIFESTS_DIR,
    PROJECT_ROOT,
    RESOURCES_DIR,
)

DEFAULT_ENV_PATH = PROJECT_ENV_FILE
EMBEDDING_MODEL_FILENAME = "qwen3-embedding-0.6b-fp16.gguf"


@dataclass(frozen=True)
class PreflightIssue:
    check: str
    message: str
    resource: str | None = None


@dataclass(frozen=True)
class PreflightReport:
    backend: Backend
    issues: tuple[PreflightIssue, ...]

    @property
    def ok(self) -> bool:
        return not self.issues


def _missing_files(project_root: Path) -> list[str]:
    required = [
        project_root / HEAVY_RAW_DIR.relative_to(PROJECT_ROOT) / "duoc-thu-quoc-gia-viet-nam.pdf",
        project_root / RESOURCES_DIR.relative_to(PROJECT_ROOT) / "curation/docling_tables.jsonl",
        project_root / RESOURCES_DIR.relative_to(PROJECT_ROOT) / "curation/table_duplicate_overrides.json",
        project_root / RESOURCES_DIR.relative_to(PROJECT_ROOT) / "colloquial_mappings.json",
        project_root / RESOURCES_DIR.relative_to(PROJECT_ROOT) / "term_glossary.json",
        project_root / RESOURCES_DIR.relative_to(PROJECT_ROOT) / "vietnamese_valid_syllables.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    snapshots = (
        project_root / MANIFESTS_DIR.relative_to(PROJECT_ROOT)
    ).glob("source/*.manifest.json")
    if not any(snapshots):
        missing.append(
            str(
                project_root
                / MANIFESTS_DIR.relative_to(PROJECT_ROOT)
                / "source/*.manifest.json"
            )
        )
    return missing


def _check_raw_inputs(project_root: Path) -> list[PreflightIssue]:
    missing = _missing_files(project_root)
    if not missing:
        return []
    return [
        PreflightIssue(
            "raw-inputs",
            f"Missing {len(missing)} required raw input(s); restore data/heavy archive and tracked resources",
            ", ".join(missing),
        )
    ]


def _check_compose(project_root: Path) -> list[PreflightIssue]:
    compose = project_root.parent / "docker-compose.yml"
    if compose.is_file():
        return []
    return [PreflightIssue("compose", "Docker Compose file is missing", str(compose))]


def _check_local_models(project_root: Path) -> list[PreflightIssue]:
    model = project_root.parent / "ai-models" / "gguf" / EMBEDDING_MODEL_FILENAME
    if model.is_file():
        return []
    return [PreflightIssue("local-models", "Embedding GGUF is missing", str(model))]


def _check_kaggle_credentials(env_file: Path) -> list[PreflightIssue]:
    values = dict(parse_env_file(env_file))
    values.update({key: value for key, value in os.environ.items() if value})
    token = values.get("KAGGLE_API_TOKEN") or values.get("KAGGLE_KEY")
    username = values.get("KAGGLE_USERNAME")
    missing = []
    if not username:
        missing.append("KAGGLE_USERNAME")
    if not token:
        missing.append("KAGGLE_API_TOKEN or KAGGLE_KEY")
    if not missing:
        return []
    return [
        PreflightIssue(
            "kaggle-credentials",
            "Missing Kaggle credential setting(s)",
            ", ".join(missing),
        )
    ]


def run_preflight(
    backend: Backend,
    *,
    project_root: Path = PROJECT_ROOT,
    env_file: Path = DEFAULT_ENV_PATH,
) -> PreflightReport:
    issues = [
        *_check_raw_inputs(Path(project_root)),
        *_check_compose(Path(project_root)),
        *_check_local_models(Path(project_root)),
    ]
    if backend is Backend.KAGGLE:
        issues.extend(_check_kaggle_credentials(Path(env_file)))
    return PreflightReport(backend, tuple(issues))
