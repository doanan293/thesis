from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from seed_pipeline.config.enums import Backend
from seed_pipeline.config.environment import PROJECT_ENV_FILE, parse_env_file
from seed_pipeline.config.paths import (
    COMPOSE_FILE,
    FORMULARY_PDF_PATH,
    LEAFLETS_MANIFEST_PATH,
    PROJECT_ROOT,
    SOURCES_CURATION_DIR,
    SOURCES_DIR,
)

DEFAULT_ENV_PATH = PROJECT_ENV_FILE
EMBEDDING_MODEL_FILENAME = "qwen3-embedding-0.6b-fp16.gguf"


class KaggleRunner(Protocol):
    def run(self, args: list[str], capture_output: bool = False) -> str: ...


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
    inputs = (
        FORMULARY_PDF_PATH,
        LEAFLETS_MANIFEST_PATH,
        SOURCES_CURATION_DIR / "docling_tables.jsonl",
        SOURCES_CURATION_DIR / "table_duplicate_overrides.json",
        SOURCES_DIR / "colloquial_mappings.json",
        SOURCES_DIR / "term_glossary.json",
        SOURCES_DIR / "vietnamese_valid_syllables.json",
    )
    required = [project_root / path.relative_to(PROJECT_ROOT) for path in inputs]
    return [str(path) for path in required if not path.is_file()]


def _check_raw_inputs(project_root: Path) -> list[PreflightIssue]:
    missing = _missing_files(project_root)
    if not missing:
        return []
    return [
        PreflightIssue(
            "raw-inputs",
            f"Missing {len(missing)} required source input(s); "
            "restore data/sources from the data archive",
            ", ".join(missing),
        )
    ]


def _check_compose(project_root: Path) -> list[PreflightIssue]:
    compose = project_root.parent / COMPOSE_FILE.name
    if compose.is_file():
        return []
    return [PreflightIssue("compose", "Docker Compose file is missing", str(compose))]


def _check_local_models(project_root: Path) -> list[PreflightIssue]:
    model = project_root.parent / "ai-models" / "gguf" / EMBEDDING_MODEL_FILENAME
    if model.is_file():
        return []
    return [PreflightIssue("local-models", "Embedding GGUF is missing", str(model))]


def _check_kaggle_credentials(
    env_file: Path,
    *,
    kaggle_account: str | None = None,
    runner_factory: Callable[[dict[str, str]], KaggleRunner] | None = None,
) -> list[PreflightIssue]:
    values = dict(parse_env_file(env_file))
    values.update({key: value for key, value in os.environ.items() if value})
    from seed_pipeline.integrations.kaggle.config import (
        profile_owner_configuration,
        profile_runner_environment,
        resolve_account_profile,
    )

    try:
        profile = resolve_account_profile(kaggle_account, values)
    except ValueError as exc:
        return [PreflightIssue("kaggle-credentials", str(exc))]
    if profile is not None:
        from seed_pipeline.integrations.kaggle.api import (
            KaggleCommandRunner,
            config_view_command,
            dataset_files_command,
        )
        from seed_pipeline.integrations.kaggle.orchestrator import (
            runtime_dataset_reference,
        )
        from seed_pipeline.integrations.kaggle.parsers import parse_kaggle_username

        environment = profile_runner_environment(profile, values)
        runner = (
            runner_factory(environment)
            if runner_factory is not None
            else KaggleCommandRunner(environment=environment)
        )
        issues: list[PreflightIssue] = []
        try:
            authenticated = parse_kaggle_username(
                runner.run(config_view_command(), capture_output=True)
            )
        except Exception:
            issues.append(
                PreflightIssue(
                    "kaggle-authentication",
                    f"Kaggle profile {profile.name} could not authenticate",
                )
            )
        else:
            if authenticated.casefold() != profile.username.casefold():
                issues.append(
                    PreflightIssue(
                        "kaggle-authentication",
                        f"Kaggle profile {profile.name} authenticated as {authenticated!r}, "
                        f"expected {profile.username!r}",
                    )
                )
        owners = profile_owner_configuration(profile, values)
        reference = runtime_dataset_reference(owners)
        try:
            output = runner.run(dataset_files_command(reference), capture_output=True)
        except Exception:
            issues.append(
                PreflightIssue(
                    "kaggle-shared-runtime",
                    f"Kaggle shared runtime dataset is not accessible: {reference}",
                    reference,
                )
            )
        else:
            if not output.strip():
                issues.append(
                    PreflightIssue(
                        "kaggle-shared-runtime",
                        f"Kaggle shared runtime dataset returned no files: {reference}",
                        reference,
                    )
                )
        return issues
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
    kaggle_account: str | None = None,
    runner_factory: Callable[[dict[str, str]], KaggleRunner] | None = None,
) -> PreflightReport:
    issues = [
        *_check_raw_inputs(Path(project_root)),
        *_check_compose(Path(project_root)),
        *_check_local_models(Path(project_root)),
    ]
    if backend is Backend.KAGGLE:
        issues.extend(
            _check_kaggle_credentials(
                Path(env_file),
                kaggle_account=kaggle_account,
                runner_factory=runner_factory,
            )
        )
    return PreflightReport(backend, tuple(issues))
