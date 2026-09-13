import tomllib
from pathlib import Path

from typer.testing import CliRunner

from seed_pipeline.cli.app import app
from seed_pipeline.config.paths import PROJECT_ROOT, WORKSPACE_ROOT_ENV
from seed_pipeline.integrations.kaggle.dependencies import BUNDLE_INPUT_DATASET_SLUG
from seed_pipeline.integrations.kaggle.workspace import (
    TemporaryWorkspace,
    managed_staging_directory,
)


def test_project_metadata_uses_seed_pipeline_names() -> None:
    config = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text("utf-8"))

    assert PROJECT_ROOT.name == "seed-pipeline"
    assert config["project"]["name"] == "seed-pipeline"
    assert config["project"]["scripts"] == {"seed": "seed_pipeline.cli.app:main"}
    assert WORKSPACE_ROOT_ENV == "SEED_PIPELINE_ROOT"


def test_cli_program_is_named_seed() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Usage: seed" in result.output


def test_kaggle_names_use_seed_pipeline(tmp_path: Path) -> None:
    assert BUNDLE_INPUT_DATASET_SLUG == "seed-pipeline-bundle"
    assert TemporaryWorkspace.PREFIX == "seed-pipeline-kaggle-"
    with managed_staging_directory(tmp_path) as staging:
        assert staging.name.startswith("seed-pipeline-stage-")
