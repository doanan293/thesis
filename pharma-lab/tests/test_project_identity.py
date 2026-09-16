import tomllib
from pathlib import Path

from typer.testing import CliRunner

from pharma_lab.cli.app import app
from pharma_lab.config.paths import PROJECT_ROOT, WORKSPACE_ROOT_ENV
from pharma_lab.integrations.kaggle.dependencies import BUNDLE_INPUT_DATASET_SLUG
from pharma_lab.integrations.kaggle.workspace import (
    TemporaryWorkspace,
    managed_staging_directory,
)


def test_project_metadata_uses_pharma_lab_names() -> None:
    config = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text("utf-8"))

    assert PROJECT_ROOT.name == "pharma-lab"
    assert config["project"]["name"] == "pharma-lab"
    assert config["project"]["scripts"] == {"pharma-lab": "pharma_lab.cli.app:main"}
    assert WORKSPACE_ROOT_ENV == "PHARMA_LAB_ROOT"


def test_cli_program_is_named_pharma_lab() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Usage: pharma-lab" in result.output


def test_kaggle_names_use_pharma_lab(tmp_path: Path) -> None:
    assert BUNDLE_INPUT_DATASET_SLUG == "seed-pipeline-bundle"
    assert TemporaryWorkspace.PREFIX == "pharma-lab-kaggle-"
    with managed_staging_directory(tmp_path) as staging:
        assert staging.name.startswith("pharma-lab-stage-")
