import re
import tomllib
from pathlib import Path

EXACT_REQUIREMENT = re.compile(r"^[A-Za-z0-9_.-]+(?:\[[^]]+\])?==[^,;<>=!~]+$")
REQUIREMENT_NAME = re.compile(r"^[A-Za-z0-9_.-]+")


def _load(path: str) -> dict:
    return tomllib.loads(Path(path).read_text(encoding="utf-8"))


def _name(requirement: str) -> str:
    match = REQUIREMENT_NAME.match(requirement)
    assert match is not None, requirement
    return match.group(0).lower()


def test_direct_dependencies_are_pinned_local_or_shared_with_backend() -> None:
    config = _load("pyproject.toml")
    backend = _load("../backend/pyproject.toml")
    path_sources = {
        name
        for name, source in config["tool"]["uv"]["sources"].items()
        if "path" in source
    }
    backend_specifiers = {
        _name(item): item for item in backend["project"]["dependencies"]
    }
    requirements = list(config["project"]["dependencies"])
    for group in config.get("dependency-groups", {}).values():
        requirements.extend(group)

    assert requirements
    for item in requirements:
        name = _name(item)
        if name in path_sources:
            assert item == name, item
        elif EXACT_REQUIREMENT.fullmatch(item) is None:
            assert backend_specifiers.get(name) == item, item


def test_backend_is_an_editable_path_dependency() -> None:
    config = _load("pyproject.toml")

    assert "pharma-agent" in config["project"]["dependencies"]
    assert "qdrant-client>=1.19,<2" in config["project"]["dependencies"]
    assert config["tool"]["uv"]["sources"]["pharma-agent"] == {
        "path": "../backend",
        "editable": True,
    }
