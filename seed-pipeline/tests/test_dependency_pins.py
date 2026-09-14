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
    local_sources = {
        name
        for name, source in config["tool"]["uv"]["sources"].items()
        if "path" in source or "workspace" in source
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
        if name in local_sources:
            assert item == name, item
        elif EXACT_REQUIREMENT.fullmatch(item) is None:
            assert backend_specifiers.get(name) == item, item


def test_backend_is_the_only_runtime_dependency() -> None:
    config = _load("pyproject.toml")
    workspace = _load("../pyproject.toml")["tool"]["uv"]["workspace"]

    # seed-pipeline never ships, so its own libraries live in the dev group.
    assert config["project"]["dependencies"] == ["pharma-agent"]
    assert "qdrant-client>=1.19,<2" in config["dependency-groups"]["dev"]
    assert config["tool"]["uv"]["sources"]["pharma-agent"] == {"workspace": True}
    assert {"backend", "seed-pipeline"} <= set(workspace["members"])
