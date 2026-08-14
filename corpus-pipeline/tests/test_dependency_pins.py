import re
import tomllib
from pathlib import Path

EXACT_REQUIREMENT = re.compile(r"^[A-Za-z0-9_.-]+(?:\[[^]]+\])?==[^,;<>=!~]+$")


def test_all_direct_dependencies_are_exactly_pinned():
    config = tomllib.loads(Path("pyproject.toml").read_text())
    requirements = list(config["project"]["dependencies"])
    for group in config.get("dependency-groups", {}).values():
        requirements.extend(group)

    assert requirements
    assert all(EXACT_REQUIREMENT.fullmatch(item) for item in requirements), requirements
