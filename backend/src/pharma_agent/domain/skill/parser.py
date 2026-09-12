"""SKILL.md parsing and validation, delegated to the Agent Skills reference library.

`skills-ref` implements https://agentskills.io/specification; using it keeps this parser
consistent with `agentskills validate` instead of re-implementing the rules. Anything this
parser accepts, the reference validator accepts too.
"""

import hashlib
import re
import unicodedata
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from skills_ref.errors import ParseError
from skills_ref.parser import parse_frontmatter
from skills_ref.validator import (
    MAX_DESCRIPTION_LENGTH,
    MAX_SKILL_NAME_LENGTH,
    validate_metadata,
)

from pharma_agent.domain.shared.errors import DomainError

MAX_NAME_CHARS: int = MAX_SKILL_NAME_LENGTH
MAX_DESCRIPTION_CHARS: int = MAX_DESCRIPTION_LENGTH
# The specification's portable subset: skills-ref also accepts non-ASCII lowercase letters,
# but other Agent Skills clients only accept a-z, 0-9 and hyphens, so names stay in this set.
PORTABLE_NAME_PATTERN = r"^[a-z0-9]+(?:-[a-z0-9]+)*$"
_PORTABLE_NAME = re.compile(PORTABLE_NAME_PATTERN)


class SkillParseError(DomainError):
    code = "SKILL_PARSE_ERROR"

    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = list(errors)


class ParsedSkill(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    instructions: str
    license: str | None = None
    compatibility: str | None = None
    allowed_tools: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    version: str


def _optional_text(value: object) -> str | None:
    return None if value is None else str(value)


def parse_skill_markdown(
    content: str, *, directory_name: str | None = None
) -> ParsedSkill:
    """Parse and validate a SKILL.md: `agentskills validate` rules plus an ASCII-only name.

    `directory_name` is the folder holding the file; when given, `name` must match it.
    The markdown body is free-form and returned as `instructions`.
    """
    try:
        frontmatter, body = parse_frontmatter(content)
    except ParseError as exc:
        raise SkillParseError([str(exc)]) from exc
    skill_dir = Path(directory_name) if directory_name else None
    errors = validate_metadata(frontmatter, skill_dir)
    metadata = frontmatter.get("metadata", {})
    if not isinstance(metadata, dict):
        errors.append(
            "Field 'metadata' must be a mapping from string keys to string values"
        )
    name = unicodedata.normalize("NFKC", str(frontmatter.get("name", "")).strip())
    if not errors and not _PORTABLE_NAME.fullmatch(name):
        errors.append(
            f"Skill name '{name}' may only contain lowercase letters a-z, digits "
            "and single hyphens"
        )
    if errors:
        raise SkillParseError(errors)
    return ParsedSkill(
        name=name,
        description=str(frontmatter["description"]).strip(),
        instructions=body,
        license=_optional_text(frontmatter.get("license")),
        compatibility=_optional_text(frontmatter.get("compatibility")),
        allowed_tools=_optional_text(frontmatter.get("allowed-tools")),
        metadata={str(key): str(value) for key, value in metadata.items()},
        version=hashlib.sha256(content.encode("utf-8")).hexdigest()[:12],
    )
