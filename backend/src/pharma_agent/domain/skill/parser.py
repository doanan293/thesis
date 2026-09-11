import hashlib
import re

import yaml
from pydantic import BaseModel, ConfigDict

from pharma_agent.domain.shared.errors import DomainError

_FRONTMATTER_KEYS = {"name", "description"}
_HEADING = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_SECTION_ALIASES = {
    "tìm kiếm": "search",
    "search": "search",
    "trả lời": "answer",
    "answer": "answer",
}


class SkillParseError(DomainError):
    code = "SKILL_PARSE_ERROR"


class ParsedSkill(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    search_guidance: str
    answer_guidance: str
    version: str


def parse_skill_markdown(text: str) -> ParsedSkill:
    """Parse a SKILL.md: YAML frontmatter (name, description) + '## Tìm kiếm' / '## Trả lời' sections."""
    lines = text.strip().splitlines()
    if not lines or lines[0].strip() != "---":
        raise SkillParseError("SKILL.md must start with a '---' frontmatter block")
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration as exc:
        raise SkillParseError("frontmatter block is not closed with '---'") from exc

    try:
        frontmatter = yaml.safe_load("\n".join(lines[1:end])) or {}
    except yaml.YAMLError as exc:
        raise SkillParseError(f"invalid YAML frontmatter: {exc}") from exc
    if not isinstance(frontmatter, dict):
        raise SkillParseError("frontmatter must be a mapping")
    extra = set(frontmatter) - _FRONTMATTER_KEYS
    missing = _FRONTMATTER_KEYS - set(frontmatter)
    if extra or missing:
        raise SkillParseError(
            f"frontmatter keys must be exactly name and description (extra={sorted(extra)}, missing={sorted(missing)})"
        )
    name = str(frontmatter["name"] or "").strip()
    description = str(frontmatter["description"] or "").strip()
    if not name or not description:
        raise SkillParseError("name and description must be non-empty")

    body = "\n".join(lines[end + 1 :])
    sections = _split_sections(body)
    search_guidance = sections.get("search", "")
    answer_guidance = sections.get("answer", "")
    if not search_guidance and not answer_guidance:
        raise SkillParseError(
            "SKILL.md needs at least one of '## Tìm kiếm' or '## Trả lời'"
        )

    version = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return ParsedSkill(
        name=name,
        description=description,
        search_guidance=search_guidance,
        answer_guidance=answer_guidance,
        version=version,
    )


def _split_sections(body: str) -> dict[str, str]:
    matches = list(_HEADING.finditer(body))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        key = _SECTION_ALIASES.get(match.group(1).strip().rstrip(":").lower())
        if key is None:
            continue
        start = match.end()
        stop = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        sections[key] = body[start:stop].strip()
    return sections
