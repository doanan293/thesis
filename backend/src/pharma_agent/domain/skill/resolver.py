from collections.abc import Sequence

from pharma_agent.domain.skill.models import MAX_SELECTED_SKILLS, SkillMetadata


def resolve_selected(
    catalog: Sequence[SkillMetadata],
    selected_names: Sequence[str],
    *,
    max_selected: int = MAX_SELECTED_SKILLS,
) -> list[str]:
    """Keep only names that exist in the catalog, in the order the model chose them, capped."""
    known = {meta.name for meta in catalog}
    resolved: list[str] = []
    for name in selected_names:
        if name in known and name not in resolved:
            resolved.append(name)
        if len(resolved) >= max_selected:
            break
    return resolved
