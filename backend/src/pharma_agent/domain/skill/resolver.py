from collections.abc import Sequence

from pharma_agent.domain.skill.models import MAX_SELECTED_SKILLS, SkillMetadata


def resolve_selected(
    catalog: Sequence[SkillMetadata],
    selected_ids: Sequence[str],
    *,
    max_selected: int = MAX_SELECTED_SKILLS,
) -> list[str]:
    """Keep only ids that exist in the catalog, in the order the model chose them, capped."""
    known = {meta.skill_id for meta in catalog}
    resolved: list[str] = []
    for skill_id in selected_ids:
        if skill_id in known and skill_id not in resolved:
            resolved.append(skill_id)
        if len(resolved) >= max_selected:
            break
    return resolved
