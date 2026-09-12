from typing import Annotated

from fastapi import APIRouter, Depends, File, Path, Response, UploadFile

from pharma_agent.api.deps import ContainerDep, UserIdDependency, require_skills
from pharma_agent.api.schemas import SKILL_ID_PATTERN, EnableSkillRequest
from pharma_agent.application.skill.service import MAX_SKILL_BYTES, SkillView

SkillId = Annotated[str, Path(pattern=SKILL_ID_PATTERN, max_length=80)]


def build_skills_router(current_user_id: UserIdDependency) -> APIRouter:
    router = APIRouter(prefix="/skills", tags=["skills"])
    UserId = Annotated[str, Depends(current_user_id)]

    @router.get("", response_model=list[SkillView])
    async def list_skills(user_id: UserId, container: ContainerDep) -> list[SkillView]:
        return await require_skills(container).list_visible(user_id)

    @router.post("", response_model=SkillView, status_code=201)
    async def upload_skill(
        user_id: UserId,
        container: ContainerDep,
        file: Annotated[UploadFile, File(description="SKILL.md, at most 64 KB")],
    ) -> SkillView:
        # Read one byte past the limit so oversized uploads are rejected without
        # buffering the whole body.
        content = await file.read(MAX_SKILL_BYTES + 1)
        return await require_skills(container).upload(
            user_id, file.filename or "", content
        )

    @router.patch("/{skill_id}", response_model=SkillView)
    async def set_skill_enabled(
        skill_id: SkillId,
        body: EnableSkillRequest,
        user_id: UserId,
        container: ContainerDep,
    ) -> SkillView:
        return await require_skills(container).set_enabled(
            user_id, skill_id, body.enabled
        )

    @router.delete("/{skill_id}", status_code=204)
    async def delete_skill(
        skill_id: SkillId, user_id: UserId, container: ContainerDep
    ) -> Response:
        await require_skills(container).delete(user_id, skill_id)
        return Response(status_code=204)

    return router
