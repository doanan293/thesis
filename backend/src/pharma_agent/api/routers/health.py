import asyncio
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from pharma_agent.api.deps import ContainerDep
from pharma_agent.api.problems import PROBLEM_MEDIA_TYPE, PROBLEM_SCHEMA_REF
from pharma_agent.api.schemas import HealthResponse

# 503 is either a degraded HealthResponse or a SERVICE_STARTING problem.
HEALTH_RESPONSES: dict[int | str, dict[str, Any]] = {
    503: {
        "model": HealthResponse,
        "description": "A dependency is down, or the service is still starting",
        "content": {PROBLEM_MEDIA_TYPE: {"schema": {"$ref": PROBLEM_SCHEMA_REF}}},
    }
}


def build_health_router() -> APIRouter:
    router = APIRouter(tags=["health"], responses=HEALTH_RESPONSES)

    @router.get("/health", response_model=HealthResponse)
    async def health(container: ContainerDep) -> JSONResponse:
        names = list(container.health_checks)
        results = await asyncio.gather(
            *(container.health_checks[name]() for name in names)
        )
        checks = dict(zip(names, results, strict=True))
        healthy = all(checks.values())
        body = HealthResponse(
            status="ok" if healthy else "degraded",
            agent=container.chat is not None,
            checks=checks,
            reasons={
                name: container.health_reasons[name]
                for name, ok in checks.items()
                if not ok and name in container.health_reasons
            },
        )
        return JSONResponse(
            status_code=200 if healthy else 503, content=body.model_dump()
        )

    return router
