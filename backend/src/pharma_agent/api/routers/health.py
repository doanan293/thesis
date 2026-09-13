import asyncio

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from pharma_agent.api.deps import ContainerDep
from pharma_agent.api.schemas import HealthResponse


def build_health_router() -> APIRouter:
    router = APIRouter(tags=["health"])

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
