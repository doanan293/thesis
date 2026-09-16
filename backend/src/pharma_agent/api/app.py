"""FastAPI application factory: `uvicorn pharma_agent.api.app:create_app --factory`."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pharma_agent.api.csrf import install_csrf
from pharma_agent.api.deps import session_factory, user_id_dependency
from pharma_agent.api.errors import install_error_handlers
from pharma_agent.api.openapi import PharmaAgentAPI
from pharma_agent.api.routers.chat import build_chat_router
from pharma_agent.api.routers.citations import build_citations_router
from pharma_agent.api.routers.conversations import build_conversations_router
from pharma_agent.api.routers.feedback import build_feedback_router
from pharma_agent.api.routers.health import build_health_router
from pharma_agent.infrastructure.auth.users import build_auth, include_auth_routes
from pharma_agent.infrastructure.container import ContainerFactory, open_container
from pharma_agent.infrastructure.settings import Settings


def create_app(
    settings: Settings | None = None,
    *,
    container_factory: ContainerFactory = open_container,
) -> FastAPI:
    resolved = settings if settings is not None else Settings()
    auth = build_auth(resolved.auth, session_factory)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        async with container_factory(resolved) as container:
            app.state.container = container
            yield

    app = PharmaAgentAPI(
        title="Pharma Agent API",
        version="0.1.0",
        lifespan=lifespan,
        generate_unique_id_function=lambda route: route.name,
    )
    app.state.auth = auth
    install_csrf(app, resolved.auth)
    if resolved.api.cors_origins:
        # Added after CSRF, so CORS is the outer layer and 403 problems carry CORS headers.
        app.add_middleware(
            CORSMiddleware,
            allow_origins=resolved.api.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    install_error_handlers(app)

    api = APIRouter(prefix="/api/v1")
    include_auth_routes(api, auth)
    current_user_id = user_id_dependency(auth)
    api.include_router(build_health_router())
    api.include_router(build_chat_router(current_user_id))
    api.include_router(build_conversations_router(current_user_id))
    api.include_router(build_feedback_router(current_user_id))
    api.include_router(build_citations_router(current_user_id))
    app.include_router(api)
    return app
