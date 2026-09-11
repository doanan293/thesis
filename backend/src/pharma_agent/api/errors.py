from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from pharma_agent.application.errors import (
    AgentUnavailable,
    ApplicationError,
    ConversationNotFound,
    InvalidInput,
)

STATUS_BY_ERROR: dict[type[ApplicationError], int] = {
    ConversationNotFound: 404,
    InvalidInput: 422,
    AgentUnavailable: 503,
}


async def application_error_handler(request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, ApplicationError):
        raise exc
    status = next(
        (code for kind, code in STATUS_BY_ERROR.items() if isinstance(exc, kind)), 400
    )
    return JSONResponse(
        status_code=status, content={"code": exc.code, "message": str(exc) or exc.code}
    )


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ApplicationError, application_error_handler)
