from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from pharma_agent.application.errors import (
    AgentUnavailable,
    ApplicationError,
    ConversationNotFound,
    InvalidInput,
    PayloadTooLarge,
)
from pharma_agent.application.feedback.service import MessageNotFound
from pharma_agent.application.skill.service import SkillNameTaken, SkillNotFound

# Subclasses must come before their base class: the first isinstance match wins.
STATUS_BY_ERROR: dict[type[ApplicationError], int] = {
    ConversationNotFound: 404,
    MessageNotFound: 404,
    SkillNotFound: 404,
    SkillNameTaken: 409,
    PayloadTooLarge: 413,
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
