"""FastAPI exception handlers: every error leaves the API as application/problem+json."""

import re
from enum import Enum
from http import HTTPStatus

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException
from starlette.responses import Response

from pharma_agent.api.problems import problem_response
from pharma_agent.api.schemas import ProblemItem
from pharma_agent.application.errors import (
    ApplicationError,
    CitationNotFound,
    ConversationNotFound,
    InvalidInput,
    PayloadTooLarge,
    ServiceUnavailable,
)
from pharma_agent.application.feedback.service import MessageNotFound

VALIDATION_ERROR = "VALIDATION_ERROR"
INTERNAL_ERROR = "INTERNAL_ERROR"
HTTP_ERROR = "HTTP_ERROR"

# fastapi-users raises HTTPException with an upper snake case ErrorCode in `detail`.
ERROR_CODE_PATTERN = re.compile(r"[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*")

# Subclasses must come before their base class: the first isinstance match wins.
STATUS_BY_ERROR: dict[type[ApplicationError], int] = {
    ConversationNotFound: 404,
    MessageNotFound: 404,
    CitationNotFound: 404,
    PayloadTooLarge: 413,
    InvalidInput: 422,
    ServiceUnavailable: 503,
}


def _status_name(status: int) -> str:
    try:
        return HTTPStatus(status).name
    except ValueError:
        return HTTP_ERROR


def _status_phrase(status: int) -> str:
    try:
        return HTTPStatus(status).phrase
    except ValueError:
        return ""


def _plain(value: object) -> object:
    return value.value if isinstance(value, Enum) else value


def _code_and_detail(exc: HTTPException) -> tuple[str, str | None]:
    fallback = _status_name(exc.status_code)
    detail = _plain(exc.detail)
    if isinstance(detail, dict):
        code, reason = _plain(detail.get("code")), detail.get("reason")
        valid = isinstance(code, str) and ERROR_CODE_PATTERN.fullmatch(code)
        return (
            code if valid and isinstance(code, str) else fallback,
            reason if isinstance(reason, str) else None,
        )
    if isinstance(detail, str):
        if ERROR_CODE_PATTERN.fullmatch(detail):
            return detail, None
        if detail != _status_phrase(exc.status_code):
            return fallback, detail
    return fallback, None


async def application_error_handler(request: Request, exc: Exception) -> Response:
    if not isinstance(exc, ApplicationError):
        raise exc
    status = next(
        (code for kind, code in STATUS_BY_ERROR.items() if isinstance(exc, kind)), 400
    )
    return problem_response(status, exc.code, detail=str(exc))


async def http_exception_handler(request: Request, exc: Exception) -> Response:
    if not isinstance(exc, HTTPException):
        raise exc
    code, detail = _code_and_detail(exc)
    return problem_response(exc.status_code, code, detail=detail, headers=exc.headers)


async def validation_error_handler(request: Request, exc: Exception) -> Response:
    if not isinstance(exc, RequestValidationError):
        raise exc
    errors = [
        ProblemItem(
            loc=list(error["loc"]), message=str(error["msg"]), type=str(error["type"])
        )
        for error in exc.errors()
    ]
    return problem_response(422, VALIDATION_ERROR, errors=errors)


async def unhandled_error_handler(request: Request, exc: Exception) -> Response:
    # Never echo the exception text: it may contain internals. Starlette re-raises
    # the exception after this response so the server still logs the traceback.
    return problem_response(500, INTERNAL_ERROR)


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ApplicationError, application_error_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)
