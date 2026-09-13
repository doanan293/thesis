"""CSRF protection for cookie sessions with starlette-csrf (double submit cookie).

Only unsafe requests that carry the session cookie are checked, so bearer clients (CLI,
tests) are unaffected. The middleware answers before routing, so FastAPI's exception
handlers never see a CSRF failure; the documented `_get_error_response` hook renders it
with the same problem helper those handlers use.
"""

from typing import Final, override

from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import Response
from starlette_csrf import CSRFMiddleware

from pharma_agent.api.problems import problem_response
from pharma_agent.infrastructure.auth.users import SESSION_COOKIE_NAME
from pharma_agent.infrastructure.settings import AuthSettings

CSRF_COOKIE_NAME: Final = "csrftoken"
CSRF_HEADER_NAME: Final = "x-csrftoken"
CSRF_FAILED_CODE: Final = "CSRF_FAILED"


class ProblemCSRFMiddleware(CSRFMiddleware):
    """starlette-csrf whose documented `_get_error_response` hook returns a problem."""

    @override
    def _get_error_response(self, request: Request) -> Response:
        return problem_response(
            403,
            CSRF_FAILED_CODE,
            detail=(
                f"Send the value of the {CSRF_COOKIE_NAME} cookie "
                f"in the {CSRF_HEADER_NAME} header."
            ),
        )


def install_csrf(app: FastAPI, settings: AuthSettings) -> None:
    app.add_middleware(
        ProblemCSRFMiddleware,
        secret=settings.require_csrf_secret(),
        sensitive_cookies={SESSION_COOKIE_NAME},
        cookie_name=CSRF_COOKIE_NAME,
        header_name=CSRF_HEADER_NAME,
        cookie_secure=settings.cookie_secure,
        cookie_samesite="lax",
    )
