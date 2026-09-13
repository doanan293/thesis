"""RFC 9457 problem details: the single response shape of every API error."""

from collections.abc import Mapping
from http import HTTPStatus
from typing import Any

from fastapi.responses import JSONResponse

from pharma_agent.api.schemas import Problem, ProblemItem

PROBLEM_MEDIA_TYPE = "application/problem+json"
PROBLEM_TYPE_PREFIX = "urn:pharma-agent:problem:"


def problem_type(code: str) -> str:
    """`CONVERSATION_NOT_FOUND` -> `urn:pharma-agent:problem:conversation-not-found`."""
    return PROBLEM_TYPE_PREFIX + code.lower().replace("_", "-")


def humanize_code(code: str) -> str:
    """`CONVERSATION_NOT_FOUND` -> `Conversation not found`."""
    return code.replace("_", " ").capitalize()


class ProblemResponse(JSONResponse):
    media_type = PROBLEM_MEDIA_TYPE


def problem_response(
    status: int,
    code: str,
    *,
    detail: str | None = None,
    errors: list[ProblemItem] | None = None,
    headers: Mapping[str, str] | None = None,
) -> ProblemResponse:
    problem = Problem(
        type=problem_type(code),
        title=humanize_code(code),
        status=status,
        detail=detail or None,
        code=code,
        errors=errors,
    )
    return ProblemResponse(
        problem.model_dump(mode="json", exclude_none=True),
        status_code=status,
        headers=headers,
    )


PROBLEM_SCHEMA_REF = "#/components/schemas/Problem"


def problem_responses(*statuses: int) -> dict[int | str, dict[str, Any]]:
    """OpenAPI `responses` declaring `Problem` for each status.

    `PharmaAgentAPI` moves them from application/json to application/problem+json.
    """
    return {
        status: {"model": Problem, "description": HTTPStatus(status).phrase}
        for status in statuses
    }
