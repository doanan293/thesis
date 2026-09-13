"""OpenAPI document of the HTTP API, consistent with the RFC 9457 error handlers."""

import json
import re
from typing import Any, override

from fastapi import FastAPI
from pydantic import SecretStr

from pharma_agent.api.problems import PROBLEM_MEDIA_TYPE, PROBLEM_SCHEMA_REF
from pharma_agent.infrastructure.settings import AuthSettings, Settings

REF_PREFIX = "#/components/schemas/"
# Error schemas declared by FastAPI and fastapi-users that the handlers never send.
REPLACED_ERROR_SCHEMAS = ("HTTPValidationError", "ValidationError", "ErrorModel")
ERROR_SCHEMA_REFS = frozenset(
    {PROBLEM_SCHEMA_REF, REF_PREFIX + "HTTPValidationError", REF_PREFIX + "ErrorModel"}
)
UNSAFE_NAME_CHARACTERS = re.compile(r"[^A-Za-z0-9_]+")
EXPORT_PLACEHOLDER = "openapi-export"
EVENT_STREAM_MEDIA_TYPE = "text/event-stream"


def _use_problem_responses(document: dict[str, Any]) -> None:
    for path_item in document.get("paths", {}).values():
        for operation in path_item.values():
            for status, response in operation.get("responses", {}).items():
                if not (status.isdigit() and int(status) >= 400):
                    continue
                content = response.setdefault("content", {})
                # FastAPI files error models under the route's media type, which is
                # text/event-stream for the chat stream, not only application/json.
                replaced = [
                    media_type
                    for media_type, body in content.items()
                    if media_type != PROBLEM_MEDIA_TYPE
                    and body.get("schema", {}).get("$ref") in ERROR_SCHEMA_REFS
                ]
                for media_type in replaced:
                    del content[media_type]
                content[PROBLEM_MEDIA_TYPE] = {"schema": {"$ref": PROBLEM_SCHEMA_REF}}


def _keep_event_stream_refs(document: dict[str, Any]) -> None:
    """Keep only the model reference of an event-stream response schema.

    FastAPI merges a `responses` model into the `{"type": "string"}` it writes for
    non-JSON response classes, which would describe a string that is also an object.
    """
    for path_item in document.get("paths", {}).values():
        for operation in path_item.values():
            for response in operation.get("responses", {}).values():
                stream = response.get("content", {}).get(EVENT_STREAM_MEDIA_TYPE)
                if not isinstance(stream, dict):
                    continue
                schema = stream.get("schema")
                if isinstance(schema, dict) and "$ref" in schema:
                    stream["schema"] = {"$ref": schema["$ref"]}


def _replace_refs(node: object, renames: dict[str, str]) -> None:
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith(REF_PREFIX):
            name = ref.removeprefix(REF_PREFIX)
            if name in renames:
                node["$ref"] = REF_PREFIX + renames[name]
        for value in node.values():
            _replace_refs(value, renames)
    elif isinstance(node, list):
        for item in node:
            _replace_refs(item, renames)


def _name_body_schemas(document: dict[str, Any]) -> None:
    """Name form bodies `Body_<operation id>` with only [A-Za-z0-9_] characters.

    FastAPI titles them after the operation id; for ids like `auth:jwt.login` it
    shortens the component to `login`, or module-qualifies it on a name clash.
    """
    schemas: dict[str, Any] = document.get("components", {}).get("schemas", {})
    renames: dict[str, str] = {}
    for name, schema in schemas.items():
        title = schema.get("title")
        if isinstance(title, str) and title.startswith("Body_"):
            wanted = UNSAFE_NAME_CHARACTERS.sub("_", title)
            if wanted != name:
                renames[name] = wanted
    for old, new in renames.items():
        schemas[new] = {**schemas.pop(old), "title": new}
    _replace_refs(document, renames)


def _drop_replaced_error_schemas(document: dict[str, Any]) -> None:
    schemas: dict[str, Any] = document.get("components", {}).get("schemas", {})
    for name in REPLACED_ERROR_SCHEMAS:
        schema = schemas.pop(name, None)
        if schema is not None and f'"{REF_PREFIX}{name}"' in json.dumps(document):
            schemas[name] = schema


def use_problem_details(document: dict[str, Any]) -> dict[str, Any]:
    """Rewrite the generated document in place; safe to run on an already rewritten one."""
    _use_problem_responses(document)
    _keep_event_stream_refs(document)
    _name_body_schemas(document)
    _drop_replaced_error_schemas(document)
    return document


class PharmaAgentAPI(FastAPI):
    """FastAPI whose OpenAPI document describes the problem+json errors it sends."""

    @override
    def openapi(self) -> dict[str, Any]:
        # FastAPI caches the document and rebuilds it when routes change, so the
        # rewrite runs on every call and must be idempotent.
        return use_problem_details(super().openapi())


def openapi_export_settings() -> Settings:
    """Settings used only to render the document: nothing is served or connected.

    Google OAuth is enabled with placeholders so its routes are always documented.
    """
    return Settings(
        _env_file=None,
        auth=AuthSettings(
            jwt_secret=SecretStr(EXPORT_PLACEHOLDER + "-" + "0" * 32),
            google_client_id=EXPORT_PLACEHOLDER,
            google_client_secret=SecretStr(EXPORT_PLACEHOLDER),
        ),
    )


def render_openapi(app: FastAPI) -> str:
    return (
        json.dumps(app.openapi(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
