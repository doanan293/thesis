import re
from collections import Counter
from typing import Any

from pharma_agent.api.app import create_app
from pharma_agent.api.openapi import openapi_export_settings

PROBLEM_CONTENT = {"schema": {"$ref": "#/components/schemas/Problem"}}

# Route function names (overview §3.5). P7 adds the cookie auth routes and moves
# Google OAuth to the cookie backend.
EXPECTED_OPERATION_IDS = {
    "health",
    "chat",
    "chat_stream",
    "create_conversation",
    "list_conversations",
    "get_conversation",
    "list_messages",
    "rename_conversation",
    "delete_conversation",
    "get_message_citation",
    "submit_feedback",
    "list_skills",
    "upload_skill",
    "set_skill_enabled",
    "delete_skill",
    "auth:jwt.login",
    "auth:jwt.logout",
    "register:register",
    "users:current_user",
    "users:patch_current_user",
    "users:user",
    "users:patch_user",
    "users:delete_user",
    "oauth:google.jwt.authorize",
    "oauth:google.jwt.callback",
}


def document() -> dict[str, Any]:
    return create_app(openapi_export_settings()).openapi()


def operations(doc: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    return [
        (method, path, operation)
        for path, item in doc["paths"].items()
        for method, operation in item.items()
    ]


def test_operation_ids_are_unique_route_names() -> None:
    ids = [operation["operationId"] for _, _, operation in operations(document())]
    assert [name for name, count in Counter(ids).items() if count > 1] == []
    assert set(ids) == EXPECTED_OPERATION_IDS


def test_every_error_response_is_a_problem() -> None:
    doc = document()
    for method, path, operation in operations(doc):
        for status, response in operation["responses"].items():
            if int(status) < 400:
                continue
            content = response["content"]
            assert content["application/problem+json"] == PROBLEM_CONTENT, (
                method,
                path,
                status,
            )
            if (path, status) != ("/api/v1/health", "503"):
                assert set(content) == {"application/problem+json"}, (
                    method,
                    path,
                    status,
                )

    health = doc["paths"]["/api/v1/health"]["get"]["responses"]["503"]["content"]
    assert health["application/json"] == {
        "schema": {"$ref": "#/components/schemas/HealthResponse"}
    }
    conversations = doc["paths"]["/api/v1/conversations"]
    assert {"401", "422", "503"} <= set(conversations["get"]["responses"])
    assert conversations["post"]["responses"]["201"]["content"]["application/json"] == {
        "schema": {"$ref": "#/components/schemas/ConversationView"}
    }
    assert conversations["get"]["responses"]["200"]["content"]["application/json"] == {
        "schema": {"$ref": "#/components/schemas/ConversationPage"}
    }
    skills = doc["paths"]["/api/v1/skills"]["post"]["responses"]
    assert {"401", "409", "413", "422", "503"} <= set(skills)
    register = doc["paths"]["/api/v1/auth/register"]["post"]["responses"]
    assert register["400"]["content"] == {"application/problem+json": PROBLEM_CONTENT}


def test_components_are_clean() -> None:
    schemas = document()["components"]["schemas"]
    assert {
        "Problem",
        "ProblemItem",
        "ConversationView",
        "ConversationPage",
        "MessagePage",
        "FeedbackView",
        "SkillView",
        "HealthResponse",
        "Body_auth_jwt_login",
        "Body_upload_skill",
    } <= set(schemas)
    assert not {"HTTPValidationError", "ValidationError", "ErrorModel"} & set(schemas)
    # OpenAPI 3.1 component keys must match ^[a-zA-Z0-9.\-_]+$.
    assert all(re.fullmatch(r"[A-Za-z0-9.\-_]+", name) for name in schemas)
    assert schemas["Problem"]["required"] == ["type", "title", "status", "code"]
    assert schemas["ProblemItem"]["required"] == ["loc", "message", "type"]


def test_post_processing_is_idempotent() -> None:
    app = create_app(openapi_export_settings())
    assert app.openapi() == app.openapi()


UI_SCHEMAS = {
    "UIMessage",
    "TextUIPart",
    "SourceDocumentUIPart",
    "MessageMetadata",
    "MessageStatus",
    "MessageUsage",
    "MessageFeedback",
    "PharmaSourceMetadata",
    "EvidenceItem",
    "PharmaDataParts",
    "PhaseData",
    "SkillRef",
    "SkillsData",
    "EvidenceData",
    "ConversationData",
}


def test_ui_message_stream_and_history_are_documented() -> None:
    doc = document()
    schemas = doc["components"]["schemas"]
    assert set(schemas) >= UI_SCHEMAS
    assert set(schemas["UIMessage"]["properties"]) == {
        "id",
        "role",
        "parts",
        "metadata",
    }
    assert "sourceId" in schemas["SourceDocumentUIPart"]["properties"]
    assert "isCurrent" in schemas["PharmaSourceMetadata"]["properties"]

    stream = doc["paths"]["/api/v1/chat/stream"]["post"]["responses"]
    assert stream["200"]["content"] == {
        "text/event-stream": {
            "schema": {"$ref": "#/components/schemas/PharmaDataParts"}
        }
    }
    for status in ("401", "404", "422", "503"):
        assert stream[status]["content"] == {
            "application/problem+json": PROBLEM_CONTENT
        }

    page = doc["paths"]["/api/v1/conversations/{conversation_id}/messages"]["get"]
    assert page["responses"]["200"]["content"]["application/json"] == {
        "schema": {"$ref": "#/components/schemas/MessagePage"}
    }


def test_citation_detail_is_documented() -> None:
    doc = document()
    operation = doc["paths"]["/api/v1/messages/{message_id}/citations/{index}"]["get"]
    assert operation["operationId"] == "get_message_citation"
    assert operation["responses"]["200"]["content"]["application/json"] == {
        "schema": {"$ref": "#/components/schemas/CitationDetail"}
    }
    for status in ("401", "404", "422", "503"):
        assert operation["responses"][status]["content"] == {
            "application/problem+json": PROBLEM_CONTENT
        }
    schemas = doc["components"]["schemas"]
    assert {"CitationDetail", "CitationChunk"} <= set(schemas)
    assert "document_title" in schemas["CitationDetail"]["properties"]
