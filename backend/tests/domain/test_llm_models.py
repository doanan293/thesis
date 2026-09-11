import pytest
from pydantic import ValidationError

from pharma_agent.domain.llm.models import ChatMessage, ChatRole, LlmRole, LlmUsage


def test_usage_adds_token_counts() -> None:
    total = LlmUsage(prompt_tokens=10, completion_tokens=5) + LlmUsage(
        prompt_tokens=1, completion_tokens=2
    )
    assert total == LlmUsage(prompt_tokens=11, completion_tokens=7)
    assert total.total_tokens == 18


def test_chat_message_is_frozen() -> None:
    message = ChatMessage(role=ChatRole.USER, content="hi")
    with pytest.raises(ValidationError):
        message.content = "changed"  # type: ignore[misc]


def test_llm_roles_cover_every_pipeline_step() -> None:
    assert {role.value for role in LlmRole} == {
        "guardrail",
        "rephrase",
        "skill_selector",
        "judge",
        "refine",
        "answer",
        "summarizer",
    }
