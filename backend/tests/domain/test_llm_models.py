from pharma_agent.domain.llm.models import ChatMessage, ChatRole, LlmRole, LlmUsage


def test_usage_adds_token_counts() -> None:
    total = LlmUsage(prompt_tokens=10, completion_tokens=5) + LlmUsage(
        prompt_tokens=1, completion_tokens=2
    )
    assert total == LlmUsage(prompt_tokens=11, completion_tokens=7)
    assert total.total_tokens == 18


def test_chat_message_is_immutable_value_object() -> None:
    message = ChatMessage(role=ChatRole.USER, content="hi")
    assert ChatMessage.model_config.get("frozen") is True
    assert {message, ChatMessage(role=ChatRole.USER, content="hi")} == {message}


def test_llm_roles_cover_every_pipeline_step() -> None:
    assert {role.value for role in LlmRole} == {
        "guardrail",
        "rephrase",
        "judge",
        "refine",
        "answer",
        "summarizer",
    }
