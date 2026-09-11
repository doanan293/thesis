from pharma_agent.domain.agent.prompts import DISCLAIMER_PHRASES
from pharma_agent.domain.conversation.models import Turn
from pharma_agent.domain.conversation.prompts import summary_messages


def test_summary_prompt_contains_previous_summary_turns_and_limit() -> None:
    messages = summary_messages(
        "Người dùng hỏi về amoxicillin.",
        [
            Turn(
                user_text="Uống lúc no hay đói?",
                assistant_text="Uống lúc nào cũng được [1].",
                status="completed",
            )
        ],
        max_chars=1500,
    )
    text = "\n".join(m.content for m in messages)
    assert "Người dùng hỏi về amoxicillin." in text
    assert "Uống lúc no hay đói?" in text and "Uống lúc nào cũng được [1]." in text
    assert "1500" in text
    assert not any(phrase in text.lower() for phrase in DISCLAIMER_PHRASES)


def test_summary_prompt_without_previous_summary() -> None:
    messages = summary_messages(
        "", [Turn(user_text="a", assistant_text="b", status="completed")], max_chars=500
    )
    assert "(chưa có)" in messages[1].content
