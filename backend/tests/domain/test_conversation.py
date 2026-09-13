from datetime import timedelta

import pytest

from pharma_agent.domain.agent.run import AnswerMode, AnswerPlan, RunStatus
from pharma_agent.domain.conversation.models import (
    Conversation,
    InvalidTitle,
    MessageRole,
)
from pharma_agent.domain.conversation.turns import build_turn_messages, pair_turns
from tests.domain.factories import NOW, make_citation, make_run


def test_start_sets_title_from_first_message() -> None:
    conversation = Conversation.start(
        user_id="u1",
        first_message="  Paracetamol   uống bao nhiêu?  " + "x" * 200,
        now=NOW,
    )
    assert conversation.title.startswith("Paracetamol uống bao nhiêu?")
    assert len(conversation.title) <= 80
    assert conversation.turn_count == 0 and conversation.summary == ""
    assert conversation.created_at == conversation.updated_at == NOW
    assert len(conversation.conversation_id) == 32


def test_summary_is_due_every_n_turns() -> None:
    conversation = Conversation.start(user_id="u1", first_message="hi", now=NOW)
    conversation.record_turn(NOW + timedelta(seconds=1))
    assert conversation.needs_summary(every=2) is False
    conversation.record_turn(NOW + timedelta(seconds=2))
    assert conversation.needs_summary(every=2) is True
    conversation.apply_summary(
        "  tóm tắt  ", covered_turns=2, now=NOW + timedelta(seconds=3)
    )
    assert conversation.summary == "tóm tắt" and conversation.summarized_turns == 2
    assert conversation.needs_summary(every=2) is False
    assert conversation.updated_at == NOW + timedelta(seconds=3)


def test_rename_validates_title() -> None:
    conversation = Conversation.start(user_id="u1", first_message="hi", now=NOW)
    conversation.rename("  Thuốc hạ sốt  ", now=NOW)
    assert conversation.title == "Thuốc hạ sốt"
    with pytest.raises(InvalidTitle):
        conversation.rename("   ", now=NOW)
    with pytest.raises(InvalidTitle):
        conversation.rename("x" * 81, now=NOW)


def test_apply_summary_cannot_cover_future_turns() -> None:
    conversation = Conversation.start(user_id="u1", first_message="hi", now=NOW)
    conversation.record_turn(NOW)
    with pytest.raises(ValueError, match="covered_turns"):
        conversation.apply_summary("s", covered_turns=2, now=NOW)


def test_build_turn_messages_and_pair_turns() -> None:
    run = make_run("Liều paracetamol?")
    run.submit_plan(AnswerPlan(mode=AnswerMode.NO_RETRIEVAL), now=NOW)
    run.complete()
    citation = make_citation("c1")
    user_msg, assistant_msg = build_turn_messages(
        conversation_id="conv1",
        run=run,
        answer_text="500 mg [1]",
        citations=[citation],
        phases=["guarding", "answering"],
        now=NOW,
    )
    assert user_msg.role is MessageRole.USER and user_msg.content == "Liều paracetamol?"
    assert user_msg.status == RunStatus.COMPLETED.value
    assert assistant_msg.role is MessageRole.ASSISTANT and assistant_msg.citations == [
        citation
    ]
    assert assistant_msg.run_id == run.run_id and assistant_msg.phases == [
        "guarding",
        "answering",
    ]
    assert assistant_msg.usage["llm_calls"] == 0
    assert assistant_msg.created_at > user_msg.created_at

    turns = pair_turns([user_msg, assistant_msg, user_msg])
    assert len(turns) == 1
    assert (
        turns[0].user_text == "Liều paracetamol?"
        and turns[0].assistant_text == "500 mg [1]"
    )
    assert turns[0].status == "completed"
