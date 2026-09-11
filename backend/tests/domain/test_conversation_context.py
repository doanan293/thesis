from pharma_agent.domain.conversation.context import context_for_rephrase
from pharma_agent.domain.conversation.models import Turn


def turn(i: int, status: str = "completed", size: int = 10) -> Turn:
    return Turn(
        user_text=f"u{i} " + "x" * size,
        assistant_text=f"a{i} " + "y" * size,
        status=status,
    )


def test_excludes_blocked_error_timeout_turns_and_keeps_last_n() -> None:
    turns = [
        turn(1),
        turn(2, "blocked"),
        turn(3, "error"),
        turn(4),
        turn(5, "timeout"),
        turn(6),
        turn(7),
        turn(8),
    ]
    context = context_for_rephrase("tóm tắt", turns, max_turns=3, max_chars=10_000)
    assert [t.user_text[:2] for t in context.turns] == ["u6", "u7", "u8"]
    assert context.summary == "tóm tắt"


def test_trims_oldest_turns_to_fit_max_chars() -> None:
    turns = [turn(1, size=100), turn(2, size=100), turn(3, size=100)]
    context = context_for_rephrase("", turns, max_turns=4, max_chars=250)
    assert [t.user_text[:2] for t in context.turns] == ["u3"]


def test_summary_alone_is_cut_to_max_chars() -> None:
    context = context_for_rephrase("s" * 500, [], max_turns=4, max_chars=100)
    assert len(context.summary) == 100 and context.turns == []
    assert context.is_empty is False
    assert context_for_rephrase("", [], max_turns=4, max_chars=100).is_empty is True
