import pytest

from pharma_agent.domain.feedback.models import (
    MAX_NOTE_CHARS,
    Feedback,
    InvalidNote,
    Rating,
)
from tests.domain.factories import NOW


def test_create_strips_note_and_assigns_id() -> None:
    feedback = Feedback.create(
        user_id="u" * 32,
        message_id="m" * 32,
        rating=Rating.UP,
        note="  hữu ích  ",
        now=NOW,
    )
    assert feedback.note == "hữu ích" and len(feedback.feedback_id) == 32
    assert feedback.rating is Rating.UP and feedback.created_at == NOW


def test_note_length_is_bounded() -> None:
    with pytest.raises(InvalidNote):
        Feedback.create(
            user_id="u",
            message_id="m",
            rating=Rating.DOWN,
            note="x" * (MAX_NOTE_CHARS + 1),
            now=NOW,
        )
    assert {r.value for r in Rating} == {"up", "down"}
