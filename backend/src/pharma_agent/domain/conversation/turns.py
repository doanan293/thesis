from collections.abc import Sequence
from datetime import datetime, timedelta

from pharma_agent.domain.agent.run import AgentRun
from pharma_agent.domain.conversation.models import Citation, Message, MessageRole, Turn

# The assistant message must sort after the user message even when both are
# created within the same clock tick.
_ASSISTANT_OFFSET = timedelta(microseconds=1)


def build_turn_messages(
    *,
    user_message_id: str,
    assistant_message_id: str,
    conversation_id: str,
    run: AgentRun,
    answer_text: str,
    citations: Sequence[Citation],
    phases: Sequence[str],
    now: datetime,
) -> tuple[Message, Message]:
    status = run.status.value
    user_message = Message(
        message_id=user_message_id,
        conversation_id=conversation_id,
        role=MessageRole.USER,
        content=run.original_query,
        status=status,
        run_id=run.run_id,
        created_at=now,
    )
    assistant_message = Message(
        message_id=assistant_message_id,
        conversation_id=conversation_id,
        role=MessageRole.ASSISTANT,
        content=answer_text,
        status=status,
        citations=list(citations),
        phases=list(phases),
        usage=run.usage.model_dump(),
        run_id=run.run_id,
        created_at=now + _ASSISTANT_OFFSET,
    )
    return user_message, assistant_message


def pair_turns(messages: Sequence[Message]) -> list[Turn]:
    """Pair consecutive user/assistant messages (oldest first) into turns."""
    turns: list[Turn] = []
    pending: Message | None = None
    for message in messages:
        if message.role is MessageRole.USER:
            pending = message
        elif pending is not None:
            turns.append(
                Turn(
                    user_text=pending.content,
                    assistant_text=message.content,
                    status=message.status,
                )
            )
            pending = None
    return turns
