"""Chat use case: run one agent turn inside a conversation and persist it atomically."""

import logging
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from pharma_agent.application.chat.runner import ChatTurnExecution, ChatTurnRunner
from pharma_agent.application.errors import ConversationNotFound
from pharma_agent.application.progress import EventType, ProgressEvent
from pharma_agent.domain.conversation.context import context_for_rephrase
from pharma_agent.domain.conversation.models import (
    Citation,
    Conversation,
    ConversationContext,
)
from pharma_agent.domain.conversation.ports import ConversationRepository
from pharma_agent.domain.conversation.turns import build_turn_messages
from pharma_agent.domain.retrieval.audit import audit_from_run
from pharma_agent.domain.shared.clock import Clock
from pharma_agent.domain.shared.ids import new_id

logger = logging.getLogger(__name__)


class MemoryPolicy(BaseModel):
    context_turns: int = 4
    context_chars: int = 4000


class ChatTurnResult(BaseModel):
    conversation_id: str
    message_id: str | None
    run_id: str
    status: str
    content: str
    citations: list[Citation] = Field(default_factory=list)
    phases: list[str] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    persisted: bool


class ChatSession:
    def __init__(
        self,
        execution: ChatTurnExecution,
        conversation: Conversation,
        *,
        created: bool,
        user_message_id: str,
        assistant_message_id: str,
        conversations: ConversationRepository,
        clock: Clock,
    ) -> None:
        self._execution = execution
        self._conversation = conversation
        self._created = created
        self._user_message_id = user_message_id
        self._assistant_message_id = assistant_message_id
        self._conversations = conversations
        self._clock = clock
        self.result: ChatTurnResult | None = None

    @property
    def conversation_id(self) -> str:
        return self._conversation.conversation_id

    @property
    def user_message_id(self) -> str:
        return self._user_message_id

    @property
    def assistant_message_id(self) -> str:
        return self._assistant_message_id

    async def events(self) -> AsyncGenerator[ProgressEvent]:
        yield ProgressEvent(
            type=EventType.CONVERSATION,
            data={
                "conversation_id": self.conversation_id,
                "title": self._conversation.title,
                "created": self._created,
                "message_id": self._assistant_message_id,
            },
        )
        phases: list[str] = []
        async for event in self._execution.events():
            if event.type is EventType.PHASE:
                phases.append(str(event.data["phase"]))
            if event.type is not EventType.DONE:
                yield event
                continue
            result = await self._persist(phases)
            yield ProgressEvent(
                type=EventType.DONE,
                data={
                    **event.data,
                    "conversation_id": self.conversation_id,
                    "message_id": result.message_id,
                    "persisted": result.persisted,
                    "created_at": result.created_at.isoformat(),
                },
            )

    async def _persist(self, phases: list[str]) -> ChatTurnResult:
        outcome = self._execution.outcome
        if outcome is None:
            raise RuntimeError("runner emitted done without an outcome")
        user_message, assistant_message = build_turn_messages(
            user_message_id=self._user_message_id,
            assistant_message_id=self._assistant_message_id,
            conversation_id=self.conversation_id,
            run=outcome.run,
            answer_text=outcome.answer_text,
            citations=outcome.citations,
            phases=phases,
            now=self._clock.now(),
        )
        self._conversation.record_turn(assistant_message.created_at)
        persisted = True
        try:
            await self._conversations.append_turn(
                self._conversation,
                user_message,
                assistant_message,
                audit_from_run(outcome.run, outcome.citations),
            )
        except Exception:
            logger.exception(
                "failed to persist turn for conversation %s", self.conversation_id
            )
            persisted = False
        self.result = ChatTurnResult(
            conversation_id=self.conversation_id,
            message_id=assistant_message.message_id if persisted else None,
            run_id=outcome.run.run_id,
            status=outcome.run.status.value,
            content=outcome.answer_text,
            citations=list(outcome.citations),
            phases=phases,
            usage=outcome.run.usage.model_dump(),
            created_at=assistant_message.created_at,
            persisted=persisted,
        )
        return self.result


class ChatService:
    def __init__(
        self,
        runner: ChatTurnRunner,
        conversations: ConversationRepository,
        clock: Clock,
        policy: MemoryPolicy,
    ) -> None:
        self._runner = runner
        self._conversations = conversations
        self._clock = clock
        self._policy = policy

    async def open_turn(
        self, *, user_id: str, message: str, conversation_id: str | None
    ) -> ChatSession:
        if conversation_id is None:
            conversation = Conversation.start(
                user_id=user_id, first_message=message, now=self._clock.now()
            )
            await self._conversations.create(conversation)
            context = ConversationContext()
            created = True
        else:
            found = await self._conversations.get(user_id, conversation_id)
            if found is None:
                raise ConversationNotFound(conversation_id)
            conversation = found
            # A conversation created by POST /conversations gets its title from
            # the first question, unless the user renamed it first.
            if conversation.title_from_first_message(message):
                await self._conversations.update_title(conversation)
            turns = await self._conversations.recent_turns(
                conversation_id, self._policy.context_turns
            )
            context = context_for_rephrase(
                conversation.summary,
                turns,
                max_turns=self._policy.context_turns,
                max_chars=self._policy.context_chars,
            )
            created = False
        execution = self._runner.start(
            user_id=user_id,
            message=message,
            conversation=context,
            conversation_id=conversation.conversation_id,
        )
        return ChatSession(
            execution,
            conversation,
            created=created,
            user_message_id=new_id(),
            assistant_message_id=new_id(),
            conversations=self._conversations,
            clock=self._clock,
        )

    async def ask(
        self, *, user_id: str, message: str, conversation_id: str | None
    ) -> ChatTurnResult:
        session = await self.open_turn(
            user_id=user_id, message=message, conversation_id=conversation_id
        )
        async for _ in session.events():
            pass
        if session.result is None:
            raise RuntimeError("chat turn finished without a result")
        return session.result
