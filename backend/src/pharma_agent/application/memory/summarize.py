import logging

from pharma_agent.domain.conversation.context import EXCLUDED_STATUSES
from pharma_agent.domain.conversation.models import ConversationSummary
from pharma_agent.domain.conversation.ports import ConversationRepository
from pharma_agent.domain.conversation.prompts import summary_messages
from pharma_agent.domain.llm.models import LlmRole
from pharma_agent.domain.llm.port import LlmError, LlmPort
from pharma_agent.domain.shared.clock import Clock

logger = logging.getLogger(__name__)


class SummarizeConversation:
    """Background job: fold new turns into the rolling summary every `every` turns."""

    def __init__(
        self,
        llm: LlmPort,
        conversations: ConversationRepository,
        clock: Clock,
        *,
        every: int,
        max_chars: int,
    ) -> None:
        self._llm = llm
        self._conversations = conversations
        self._clock = clock
        self._every = every
        self._max_chars = max_chars

    async def run_if_needed(self, *, user_id: str, conversation_id: str) -> bool:
        conversation = await self._conversations.get(user_id, conversation_id)
        if conversation is None or not conversation.needs_summary(self._every):
            return False
        turns = await self._conversations.turns_since(
            conversation_id, conversation.summarized_turns
        )
        covered = conversation.summarized_turns + len(turns)
        usable = [turn for turn in turns if turn.status not in EXCLUDED_STATUSES]
        if not usable:
            conversation.apply_summary(
                conversation.summary, covered_turns=covered, now=self._clock.now()
            )
            await self._conversations.update_summary(conversation)
            return False
        try:
            result, _ = await self._llm.structured(
                LlmRole.SUMMARIZER,
                summary_messages(
                    conversation.summary, usable, max_chars=self._max_chars
                ),
                ConversationSummary,
            )
        except LlmError:
            logger.warning(
                "summary failed for conversation %s", conversation_id, exc_info=True
            )
            return False
        conversation.apply_summary(
            result.summary[: self._max_chars],
            covered_turns=covered,
            now=self._clock.now(),
        )
        await self._conversations.update_summary(conversation)
        return True
