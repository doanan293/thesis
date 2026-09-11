from pydantic import BaseModel, Field

from pharma_agent.domain.agent.run import AgentRun
from pharma_agent.domain.conversation.models import Citation
from pharma_agent.domain.retrieval.models import Query


class ChatTurnState(BaseModel):
    run: AgentRun
    pending_queries: list[Query] = Field(default_factory=list)
    last_gaps: list[str] = Field(default_factory=list)
    answer_text: str = ""
    citations: list[Citation] = Field(default_factory=list)
