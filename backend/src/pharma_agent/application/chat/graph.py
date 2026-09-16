from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from pharma_agent.application.chat import nodes, routing
from pharma_agent.application.chat.context import TurnContext
from pharma_agent.application.chat.state import ChatTurnState

ChatGraph = CompiledStateGraph[ChatTurnState, TurnContext, ChatTurnState, ChatTurnState]


def build_chat_graph(checkpointer: BaseCheckpointSaver | None = None) -> ChatGraph:
    """guard → rephrase → search ⇄ (judge → refine) → answer, with fallback."""
    builder = StateGraph(ChatTurnState, context_schema=TurnContext)
    builder.add_node("guard", nodes.guard_node)
    builder.add_node("rephrase", nodes.rephrase_node)
    builder.add_node("search", nodes.search_node)
    builder.add_node("judge", nodes.judge_node)
    builder.add_node("refine", nodes.refine_node)
    builder.add_node("answer", nodes.answer_node)
    builder.add_node("fallback", nodes.fallback_node)

    builder.add_edge(START, "guard")
    builder.add_conditional_edges("guard", routing.route_after_guard)
    builder.add_conditional_edges("rephrase", routing.route_after_rephrase)
    builder.add_conditional_edges("search", routing.route_after_search)
    builder.add_conditional_edges("judge", routing.route_after_judge)
    builder.add_conditional_edges("refine", routing.route_after_refine)
    builder.add_conditional_edges("answer", routing.route_after_answer)
    builder.add_edge("fallback", END)
    return builder.compile(checkpointer=checkpointer)
