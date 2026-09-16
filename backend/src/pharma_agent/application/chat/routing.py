from typing import Literal

from pharma_agent.application.chat.state import ChatTurnState
from pharma_agent.domain.agent.run import RunStatus, Step

FALLBACK = "fallback"


def _failed(state: ChatTurnState) -> bool:
    return state.run.status in (RunStatus.ERROR, RunStatus.TIMEOUT)


def route_after_guard(
    state: ChatTurnState,
) -> Literal["rephrase", "answer", "fallback"]:
    if _failed(state):
        return FALLBACK
    verdict = state.run.guard_verdict
    return "rephrase" if verdict is not None and verdict.allows_processing else "answer"


def route_after_rephrase(
    state: ChatTurnState,
) -> Literal["search", "answer", "fallback"]:
    if _failed(state):
        return FALLBACK
    return "answer" if Step.ANSWER in state.run.allowed_steps() else "search"


def route_after_search(state: ChatTurnState) -> Literal["judge", "fallback"]:
    return FALLBACK if _failed(state) else "judge"


def route_after_judge(state: ChatTurnState) -> Literal["refine", "answer", "fallback"]:
    if _failed(state):
        return FALLBACK
    return "refine" if Step.REFINE in state.run.allowed_steps() else "answer"


def route_after_refine(state: ChatTurnState) -> Literal["search", "answer", "fallback"]:
    if _failed(state):
        return FALLBACK
    return "search" if Step.SEARCH in state.run.allowed_steps() else "answer"


def route_after_answer(state: ChatTurnState) -> Literal["fallback", "__end__"]:
    return "fallback" if _failed(state) else "__end__"
