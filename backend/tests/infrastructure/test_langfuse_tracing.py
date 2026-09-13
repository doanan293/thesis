"""Langfuse adapter against an in-memory OTel exporter and a mocked ingestion endpoint."""

import json
from collections.abc import Generator, Sequence
from uuid import uuid4

import httpx
import pytest
from langfuse import Langfuse
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from pharma_agent.application.chat.graph import build_chat_graph
from pharma_agent.application.chat.runner import ChatTurnRunner
from pharma_agent.domain.agent.budget import BudgetLimits
from pharma_agent.domain.agent.schemas import JudgeDecision, JudgeOutcome
from pharma_agent.domain.llm.models import LlmRole
from pharma_agent.domain.retrieval.models import Hit
from pharma_agent.domain.retrieval.ports import RetrievalError
from pharma_agent.infrastructure.observability.langfuse_retrieval import (
    RERANK_OBSERVATION_NAME,
    LangfuseTracedReranker,
)
from pharma_agent.infrastructure.observability.langfuse_tracing import (
    FEEDBACK_SCORE_NAME,
    TURN_OBSERVATION_NAME,
    LangfuseTracing,
)
from tests.application.test_chat_graph import QUESTION, passing_llm
from tests.domain.factories import chunk_uuid, make_hit
from tests.fakes import FakeReranker, FakeRetriever, build_deps


def trace_id_of(span: ReadableSpan) -> str:
    context = span.context
    assert context is not None
    return format(context.trace_id, "032x")


class FakeLangfuseServer:
    def __init__(self) -> None:
        self.exporter = InMemorySpanExporter()
        self.ingested: list[dict[str, object]] = []
        self.public_key = f"pk-test-{uuid4().hex}"

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/public/ingestion":
            self.ingested.extend(json.loads(request.content)["batch"])
        return httpx.Response(207, json={"successes": [], "errors": []})

    def client(self) -> Langfuse:
        # Langfuse keeps one resource manager per public key for the whole process,
        # and shutdown() closes it; a unique key keeps every test's client isolated.
        return Langfuse(
            public_key=self.public_key,
            secret_key="sk-test",
            base_url="http://langfuse.test",
            span_exporter=self.exporter,
            httpx_client=httpx.Client(transport=httpx.MockTransport(self.handle)),
            flush_at=1,
            flush_interval=0.05,
        )


@pytest.fixture
def server() -> Generator[FakeLangfuseServer]:
    fake = FakeLangfuseServer()
    yield fake


async def test_turn_becomes_one_trace_with_graph_spans(
    server: FakeLangfuseServer,
) -> None:
    client = server.client()
    tracing = LangfuseTracing(client, public_key=server.public_key)
    llm = passing_llm()
    llm.script(
        LlmRole.JUDGE, JudgeDecision(decision=JudgeOutcome.ANSWER, gaps=[], reason="đủ")
    )
    runner = ChatTurnRunner(
        build_chat_graph(),
        build_deps(llm, FakeRetriever([make_hit("c1", fusion=0.9)])),
        BudgetLimits(),
        tracer=tracing,
    )
    execution = runner.start(user_id="u1", message=QUESTION, conversation_id="c" * 32)
    async for _ in execution.events():
        pass
    tracing.flush()

    expected_trace = tracing.trace_id_for(execution.run.run_id)
    spans = server.exporter.get_finished_spans()
    by_name = {span.name: span for span in spans}
    turn = by_name[TURN_OBSERVATION_NAME]
    assert trace_id_of(turn) == expected_trace
    attributes = dict(turn.attributes or {})
    assert attributes["user.id"] == "u1" and attributes["session.id"] == "c" * 32
    assert execution.outcome is not None
    assert attributes["langfuse.observation.output"] == execution.outcome.answer_text
    assert "guard" in by_name and "answer" in by_name
    assert all(trace_id_of(span) == expected_trace for span in spans)
    client.shutdown()


def test_feedback_is_scored_on_the_turn_trace(server: FakeLangfuseServer) -> None:
    client = server.client()
    tracing = LangfuseTracing(client, public_key=server.public_key)
    tracing.record_feedback(run_id="run-1", rating="down", note="sai liều")
    tracing.flush()
    scores = [item for item in server.ingested if item.get("type") == "score-create"]
    assert len(scores) == 1
    body = scores[0]["body"]
    assert isinstance(body, dict)
    assert body["name"] == FEEDBACK_SCORE_NAME and body["value"] == 0.0
    assert body["traceId"] == tracing.trace_id_for("run-1")
    assert body["comment"] == "sai liều"
    client.shutdown()


async def test_rerank_is_a_retriever_observation_inside_the_turn_trace(
    server: FakeLangfuseServer,
) -> None:
    client = server.client()
    tracing = LangfuseTracing(client, public_key=server.public_key)
    llm = passing_llm()
    llm.script(
        LlmRole.JUDGE, JudgeDecision(decision=JudgeOutcome.ANSWER, gaps=[], reason="đủ")
    )
    reranker = LangfuseTracedReranker(
        FakeReranker(), client, protocol="completion_logprobs", model="qwen3-reranker"
    )
    runner = ChatTurnRunner(
        build_chat_graph(),
        build_deps(llm, FakeRetriever([make_hit("c1", fusion=0.9)]), reranker=reranker),
        BudgetLimits(),
        tracer=tracing,
    )
    execution = runner.start(user_id="u1", message=QUESTION)
    async for _ in execution.events():
        pass
    tracing.flush()

    rerank = next(
        span
        for span in server.exporter.get_finished_spans()
        if span.name == RERANK_OBSERVATION_NAME
    )
    assert trace_id_of(rerank) == tracing.trace_id_for(execution.run.run_id)
    attributes = dict(rerank.attributes or {})
    assert attributes["langfuse.observation.type"] == "retriever"
    assert json.loads(str(attributes["langfuse.observation.output"])) == [
        {"chunk_version_id": str(chunk_uuid("c1")), "rerank_score": 1.0}
    ]
    client.shutdown()


class DownReranker:
    async def rerank(self, query: str, hits: Sequence[Hit], top_n: int) -> list[Hit]:
        raise RetrievalError("rerank down")


async def test_rerank_failure_is_recorded_as_an_error_and_reraised(
    server: FakeLangfuseServer,
) -> None:
    client = server.client()
    reranker = LangfuseTracedReranker(
        DownReranker(), client, protocol="native_rerank", model="m"
    )
    with pytest.raises(RetrievalError, match="rerank down"):
        await reranker.rerank("q", [make_hit("c1")], top_n=1)
    client.flush()

    span = next(
        span
        for span in server.exporter.get_finished_spans()
        if span.name == RERANK_OBSERVATION_NAME
    )
    attributes = dict(span.attributes or {})
    assert attributes["langfuse.observation.level"] == "ERROR"
    assert attributes["langfuse.observation.status_message"] == "rerank down"
    client.shutdown()
