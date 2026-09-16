import asyncio
from pathlib import Path

import pytest
from pharma_agent.application.chat.graph import build_chat_graph
from pharma_agent.domain.agent.budget import BudgetLimits
from pharma_agent.domain.agent.schemas import (
    Audience,
    Intent,
    JudgeDecision,
    JudgeOutcome,
    Language,
    RephraseResult,
)
from pharma_agent.domain.guardrail.models import LlmGuardVerdict
from pharma_agent.domain.llm.models import LlmRole
from pharma_agent.domain.llm.port import LlmError
from pharma_agent.infrastructure.settings import Settings
from tests.e2e.factories import (
    FakeRetriever,
    ScriptedLlm,
    make_hit,
    retrieval_service,
)

from pharma_lab.e2e.configs import E2EConfig, pipeline_for
from pharma_lab.e2e.executor import BackendTurnExecutor, conversation_for
from pharma_lab.e2e.golden import GoldenItem
from pharma_lab.e2e.harness import run_items, spread, with_deadline
from pharma_lab.e2e.records import AnswerRecord, JsonlStore

SECTION = "drug:paracetamol:lieu-dung"


def item(item_id: str = "e2e-mt-0001") -> GoldenItem:
    return GoldenItem.model_validate(
        {
            "item_id": item_id,
            "category": "multi_turn",
            "turns": [
                {"role": "user", "text": "Paracetamol dùng làm gì?"},
                {"role": "assistant", "text": "Giảm đau, hạ sốt."},
                {"role": "user", "text": "Người lớn uống tối đa bao nhiêu?"},
            ],
            "expected_behavior": "grounded",
            "gold_section_ids": [SECTION],
            "reference": {
                "answer": "4 g/ngày",
                "key_facts": [
                    {"fact": "4 g", "evidence_quote": "4 g", "section_id": SECTION}
                ],
            },
        }
    )


def llm(*, judge: bool, rephrase: bool) -> ScriptedLlm:
    fake = ScriptedLlm(answer="Tối đa 4 g mỗi ngày [1].")
    fake.script(
        LlmRole.GUARDRAIL, LlmGuardVerdict(is_attack=False, in_scope=True, reason="ok")
    )
    if rephrase:
        fake.script(
            LlmRole.REPHRASE,
            RephraseResult(
                standalone_query="Liều tối đa paracetamol người lớn",
                audience=Audience.GENERAL_PUBLIC,
                language=Language.VI,
                intent=Intent.PHARMA_QUESTION,
            ),
        )
    if judge:
        fake.script(
            LlmRole.JUDGE,
            JudgeDecision(decision=JudgeOutcome.ANSWER, gaps=[], reason="đủ"),
        )
    return fake


def executor(fake: ScriptedLlm, retriever: FakeRetriever, config: E2EConfig):
    return BackendTurnExecutor(
        graph=build_chat_graph(),
        llm=fake,
        retrieval=retrieval_service(retriever),
        limits=BudgetLimits(),
        pipeline=pipeline_for(config),
        config=str(config),
    )


def test_history_becomes_the_conversation_context() -> None:
    context = conversation_for(item())
    assert [(t.user_text, t.assistant_text) for t in context.turns] == [
        ("Paracetamol dùng làm gì?", "Giảm đau, hạ sốt.")
    ]


async def test_full_agent_record_maps_citations_to_gold_labels() -> None:
    fake = llm(judge=True, rephrase=True)
    retriever = FakeRetriever(
        [
            make_hit("a", SECTION, ordinal=2, score=0.9),
            make_hit("b", "other", score=0.1),
        ]
    )

    record = await executor(fake, retriever, E2EConfig.FULL).execute(item())

    assert record.status == "completed"
    assert record.answer_mode == "grounded"
    assert record.standalone_query == "Liều tối đa paracetamol người lớn"
    assert record.search_queries == [["Liều tối đa paracetamol người lớn"]]
    assert record.judge_outcomes == ["answer"]
    assert [(c.index, c.chunk_id, c.section_id) for c in record.citations] == [
        (1, f"{SECTION}:chunk-002", SECTION)
    ]
    assert record.retrieved_chunk_ids[0] == f"{SECTION}:chunk-002"
    assert record.context_text.startswith("[1] Paracetamol > Liều dùng")
    assert record.llm_calls == 4
    assert sorted(record.usage_by_role) == ["answer", "guardrail", "judge", "rephrase"]
    assert record.usage_by_role["answer"].prompt_tokens == 100
    assert not record.retryable
    assert "Người lớn uống tối đa" in fake.calls[1][1][-1].content


async def test_one_step_searches_the_raw_question_without_a_judge() -> None:
    fake = llm(judge=False, rephrase=False)
    retriever = FakeRetriever([make_hit("a", SECTION, score=0.9)])

    record = await executor(fake, retriever, E2EConfig.ONE_STEP).execute(item())

    assert retriever.queries == [["Người lớn uống tối đa bao nhiêu?"]]
    assert record.judge_outcomes == ["skipped"]
    assert record.llm_calls == 2
    assert record.answer_mode == "grounded"


async def test_a_failed_answer_is_a_retryable_record() -> None:
    fake = llm(judge=False, rephrase=False)
    fake.stream_error = LlmError("endpoint down")
    retriever = FakeRetriever([make_hit("a", SECTION)])

    record = await executor(fake, retriever, E2EConfig.ONE_STEP).execute(item())

    assert record.status == "error"
    assert record.retryable
    assert record.error is not None and record.error.startswith("ANSWER_FAILED")


class CountingExecutor:
    def __init__(self, failing: set[str]) -> None:
        self.failing = failing
        self.seen: list[str] = []
        self.active = 0
        self.peak = 0

    async def execute(self, item: GoldenItem) -> AnswerRecord:
        golden = item
        self.seen.append(golden.item_id)
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        if golden.item_id in self.failing:
            raise RuntimeError("endpoint down")
        return AnswerRecord(item_id=golden.item_id, config="full", status="completed")


async def test_run_items_resumes_and_retries_only_on_request(tmp_path: Path) -> None:
    items = [item(f"e2e-mt-{n:04d}") for n in range(1, 7)]
    store = JsonlStore(tmp_path / "answers.jsonl", AnswerRecord)
    first = CountingExecutor(failing={"e2e-mt-0002"})

    summary = await run_items(
        items, first, store, config="full", concurrency=2, retry_errors=False
    )

    assert (summary.total, summary.ran, summary.errors) == (6, 6, 1)
    assert first.peak == 2
    again = CountingExecutor(failing=set())
    summary = await run_items(
        items, again, store, config="full", concurrency=2, retry_errors=False
    )
    assert again.seen == []
    assert summary.errors == 1
    summary = await run_items(
        items, again, store, config="full", concurrency=2, retry_errors=True
    )
    assert again.seen == ["e2e-mt-0002"]
    assert summary.errors == 0
    assert store.latest()["e2e-mt-0002"].status == "completed"


def test_spread_takes_items_across_the_set() -> None:
    items = [item(f"e2e-mt-{n:04d}") for n in range(1, 11)]
    assert [i.item_id[-2:] for i in spread(items, 3)] == ["01", "05", "09"]
    assert spread(items, None) == items
    assert spread(items, 50) == items


def test_deadline_override_changes_only_the_budget() -> None:
    base = Settings(_env_file=None)
    assert with_deadline(base, None) is base
    longer = with_deadline(base, 600)
    assert longer.budget.deadline_seconds == 600
    assert base.budget.deadline_seconds == 90
    assert longer.budget.max_llm_calls == base.budget.max_llm_calls
    with pytest.raises(ValueError, match="positive"):
        with_deadline(base, 0)
