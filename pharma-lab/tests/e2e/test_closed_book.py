import asyncio

from pharma_agent.domain.llm.models import ChatRole, LlmRole
from pharma_agent.domain.llm.port import LlmError
from tests.e2e.factories import ScriptedLlm
from tests.e2e.test_executor import item

from pharma_lab.e2e.configs import E2EConfig, uses_retrieval
from pharma_lab.e2e.executor import ClosedBookExecutor


def test_closed_book_asks_the_answer_model_with_history_and_no_documents():
    llm = ScriptedLlm(answer="Người lớn tối đa 4 g mỗi ngày.")
    executor = ClosedBookExecutor(llm=llm, config=E2EConfig.CLOSED_BOOK.value)

    record = asyncio.run(executor.execute(item()))

    [(role, messages)] = llm.calls
    assert role is LlmRole.ANSWER
    assert [m.role for m in messages] == [
        ChatRole.SYSTEM,
        ChatRole.USER,
        ChatRole.ASSISTANT,
        ChatRole.USER,
    ]
    assert messages[-1].content == "Người lớn uống tối đa bao nhiêu?"
    assert "Tài liệu" not in "\n".join(m.content for m in messages)
    assert record.answer_text == "Người lớn tối đa 4 g mỗi ngày."
    # The answer counts as an attempt, so the judge scores its content.
    assert record.answer_mode == "grounded"
    assert record.context_text == ""
    assert record.citations == [] and record.retrieved_chunk_ids == []
    assert record.llm_calls == 1
    assert record.usage_by_role["answer"].prompt_tokens == 100
    assert record.error is None and not record.retryable


def test_closed_book_records_a_failed_call_as_retryable():
    llm = ScriptedLlm(stream_error=LlmError("upstream down"))
    executor = ClosedBookExecutor(llm=llm, config=E2EConfig.CLOSED_BOOK.value)

    record = asyncio.run(executor.execute(item()))

    assert record.status == "error" and record.retryable
    assert record.answer_mode is None
    assert "upstream down" in (record.error or "")


def test_only_the_closed_book_baseline_skips_retrieval():
    assert not uses_retrieval(E2EConfig.CLOSED_BOOK)
    assert all(uses_retrieval(c) for c in E2EConfig if c is not E2EConfig.CLOSED_BOOK)
