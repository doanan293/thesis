import hashlib
import json

from pharma_agent.domain.agent.prompts import (
    ASSISTANT_ROLE,
    DISCLAIMER_PHRASES,
    answer_messages,
    fallback_text,
    judge_messages,
    refine_messages,
    rephrase_messages,
)
from pharma_agent.domain.agent.run import (
    AnswerMode,
    AnswerPlan,
    RunStatus,
)
from pharma_agent.domain.agent.schemas import Audience
from pharma_agent.domain.conversation.models import ConversationContext, Turn
from pharma_agent.domain.retrieval.models import Query, QueryOrigin
from tests.domain.factories import NOW, make_run, search_result


def all_prompt_text() -> str:
    run = make_run()
    run.record_search(
        [Query(text="q", origin=QueryOrigin.INITIAL)], search_result("c1"), now=NOW
    )
    parts = [
        *rephrase_messages(
            "thuốc đó uống lúc nào",
            ConversationContext(
                summary="hỏi về amoxicillin",
                turns=[Turn(user_text="u", assistant_text="a", status="completed")],
            ),
        ),
        *judge_messages(run, run.evidence.summary_view()),
        *refine_messages(run, ["liều tối đa"], ["Panadol"]),
    ]
    for mode in AnswerMode:
        parts.extend(
            answer_messages(
                run,
                AnswerPlan(mode=mode, partial=mode is AnswerMode.GROUNDED),
                "[1] ctx",
            )
        )
    text = "\n".join(m.content for m in parts)
    return text + "\n" + "\n".join(fallback_text(s) for s in RunStatus)


def test_no_medical_disclaimer_anywhere() -> None:
    text = all_prompt_text().lower()
    for phrase in DISCLAIMER_PHRASES:
        assert phrase not in text, phrase


def test_rephrase_prompt_carries_summary_and_turns() -> None:
    messages = rephrase_messages(
        "thuốc đó uống lúc nào",
        ConversationContext(
            summary="về amoxicillin",
            turns=[
                Turn(
                    user_text="amoxicillin là gì",
                    assistant_text="kháng sinh",
                    status="completed",
                )
            ],
        ),
    )
    joined = "\n".join(m.content for m in messages)
    assert (
        "về amoxicillin" in joined
        and "amoxicillin là gì" in joined
        and "thuốc đó uống lúc nào" in joined
    )


def test_judge_and_refine_prompts_include_evidence_hints_and_used_queries() -> None:
    run = make_run()
    run.record_search(
        [Query(text="paracetamol liều", origin=QueryOrigin.INITIAL)],
        search_result("c1"),
        now=NOW,
    )
    judge = "\n".join(
        m.content for m in judge_messages(run, "E1 | Paracetamol > Liều dùng")
    )
    assert "E1 | Paracetamol" in judge
    refine = "\n".join(
        m.content for m in refine_messages(run, ["liều tối đa"], ["Panadol"])
    )
    assert (
        "paracetamol liều" in refine and "Panadol" in refine and "liều tối đa" in refine
    )


def test_answer_prompt_switches_on_mode_and_audience() -> None:
    run = make_run()
    run.audience = Audience.PROFESSIONAL
    grounded = "\n".join(
        m.content
        for m in answer_messages(
            run, AnswerPlan(mode=AnswerMode.GROUNDED), "[1] Paracetamol"
        )
    )
    assert (
        "[1] Paracetamol" in grounded
        and "[n]" in grounded
        and "chuyên môn" in grounded.lower()
    )
    partial = "\n".join(
        m.content
        for m in answer_messages(
            run, AnswerPlan(mode=AnswerMode.GROUNDED, partial=True), "[1] x"
        )
    )
    assert "chưa đủ" in partial
    abstain = "\n".join(
        m.content for m in answer_messages(run, AnswerPlan(mode=AnswerMode.ABSTAIN), "")
    )
    assert "không tìm thấy" in abstain.lower() and "[1]" not in abstain
    assert "thuốc" in fallback_text(RunStatus.TIMEOUT).lower()


def test_prompts_describe_the_leaflet_corpus_by_content() -> None:
    assert "tờ hướng dẫn sử dụng thuốc" in ASSISTANT_ROLE
    assert "an khang" not in all_prompt_text().casefold()


def _grounded_prompt(run) -> str:
    return "\n".join(
        m.content
        for m in answer_messages(run, AnswerPlan(mode=AnswerMode.GROUNDED), "[1] x")
    )


def test_answer_prompt_answers_the_users_own_question() -> None:
    # The rewritten question steers retrieval, but a rewrite of a self-contained
    # question can drop details, so the answer step also sees the user's own words.
    run = make_run("paracetamol uong may vien 1 lan")
    run.standalone_query = "Liều dùng paracetamol cho người lớn"

    prompt = _grounded_prompt(run)

    assert "paracetamol uong may vien 1 lan" in prompt
    assert "Liều dùng paracetamol cho người lớn" in prompt


def test_answer_prompt_is_unchanged_when_the_question_was_not_rewritten() -> None:
    # One-step RAG and the no-rephrase ablation never rewrite the question. Their
    # prompts must stay byte-identical so earlier answers of those configurations
    # remain valid for the current pipeline.
    run = make_run("paracetamol uống mấy viên")
    messages = answer_messages(run, AnswerPlan(mode=AnswerMode.GROUNDED), "[1] x")
    blob = json.dumps([[m.role, m.content] for m in messages], ensure_ascii=False)

    assert hashlib.sha256(blob.encode()).hexdigest() == (
        "dd7d2329035fb29e6005f5d679fb16a354b7aa3f8e9f9fa608be931a373d14fa"
    )


def test_audience_changes_wording_but_keeps_every_fact() -> None:
    for audience in (Audience.GENERAL_PUBLIC, Audience.PROFESSIONAL):
        run = make_run()
        run.audience = audience

        prompt = _grounded_prompt(run).lower()

        assert "ngắn gọn" not in prompt
        assert "đầy đủ" in prompt
