from pharma_agent.domain.agent.prompts import (
    ASSISTANT_ROLE,
    DISCLAIMER_PHRASES,
    answer_messages,
    fallback_text,
    judge_messages,
    refine_messages,
    rephrase_messages,
    skill_selection_messages,
)
from pharma_agent.domain.agent.run import (
    AnswerMode,
    AnswerPlan,
    RunStatus,
    SelectedSkill,
)
from pharma_agent.domain.agent.schemas import Audience
from pharma_agent.domain.conversation.models import ConversationContext, Turn
from pharma_agent.domain.retrieval.models import Query, QueryOrigin
from pharma_agent.domain.skill.models import SkillMetadata
from tests.domain.factories import NOW, make_run, search_result


def all_prompt_text() -> str:
    run = make_run()
    run.skills = [
        SelectedSkill(name="s", title="S", instructions="tìm mục liều\ntrả lời bảng")
    ]
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
        *skill_selection_messages("q", [SkillMetadata(name="s", description="d")]),
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


def test_judge_and_refine_prompts_include_skill_guidance_hints_and_used_queries() -> (
    None
):
    run = make_run()
    run.skills = [SelectedSkill(name="s", title="S", instructions="tìm mục Liều dùng")]
    run.record_search(
        [Query(text="paracetamol liều", origin=QueryOrigin.INITIAL)],
        search_result("c1"),
        now=NOW,
    )
    judge = "\n".join(
        m.content for m in judge_messages(run, "E1 | Paracetamol > Liều dùng")
    )
    assert "tìm mục Liều dùng" in judge and "E1 | Paracetamol" in judge
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
