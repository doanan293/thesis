from collections.abc import Sequence

from pharma_agent.domain.conversation.models import Turn
from pharma_agent.domain.llm.models import ChatMessage, system, user

SUMMARY_SYSTEM = """Bạn tóm tắt hội thoại giữa người dùng và trợ lý tra cứu thuốc để dùng làm ngữ cảnh cho các câu hỏi sau.
Trả về JSON {{"summary": "..."}}.

Quy tắc:
- Giữ lại: tên thuốc/hoạt chất/biệt dược đã nhắc, đối tượng (trẻ em, thai kỳ, bệnh nền), triệu chứng, các con số liều đã trao đổi, và điều người dùng còn muốn biết.
- Bỏ: lời chào, câu lặp lại, số trích dẫn [n].
- Viết tiếng Việt, câu ngắn, tối đa {max_chars} ký tự.
- Gộp tóm tắt cũ với các lượt mới thành một bản duy nhất; thông tin mới hơn thay thế thông tin cũ mâu thuẫn."""


def summary_messages(
    previous_summary: str, turns: Sequence[Turn], *, max_chars: int
) -> list[ChatMessage]:
    rendered = "\n\n".join(
        f"Người dùng: {turn.user_text}\nTrợ lý: {turn.assistant_text}" for turn in turns
    )
    content = (
        f"Tóm tắt cũ:\n{previous_summary.strip() or '(chưa có)'}\n\n"
        f"Các lượt mới:\n{rendered}"
    )
    return [system(SUMMARY_SYSTEM.format(max_chars=max_chars)), user(content)]
