import re
from dataclasses import dataclass

from pharma_agent.domain.guardrail.models import Verdict, VerdictSource


@dataclass(frozen=True)
class PatternGroup:
    label: str
    patterns: tuple[re.Pattern[str], ...]

    def check(self, text_lower: str) -> tuple[str, ...] | None:
        hits = tuple(p.pattern for p in self.patterns if p.search(text_lower))
        return hits or None


def _compile(*patterns: str) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(p, re.IGNORECASE) for p in patterns)


PROMPT_INJECTION = PatternGroup(
    "prompt_injection",
    _compile(
        r"ignore (all |the |any )?(previous|above|prior|earlier) (instructions|prompts?|rules)",
        r"disregard (all |the |your )?(previous|above|prior) instructions",
        r"bỏ qua (mọi|các|tất cả|những)? ?(hướng dẫn|chỉ dẫn|lệnh|quy tắc) (trước|ở trên|phía trên)",
        r"quên (hết|toàn bộ|mọi) (hướng dẫn|chỉ dẫn|lệnh)",
        r"new instructions?:",
        r"\bsystem prompt\b",
    ),
)

JAILBREAK = PatternGroup(
    "jailbreak",
    _compile(
        r"\bdo anything now\b",
        r"\bdan mode\b",
        r"developer mode",
        r"you are now (an? )?(unrestricted|unfiltered|uncensored)",
        r"(không|chẳng) (có|còn) (bất kỳ )?(giới hạn|kiểm duyệt|quy tắc) nào",
        r"chế độ (không giới hạn|nhà phát triển)",
    ),
)

DATA_LEAK = PatternGroup(
    "data_leak",
    _compile(
        r"(reveal|print|show|display|repeat|output) (me )?(your|the) (system )?(prompt|instructions|rules)",
        r"(in|hiện|hiển thị|tiết lộ|cho xem) (ra )?(system prompt|prompt hệ thống|hướng dẫn hệ thống)",
        r"\bapi[ _-]?key\b",
        r"\b(secret|mật khẩu|password)s? (của|of) (hệ thống|the system|server)",
    ),
)

BYPASS = PatternGroup(
    "bypass",
    _compile(
        r"pretend (that )?(you are|you're|to be) (not|an? )",
        r"giả vờ (rằng )?(bạn|mày|cậu) (là|không phải)",
        r"role[- ]?play as (an? )?(unrestricted|evil|different)",
        r"hypothetically,? (there are|you have) no rules",
        r"đóng vai (một )?(ai|trợ lý|chatbot) (không|chẳng) (có|bị) (giới hạn|kiểm duyệt)",
    ),
)

ALL_GROUPS: tuple[PatternGroup, ...] = (PROMPT_INJECTION, JAILBREAK, DATA_LEAK, BYPASS)


def regex_screen(query: str) -> Verdict | None:
    """Return a blocking verdict when a high-risk pattern matches, else None."""
    text = " ".join(query.lower().split())
    for group in ALL_GROUPS:
        hits = group.check(text)
        if hits:
            return Verdict(
                passed=False,
                in_scope=False,
                source=VerdictSource.REGEX,
                label=group.label,
                reason="; ".join(hits),
            )
    return None
