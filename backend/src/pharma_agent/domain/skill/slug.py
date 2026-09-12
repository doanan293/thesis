import re
import unicodedata

MAX_SLUG_CHARS = 60
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def slugify(text: str) -> str:
    """Lowercase ASCII kebab-case; Vietnamese letters are folded (đ → d, ệ → e)."""
    folded = text.replace("đ", "d").replace("Đ", "D")
    ascii_text = (
        unicodedata.normalize("NFKD", folded).encode("ascii", "ignore").decode("ascii")
    )
    slug = (
        _NON_ALNUM.sub("-", ascii_text.lower()).strip("-")[:MAX_SLUG_CHARS].strip("-")
    )
    return slug or "skill"
