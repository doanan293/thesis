from __future__ import annotations

import hashlib


def document_hash(document_text: str) -> str:
    return hashlib.sha256(document_text.encode("utf-8")).hexdigest()
