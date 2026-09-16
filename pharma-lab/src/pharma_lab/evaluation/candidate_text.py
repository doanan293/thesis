from pharma_lab.corpus.metadata.payload_layers import (
    compact_colloquial_mapping,
    format_colloquial_mapping,
)


def candidate_document_text(payload: dict) -> str:
    embedding_text = str(payload.get("embedding_text") or "").strip()
    if embedding_text:
        return embedding_text
    context_header = str(payload.get("context_header") or "").strip()
    text = str(payload.get("chunk_text") or "").strip()
    visible_text = (
        f"{context_header}\n\n{text}"
        if context_header and text
        else context_header or text
    )
    colloquial_text = format_colloquial_mapping(
        compact_colloquial_mapping(payload), visible_text
    )
    parts = []
    if context_header:
        parts.append(context_header)
    if colloquial_text:
        parts.append(colloquial_text)
    if text:
        parts.append(text)
    return "\n\n".join(parts).strip()
