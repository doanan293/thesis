QDRANT_RUNTIME_REQUIRED_PAYLOAD_FIELDS = (
    "chunk_id",
    "section_id",
    "chunk_index",
    "hydrate_strategy",
    "source",
    "title",
    "section",
    "start_page",
    "end_page",
    "context_header",
    "chunk_text",
    "embedding_text",
)

QDRANT_RUNTIME_OPTIONAL_PAYLOAD_FIELDS = (
    "colloquial_mapping",
    "term_annotations",
    "content_type",
    "table_id",
)

QDRANT_RUNTIME_PAYLOAD_FIELDS = (
    *QDRANT_RUNTIME_REQUIRED_PAYLOAD_FIELDS,
    *QDRANT_RUNTIME_OPTIONAL_PAYLOAD_FIELDS,
)

QDRANT_RUNTIME_INDEX_FIELDS = (
    "section_id",
    "chunk_id",
    "chunk_index",
    "hydrate_strategy",
    "content_type",
    "table_id",
)

QDRANT_INTEGER_INDEX_FIELDS = frozenset({"chunk_index"})
QDRANT_RUNTIME_INTEGER_PAYLOAD_FIELDS = frozenset(
    {"chunk_index", "start_page", "end_page"}
)
QDRANT_TERM_ANNOTATION_FIELDS = frozenset({"term", "vi", "en"})
QDRANT_COLLOQUIAL_MAPPING_FIELDS = frozenset(
    {"key", "aliases", "visual_sign", "product_names"}
)


def compact_runtime_payload(record: dict) -> dict:
    return {
        field: record.get(field)
        for field in QDRANT_RUNTIME_PAYLOAD_FIELDS
        if record.get(field) not in ("", [], {}, None)
    }


def validate_runtime_payload(record: dict, label: str = "runtime payload") -> None:
    missing = sorted(set(QDRANT_RUNTIME_REQUIRED_PAYLOAD_FIELDS) - set(record))
    if missing:
        raise ValueError(f"{label} missing required field(s): {missing}")

    extra = sorted(set(record) - set(QDRANT_RUNTIME_PAYLOAD_FIELDS))
    if extra:
        raise ValueError(f"{label} contains unexpected field(s): {extra}")

    for field in QDRANT_RUNTIME_REQUIRED_PAYLOAD_FIELDS:
        if record.get(field) in (None, ""):
            raise ValueError(f"{label} field {field} is empty")

    for field in QDRANT_RUNTIME_INTEGER_PAYLOAD_FIELDS:
        if not isinstance(record.get(field), int) or isinstance(
            record.get(field), bool
        ):
            raise ValueError(f"{label} field {field} must be an integer")

    mapping = record.get("colloquial_mapping")
    if mapping not in (None, {}, ""):
        if not isinstance(mapping, dict):
            raise ValueError(f"{label} field colloquial_mapping must be an object")
        extra_mapping = sorted(set(mapping) - QDRANT_COLLOQUIAL_MAPPING_FIELDS)
        if extra_mapping:
            raise ValueError(
                f"{label} field colloquial_mapping has unexpected key(s): {extra_mapping}"
            )
        for field in ("aliases", "product_names"):
            if field in mapping and not isinstance(mapping[field], list):
                raise ValueError(
                    f"{label} field colloquial_mapping.{field} must be a list"
                )

    annotations = record.get("term_annotations")
    if annotations in (None, [], ""):
        return
    if not isinstance(annotations, list):
        raise ValueError(f"{label} field term_annotations must be a list")
    for index, annotation in enumerate(annotations, start=1):
        if not isinstance(annotation, dict):
            raise ValueError(f"{label} term_annotations[{index}] must be an object")
        extra_annotation = sorted(set(annotation) - QDRANT_TERM_ANNOTATION_FIELDS)
        if extra_annotation:
            raise ValueError(
                f"{label} term_annotations[{index}] has unexpected key(s): {extra_annotation}"
            )
        term = str(annotation.get("term") or "").strip()
        if not term:
            raise ValueError(f"{label} term_annotations[{index}] missing term")
        for field in ("vi", "en"):
            if field in annotation and not isinstance(annotation[field], list):
                raise ValueError(
                    f"{label} term_annotations[{index}].{field} must be a list"
                )


def compact_validated_runtime_payload(
    record: dict, label: str = "runtime payload"
) -> dict:
    payload = compact_runtime_payload(record)
    missing = sorted(set(QDRANT_RUNTIME_REQUIRED_PAYLOAD_FIELDS) - set(payload))
    if missing:
        raise ValueError(f"{label} missing required field(s): {missing}")
    validate_runtime_payload(payload, label=label)
    return payload
