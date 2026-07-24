from collections import defaultdict
from collections.abc import Iterable

DEFAULT_METRIC_KS = (3, 5, 10)


def expected_section_ids(row: dict) -> list[str]:
    return [
        section_id
        for section_id in row.get("expected_section_ids", "").split("|")
        if section_id
    ]


def is_hit(retrieved_payload: dict, expected: dict, window_size: int = 3) -> bool:
    granularity = expected.get("retrieval_granularity", "section")
    answer_mode = expected.get("answer_mode", "single")
    expected_section_id = expected.get("expected_section_id", "")
    expected_sections = expected_section_ids(expected)
    retrieved_section_id = retrieved_payload.get("section_id")

    if granularity == "multi_section" or answer_mode == "multi_required":
        return retrieved_section_id in expected_sections

    if answer_mode == "any_acceptable":
        return retrieved_section_id in expected_sections

    if retrieved_section_id != expected_section_id:
        return False

    if granularity == "section":
        return True

    if granularity == "chunk_window":
        if retrieved_payload.get("chunk_index") in (None, ""):
            return False
        try:
            retrieved_idx = int(retrieved_payload.get("chunk_index"))
            expected_idx = int(expected.get("expected_chunk_index", -1))
        except (ValueError, TypeError):
            return False
        if expected_idx == -1:
            return False
        return abs(retrieved_idx - expected_idx) <= window_size // 2

    if granularity == "chunk_exact":
        retrieved_chunk_id = retrieved_payload.get("chunk_id") or str(
            retrieved_payload.get("chunk_key") or ""
        )
        return retrieved_chunk_id == expected.get("expected_chunk_id", "")

    return False


def score_ranked_payloads(
    retrieved_payloads: list[dict],
    query_row: dict,
    top_k: int = 10,
    window_size: int = 3,
    metric_ks: Iterable[int] = DEFAULT_METRIC_KS,
) -> dict[str, float]:
    hits: dict[str, float] = {}
    expected_sections = expected_section_ids(query_row)
    retrieved_sections = [
        p.get("section_id") for p in retrieved_payloads if p.get("section_id")
    ]

    for k in metric_ks:
        if k > top_k:
            continue
        payloads_at_k = retrieved_payloads[:k]
        hits[f"hit@{k}"] = (
            1 if any(is_hit(p, query_row, window_size) for p in payloads_at_k) else 0
        )

        if query_row.get("answer_mode") == "multi_required":
            retrieved_set_at_k = set(retrieved_sections[:k])
            matched_count = sum(
                1
                for section_id in expected_sections
                if section_id in retrieved_set_at_k
            )
            total = len(expected_sections) or 1
            hits[f"multi_section_recall@{k}"] = matched_count / total
            hits[f"multi_all_hit@{k}"] = (
                1
                if matched_count == len(expected_sections) and expected_sections
                else 0
            )

    first_hit_rank = None
    for rank, payload in enumerate(retrieved_payloads, start=1):
        if is_hit(payload, query_row, window_size):
            first_hit_rank = rank
            break
    hits["mrr"] = 1.0 / first_hit_rank if first_hit_rank else 0.0
    return hits


def new_metric_bucket() -> defaultdict[str, float]:
    return defaultdict(float)


def accumulate_metrics(
    target: defaultdict[str, float], hits: dict[str, float], query_row: dict
) -> None:
    target["count"] += 1
    for key, value in hits.items():
        target[key] += value
    if query_row.get("answer_mode") == "multi_required":
        target["multi_count"] += 1


def metric_average(metrics: dict, key: str, count_key: str = "count") -> float:
    count = metrics.get(count_key, 0)
    return 0.0 if count == 0 else metrics.get(key, 0.0) / count
