import math
from collections import defaultdict
from collections.abc import Iterable, Mapping

DEFAULT_METRIC_KS = (3, 5, 10, 30)
# Cutoffs of the BEIR/MTEB-style metrics reported next to Hit@k and MRR@top_k.
NDCG_K = 10
MRR_K = 10


def expected_section_ids(row: dict) -> list[str]:
    raw = row.get("expected_section_ids")
    if isinstance(raw, list):
        return [str(s) for s in raw if s]
    if isinstance(raw, str):
        return [s for s in raw.split("|") if s]
    return []


def expected_chunk_ids(row: dict) -> list[str]:
    raw = row.get("expected_chunk_ids")
    values = [str(value) for value in raw if value] if isinstance(raw, list) else []
    legacy = str(row.get("expected_chunk_id") or "")
    if legacy and legacy not in values:
        values.insert(0, legacy)
    return values


def _chunk_id(payload: dict) -> str:
    return str(payload.get("chunk_id") or payload.get("chunk_key") or "")


def is_hit(
    retrieved_payload: dict,
    expected: dict,
    window_size: int = 3,
    accepted_chunk_ids: frozenset[str] = frozenset(),
) -> bool:
    """Whether a retrieved chunk answers the gold row.

    `accepted_chunk_ids` are chunks outside the gold sections that relevance judgments
    accept for this row (see evaluation/relevance_judgments.py).
    """
    if accepted_chunk_ids and _chunk_id(retrieved_payload) in accepted_chunk_ids:
        return True
    granularity = expected.get("retrieval_granularity", "section")
    answer_mode = expected.get("answer_mode", "single")
    expected_section_id = expected.get("expected_section_id", "")
    expected_sections = expected_section_ids(expected)
    retrieved_section_id = retrieved_payload.get("section_id")

    if granularity == "multi_section" or answer_mode == "multi_required":
        return retrieved_section_id in expected_sections

    if answer_mode == "any_acceptable":
        return retrieved_section_id in expected_sections

    if granularity == "chunk_exact":
        return _chunk_id(retrieved_payload) in expected_chunk_ids(expected)

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

    return False


def evidence_units(query_row: dict) -> tuple[str, ...]:
    """What a query needs found: every section of a multi-part question, otherwise one
    unit (the expected section, or any of the acceptable ones)."""
    sections = expected_section_ids(query_row)
    multi = (
        query_row.get("answer_mode") == "multi_required"
        or query_row.get("retrieval_granularity") == "multi_section"
    )
    if multi and sections:
        return tuple(sections)
    return ("answer",)


def satisfied_unit(
    payload: dict,
    query_row: dict,
    window_size: int,
    judged: Mapping[str, frozenset[str]],
) -> str | None:
    """The evidence unit a retrieved chunk satisfies, if any."""
    units = evidence_units(query_row)
    if units == ("answer",):
        accepted = frozenset().union(*judged.values()) if judged else frozenset()
        return "answer" if is_hit(payload, query_row, window_size, accepted) else None
    section = payload.get("section_id")
    if section in units:
        return section
    chunk = _chunk_id(payload)
    return next((unit for unit in units if chunk in judged.get(unit, ())), None)


def ranking_scores(
    retrieved_payloads: list[dict],
    query_row: dict,
    *,
    window_size: int,
    judged: Mapping[str, frozenset[str]],
    ndcg_k: int = NDCG_K,
    mrr_k: int = MRR_K,
) -> dict[str, float]:
    """nDCG@k with binary gains and MRR@k over evidence units.

    A unit earns gain once, at the first chunk that satisfies it, so several chunks of
    one gold section do not inflate the score; the ideal ranking finds every unit in
    the first positions. For single-answer queries nDCG@k is 1/log2(rank+1) of the
    first hit, as in known-item retrieval benchmarks.
    """
    units = evidence_units(query_row)
    seen: set[str] = set()
    dcg = 0.0
    first_hit: int | None = None
    for rank, payload in enumerate(retrieved_payloads[:ndcg_k], start=1):
        unit = satisfied_unit(payload, query_row, window_size, judged)
        if unit is None or unit in seen:
            continue
        seen.add(unit)
        dcg += 1.0 / math.log2(rank + 1)
    for rank, payload in enumerate(retrieved_payloads[:mrr_k], start=1):
        if satisfied_unit(payload, query_row, window_size, judged) is not None:
            first_hit = rank
            break
    ideal = sum(
        1.0 / math.log2(rank + 1) for rank in range(1, min(len(units), ndcg_k) + 1)
    )
    return {
        f"ndcg@{ndcg_k}": dcg / ideal,
        f"mrr@{mrr_k}": 0.0 if first_hit is None else 1.0 / first_hit,
    }


def score_ranked_payloads(
    retrieved_payloads: list[dict],
    query_row: dict,
    top_k: int = 10,
    window_size: int = 3,
    metric_ks: Iterable[int] = DEFAULT_METRIC_KS,
    judgments: Mapping[str, frozenset[str]] | None = None,
) -> dict[str, float]:
    """Hit@k, MRR and multi-section recall of one ranked candidate list.

    `judgments` maps a gold section id to the extra chunk ids accepted for it.
    """
    judged = judgments or {}
    accepted = frozenset().union(*judged.values()) if judged else frozenset()
    hits: dict[str, float] = {}
    expected_sections = expected_section_ids(query_row)
    retrieved_sections = [
        p.get("section_id") for p in retrieved_payloads if p.get("section_id")
    ]
    retrieved_chunks = [_chunk_id(p) for p in retrieved_payloads]

    for k in metric_ks:
        if k > top_k:
            continue
        payloads_at_k = retrieved_payloads[:k]
        hits[f"hit@{k}"] = (
            1
            if any(is_hit(p, query_row, window_size, accepted) for p in payloads_at_k)
            else 0
        )

        if query_row.get("answer_mode") == "multi_required":
            retrieved_set_at_k = set(retrieved_sections[:k])
            chunks_at_k = set(retrieved_chunks[:k])
            matched_count = sum(
                1
                for section_id in expected_sections
                if section_id in retrieved_set_at_k
                or chunks_at_k & judged.get(section_id, frozenset())
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
        if is_hit(payload, query_row, window_size, accepted):
            first_hit_rank = rank
            break
    hits["mrr"] = 1.0 / first_hit_rank if first_hit_rank else 0.0
    if top_k >= NDCG_K:
        hits.update(
            ranking_scores(
                retrieved_payloads, query_row, window_size=window_size, judged=judged
            )
        )
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
