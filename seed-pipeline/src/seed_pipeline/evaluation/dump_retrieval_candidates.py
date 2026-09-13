import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from seed_pipeline.config.paths import (
    HEAVY_RETRIEVAL_EVAL_DIR,
    PROCESSED_EVALUATION_DIR,
)
from seed_pipeline.evaluation.retrieval_candidate_artifact import (
    CandidateArtifact,
    build_candidate_artifact,
)
from seed_pipeline.evaluation.retrieval_types import RetrievalCandidate

DEFAULT_EVAL_JSONL = PROCESSED_EVALUATION_DIR / "section_retrieval_eval.jsonl"
DEFAULT_OUTPUT_DIR = HEAVY_RETRIEVAL_EVAL_DIR / "candidates"


def build_dump_payload(
    queries_data: Sequence[dict[str, Any]],
    retriever_type: str,
    top_k: int,
) -> dict[str, Any]:
    queries_dict: dict[str, Any] = {}

    for item in queries_data:
        row = item["row"]
        candidates: list[RetrievalCandidate] = item["candidates"]
        q_id = str(row.get("query_id") or "").strip()
        q_text = str(row.get("query") or "").strip()

        serialized_candidates = [
            {
                "chunk_id": c.resolved_chunk_id,
                "score": c.score,
                "rank": c.rank,
                "source": c.source,
                "payload": c.payload,
            }
            for c in candidates
        ]

        key = q_id or q_text
        queries_dict[key] = {
            "query_id": q_id,
            "query": q_text,
            "candidates": serialized_candidates,
        }

    return {
        "metadata": {
            "version": "1.0",
            "created_at": datetime.now(UTC).isoformat(),
            "retriever_type": retriever_type,
            "top_k_dumped": top_k,
            "total_queries": len(queries_dict),
        },
        "queries": queries_dict,
    }


def dump_retrieval_candidates(
    eval_jsonl_path: Path,
    retriever_type: str,
    top_k: int,
    output_json_path: Path,
    retriever_instance: Any = None,
):
    rows = []
    with open(eval_jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    queries_data = []
    for row in rows:
        if retriever_instance is not None:
            if getattr(retriever_instance, "accepts_query_row", False):
                candidates = retriever_instance.search_query_row(row, limit=top_k)
            else:
                candidates = retriever_instance.search(
                    row.get("query", ""), limit=top_k
                )
        else:
            candidates = []
        queries_data.append({"row": row, "candidates": candidates})

    dump_payload = build_dump_payload(
        queries_data=queries_data, retriever_type=retriever_type, top_k=top_k
    )

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(dump_payload, f, ensure_ascii=False, indent=2)

    return dump_payload


def dump_candidate_artifact(
    *,
    eval_jsonl_path: Path,
    retriever_instance: Any,
    output_jsonl_path: Path,
    identity: dict[str, Any],
    candidate_k: int,
) -> CandidateArtifact:
    rows = []
    with Path(eval_jsonl_path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return build_candidate_artifact(
        rows=rows,
        retriever=retriever_instance,
        output_path=output_jsonl_path,
        identity=identity,
        candidate_k=candidate_k,
    )


def load_query_rows(path: Path, limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    try:
        with Path(path).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid evaluation JSON at {path}:{line_number}"
                    ) from exc
                if (
                    not isinstance(row, dict)
                    or not row.get("query_id")
                    or not str(row.get("query") or "").strip()
                ):
                    raise ValueError(
                        "Evaluation row requires query_id and query at "
                        f"{path}:{line_number}"
                    )
                rows.append(row)
                if limit is not None and len(rows) >= limit:
                    break
    except OSError as exc:
        raise ValueError(f"Evaluation JSONL is missing: {path}") from exc
    if not rows:
        raise ValueError(f"Evaluation JSONL is empty: {path}")
    return rows
