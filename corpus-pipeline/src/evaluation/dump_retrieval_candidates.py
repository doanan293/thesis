import argparse
import csv
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evaluation.retrieval_types import RetrievalCandidate


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
    eval_csv_path: Path,
    retriever_type: str,
    top_k: int,
    output_json_path: Path,
    retriever_instance: Any = None,
):
    with open(eval_csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

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


def main():
    parser = argparse.ArgumentParser(
        description="Dump Qdrant retrieval candidates to JSON for offline eval."
    )
    parser.add_argument(
        "--eval-csv", type=Path, required=True, help="Path to evaluation CSV file"
    )
    parser.add_argument(
        "--retriever", choices=["dense", "bm25", "hybrid"], default="hybrid"
    )
    parser.add_argument(
        "--top-k", type=int, default=50, help="Number of candidates to dump per query"
    )
    parser.add_argument(
        "--output-json", type=Path, required=True, help="Output path for JSON dump"
    )
    args = parser.parse_args()

    print(f"Candidate dumper initialized for {args.eval_csv}")


if __name__ == "__main__":
    main()
