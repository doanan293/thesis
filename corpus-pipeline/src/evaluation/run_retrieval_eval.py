import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests
from qdrant_client import QdrantClient

from config.paths import PROCESSED_EVALUATION_DIR, PROJECT_ROOT, RETRIEVAL_EVAL_RUNS_DIR
from evaluation.query_embedding_cache import (
    QueryEmbeddingCache,
    default_query_embedding_cache_path,
    query_hash,
)
from evaluation.rerankers import LlamaCppReranker, NoopReranker
from evaluation.retrieval_metrics import (
    accumulate_metrics,
    new_metric_bucket,
    score_ranked_payloads,
)
from evaluation.retrievers import (
    DenseQdrantRetriever,
    QdrantBm25Retriever,
    QdrantHybridRetriever,
)
from evaluation.section_eval_schema import (
    ALLOWED_ANSWER_MODES,
    ALLOWED_QUERY_FORMS,
    ALLOWED_RETRIEVAL_GRANULARITIES,
    EVAL_HEADER,
)
from model_runtime.catalog import EMBEDDING_MODELS, RERANKER_MODELS, require_model
from model_runtime.client import LlamaCppClient
from model_runtime.compose import LlamaCppComposeManager, resolve_server

DEFAULT_EVAL_CSV = PROCESSED_EVALUATION_DIR / "section_retrieval_eval.csv"
DEFAULT_QUERY_INPUT_BATCH_SIZE = 1
DEFAULT_COMPOSE_FILE = PROJECT_ROOT.parent / "docker-compose.yml"
DEFAULT_GGUF_ROOT = PROJECT_ROOT.parent / "ai-models" / "gguf"
DEFAULT_EVAL_MAX_RETRIES = 3
DEFAULT_EVAL_RETRY_BASE_SLEEP = 5
DISPLAY_HIT_KS = [3, 5, 10]
TRANSIENT_ERRNOS = {32, 54, 104, 110, 111}
TRANSIENT_HTTP_STATUS_CODES = {408, 429, 500, 502, 503, 504}
GROUP_DIMENSIONS = [
    ("by_eval_group", "eval_group", "Metrics By Eval Group"),
    ("by_source_family", "source_family", "Metrics By Source Family"),
    ("by_query_form", "query_form", "Metrics By Query Form"),
    ("by_intent_category", "intent_category", "Metrics By Intent Category"),
    ("by_source_subcategory", "source_subcategory", "Metrics By Source Subcategory"),
    ("by_answer_mode", "answer_mode", "Metrics By Answer Mode"),
    (
        "by_retrieval_granularity",
        "retrieval_granularity",
        "Metrics By Retrieval Granularity",
    ),
    ("by_difficulty", "difficulty", "Metrics By Difficulty"),
]
EVAL_METADATA_FIELDS = [
    "eval_group",
    "source_family",
    "query_form",
    "intent_category",
    "source_subcategory",
    "answer_mode",
    "retrieval_granularity",
    "difficulty",
    "expected_section_id",
    "expected_section_ids",
    "expected_chunk_id",
    "expected_chunk_index",
    "expected_chunk_role",
    "expected_title",
]
EVAL_ROW_HASH_FIELDS = [
    "query_id",
    "query",
    "query_intent_count",
    "eval_tags",
    *EVAL_METADATA_FIELDS,
]


def _exception_chain(exc: BaseException):
    current = exc
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def is_transient_evaluation_error(exc: BaseException) -> bool:
    for current in _exception_chain(exc):
        if isinstance(current, requests.exceptions.HTTPError):
            response = current.response
            if response is None:
                return True
            return response.status_code in TRANSIENT_HTTP_STATUS_CODES
        if isinstance(
            current,
            ConnectionError | TimeoutError | requests.exceptions.RequestException,
        ):
            return True
        if getattr(current, "errno", None) in TRANSIENT_ERRNOS:
            return True

        module = type(current).__module__
        name = type(current).__name__
        if module.startswith("httpx") and any(
            marker in name
            for marker in (
                "Connect",
                "Network",
                "Read",
                "RemoteProtocol",
                "Timeout",
                "Transport",
            )
        ):
            return True
    return False


def evaluate_row_with_retries(
    runner,
    query_row: dict,
    max_retries: int = DEFAULT_EVAL_MAX_RETRIES,
    retry_sleep=time.sleep,
    base_sleep: float = DEFAULT_EVAL_RETRY_BASE_SLEEP,
    on_retry=None,
) -> dict:
    last_error = None
    attempts = max(1, max_retries)
    for attempt in range(1, attempts + 1):
        try:
            return runner.evaluate_row(query_row)
        except Exception as exc:
            last_error = exc
            if attempt >= attempts or not is_transient_evaluation_error(exc):
                raise
            delay = min(base_sleep * (2 ** (attempt - 1)), 30)
            if on_retry is not None:
                on_retry(attempt, attempts, delay, exc)
            retry_sleep(delay)
    raise last_error


def format_evaluation_error_report(failed_queries: list[dict]) -> str:
    lines = ["=== EVALUATION ERROR REPORT ==="]
    if failed_queries:
        lines.append(f"Failed queries skipped from metrics: {len(failed_queries)}")
        for failure in failed_queries:
            lines.append(f"  - {failure['query_id']}: {failure['error']}")
    else:
        lines.append("Failed queries skipped from metrics: 0")
    return "\n".join(lines)


class ExperimentRunner:
    def __init__(
        self,
        retriever,
        reranker=None,
        candidate_k: int = 10,
        top_k: int = 10,
        window_size: int = 3,
    ):
        self.retriever = retriever
        self.reranker = reranker or NoopReranker()
        self.candidate_k = candidate_k
        self.top_k = top_k
        self.window_size = window_size

    def evaluate_row(self, query_row: dict) -> dict:
        search_input = (
            query_row
            if getattr(self.retriever, "accepts_query_row", False)
            else query_row["query"]
        )
        candidates = self.retriever.search(search_input, limit=self.candidate_k)
        reranked = self.reranker.rerank(query_row["query"], candidates)
        final_candidates = reranked[: self.top_k]
        payloads = [candidate.payload for candidate in final_candidates]
        hits = score_ranked_payloads(
            payloads,
            query_row,
            top_k=self.top_k,
            window_size=self.window_size,
        )
        return {
            "query_id": query_row.get("query_id", ""),
            "query": query_row.get("query", ""),
            "hits": hits,
            "ranked_chunk_ids": [
                candidate.resolved_chunk_id for candidate in final_candidates
            ],
            "ranked_section_ids": [
                candidate.section_id for candidate in final_candidates
            ],
            "candidates": final_candidates,
        }


def model_collection_name(model_name: str) -> str:
    model_suffix = model_name.replace(":", "_").replace("-", "_").replace(".", "_")
    return f"thesis_chunks_{model_suffix}"


def slug_name(value: str) -> str:
    return (
        str(value)
        .replace(":", "_")
        .replace("-", "_")
        .replace(".", "_")
        .replace("/", "_")
    )


def default_output_jsonl_path(args) -> Path:
    parts = [args.retriever]
    if args.retriever in {"dense", "bm25", "hybrid"}:
        parts.append(slug_name(args.model))
    if args.reranker != "none":
        parts.append(slug_name(args.reranker))
    parts.append(f"ck{args.candidate_k}")
    parts.append(f"tk{args.top_k}")
    return RETRIEVAL_EVAL_RUNS_DIR / f"{'_'.join(parts)}.jsonl"


def compare_configs(config1: dict, config2: dict) -> bool:
    if not config1 or not config2:
        return False
    keys_to_compare = [
        "retriever",
        "model",
        "candidate_k",
        "top_k",
        "window_size",
        "rrf_k",
        "reranker",
    ]
    return all(config1.get(key) == config2.get(key) for key in keys_to_compare)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run RAG retrieval evaluation.")
    parser.add_argument(
        "--eval-csv",
        type=Path,
        default=DEFAULT_EVAL_CSV,
        help="Path to evaluation CSV.",
    )
    parser.add_argument(
        "--model",
        choices=sorted(EMBEDDING_MODELS),
        default="qwen3-embedding:0.6b-fp16",
        help="Embedding model name.",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default=None,
        help="Qdrant collection name. Defaults to model-derived thesis_chunks_*.",
    )
    parser.add_argument(
        "--qdrant-host", type=str, default="localhost", help="Qdrant host."
    )
    parser.add_argument("--qdrant-port", type=int, default=6333, help="Qdrant port.")
    parser.add_argument(
        "--retriever",
        choices=["dense", "bm25", "hybrid"],
        default="dense",
        help="Candidate retrieval strategy.",
    )
    parser.add_argument(
        "--rrf-k", type=int, default=60, help="RRF constant for hybrid retrieval."
    )
    parser.add_argument(
        "--top-k", type=int, default=10, help="Top K final chunks to evaluate."
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=None,
        help="Candidate count before reranking. Defaults to --top-k.",
    )
    parser.add_argument(
        "--reranker",
        choices=["none", *sorted(RERANKER_MODELS)],
        default="none",
        help="Optional llama.cpp reranker model.",
    )
    parser.add_argument(
        "--server-mode",
        choices=["compose", "external"],
        default="compose",
        help="Start local Compose services or use external llama.cpp endpoints.",
    )
    parser.add_argument(
        "--llama-server-url",
        action="append",
        default=[],
        help="External embedding llama.cpp base URL.",
    )
    parser.add_argument(
        "--reranker-server-url",
        default=None,
        help="External reranker llama.cpp base URL.",
    )
    parser.add_argument("--compose-file", type=Path, default=DEFAULT_COMPOSE_FILE)
    parser.add_argument("--gguf-root", type=Path, default=DEFAULT_GGUF_ROOT)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument(
        "--window-size",
        type=int,
        default=3,
        help="Hydration window size for chunk_window strategy.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of queries to run for quick evaluation.",
    )
    parser.add_argument(
        "--output-json", type=Path, default=None, help="Optional JSON result path."
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=None,
        help="Optional per-query JSONL result path.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Optional Markdown metrics summary path. Defaults to --output-jsonl with .md suffix.",
    )
    parser.add_argument(
        "--query-embedding-cache",
        type=Path,
        default=None,
        help="Optional JSONL cache path for query embeddings.",
    )
    parser.add_argument(
        "--query-input-batch-size",
        type=int,
        default=None,
        help="Queries per llama.cpp embedding request; defaults to the catalog value.",
    )
    parser.add_argument(
        "--from-dump",
        type=Path,
        default=None,
        help="Path to precomputed retrieval candidates JSON dump for offline evaluation without Qdrant.",
    )
    parser.add_argument(
        "--eval-max-retries",
        type=int,
        default=DEFAULT_EVAL_MAX_RETRIES,
        help="Maximum attempts for each query evaluation when transient network errors occur.",
    )
    parser.add_argument(
        "--eval-retry-base-sleep",
        type=float,
        default=DEFAULT_EVAL_RETRY_BASE_SLEEP,
        help="Initial retry sleep in seconds for transient per-query evaluation errors.",
    )
    return parser


build_eval_parser = build_arg_parser


def normalize_args(args):
    args.candidate_k = args.candidate_k or args.top_k
    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")
    if args.candidate_k < 1:
        raise ValueError("--candidate-k must be >= 1")
    if args.reranker != "none" and args.candidate_k < args.top_k:
        raise ValueError("--candidate-k must be >= --top-k when reranking is enabled")
    if args.query_input_batch_size is None:
        args.query_input_batch_size = require_model(args.model).local_request_batch_size
    if args.query_input_batch_size < 1:
        raise ValueError("--query-input-batch-size must be >= 1")
    if args.eval_max_retries < 1:
        raise ValueError("--eval-max-retries must be >= 1")
    if args.eval_retry_base_sleep < 0:
        raise ValueError("--eval-retry-base-sleep must be >= 0")
    if args.request_timeout <= 0:
        raise ValueError("--request-timeout must be > 0")
    if args.server_mode == "external":
        if args.retriever in {"dense", "hybrid"} and not args.llama_server_url:
            raise ValueError(
                "external mode requires --llama-server-url for dense/hybrid retrieval"
            )
        if args.reranker != "none" and not args.reranker_server_url:
            raise ValueError(
                "external mode requires --reranker-server-url when reranking"
            )
    if args.output_jsonl is None:
        args.output_jsonl = default_output_jsonl_path(args)
    if args.output_md is None and args.output_jsonl is not None:
        args.output_md = Path(args.output_jsonl).with_suffix(".md")
    return args


def experiment_config(args, collection_name: str | None) -> dict:
    cache_enabled = not args.no_query_embedding_cache and args.retriever in {
        "dense",
        "hybrid",
    }
    return {
        "retriever": args.retriever,
        "reranker": args.reranker,
        "model": args.model,
        "collection_name": collection_name,
        "candidate_k": args.candidate_k,
        "top_k": args.top_k,
        "window_size": args.window_size,
        "rrf_k": args.rrf_k,
        "query_input_batch_size": args.query_input_batch_size,
        "eval_max_retries": args.eval_max_retries,
        "eval_retry_base_sleep": args.eval_retry_base_sleep,
        "query_embedding_cache_enabled": cache_enabled,
        "query_embedding_cache_path": str(resolve_query_embedding_cache_path(args))
        if cache_enabled
        else None,
        "query_embedding_cache_hits": getattr(args, "query_embedding_cache_hits", 0),
        "query_embedding_cache_misses": getattr(
            args, "query_embedding_cache_misses", 0
        ),
    }


def expected_sections_for_eval_row(row: dict) -> list[str]:
    return [
        section_id
        for section_id in str(row.get("expected_section_ids") or "").split("|")
        if section_id
    ]


def validate_query_rows(rows: list[dict], fieldnames: list[str] | None = None) -> None:
    header = list(fieldnames or [])
    duplicate_columns = sorted({field for field in header if header.count(field) > 1})
    if duplicate_columns:
        raise ValueError(f"duplicate eval column(s): {duplicate_columns}")

    missing_columns = [column for column in EVAL_HEADER if column not in header]
    if missing_columns:
        raise ValueError(f"missing required eval column(s): {missing_columns}")

    for index, row in enumerate(rows, start=1):
        query_id = str(row.get("query_id") or f"row-{index}")
        answer_mode = str(row.get("answer_mode") or "")
        retrieval_granularity = str(row.get("retrieval_granularity") or "")
        query_form = str(row.get("query_form") or "")
        expected_ids = expected_sections_for_eval_row(row)

        if answer_mode not in ALLOWED_ANSWER_MODES:
            raise ValueError(
                f"row {index} {query_id} has unknown answer_mode: {answer_mode}"
            )
        if query_form not in ALLOWED_QUERY_FORMS:
            raise ValueError(
                f"row {index} {query_id} has unknown query_form: {query_form}"
            )
        if retrieval_granularity not in ALLOWED_RETRIEVAL_GRANULARITIES:
            raise ValueError(
                f"row {index} {query_id} has unknown retrieval_granularity: {retrieval_granularity}"
            )
        try:
            query_intent_count = int(row.get("query_intent_count") or "0")
        except ValueError as exc:
            raise ValueError(
                f"row {index} {query_id} has non-integer query_intent_count"
            ) from exc
        if query_intent_count != len(expected_ids):
            raise ValueError(
                f"row {index} {query_id} query_intent_count {query_intent_count} "
                f"does not match {len(expected_ids)} expected sections"
            )
        if answer_mode == "single" and len(expected_ids) != 1:
            raise ValueError(
                f"row {index} {query_id} single answer_mode requires exactly one expected section"
            )
        if answer_mode == "any_acceptable" and len(expected_ids) < 2:
            raise ValueError(
                f"row {index} {query_id} any_acceptable answer_mode requires multiple expected sections"
            )
        if answer_mode == "multi_required" and len(expected_ids) < 2:
            raise ValueError(
                f"row {index} {query_id} multi_required answer_mode requires multiple expected sections"
            )
        if answer_mode == "multi_required" and retrieval_granularity != "multi_section":
            raise ValueError(
                f"row {index} {query_id} multi_required answer_mode requires multi_section retrieval"
            )
        if retrieval_granularity == "multi_section" and answer_mode != "multi_required":
            raise ValueError(
                f"row {index} {query_id} multi_section retrieval requires multi_required answer_mode"
            )


def load_queries(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    validate_query_rows(rows, reader.fieldnames or [])
    return rows


def resolve_query_embedding_cache_path(args) -> Path:
    if args.query_embedding_cache is not None:
        return Path(args.query_embedding_cache)
    return default_query_embedding_cache_path(args.eval_csv, args.model)


def build_query_embedder(model: str, cache: QueryEmbeddingCache | None, embed_fn):
    def embed_query(query_row_or_text):
        if isinstance(query_row_or_text, dict):
            query_id = str(query_row_or_text.get("query_id") or "")
            query_text = str(query_row_or_text.get("query") or "")
        else:
            query_id = ""
            query_text = str(query_row_or_text)
        if cache is None:
            return embed_fn(query_text, model)
        return cache.get_or_embed(
            model,
            query_id,
            query_text,
            lambda text: embed_fn(text, model),
        )

    return embed_query


def chunked(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def warm_query_embedding_cache(
    rows: list[dict],
    model: str,
    cache: QueryEmbeddingCache,
    embed_batch_fn=None,
    input_batch_size: int = DEFAULT_QUERY_INPUT_BATCH_SIZE,
) -> dict:
    if embed_batch_fn is None:
        raise ValueError("embed_batch_fn is required to warm query embeddings")
    start = time.time()
    pending: list[tuple[str, str]] = []
    seen_pending: set[tuple[str, str, str]] = set()
    warmed = 0

    def print_progress() -> None:
        elapsed = time.time() - start
        qps = warmed / max(elapsed, 0.001)
        sys.stdout.write(
            f"\rWarmed {warmed}/{len(rows)} query embeddings "
            f"(hits={cache.hits}, misses={cache.misses}, {qps:.2f} queries/sec)"
        )
        sys.stdout.flush()

    for row in rows:
        query_id = str(row.get("query_id") or "")
        query_text = str(row.get("query") or "")
        key = (str(model), query_id, query_hash(query_text))
        if cache.get(model, query_id, query_text) is not None:
            cache.hits += 1
            warmed += 1
            continue
        if key in seen_pending:
            cache.hits += 1
            warmed += 1
            continue
        seen_pending.add(key)
        cache.misses += 1
        pending.append((query_id, query_text))

    for batch in chunked(pending, input_batch_size):
        texts = [query_text for _, query_text in batch]
        embeddings = embed_batch_fn(texts, model)
        if len(embeddings) != len(batch):
            raise RuntimeError(
                f"Expected {len(batch)} query embeddings, got {len(embeddings)}"
            )
        for (query_id, query_text), embedding in zip(batch, embeddings, strict=False):
            cache.set(model, query_id, query_text, embedding)
            warmed += 1
        if warmed % 50 == 0 or warmed == len(rows):
            print_progress()

    if rows and (not pending or warmed < len(rows)):
        print_progress()
    if rows:
        print()
    return {"total": len(rows), "hits": cache.hits, "misses": cache.misses}


def build_reranker(args):
    if args.reranker == "none":
        return NoopReranker()
    return LlamaCppReranker(require_model(args.reranker), args.reranker_client)


def build_retriever(
    args,
    qdrant_client: QdrantClient | None,
    collection_name: str | None,
    query_embedding_cache=None,
):
    if getattr(args, "from_dump", None) is not None:
        from evaluation.retrievers import PrecomputedJsonRetriever

        return PrecomputedJsonRetriever(args.from_dump)
    if args.retriever in {"dense", "bm25", "hybrid"}:
        if qdrant_client is None or collection_name is None:
            raise ValueError(
                "Qdrant client and collection are required for dense, bm25, or hybrid retrieval"
            )
        if args.retriever == "bm25":
            return QdrantBm25Retriever(
                qdrant_client,
                collection_name,
            )
        retriever_cls = (
            QdrantHybridRetriever
            if args.retriever == "hybrid"
            else DenseQdrantRetriever
        )
        retriever_kwargs = {"rrf_k": args.rrf_k} if args.retriever == "hybrid" else {}
        return retriever_cls(
            qdrant_client,
            collection_name,
            embed_query=build_query_embedder(
                args.model,
                None if args.no_query_embedding_cache else query_embedding_cache,
                lambda texts: args.embedding_client.embed(
                    [texts], args.model, require_model(args.model).vector_dimension
                )[0],
            ),
            **retriever_kwargs,
        )
    raise ValueError(f"Unsupported retriever: {args.retriever}")


def serialize_metrics(metrics: dict) -> dict:
    return {
        key: serialize_metrics(value) if isinstance(value, dict) else value
        for key, value in metrics.items()
    }


def eval_row_hash(query_row: dict) -> str:
    payload = {field: str(query_row.get(field) or "") for field in EVAL_ROW_HASH_FIELDS}
    return query_hash(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def eval_row_metadata(query_row: dict) -> dict:
    return {field: str(query_row.get(field) or "") for field in EVAL_METADATA_FIELDS}


def per_query_output(result: dict, query_row: dict | None = None) -> dict:
    row = query_row or result
    output = {
        "query_id": result["query_id"],
        "query": result["query"],
        "query_hash": query_hash(result["query"]),
        "eval_row_hash": eval_row_hash(row),
        "hits": result["hits"],
        "ranked_chunk_ids": result["ranked_chunk_ids"],
        "ranked_section_ids": result["ranked_section_ids"],
    }
    output.update(eval_row_metadata(row))
    return output


def write_jsonl_checkpoint(path: Path, config: dict, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(
            json.dumps({"type": "metadata", "config": config}, ensure_ascii=False)
            + "\n"
        )
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json_output(
    path: Path,
    config: dict,
    metrics: dict,
    grouped_metrics: dict,
    per_query: list[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        "config": config,
        "metrics": serialize_metrics(metrics),
        "grouped_metrics": serialize_metrics(grouped_metrics),
        "queries": per_query,
    }
    path.write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def new_grouped_metric_buckets() -> dict:
    return {
        group_key: defaultdict(new_metric_bucket)
        for group_key, _field, _title in GROUP_DIMENSIONS
    }


def accumulate_grouped_metrics(
    grouped_metrics: dict, hits: dict, query_row: dict
) -> None:
    for group_key, field, _title in GROUP_DIMENSIONS:
        value = str(query_row.get(field) or "unknown")
        accumulate_metrics(grouped_metrics[group_key][value], hits, query_row)


def markdown_metric_table(metrics: dict) -> str:
    lines = [
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Total Queries | {metrics['count']} |",
    ]
    if metrics["count"] == 0:
        return "\n".join(lines)
    for k in DISPLAY_HIT_KS:
        hit_key = f"hit@{k}"
        if hit_key in metrics:
            value = metrics[hit_key] / metrics["count"] * 100
            lines.append(f"| Hit@{k} | {value:.2f}% |")

    multi_count = metrics.get("multi_count", 0)
    if multi_count > 0:
        for k in DISPLAY_HIT_KS:
            multi_hit_key = f"multi_all_hit@{k}"
            if multi_hit_key in metrics:
                value = metrics[multi_hit_key] / multi_count * 100
                lines.append(f"| All-Hit@{k} | {value:.2f}% |")
            recall_key = f"multi_section_recall@{k}"
            if recall_key in metrics:
                value = metrics[recall_key] / multi_count * 100
                lines.append(f"| Section-Recall@{k} | {value:.2f}% |")

    lines.append(f"| MRR | {metrics['mrr'] / metrics['count']:.4f} |")
    return "\n".join(lines)


def write_markdown_output(
    path: Path, config: dict, metrics: dict, grouped_metrics: dict
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Retrieval Evaluation Summary", ""]
    lines.extend(["## Configuration", ""])
    for key, value in sorted(config.items()):
        lines.append(f"- `{key}`: `{value}`")

    lines.extend(["", "## Overall Metrics", "", markdown_metric_table(metrics), ""])
    for group_key, _field, title in GROUP_DIMENSIONS:
        group = grouped_metrics.get(group_key, {})
        if not group:
            continue
        lines.extend([f"## {title}", ""])
        for name, group_metrics in sorted(group.items()):
            lines.extend([f"### {name}", "", markdown_metric_table(group_metrics), ""])

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def print_metrics(title: str, metrics: dict):
    print(f"\n=== {title} ===")
    print(f"  Total Queries: {metrics['count']}")
    if metrics["count"] == 0:
        return
    for k in DISPLAY_HIT_KS:
        hit_key = f"hit@{k}"
        if hit_key in metrics:
            val = metrics[hit_key] / metrics["count"] * 100
            print(f"  Hit@{k:<2}: {val:.2f}%")

    multi_count = metrics.get("multi_count", 0)
    if multi_count > 0:
        print("  --- Multi-Required (All-Hit / Section Recall) ---")
        for k in DISPLAY_HIT_KS:
            multi_hit_key = f"multi_all_hit@{k}"
            if multi_hit_key in metrics:
                val = metrics[multi_hit_key] / multi_count * 100
                print(f"  All-Hit@{k:<2}: {val:.2f}%")
            recall_key = f"multi_section_recall@{k}"
            if recall_key in metrics:
                val = metrics[recall_key] / multi_count * 100
                print(f"  Section-Recall@{k:<2}: {val:.2f}%")

    print(f"  MRR   : {metrics['mrr'] / metrics['count']:.4f}")


def configure_model_runtime(args) -> None:
    manager = LlamaCppComposeManager(args.compose_file)
    if args.from_dump is None and args.retriever in {"dense", "hybrid"}:
        spec = require_model(args.model)
        endpoints = resolve_server(
            args.server_mode, args.llama_server_url, spec, manager, args.gguf_root
        )
        if len(endpoints) != 1:
            raise ValueError(
                "retrieval evaluation currently requires one embedding server"
            )
        args.embedding_client = LlamaCppClient(
            endpoints[0], timeout=args.request_timeout
        )
    if args.reranker != "none":
        spec = require_model(args.reranker)
        external = [args.reranker_server_url] if args.reranker_server_url else []
        endpoint = resolve_server(
            args.server_mode, external, spec, manager, args.gguf_root
        )[0]
        args.reranker_client = LlamaCppClient(endpoint, timeout=args.request_timeout)


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        normalize_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    if not args.eval_csv.exists():
        print(
            f"Error: Evaluation CSV not found at {args.eval_csv}. Please run build_section_retrieval_eval.py first."
        )
        sys.exit(1)

    print(f"Loading queries from {args.eval_csv}...")
    all_queries = load_queries(args.eval_csv)
    queries = all_queries

    if args.limit:
        queries = queries[: args.limit]
        print(f"Limited evaluation to first {args.limit} queries.")

    collection_name = args.collection_name or model_collection_name(args.model)

    current_config = experiment_config(args, collection_name)

    completed_results = []
    config_matched = False

    if args.output_jsonl.exists():
        try:
            with args.output_jsonl.open("r", encoding="utf-8") as f:
                lines = f.readlines()
            if lines:
                first_line = json.loads(lines[0])
                if first_line.get("type") == "metadata":
                    if compare_configs(first_line.get("config"), current_config):
                        config_matched = True
                        for line in lines[1:]:
                            if not line.strip():
                                continue
                            record = json.loads(line)
                            if record.get("type") == "result":
                                completed_results.append(record)
        except Exception as e:
            print(f"Warning: Failed to parse existing output file: {e}")

    valid_completed_results = []
    if config_matched and completed_results:
        query_by_id = {q["query_id"]: q for q in queries}
        completed_by_id = {}
        for record in completed_results:
            query_id = record.get("query_id")
            query_row = query_by_id.get(query_id)
            if not query_row:
                continue
            expected_hash = query_hash(str(query_row.get("query") or ""))
            expected_eval_hash = eval_row_hash(query_row)
            if (
                record.get("query_hash") == expected_hash
                and record.get("eval_row_hash") == expected_eval_hash
            ):
                completed_by_id[query_id] = record
        valid_completed_results = [
            completed_by_id[q["query_id"]]
            for q in queries
            if q["query_id"] in completed_by_id
        ]
        invalid_count = len(completed_results) - len(valid_completed_results)
        if invalid_count:
            print(
                f"Found {invalid_count} stale result records with changed or missing query hashes."
            )
        if len(valid_completed_results) < len(queries):
            print(
                f"Resuming run with {len(valid_completed_results)}/{len(queries)} "
                "query results still valid."
            )
        else:
            print(
                "Previous run is fully completed and query hashes still match. Reusing existing results."
            )
    elif args.output_jsonl.exists():
        print("Existing output config does not match. Starting a fresh run.")

    overall = new_metric_bucket()
    grouped_metrics = new_grouped_metric_buckets()
    per_query_results: list[dict] = []
    failed_queries: list[dict] = []

    if config_matched:
        # Create query map for quick lookup
        query_by_id = {q["query_id"]: q for q in queries}
        for record in valid_completed_results:
            query_id = record["query_id"]
            query_row = query_by_id.get(query_id)
            if query_row:
                hits = record["hits"]
                accumulate_metrics(overall, hits, query_row)
                accumulate_grouped_metrics(grouped_metrics, hits, query_row)
                per_query_results.append(record)

        completed_ids = {r["query_id"] for r in valid_completed_results}
        queries_to_run = [q for q in queries if q["query_id"] not in completed_ids]
        write_jsonl_checkpoint(
            args.output_jsonl, current_config, valid_completed_results
        )
    else:
        # Start fresh: write metadata line
        write_jsonl_checkpoint(args.output_jsonl, current_config, [])
        queries_to_run = queries

    if queries_to_run:
        configure_model_runtime(args)

    query_embedding_cache = None
    if (
        args.from_dump is None
        and not args.no_query_embedding_cache
        and args.retriever in {"dense", "hybrid"}
    ):
        query_embedding_cache = QueryEmbeddingCache(
            resolve_query_embedding_cache_path(args)
        )
        print(f"Using query embedding cache at {query_embedding_cache.path}")
        prune_stats = query_embedding_cache.prune_to_queries(args.model, all_queries)
        if prune_stats["removed"]:
            print(
                f"Pruned query embedding cache: kept {prune_stats['kept']}, "
                f"removed {prune_stats['removed']} stale records."
            )
        stats = warm_query_embedding_cache(
            queries_to_run,
            args.model,
            query_embedding_cache,
            embed_batch_fn=lambda texts, model: args.embedding_client.embed(
                texts, model, require_model(model).vector_dimension
            ),
            input_batch_size=args.query_input_batch_size,
        )
        print(f"Warmed query embedding cache at {query_embedding_cache.path}")
        print(f"Remaining queries to embed: {stats['total']}")
        print(f"Cache hits: {stats['hits']}")
        print(f"Cache misses: {stats['misses']}")

    start_time = time.time()
    run_processed_count = 0
    if queries_to_run:
        qdrant_client = None
        if args.from_dump is None and args.retriever in {"dense", "bm25", "hybrid"}:
            print(f"Connecting to Qdrant at {args.qdrant_host}:{args.qdrant_port}...")
            qdrant_client = QdrantClient(host=args.qdrant_host, port=args.qdrant_port)

            collections = [c.name for c in qdrant_client.get_collections().collections]
            if collection_name not in collections:
                print(
                    f"Error: Qdrant collection '{collection_name}' not found. Please run ingest_vectors.py first."
                )
                sys.exit(1)

        print(f"Building retriever '{args.retriever}'...")
        retriever = build_retriever(
            args, qdrant_client, collection_name, query_embedding_cache
        )
        reranker = build_reranker(args)
        runner = ExperimentRunner(
            retriever,
            reranker=reranker,
            candidate_k=args.candidate_k,
            top_k=args.top_k,
            window_size=args.window_size,
        )

        target = collection_name
        print(
            f"Starting evaluation of {len(queries_to_run)} queries (total {len(queries)}) against '{target}'..."
        )

        for idx, query_row in enumerate(
            queries_to_run, start=len(valid_completed_results) + 1
        ):
            qid = query_row["query_id"]
            try:

                def log_retry(
                    attempt: int,
                    attempts: int,
                    delay: float,
                    exc: BaseException,
                    _qid: str = qid,
                ) -> None:
                    print(
                        f"\nTransient error evaluating query {_qid} "
                        f"(attempt {attempt}/{attempts}): {exc}; retrying in {delay:.1f}s"
                    )

                result = evaluate_row_with_retries(
                    runner,
                    query_row,
                    max_retries=args.eval_max_retries,
                    base_sleep=args.eval_retry_base_sleep,
                    on_retry=log_retry,
                )
                hits = result["hits"]
            except Exception as e:
                print(f"\nError evaluating query {query_row['query_id']}: {e}")
                failed_queries.append(
                    {"query_id": query_row["query_id"], "error": str(e)}
                )
                continue

            accumulate_metrics(overall, hits, query_row)
            accumulate_grouped_metrics(grouped_metrics, hits, query_row)

            res_output = per_query_output(result, query_row)
            per_query_results.append(res_output)

            # Write incrementally to output_jsonl
            with args.output_jsonl.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps({"type": "result", **res_output}, ensure_ascii=False)
                    + "\n"
                )

            run_processed_count += 1
            if idx % 50 == 0 or idx == len(queries):
                elapsed = time.time() - start_time
                qps = run_processed_count / max(elapsed, 0.001)
                sys.stdout.write(
                    f"\rProcessed {idx}/{len(queries)} queries... ({qps:.2f} queries/sec)"
                )
                sys.stdout.flush()
    else:
        print("No queries to evaluate; reused existing valid results.")

    print("\n\nEvaluation Complete!")
    print(f"Time taken: {time.time() - start_time:.2f} seconds")
    print()
    print(format_evaluation_error_report(failed_queries))

    # Print Results
    print_metrics("OVERALL METRICS", overall)

    for group_key, _field, title in GROUP_DIMENSIONS:
        for name, metrics in sorted(grouped_metrics[group_key].items()):
            print_metrics(f"{title.upper()}: {name.upper()}", metrics)
    args.query_embedding_cache_hits = getattr(query_embedding_cache, "hits", 0)
    args.query_embedding_cache_misses = getattr(query_embedding_cache, "misses", 0)
    config = experiment_config(args, collection_name)
    if args.output_json:
        write_json_output(
            args.output_json, config, overall, grouped_metrics, per_query_results
        )
        print(f"Wrote JSON results to {args.output_json}")
    print(f"Wrote JSONL results to {args.output_jsonl}")
    if args.output_md:
        write_markdown_output(args.output_md, config, overall, grouped_metrics)
        print(f"Wrote Markdown summary to {args.output_md}")


if __name__ == "__main__":
    main()
