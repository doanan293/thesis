from __future__ import annotations

import csv
import io
import json

from kaggle_vector_cache.models import require_model


def parse_gpu_quota_hours(csv_text: str) -> float:
    for row in csv.DictReader(io.StringIO(csv_text)):
        if str(row.get("resource", "")).strip().upper() == "GPU":
            remaining = str(row["remaining"]).strip().removesuffix("h")
            return max(0.0, float(remaining))
    raise RuntimeError("Kaggle quota output did not contain a GPU row")


def parse_kernel_status(output: str) -> str:
    value = output.casefold()
    if "error" in value or "fail" in value:
        return "ERROR"
    if "complete" in value:
        return "COMPLETE"
    if "running" in value:
        return "RUNNING"
    if "queue" in value or "pending" in value:
        return "QUEUED"
    raise RuntimeError(f"Unrecognized Kaggle kernel status: {output.strip()}")


def parse_kernel_log_entries(output: str) -> list[str]:
    decoder = json.JSONDecoder()
    payload = None
    for index, character in enumerate(output):
        if character != "[":
            continue
        try:
            candidate, _end = decoder.raw_decode(output, index)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, list):
            payload = candidate
            break
    if payload is None:
        raise ValueError("Kaggle kernel logs did not contain a JSON list")
    if not isinstance(payload, list):
        raise ValueError("Kaggle kernel logs must be a JSON list")
    entries = []
    for record in payload:
        if not isinstance(record, dict) or not isinstance(record.get("data"), str):
            raise ValueError("Kaggle kernel log entry is invalid")
        entries.append(record["data"])
    return entries


def format_elapsed(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, sec = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def format_timed_log_lines(entry: str, elapsed: float) -> list[str]:
    prefix = f"[{format_elapsed(elapsed)}]"
    return [f"{prefix} {line.strip()}" for line in entry.splitlines() if line.strip()]


def format_compact_elapsed(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, sec = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{sec:02d}s"
    return f"{minutes}m{sec:02d}s"


def parse_dataset_status(output: str) -> str:
    decoder = json.JSONDecoder()
    payload = None
    for index, character in enumerate(output):
        if character != "{":
            continue
        try:
            candidate, _end = decoder.raw_decode(output, index)
            if isinstance(candidate, dict) and "status" in candidate:
                payload = candidate
                break
        except json.JSONDecodeError:
            continue
    if payload is None:
        try:
            payload = json.loads(output)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise RuntimeError(
                f"Unrecognized Kaggle dataset status: {output.strip()}"
            ) from exc
    status = str(payload.get("status", "")).strip().upper()
    if not status:
        raise RuntimeError(f"Unrecognized Kaggle dataset status: {output.strip()}")
    return status


def checkpoint_dataset_slug(model: str) -> str:
    slug = require_model(model).slug.replace("_", "-")
    return f"vector-cache-checkpoint-{slug}"


def kernel_slug(model: str) -> str:
    return f"vector-cache-ingest-{require_model(model).slug.replace('_', '-')}"
