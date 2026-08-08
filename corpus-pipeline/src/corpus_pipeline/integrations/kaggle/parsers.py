from __future__ import annotations

import csv
import io
import json


def parse_gpu_quota_hours(csv_text: str) -> float:
    for row in csv.DictReader(io.StringIO(csv_text)):
        if str(row.get("resource", "")).strip().upper() == "GPU":
            return max(0.0, float(str(row["remaining"]).strip().removesuffix("h")))
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
    for index, character in enumerate(output):
        if character != "[":
            continue
        try:
            payload, _ = decoder.raw_decode(output, index)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, list):
            entries = []
            for record in payload:
                if not isinstance(record, dict) or not isinstance(
                    record.get("data"), str
                ):
                    raise ValueError("Kaggle kernel log entry is invalid")
                entries.append(record["data"])
            return entries
    raise ValueError("Kaggle kernel logs did not contain a JSON list")


def format_elapsed(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_timed_log_lines(entry: str, elapsed: float) -> list[str]:
    prefix = f"[{format_elapsed(elapsed)}]"
    return [f"{prefix} {line.strip()}" for line in entry.splitlines() if line.strip()]


def parse_dataset_status(output: str) -> str:
    payload = parse_dataset_status_payload(output)
    status = str(payload["status"]).strip().upper()
    if status:
        return status
    raise RuntimeError(f"Unrecognized Kaggle dataset status: {output.strip()}")


def parse_dataset_status_payload(output: str) -> dict:
    decoder = json.JSONDecoder()
    for index, character in enumerate(output):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(output, index)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "status" in payload:
            return payload
    raise RuntimeError(f"Unrecognized Kaggle dataset status: {output.strip()}")


def parse_dataset_references(output: str) -> set[str]:
    lines = output.splitlines()
    header_index = next(
        (
            index
            for index, line in enumerate(lines)
            if next(csv.reader([line]), [None])[0].strip().casefold() == "ref"
        ),
        None,
    )
    if header_index is None:
        return set()
    return {
        str(row.get("ref", "")).strip()
        for row in csv.DictReader(io.StringIO("\n".join(lines[header_index:])))
        if str(row.get("ref", "")).strip()
    }


def parse_kaggle_username(output: str) -> str:
    for line in output.splitlines():
        key, separator, value = line.lstrip("- ").partition(":")
        if separator and key.strip().casefold() == "username" and value.strip():
            return value.strip()
    raise RuntimeError("Kaggle config did not contain an authenticated username")
