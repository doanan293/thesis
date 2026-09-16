"""Choose the gold queries the E2E golden set is authored from (spec §3.2)."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pharma_lab.e2e.corpus_text import CorpusText
from pharma_lab.e2e.golden import EVAL_GROUPS, ID_PREFIX, Category
from pharma_lab.evaluation.backend_retrieval import sample_quotas

Row = dict[str, Any]


def _sections(row: Row) -> list[str]:
    return [str(section) for section in row.get("expected_section_ids") or []]


def _usable(row: Row, corpus: CorpusText) -> bool:
    sections = _sections(row)
    return bool(sections) and all(section in corpus.sections for section in sections)


def sample_answerable(
    rows: Sequence[Row], *, per_group: int, seed: int, corpus: CorpusText
) -> list[Row]:
    """`per_group` rows per eval group, proportional over difficulty x answer_mode."""
    rng = random.Random(seed)
    chosen: set[int] = set()
    for group in EVAL_GROUPS:
        strata: dict[tuple[str, str], list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            if row.get("eval_group") == group and _usable(row, corpus):
                key = (str(row.get("difficulty")), str(row.get("answer_mode")))
                strata[key].append(index)
        available = sum(len(indices) for indices in strata.values())
        if available < per_group:
            raise ValueError(
                f"eval group {group} has {available} usable rows, need {per_group}"
            )
        quotas = sample_quotas(
            {key: len(indices) for key, indices in strata.items()}, per_group
        )
        for key, indices in strata.items():
            chosen.update(rng.sample(indices, quotas[key]))
    return [rows[index] for index in sorted(chosen)]


def sample_multi_turn(
    rows: Sequence[Row],
    *,
    count: int,
    seed: int,
    corpus: CorpusText,
    exclude: set[str],
) -> list[tuple[Row, Row]]:
    """Pairs of single-answer rows on different sections of one document."""
    by_document: dict[str, dict[str, Row]] = defaultdict(dict)
    for row in rows:
        if (
            row.get("answer_mode") != "single"
            or row["query_id"] in exclude
            or not _usable(row, corpus)
        ):
            continue
        section = _sections(row)[0]
        by_document[corpus.section_documents[section]].setdefault(section, row)
    candidates = sorted(
        document for document, sections in by_document.items() if len(sections) >= 2
    )
    if len(candidates) < count:
        raise ValueError(
            f"only {len(candidates)} documents have two gold sections, need {count}"
        )
    rng = random.Random(seed)
    pairs: list[tuple[Row, Row]] = []
    for document in sorted(rng.sample(candidates, count)):
        sections = sorted(by_document[document])
        first, second = rng.sample(sections, 2)
        pairs.append((by_document[document][first], by_document[document][second]))
    return pairs


def _focus(chunks: dict[str, str], *source_rows: Row) -> dict[str, str]:
    """The expected chunk of each row, when its text is known."""
    labels = (str(row.get("expected_chunk_id")) for row in source_rows)
    return {label: chunks[label] for label in labels if label in chunks}


def _slot(category: Category, number: int) -> str:
    return f"{ID_PREFIX[category]}{number:04d}"


def _write_batches(
    output_dir: Path, name: str, slots: Sequence[Row], batch_size: int
) -> list[Path]:
    paths: list[Path] = []
    for start in range(0, len(slots), batch_size):
        path = output_dir / f"{name}-{start // batch_size + 1:02d}.todo.jsonl"
        path.write_text(
            "".join(
                json.dumps(slot, ensure_ascii=False) + "\n"
                for slot in slots[start : start + batch_size]
            ),
            encoding="utf-8",
        )
        paths.append(path)
    return paths


def write_authoring_batches(
    output_dir: Path,
    answerable: Sequence[Row],
    pairs: Sequence[tuple[Row, Row]],
    corpus: CorpusText,
    *,
    batch_size: int = 35,
    force: bool = False,
    chunks: dict[str, str] | None = None,
) -> list[Path]:
    """Write `*.todo.jsonl` slots carrying the source rows and their section texts.

    `chunks` (gold chunk label -> text) adds each row's expected chunk as
    `focus_chunks`, the part of a long section the question is about.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob("*.todo.jsonl"))
    if existing and not force:
        raise ValueError(
            f"{output_dir} already has {len(existing)} authoring batches; "
            "pass --force to replace them"
        )
    for path in existing:
        path.unlink()

    def texts(*source_rows: Row) -> dict[str, str]:
        return {
            section: corpus.sections[section]
            for row in source_rows
            for section in _sections(row)
        }

    def focus(*source_rows: Row) -> dict[str, str]:
        return _focus(chunks or {}, *source_rows)

    answerable_slots = [
        {
            "slot_id": _slot(Category.ANSWERABLE, number),
            "category": str(Category.ANSWERABLE),
            "source_rows": [row],
            "sections": texts(row),
            "focus_chunks": focus(row),
        }
        for number, row in enumerate(answerable, start=1)
    ]
    dialogue_slots = [
        {
            "slot_id": _slot(Category.MULTI_TURN, number),
            "category": str(Category.MULTI_TURN),
            "source_rows": [first, second],
            "sections": texts(first, second),
            "focus_chunks": focus(first, second),
        }
        for number, (first, second) in enumerate(pairs, start=1)
    ]
    return [
        *_write_batches(output_dir, "answerable", answerable_slots, batch_size),
        *_write_batches(output_dir, "multi-turn", dialogue_slots, batch_size),
    ]


def _stratum(row: Row) -> tuple[str, str, str]:
    return (
        str(row.get("eval_group")),
        str(row.get("difficulty")),
        str(row.get("answer_mode")),
    )


def replacement_candidates(
    rows: Sequence[Row],
    slots: dict[str, Row],
    *,
    used: set[str],
    per_slot: int,
    seed: int,
    corpus: CorpusText,
) -> dict[str, list[Row]]:
    """Unused rows from each slot's stratum, to replace a question that its gold
    section does not answer. Every candidate is offered to one slot only."""
    rng = random.Random(seed)
    taken = set(used)
    result: dict[str, list[Row]] = {}
    for slot_id, source in sorted(slots.items()):
        pool = [
            row
            for row in rows
            if _stratum(row) == _stratum(source)
            and row["query_id"] not in taken
            and _usable(row, corpus)
        ]
        if len(pool) < per_slot:
            raise ValueError(
                f"{slot_id}: only {len(pool)} unused rows in stratum {_stratum(source)}"
            )
        chosen = rng.sample(pool, per_slot)
        taken.update(row["query_id"] for row in chosen)
        result[slot_id] = chosen
    return result


def write_replacement_batch(
    output_dir: Path,
    candidates: dict[str, list[Row]],
    corpus: CorpusText,
    *,
    chunks: dict[str, str] | None = None,
) -> Path:
    """`replacement-NN.todo.jsonl`: per slot, candidate rows with their texts."""
    known = chunks or {}
    number = len(list(output_dir.glob("replacement-*.todo.jsonl"))) + 1
    path = output_dir / f"replacement-{number:02d}.todo.jsonl"
    lines = []
    for slot_id, rows in sorted(candidates.items()):
        options = [
            {
                "source_rows": [row],
                "sections": {
                    section: corpus.sections[section] for section in _sections(row)
                },
                "focus_chunks": _focus(known, row),
            }
            for row in rows
        ]
        lines.append(
            json.dumps(
                {
                    "slot_id": slot_id,
                    "category": str(Category.ANSWERABLE),
                    "candidates": options,
                },
                ensure_ascii=False,
            )
        )
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return path
