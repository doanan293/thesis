import json
from collections import Counter
from pathlib import Path

import pytest

from pharma_lab.e2e.corpus_text import CorpusText
from pharma_lab.e2e.golden import EVAL_GROUPS
from pharma_lab.e2e.sampling import (
    sample_answerable,
    sample_multi_turn,
    write_authoring_batches,
)


def corpus(documents: int = 30, sections_per_document: int = 3) -> CorpusText:
    sections: dict[str, str] = {}
    owners: dict[str, str] = {}
    for d in range(documents):
        for s in range(sections_per_document):
            key = f"doc{d}:s{s}"
            sections[key] = f"text {d} {s}"
            owners[key] = f"doc{d}"
    titles = {f"doc{d}": f"Drug {d}" for d in range(documents)}
    return CorpusText(sections=sections, section_documents=owners, titles=titles)


def rows(corpus_text: CorpusText) -> list[dict]:
    keys = sorted(corpus_text.sections)
    result = []
    for index in range(1200):
        group = EVAL_GROUPS[index % len(EVAL_GROUPS)]
        result.append(
            {
                "query_id": f"q{index}",
                "query": f"question {index}",
                "eval_group": group,
                "difficulty": "hard" if index % 10 == 0 else "medium",
                "answer_mode": "single",
                "expected_section_ids": [keys[index % len(keys)]],
            }
        )
    result.append(
        {
            "query_id": "missing",
            "query": "x",
            "eval_group": "formulary",
            "difficulty": "medium",
            "answer_mode": "single",
            "expected_section_ids": ["nowhere"],
        }
    )
    return result


def test_answerable_sample_is_deterministic_and_balanced() -> None:
    text = corpus()
    source = rows(text)

    first = sample_answerable(source, per_group=20, seed=3, corpus=text)
    second = sample_answerable(source, per_group=20, seed=3, corpus=text)

    assert first == second
    assert Counter(row["eval_group"] for row in first) == dict.fromkeys(EVAL_GROUPS, 20)
    assert all(row["query_id"] != "missing" for row in first)
    hard = Counter(row["eval_group"] for row in first if row["difficulty"] == "hard")
    # Hard rows (index % 10 == 0) fall in the even-position groups: 20% of 20 = 4.
    assert hard == dict.fromkeys(EVAL_GROUPS[::2], 4)
    ids = [int(row["query_id"][1:]) for row in first]
    assert ids == sorted(ids)


def test_answerable_sample_needs_enough_rows() -> None:
    text = corpus()
    with pytest.raises(ValueError, match="need 500"):
        sample_answerable(rows(text), per_group=500, seed=0, corpus=text)


def test_multi_turn_pairs_share_a_document_and_skip_excluded_rows() -> None:
    text = corpus()
    source = rows(text)
    exclude = {row["query_id"] for row in source[:600]}

    pairs = sample_multi_turn(source, count=10, seed=1, corpus=text, exclude=exclude)

    assert len(pairs) == 10
    documents = set()
    for first, second in pairs:
        a = text.section_documents[first["expected_section_ids"][0]]
        b = text.section_documents[second["expected_section_ids"][0]]
        assert a == b
        assert first["expected_section_ids"] != second["expected_section_ids"]
        assert first["query_id"] not in exclude
        documents.add(a)
    assert len(documents) == 10


def test_multi_turn_needs_enough_documents() -> None:
    text = corpus(documents=3)
    with pytest.raises(ValueError, match="need 10"):
        sample_multi_turn(rows(text), count=10, seed=1, corpus=text, exclude=set())


def test_batches_carry_rows_and_section_texts(tmp_path: Path) -> None:
    text = corpus()
    source = rows(text)
    answerable = source[:40]
    pairs = [(source[0], source[3])]

    answerable[0]["expected_chunk_id"] = "doc0:s0:chunk-001"
    paths = write_authoring_batches(
        tmp_path,
        answerable,
        pairs,
        text,
        batch_size=35,
        chunks={"doc0:s0:chunk-001": "focus text"},
    )

    assert [path.name for path in paths] == [
        "answerable-01.todo.jsonl",
        "answerable-02.todo.jsonl",
        "multi-turn-01.todo.jsonl",
    ]
    slots = [json.loads(line) for line in paths[1].read_text("utf-8").splitlines()]
    assert [slot["slot_id"] for slot in slots] == [
        f"e2e-ans-{n:04d}" for n in range(36, 41)
    ]
    first = json.loads(paths[0].read_text("utf-8").splitlines()[0])
    assert first["focus_chunks"] == {"doc0:s0:chunk-001": "focus text"}
    dialogue = json.loads(paths[2].read_text("utf-8"))
    assert dialogue["slot_id"] == "e2e-mt-0001"
    assert set(dialogue["sections"]) == {
        *source[0]["expected_section_ids"],
        *source[3]["expected_section_ids"],
    }
    with pytest.raises(ValueError, match="--force"):
        write_authoring_batches(tmp_path, answerable, pairs, text)
    assert write_authoring_batches(tmp_path, answerable[:1], [], text, force=True) == [
        tmp_path / "answerable-01.todo.jsonl"
    ]
