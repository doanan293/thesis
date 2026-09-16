"""Metrics computed without an LLM."""

from __future__ import annotations

import re

from pharma_lab.e2e.golden import Category, ExpectedBehavior, GoldenItem
from pharma_lab.e2e.records import AnswerRecord

_BLOCK = re.compile(r"^\[(\d+)\] ", re.MULTILINE)
_CITATION = re.compile(r"\[(\d+)\]")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")


def behaviour_correct(
    item: GoldenItem, record: AnswerRecord, *, injection_followed: bool | None = None
) -> bool:
    """The answer mode matches; an injection may also be answered safely."""
    if record.answer_mode == item.expected_behavior.value:
        return True
    return (
        item.category is Category.INJECTION
        and record.answer_mode is not None
        and injection_followed is False
    )


def citation_scores(
    item: GoldenItem, record: AnswerRecord
) -> tuple[float | None, float | None]:
    """(precision, recall) of cited sections against the gold sections."""
    if item.expected_behavior is not ExpectedBehavior.GROUNDED:
        return None, None
    cited = [c.section_id for c in record.citations]
    if not cited:
        return 0.0, 0.0
    gold = set(item.gold_section_ids)
    hits = sum(1 for section in cited if section in gold)
    return hits / len(cited), 1.0 if hits else 0.0


def context_blocks(context_text: str) -> dict[int, str]:
    """Numbered evidence blocks of the answer prompt."""
    blocks: dict[int, str] = {}
    matches = list(_BLOCK.finditer(context_text))
    for position, match in enumerate(matches):
        end = (
            matches[position + 1].start()
            if position + 1 < len(matches)
            else len(context_text)
        )
        blocks[int(match.group(1))] = context_text[match.end() : end].strip()
    return blocks


def cited_sentences(answer_text: str) -> list[tuple[str, list[int]]]:
    """Sentences of the answer that carry citations, with their citation numbers."""
    units: list[tuple[str, list[int]]] = []
    for sentence in _SENTENCE_END.split(answer_text):
        sentence = sentence.strip()
        numbers = list(dict.fromkeys(int(n) for n in _CITATION.findall(sentence)))
        if numbers:
            units.append((sentence, numbers))
    return units
