"""The E2E golden set: item schema, validation and freezing (spec §3)."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from pharma_lab.e2e.corpus_text import CorpusText, normalize_space
from pharma_lab.evaluation.artifact_contracts import sha256_file, write_json

SCHEMA = "e2e-golden-v1"
ANSWERABLE_PER_GROUP = 70
EVAL_GROUPS = (
    "formulary",
    "leaflet",
    "chunk_risk",
    "patient_natural",
    "noisy_confuser",
    "multi_intent",
)


class Category(StrEnum):
    ANSWERABLE = "answerable"
    MULTI_TURN = "multi_turn"
    UNANSWERABLE = "unanswerable"
    OUT_OF_SCOPE = "out_of_scope"
    INJECTION = "injection"


class ExpectedBehavior(StrEnum):
    GROUNDED = "grounded"
    ABSTAIN = "abstain"
    REDIRECT = "redirect"
    BLOCKED = "blocked"


QUOTAS: dict[Category, int] = {
    Category.ANSWERABLE: ANSWERABLE_PER_GROUP * len(EVAL_GROUPS),
    Category.MULTI_TURN: 50,
    Category.UNANSWERABLE: 10,
    Category.OUT_OF_SCOPE: 10,
    Category.INJECTION: 10,
}
ID_PREFIX: dict[Category, str] = {
    Category.ANSWERABLE: "e2e-ans-",
    Category.MULTI_TURN: "e2e-mt-",
    Category.UNANSWERABLE: "e2e-una-",
    Category.OUT_OF_SCOPE: "e2e-oos-",
    Category.INJECTION: "e2e-inj-",
}
BEHAVIOR: dict[Category, ExpectedBehavior] = {
    Category.ANSWERABLE: ExpectedBehavior.GROUNDED,
    Category.MULTI_TURN: ExpectedBehavior.GROUNDED,
    Category.UNANSWERABLE: ExpectedBehavior.ABSTAIN,
    Category.OUT_OF_SCOPE: ExpectedBehavior.REDIRECT,
    Category.INJECTION: ExpectedBehavior.BLOCKED,
}

_STRICT = ConfigDict(extra="forbid", frozen=True)


class Turn(BaseModel):
    model_config = _STRICT

    role: Literal["user", "assistant"]
    text: str = Field(min_length=1)


class KeyFact(BaseModel):
    model_config = _STRICT

    fact: str = Field(min_length=1)
    evidence_quote: str = Field(min_length=1)
    section_id: str


class Reference(BaseModel):
    model_config = _STRICT

    answer: str = ""
    key_facts: list[KeyFact] = Field(default_factory=list)


class GoldenItem(BaseModel):
    model_config = _STRICT

    item_id: str
    category: Category
    source_query_id: str | None = None
    turns: list[Turn]
    expected_behavior: ExpectedBehavior
    gold_section_ids: list[str] = Field(default_factory=list)
    gold_chunk_ids: list[str] = Field(default_factory=list)
    reference: Reference = Field(default_factory=Reference)
    absent_terms: list[str] = Field(default_factory=list)
    eval_group: str | None = None
    difficulty: str | None = None
    tags: list[str] = Field(default_factory=list)

    @property
    def question(self) -> str:
        return self.turns[-1].text

    @property
    def history(self) -> list[tuple[str, str]]:
        """(user, assistant) pairs before the evaluated question."""
        earlier = self.turns[:-1]
        return [
            (earlier[index].text, earlier[index + 1].text)
            for index in range(0, len(earlier) - 1, 2)
        ]


class GoldenManifest(BaseModel):
    model_config = _STRICT

    schema_name: str = Field(default=SCHEMA, alias="schema")
    sha256: str
    counts: dict[str, int]
    answerable_per_group: dict[str, int]
    gold_sha256: str
    bundle_manifest_sha256: str
    created_at: str


def manifest_path(golden_path: Path) -> Path:
    return golden_path.with_name(golden_path.stem + ".manifest.json")


def archive_path(golden_path: Path, sha256: str) -> Path:
    """Where `golden build` keeps the version it replaced, by content hash."""
    return golden_path.with_name(f"{golden_path.stem}.{sha256}.jsonl")


def read_items(path: Path) -> tuple[list[GoldenItem], list[str]]:
    items: list[GoldenItem] = []
    problems: list[str] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            items.append(GoldenItem.model_validate_json(line))
        except ValidationError as exc:
            problems.append(f"{path}:{line_number}: {exc.errors()[0]['msg']}")
    return items, problems


def load_golden(path: Path) -> list[GoldenItem]:
    items, problems = read_items(path)
    if problems:
        raise ValueError("invalid golden set:\n" + "\n".join(problems))
    if not items:
        raise ValueError(f"golden set is empty: {path}")
    return items


def _turn_problems(item: GoldenItem) -> list[str]:
    roles = [turn.role for turn in item.turns]
    if item.category is Category.MULTI_TURN:
        expected = ["user", "assistant"] * (len(roles) // 2) + ["user"]
        if len(roles) < 3 or roles != expected:
            return [
                f"{item.item_id}: multi_turn needs alternating user/assistant turns "
                "ending with user (at least 3)"
            ]
        return []
    if roles != ["user"]:
        return [f"{item.item_id}: {item.category} needs exactly one user turn"]
    return []


def _item_problems(item: GoldenItem, corpus: CorpusText) -> list[str]:
    problems: list[str] = []
    name = item.item_id
    if not name.startswith(ID_PREFIX[item.category]):
        problems.append(f"{name}: id must start with {ID_PREFIX[item.category]}")
    if item.expected_behavior is not BEHAVIOR[item.category]:
        problems.append(
            f"{name}: {item.category} expects {BEHAVIOR[item.category]}, "
            f"not {item.expected_behavior}"
        )
    problems.extend(_turn_problems(item))
    facts = item.reference.key_facts
    if item.expected_behavior is ExpectedBehavior.GROUNDED:
        if not facts or not item.gold_section_ids:
            problems.append(f"{name}: grounded items need key facts and gold sections")
    elif facts or item.gold_section_ids or item.gold_chunk_ids:
        problems.append(f"{name}: non-grounded items have no key facts or gold ids")
    for section_id in item.gold_section_ids:
        if section_id not in corpus.sections:
            problems.append(f"{name}: gold section {section_id} is not in the bundle")
    for index, fact in enumerate(facts, start=1):
        if fact.section_id not in item.gold_section_ids:
            problems.append(
                f"{name}: key fact {index} cites {fact.section_id}, "
                "which is not a gold section"
            )
            continue
        text = corpus.normalized_section(fact.section_id)
        if text is None or normalize_space(fact.evidence_quote) not in text:
            problems.append(
                f"{name}: key fact {index} quote is not in section {fact.section_id}"
            )
    if item.category is Category.UNANSWERABLE:
        if not item.absent_terms:
            problems.append(f"{name}: unanswerable items need absent_terms")
        for term in item.absent_terms:
            if corpus.contains(term):
                problems.append(f"{name}: absent term {term!r} occurs in the corpus")
    elif item.absent_terms:
        problems.append(f"{name}: only unanswerable items carry absent_terms")
    return problems


def _count_problems(items: Sequence[GoldenItem]) -> list[str]:
    problems: list[str] = []
    counts = Counter(item.category for item in items)
    for category, quota in QUOTAS.items():
        if counts[category] != quota:
            problems.append(f"{category}: {counts[category]} items, expected {quota}")
    groups = Counter(
        item.eval_group for item in items if item.category is Category.ANSWERABLE
    )
    for group in EVAL_GROUPS:
        if groups[group] != ANSWERABLE_PER_GROUP:
            problems.append(
                f"answerable {group}: {groups[group]} items, "
                f"expected {ANSWERABLE_PER_GROUP}"
            )
    return problems


def validate_items(
    items: Sequence[GoldenItem], corpus: CorpusText, *, complete: bool
) -> list[str]:
    problems: list[str] = []
    ids = Counter(item.item_id for item in items)
    problems.extend(f"{name}: duplicate item_id" for name, n in ids.items() if n > 1)
    for item in items:
        problems.extend(_item_problems(item, corpus))
    if complete:
        problems.extend(_count_problems(items))
    return problems


def build_golden(
    sources: Sequence[Path],
    output: Path,
    corpus: CorpusText,
    *,
    bundle_manifest_sha256: str,
    gold_sha256: str,
) -> GoldenManifest:
    """Validate authored batches and freeze them into one sorted golden file."""
    if not sources:
        raise ValueError("no authored batches to build from")
    items: list[GoldenItem] = []
    problems: list[str] = []
    for source in sorted(sources):
        batch, batch_problems = read_items(source)
        items.extend(batch)
        problems.extend(batch_problems)
    problems.extend(validate_items(items, corpus, complete=True))
    if problems:
        raise ValueError(
            f"golden set has {len(problems)} problem(s):\n" + "\n".join(problems)
        )
    items.sort(key=lambda item: item.item_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.tmp")
    staging.write_text(
        "".join(
            json.dumps(item.model_dump(mode="json"), ensure_ascii=False) + "\n"
            for item in items
        ),
        encoding="utf-8",
    )
    if output.is_file():
        # Runs made with the replaced version resume against it item by item.
        previous = archive_path(output, sha256_file(output))
        if not previous.is_file():
            shutil.copy2(output, previous)
    staging.replace(output)
    counts = Counter(str(item.category) for item in items)
    groups = Counter(
        str(item.eval_group) for item in items if item.category is Category.ANSWERABLE
    )
    manifest = GoldenManifest.model_validate(
        {
            "schema": SCHEMA,
            "sha256": sha256_file(output),
            "counts": dict(sorted(counts.items())),
            "answerable_per_group": dict(sorted(groups.items())),
            "gold_sha256": gold_sha256,
            "bundle_manifest_sha256": bundle_manifest_sha256,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
    )
    write_json(manifest_path(output), manifest.model_dump(mode="json", by_alias=True))
    return manifest
