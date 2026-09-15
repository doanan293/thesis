"""Relevance judgments that accept same-ingredient leaflet chunks for formulary queries.

A formulary question (for example the dosage of ACARBOSE) is also answered by the
matching section of a drug leaflet whose only active ingredient is that drug. The gold
file names the formulary section; these judgments add the leaflet chunks that answer the
same intent, so a ranking that puts such a chunk first is not scored as a miss. They live
in a file next to the gold file, so changing them only requires `seed metrics --force`.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pharma_agent.domain.corpus.bundle import read_bundle

from seed_pipeline.artifacts.jsonl import iter_jsonl_objects
from seed_pipeline.bundle.evaluation_chunks import evaluation_chunk_rows
from seed_pipeline.evaluation.artifact_contracts import (
    ArtifactContractError,
    sha256_file,
    write_json,
)
from seed_pipeline.evaluation.retrieval_metrics import expected_section_ids

RULES_VERSION = 1
JUDGMENTS_SCHEMA_VERSION = 1
JUDGED_EVAL_GROUPS = frozenset({"formulary", "multi_intent"})
INGREDIENT = "ingredient"

FORMULARY_SECTION_INTENTS = {
    "chi-dinh": "indication",
    "lieu-luong-va-cach-dung": "dosage",
    "chong-chi-dinh": "contraindication",
    "tac-dung-khong-mong-muon-adr": "adr",
    "qua-lieu-va-xu-tri": "overdose",
    "than-trong": "precaution",
    "duoc-ly-va-co-che-tac-dung": "pharmacology",
    "tuong-tac-thuoc": "interaction",
    "thoi-ky-mang-thai": "pregnancy_lactation",
    "thoi-ky-cho-con-bu": "pregnancy_lactation",
}

# Normalised leaflet heading text (parenthesised notes removed) -> intent.
# An empty intent marks a heading whose text answers none of the judged intents.
_HEADINGS: dict[str, tuple[str, ...]] = {
    INGREDIENT: ("thanh phan", "thanh phan hoat chat"),
    "indication": ("cong dung", "chi dinh", "cong dung chi dinh"),
    "dosage": (
        "cach dung lieu dung",
        "cach dung",
        "lieu dung",
        "lieu luong",
        "lieu khuyen cao",
        "lieu luong va cach dung",
        "cach dung va lieu dung",
        "quen lieu",
        "tang lieu",
    ),
    "contraindication": ("chong chi dinh",),
    "adr": (
        "tac dung phu",
        "tac dung khong mong muon",
        "tac dung khong mong muon adr",
    ),
    "overdose": ("qua lieu", "qua lieu va xu tri", "xu tri qua lieu"),
    "precaution": (
        "luu y",
        "than trong",
        "than trong khi su dung",
        "luu y khi su dung",
        "canh bao va than trong",
        "canh bao",
    ),
    "pharmacology": (
        "duoc ly",
        "duoc luc hoc",
        "duoc ly va co che tac dung",
        "co che tac dung",
    ),
    "interaction": ("tuong tac thuoc", "tuong tac", "tuong tac voi cac thuoc khac"),
    "pregnancy_lactation": (
        "thai ky va cho con bu",
        "phu nu co thai va cho con bu",
        "phu nu mang thai va cho con bu",
        "thoi ky mang thai",
        "thoi ky cho con bu",
    ),
    "": (
        "thong tin them",
        "thong tin khac",
        "thong tin chung",
        "dac diem",
        "quy cach dong goi",
        "han su dung",
        "doi tuong su dung",
        "doi tuong dac biet",
        "duoc dong hoc",
        "bao quan",
        "nha san xuat",
        "dang bao che",
        "nghien cuu tien lam sang",
    ),
}
HEADING_INTENTS = {
    text: intent for intent, texts in _HEADINGS.items() for text in texts
}
PSEUDO_HEADING_MAX_CHARS = 60

_MARKDOWN_HEADING = re.compile(r"^#{1,6}\s*(.*)$")
_PARENTHESES = re.compile(r"\([^()]*\)")
_HEADING_PREFIX = re.compile(r"^[\s\-*•+]*(?:\d+\s*[.)]\s*)?[\s\-*•+]*")
_DOSE = re.compile(
    r"\d[\d.,]*\s*(?:mg|g|mcg|µg|μg|iu|ui|ml|%|đơn vị|dv|mmol|mmol/l)(?![a-zà-ỹ])",
    re.IGNORECASE,
)
# Ingredients on one line are separated by ";", "," (not a decimal comma) or a sentence.
_SEGMENT_SPLIT = re.compile(r"[;,](?!\d)|\.\s+(?=[A-ZÀ-Ỹ])")
_CONNECTOR_TOKENS = ("va", "and", "voi")
# Phrases between two doses that restate an amount instead of naming an ingredient.
_FILLER_PREFIXES = (
    "bu ham luong",
    "tuong duong",
    "duoi dang",
    "ham luong",
    "tuong ung",
)
_LABEL_PREFIX = re.compile(
    r"^[\s\-*•+]*(?:thành phần hoạt chất|thành phần|hoạt chất|dược chất)\s*:?\s*",
    re.IGNORECASE,
)

# Tokens that may follow a drug name without changing the active moiety.
_SALT_TOKENS = {
    "hydroclorid",
    "hydrochlorid",
    "hcl",
    "hydrobromid",
    "bromid",
    "clorid",
    "natri",
    "dinatri",
    "kali",
    "calci",
    "magnesi",
    "magnesium",
    "sodium",
    "potassium",
    "calcium",
    "besilat",
    "besylat",
    "maleat",
    "mesylat",
    "mesilat",
    "tosylat",
    "sulfat",
    "sulphat",
    "fumarat",
    "acetat",
    "tartrat",
    "bitartrat",
    "citrat",
    "phosphat",
    "succinat",
    "nitrat",
    "mononitrat",
    "propionat",
    "dipropionat",
    "valerat",
    "lactat",
    "gluconat",
    "carbonat",
    "stearat",
    "palmitat",
    "axetil",
    "pivoxil",
    "proxetil",
    "trometamol",
    "xinafoat",
    "pamoat",
    "embonat",
    "oxalat",
    "hydrat",
    "monohydrat",
    "dihydrat",
    "trihydrat",
    "hemihydrat",
    "sesquihydrat",
    "pentahydrat",
    "heptahydrat",
    "khan",
    "base",
}
_TOKEN_ALIASES = {"acetaminophen": "paracetamol"}


def normalize_text(value: str) -> str:
    """Lowercase ASCII words of `value`: diacritics removed, đ -> d, punctuation -> space."""
    lowered = unicodedata.normalize("NFD", value.lower()).replace("đ", "d")
    stripped = "".join(char for char in lowered if unicodedata.category(char) != "Mn")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", stripped).split())


def formulary_intent(section_id: str) -> str | None:
    """Intent of a formulary monograph section that a drug leaflet can also answer."""
    parts = section_id.split(":")
    if len(parts) != 3 or parts[0] != "drug":
        return None
    return FORMULARY_SECTION_INTENTS.get(parts[2])


def heading_intent(line: str) -> str | None:
    """Intent a heading line opens, "" for other headings, None for a content line.

    Leaflets mark sections with markdown headings, but some pages lost a heading and keep
    only a short line such as "- Thận trọng khi sử dụng"; such lines count as headings when
    their whole text is a known heading.
    """
    stripped = line.strip()
    markdown = _MARKDOWN_HEADING.match(stripped)
    text = markdown.group(1) if markdown else stripped
    key = normalize_text(_HEADING_PREFIX.sub("", _PARENTHESES.sub(" ", text)))
    if key in HEADING_INTENTS:
        return HEADING_INTENTS[key]
    if markdown:
        return ""
    if len(stripped) <= PSEUDO_HEADING_MAX_CHARS and key.rstrip() in HEADING_INTENTS:
        return HEADING_INTENTS[key]
    return None


def chunk_intents(chunk_texts: Sequence[str]) -> list[frozenset[str]]:
    """Intents of the content lines of each chunk of one section, in chunk order."""
    state = ""
    result: list[frozenset[str]] = []
    for text in chunk_texts:
        intents: set[str] = set()
        for line in text.splitlines():
            if not line.strip():
                continue
            heading = heading_intent(line)
            if heading is not None:
                state = heading
            elif state:
                intents.add(state)
        result.append(frozenset(intents))
    return result


def _ingredient_names(line: str) -> list[str] | None:
    """Names of the dosed ingredients on one composition line; None if one is unnamed."""
    text = _PARENTHESES.sub(" ", line)
    if "|" in text:
        cells = [cell.strip() for cell in text.strip().strip("|").split("|")]
        cells = [cell for cell in cells if cell]
        for index, cell in enumerate(cells):
            match = _DOSE.search(cell)
            if match is None:
                continue
            name = cell[: match.start()].strip() or (cells[index - 1] if index else "")
            key = _name_key(name)
            return None if key is None else ([key] if key else [])
        return []
    names: list[str] = []
    for segment in _SEGMENT_SPLIT.split(text):
        start = 0
        # Every dose is preceded by the name it measures: "X 40 mg và Y 5 mg".
        for match in _DOSE.finditer(segment):
            key = _name_key(segment[start : match.start()])
            start = match.end()
            if key is None:
                return None
            if key:
                names.append(key)
    return names


def _name_key(name: str) -> str | None:
    """Normalised ingredient name, "" for a phrase that names none, None when blank."""
    tokens = normalize_text(_LABEL_PREFIX.sub("", name)).split()
    while tokens and tokens[0] in _CONNECTOR_TOKENS:
        tokens = tokens[1:]
    key = " ".join(tokens)
    if not re.search(r"[a-z]", key):
        return None
    if (
        key.startswith(("moi ", "trong moi ", "trong "))
        or key.endswith(" chua")
        or key.startswith(_FILLER_PREFIXES)
    ):
        return ""
    return key


def active_ingredients(text: str) -> tuple[str, ...] | None:
    """Active ingredients listed in a leaflet's composition block, in listed order.

    Returns None when the page has no composition block, lists no dosed ingredient, or
    has a dosed line whose ingredient name cannot be read.
    """
    names: list[str] = []
    in_block = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        heading = heading_intent(stripped)
        if heading == INGREDIENT:
            in_block = True
            continue
        if heading is not None:
            if in_block:
                break
            continue
        if not in_block or "ta duoc" in normalize_text(stripped):
            continue
        line_names = _ingredient_names(stripped)
        if line_names is None:
            return None
        names.extend(line_names)
    unique = tuple(dict.fromkeys(names))
    return unique or None


def _canonical_tokens(text: str) -> tuple[str, ...]:
    tokens = []
    for token in normalize_text(text).split():
        token = _TOKEN_ALIASES.get(token, token)
        token = (
            token.replace("ph", "f")
            .replace("th", "t")
            .replace("y", "i")
            .replace("ll", "l")
            .replace("mm", "m")
        )
        if len(token) > 4 and token.endswith("e"):
            token = token[:-1]
        tokens.append(token)
    return tuple(tokens)


_CANONICAL_SALTS = frozenset(_canonical_tokens(" ".join(sorted(_SALT_TOKENS))))


def _components(title: str) -> tuple[tuple[str, ...], ...]:
    parts = re.split(r"\s+va\s+|\s*\+\s*", normalize_text(title).replace(" + ", " va "))
    return tuple(_canonical_tokens(part) for part in parts if part.strip())


def _is_same_moiety(ingredient: tuple[str, ...], component: tuple[str, ...]) -> bool:
    if ingredient[: len(component)] != component:
        return False
    return all(token in _CANONICAL_SALTS for token in ingredient[len(component) :])


def same_active_ingredients(ingredients: Iterable[str], title: str) -> bool:
    """Whether a leaflet's active ingredients are exactly the monograph's components."""
    remaining = [_canonical_tokens(name) for name in dict.fromkeys(ingredients)]
    components = _components(title)
    if not components or len(remaining) != len(components):
        return False
    for component in components:
        match = next(
            (item for item in remaining if _is_same_moiety(item, component)), None
        )
        if match is None:
            return False
        remaining.remove(match)
    return not remaining


def _accepted_leaflet_chunks(
    chunk_rows: Iterable[Mapping[str, Any]],
) -> dict[tuple[str, str], set[str]]:
    """Leaflet chunk ids per (formulary drug slug, intent)."""
    sections: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    drug_titles: dict[str, str] = {}
    for row in chunk_rows:
        section_id = str(row["section_id"])
        sections[section_id].append(row)
        if section_id.startswith("drug:"):
            drug_titles.setdefault(section_id.split(":")[1], str(row["title"]))
    drugs_by_first_token: defaultdict[str, list[str]] = defaultdict(list)
    for slug, title in drug_titles.items():
        components = _components(title)
        if components and components[0]:
            drugs_by_first_token[components[0][0]].append(slug)
    accepted: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    for section_id, rows in sections.items():
        if not section_id.startswith("leaflet:"):
            continue
        ordered = sorted(rows, key=lambda row: int(row["chunk_index"]))
        texts = [str(row["chunk_text"]) for row in ordered]
        ingredients = active_ingredients("\n".join(texts))
        if ingredients is None:
            continue
        first_tokens = {
            tokens[0] for tokens in map(_canonical_tokens, ingredients) if tokens
        }
        slugs = [
            slug
            for token in sorted(first_tokens)
            for slug in drugs_by_first_token.get(token, ())
            if same_active_ingredients(ingredients, drug_titles[slug])
        ]
        for row, intents in zip(ordered, chunk_intents(texts), strict=True):
            for slug in slugs:
                for intent in intents:
                    accepted[(slug, intent)].add(str(row["chunk_id"]))
    return accepted


def build_judgments(
    evaluation_rows: Iterable[Mapping[str, Any]],
    chunk_rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Accepted leaflet chunks per judged query, one entry per formulary section intent."""
    accepted = _accepted_leaflet_chunks(chunk_rows)
    judgments: list[dict[str, Any]] = []
    for row in evaluation_rows:
        if row.get("eval_group") not in JUDGED_EVAL_GROUPS:
            continue
        granularity = row.get("retrieval_granularity")
        answer_mode = row.get("answer_mode")
        if granularity == "section" and answer_mode == "single":
            sections = [str(row["expected_section_id"])]
        elif granularity == "multi_section" and answer_mode == "multi_required":
            sections = expected_section_ids(dict(row))
        else:
            continue
        intents = []
        for section_id in sections:
            intent = formulary_intent(section_id)
            if intent is None:
                continue
            chunk_ids = sorted(accepted.get((section_id.split(":")[1], intent), ()))
            if chunk_ids:
                intents.append(
                    {
                        "section_id": section_id,
                        "intent": intent,
                        "accepted_chunk_ids": chunk_ids,
                    }
                )
        if intents:
            judgments.append({"query_id": str(row["query_id"]), "intents": intents})
    return sorted(judgments, key=lambda item: item["query_id"])


def judgments_path(evaluation_path: Path) -> Path:
    """The judgments file that belongs to a gold evaluation file."""
    evaluation_path = Path(evaluation_path)
    return evaluation_path.with_name(f"{evaluation_path.stem}.judgments.jsonl")


def _manifest_path(path: Path) -> Path:
    return Path(path).with_name(f"{Path(path).stem}.manifest.json")


def write_judgments(
    judgments: Iterable[Mapping[str, Any]],
    *,
    evaluation_path: Path,
    output_path: Path,
) -> Path:
    """Write the judgments and a manifest binding them to the gold file's digest."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for judgment in judgments:
            handle.write(
                json.dumps(judgment, ensure_ascii=False, sort_keys=True) + "\n"
            )
            count += 1
    os.replace(temporary, output_path)
    write_json(
        _manifest_path(output_path),
        {
            "schema_version": JUDGMENTS_SCHEMA_VERSION,
            "rules_version": RULES_VERSION,
            "evaluation_sha256": sha256_file(evaluation_path),
            "judgments_sha256": sha256_file(output_path),
            "query_count": count,
        },
    )
    return output_path


@dataclass(frozen=True)
class LoadedJudgments:
    sha256: str
    queries: Mapping[str, Mapping[str, frozenset[str]]]

    def for_query(self, query_id: str) -> dict[str, frozenset[str]]:
        """Accepted chunk ids per gold section id of one query ({} when none)."""
        return dict(self.queries.get(query_id, {}))


def load_judgments(path: Path, *, evaluation_sha256: str) -> LoadedJudgments:
    """Read judgments built for the gold file with `evaluation_sha256`."""
    path = Path(path)
    manifest_path = _manifest_path(path)
    if not path.is_file() or not manifest_path.is_file():
        raise ArtifactContractError(
            f"Relevance judgments {path} are missing; run `seed evaluation judgments`"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("evaluation_sha256") != evaluation_sha256:
        raise ArtifactContractError(
            f"Relevance judgments {path} were built for a different evaluation file; "
            "run `seed evaluation judgments`"
        )
    if (
        manifest.get("schema_version") != JUDGMENTS_SCHEMA_VERSION
        or manifest.get("rules_version") != RULES_VERSION
    ):
        raise ArtifactContractError(
            f"Relevance judgments {path} use rules version "
            f"{manifest.get('rules_version')}, expected {RULES_VERSION}; "
            "run `seed evaluation judgments`"
        )
    digest = sha256_file(path)
    if manifest.get("judgments_sha256") != digest:
        raise ArtifactContractError(
            f"Relevance judgments {path} do not match the checksum in {manifest_path}"
        )
    queries: dict[str, dict[str, frozenset[str]]] = {}
    for record in iter_jsonl_objects(path):
        queries[str(record["query_id"])] = {
            str(intent["section_id"]): frozenset(map(str, intent["accepted_chunk_ids"]))
            for intent in record["intents"]
        }
    return LoadedJudgments(digest, queries)


def build_evaluation_judgments(evaluation_path: Path, bundle_dir: Path) -> Path:
    """Build the judgments of a gold file from the knowledge bundle it was built from."""
    evaluation_path = Path(evaluation_path)
    rows = list(iter_jsonl_objects(evaluation_path))
    chunks = evaluation_chunk_rows(read_bundle(bundle_dir))
    return write_judgments(
        build_judgments(rows, chunks),
        evaluation_path=evaluation_path,
        output_path=judgments_path(evaluation_path),
    )
