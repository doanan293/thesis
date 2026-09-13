"""The small knowledge bundle shared by the backend tests and seed-pipeline's contract test.

`build_small_bundle()` is the source of truth. After changing it, or after P1's chunker or
enrichment changes, regenerate the committed copy with `uv run python -m tests.corpus_fixtures`;
`tests/test_corpus_fixtures.py` fails while the committed copy is stale.
"""

import shutil
from pathlib import Path

from pharma_agent.domain.corpus.bundle import (
    BlockKind,
    BlockRecord,
    BundleCollection,
    BundleEmbeddingFile,
    BundleGenerator,
    BundleManifest,
    ColloquialMappingRecord,
    DocumentKind,
    DocumentRecord,
    GlossaryEntry,
    KnowledgeBundle,
    RetrievalMode,
    SectionRecord,
    SourceInfo,
    model_slug,
    read_bundle,
    write_bundle,
)
from pharma_agent.domain.corpus.chunking import chunk_section
from pharma_agent.domain.corpus.identity import sha256_hex
from tests.fakes import FAKE_EMBEDDING_DIMENSION, FAKE_EMBEDDING_MODEL, fake_vector

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "knowledge_bundle_small"
COLLECTION_KEY = "formulary"
COLLECTION_TITLE = "Dược thư Quốc gia Việt Nam (fixture)"

PARACETAMOL = "drug:paracetamol"
LEAFLET = "leaflet:ankhang:thuoc-giam-dau:panadol-extra"
DOSAGE_SECTION = "drug:paracetamol:lieu-luong-va-cach-dung"
PHARMACOLOGY_SECTION = "drug:paracetamol:duoc-ly-va-co-che-tac-dung"
INTERACTIONS_SECTION = "drug:paracetamol:tuong-tac-thuoc"
BRANDS_SECTION = "drug:paracetamol:biet-duoc"
LEAFLET_SECTION = "leaflet:ankhang:thuoc-giam-dau:panadol-extra:cong-dung"

FORMULARY_SOURCE = SourceInfo(title="Dược thư Quốc gia Việt Nam 2022", url=None)

INTERACTIONS_TABLE = """| Thuốc phối hợp | Tương tác | Xử trí |
| --- | --- | --- |
| Warfarin | Tăng tác dụng chống đông khi dùng paracetamol liều cao kéo dài | Theo dõi INR |
| Rượu | Tăng nguy cơ độc tính trên gan | Tránh uống rượu |
| NSAID | Tăng nguy cơ tác dụng phụ trên thận khi phối hợp lâu dài | Hạn chế phối hợp kéo dài |"""

BRAND_INDEX = """Efferalgan (UPSA) - viên sủi 500 mg
Hapacol (DHG Pharma) - gói bột 250 mg
Panadol (GSK) - viên nén 500 mg
Tylenol (Johnson & Johnson) - viên nén 325 mg"""


def _pharmacology_paragraphs(first: int, count: int) -> str:
    return "\n\n".join(
        " ".join(
            f"Đoạn {paragraph}, ý {sentence}: paracetamol ức chế tổng hợp prostaglandin "
            "ở hệ thần kinh trung ương nên hạ sốt và giảm đau, còn tác dụng chống viêm "
            "ngoại vi rất yếu."
            for sentence in range(1, 7)
        )
        for paragraph in range(first, first + count)
    )


def _documents() -> list[DocumentRecord]:
    return [
        DocumentRecord(
            key=PARACETAMOL,
            kind=DocumentKind.DRUG_MONOGRAPH,
            title="Paracetamol",
            source=FORMULARY_SOURCE,
            attributes={"atc": "N02BE01"},
        ),
        DocumentRecord(
            key=LEAFLET,
            kind=DocumentKind.LEAFLET,
            title="Panadol Extra",
            source=SourceInfo(
                title="Nhà thuốc An Khang",
                url="https://www.nhathuocankhang.com/thuoc-giam-dau/panadol-extra",
            ),
            attributes={"category": "thuoc-giam-dau"},
        ),
    ]


def _sections() -> list[SectionRecord]:
    return [
        SectionRecord(
            key=DOSAGE_SECTION,
            document_key=PARACETAMOL,
            heading="Liều lượng và cách dùng",
            context_path=["Liều lượng và cách dùng"],
            ordinal=1,
            start_page=120,
            end_page=120,
            blocks=[
                BlockRecord(
                    kind=BlockKind.PROSE,
                    markdown="Người lớn: uống 500 mg đến 1 g mỗi 4 đến 6 giờ khi cần, "
                    "tối đa 4 g mỗi ngày.",
                    start_page=120,
                    end_page=120,
                ),
                BlockRecord(
                    kind=BlockKind.PROSE,
                    markdown="Trẻ em: 10 đến 15 mg/kg mỗi 4 đến 6 giờ, không quá 5 lần "
                    "trong 24 giờ. Người suy gan cần giảm liều.",
                    start_page=120,
                    end_page=120,
                ),
            ],
        ),
        SectionRecord(
            key=PHARMACOLOGY_SECTION,
            document_key=PARACETAMOL,
            heading="Dược lý và cơ chế tác dụng",
            context_path=["Dược lý và cơ chế tác dụng"],
            ordinal=2,
            start_page=121,
            end_page=124,
            blocks=[
                BlockRecord(
                    kind=BlockKind.PROSE,
                    markdown=_pharmacology_paragraphs(1, 5),
                    start_page=121,
                    end_page=121,
                ),
                BlockRecord(
                    kind=BlockKind.PROSE,
                    markdown=_pharmacology_paragraphs(6, 5),
                    start_page=122,
                    end_page=122,
                ),
                BlockRecord(
                    kind=BlockKind.PROSE,
                    markdown=_pharmacology_paragraphs(11, 5),
                    start_page=123,
                    end_page=123,
                ),
                BlockRecord(
                    kind=BlockKind.PROSE,
                    markdown=_pharmacology_paragraphs(16, 5)
                    + "\n\nNgười thiếu G6PD dùng liều điều trị thường dung nạp tốt; "
                    "quá liều gây hoại tử tế bào gan do tích lũy "
                    "N-acetyl-p-benzoquinon imin.",
                    start_page=124,
                    end_page=124,
                ),
            ],
        ),
        SectionRecord(
            key=INTERACTIONS_SECTION,
            document_key=PARACETAMOL,
            heading="Tương tác thuốc",
            context_path=["Tương tác thuốc"],
            ordinal=3,
            start_page=125,
            end_page=125,
            blocks=[
                BlockRecord(
                    kind=BlockKind.TABLE,
                    markdown=INTERACTIONS_TABLE,
                    start_page=125,
                    end_page=125,
                    table_key="paracetamol-tuong-tac-thuoc",
                )
            ],
        ),
        SectionRecord(
            key=BRANDS_SECTION,
            document_key=PARACETAMOL,
            heading="Biệt dược",
            context_path=["Biệt dược"],
            ordinal=4,
            start_page=126,
            end_page=126,
            retrieval=RetrievalMode.INDEX_ONLY,
            blocks=[
                BlockRecord(
                    kind=BlockKind.INDEX_ENTRIES,
                    markdown=BRAND_INDEX,
                    start_page=126,
                    end_page=126,
                )
            ],
        ),
        SectionRecord(
            key=LEAFLET_SECTION,
            document_key=LEAFLET,
            heading="Công dụng",
            context_path=["Công dụng"],
            ordinal=1,
            blocks=[
                BlockRecord(
                    kind=BlockKind.PROSE,
                    markdown="Panadol Extra chứa paracetamol 500 mg và cafein 65 mg, "
                    "dùng giảm đau đầu, đau răng, đau bụng kinh và hạ sốt.",
                )
            ],
        ),
    ]


def _glossary() -> list[GlossaryEntry]:
    return [
        GlossaryEntry(
            term="NSAID",
            case_sensitive=True,
            vietnamese_expansions=["thuốc chống viêm không steroid"],
            english_expansions=["non-steroidal anti-inflammatory drug"],
            category="drug_class",
            confidence="high",
            source="fixture",
        ),
        GlossaryEntry(
            term="G6PD",
            case_sensitive=True,
            vietnamese_expansions=["glucose-6-phosphat dehydrogenase"],
            english_expansions=["glucose-6-phosphate dehydrogenase"],
            category="enzyme",
            confidence="high",
            source="fixture",
        ),
    ]


def _colloquial_mappings() -> list[ColloquialMappingRecord]:
    return [
        ColloquialMappingRecord(
            key="panadol-extra",
            aliases=["panadol đỏ", "thuốc giảm đau viên đỏ"],
            visual_sign="viên nén dài bao phim màu đỏ",
            product_names=["Panadol Extra"],
            section_keys=[LEAFLET_SECTION],
        )
    ]


def build_small_bundle() -> KnowledgeBundle:
    documents = _documents()
    sections = _sections()
    glossary = _glossary()
    mappings = _colloquial_mappings()
    documents_by_key = {document.key: document for document in documents}
    vectors: dict[str, list[float]] = {}
    for section in sections:
        for draft in chunk_section(
            documents_by_key[section.document_key], section, glossary, mappings
        ):
            vectors[draft.embedding_text_sha256] = fake_vector(draft.embedding_text)
    manifest = BundleManifest(
        schema_version="knowledge-bundle/v1",
        collection=BundleCollection(key=COLLECTION_KEY, title=COLLECTION_TITLE),
        generator=BundleGenerator(
            name="pharma-agent-tests", version="1", build_id="knowledge-bundle-small"
        ),
        source_digests={
            "source_pdf_sha256": sha256_hex("fixture source pdf"),
            "snapshot_sha256": sha256_hex("fixture ankhang snapshot"),
        },
        document_count=len(documents),
        section_count=len(sections),
        files={},
        embeddings=[
            BundleEmbeddingFile(
                model=FAKE_EMBEDDING_MODEL,
                dims=FAKE_EMBEDDING_DIMENSION,
                file=f"embeddings/{model_slug(FAKE_EMBEDDING_MODEL)}.jsonl",
            )
        ],
    )
    return KnowledgeBundle(
        manifest=manifest,
        documents=documents,
        sections=sections,
        glossary=glossary,
        colloquial_mappings=mappings,
        embeddings={FAKE_EMBEDDING_MODEL: vectors},
    )


def small_bundle() -> KnowledgeBundle:
    """The committed fixture, read and validated by P1's `read_bundle`."""
    return read_bundle(FIXTURE_DIR)


def with_section_text(
    bundle: KnowledgeBundle, section_key: str, extra: str
) -> KnowledgeBundle:
    """Copy of `bundle` whose section's first block ends with `extra` (a content change)."""
    if all(section.key != section_key for section in bundle.sections):
        raise KeyError(section_key)
    sections: list[SectionRecord] = []
    for section in bundle.sections:
        if section.key == section_key:
            first, *rest = section.blocks
            edited = first.model_copy(
                update={"markdown": f"{first.markdown}\n\n{extra}"}
            )
            section = section.model_copy(update={"blocks": [edited, *rest]})
        sections.append(section)
    return bundle.model_copy(update={"sections": sections})


def regenerate(directory: Path = FIXTURE_DIR) -> None:
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    write_bundle(build_small_bundle(), directory)


if __name__ == "__main__":
    regenerate()
