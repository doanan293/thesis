from pharma_agent.domain.corpus.bundle import ColloquialMappingRecord, GlossaryEntry
from pharma_agent.domain.corpus.enrichment import (
    build_context_header,
    compose_embedding_text,
    detect_terms,
    mapping_for_section,
)
from pharma_agent.domain.retrieval.models import ColloquialMapping, TermAnnotation

ADR = GlossaryEntry(
    term="ADR",
    case_sensitive=True,
    vietnamese_expansions=["tác dụng không mong muốn", "phản ứng có hại của thuốc"],
    english_expansions=["Adverse Drug Reactions"],
    aliases=["adverse drug reaction"],
    category="safety",
    confidence="high",
    source="curated",
)
NSAID = GlossaryEntry(
    term="NSAID",
    case_sensitive=True,
    vietnamese_expansions=[
        "thuốc chống viêm không steroid",
        "thuốc kháng viêm không steroid",
    ],
    english_expansions=["Nonsteroidal Anti-inflammatory Drug"],
    aliases=["NSAIDs"],
    category="drug_class",
    confidence="high",
    source="curated",
)
G6PD = GlossaryEntry(
    term="G6PD",
    case_sensitive=True,
    vietnamese_expansions=["glucose-6-phosphate dehydrogenase"],
    english_expansions=["glucose-6-phosphate dehydrogenase"],
    aliases=["thiếu men G6PD", "thiếu enzym G6PD"],
    category="laboratory",
    confidence="high",
    source="curated",
)
GLOSSARY = [ADR, NSAID, G6PD]
ADR_ANNOTATION = TermAnnotation(
    term="ADR",
    vi=["tác dụng không mong muốn", "phản ứng có hại của thuốc"],
    en=["Adverse Drug Reactions"],
)
NSAID_ANNOTATION = TermAnnotation(
    term="NSAID",
    vi=["thuốc chống viêm không steroid", "thuốc kháng viêm không steroid"],
    en=["Nonsteroidal Anti-inflammatory Drug"],
)
G6PD_ANNOTATION = TermAnnotation(
    term="G6PD",
    vi=["glucose-6-phosphate dehydrogenase"],
    en=["glucose-6-phosphate dehydrogenase"],
)
LEAFLET_SECTION = "leaflet:thuoc-giam-dau-ha-sot:panadol-extra-gsk-150-vien-11440"
LEAFLET_HEADER = (
    "Panadol Extra GSK giảm đau, hạ sốt (15 vỉ x 12 viên)\n> Thông tin chi tiết"
)
PANADOL = ColloquialMappingRecord(
    key="panadol-extra-gsk-150-vien-11440",
    aliases=["Panadol đỏ", " Panadol  extra đỏ ", "panadol đỏ", "Panadol hộp đỏ"],
    visual_sign=" Hộp màu đỏ, vỉ thuốc màu đỏ ",
    product_names=["Panadol Extra GSK giảm đau, hạ sốt", "Panadol Extra GSK"],
    section_keys=[LEAFLET_SECTION],
)


def test_build_context_header_joins_title_and_non_empty_path_items() -> None:
    assert (
        build_context_header("PARACETAMOL", ["Liều lượng và cách dùng"])
        == "PARACETAMOL\n> Liều lượng và cách dùng"
    )
    assert build_context_header(
        "PHỤ LỤC 2. PHA THÊM THUỐC TIÊM VÀO DỊCH TRUYỀN TĨNH MẠCH",
        ["7. Các thuốc đưa vào bằng đường truyền tĩnh mạch", ""],
    ) == (
        "PHỤ LỤC 2. PHA THÊM THUỐC TIÊM VÀO DỊCH TRUYỀN TĨNH MẠCH\n"
        "> 7. Các thuốc đưa vào bằng đường truyền tĩnh mạch"
    )
    assert build_context_header("", []) == ""


def test_build_context_header_drops_colloquial_label_lines_and_blank_lines() -> None:
    title = "Panadol Extra GSK\nTên gọi khác: Panadol đỏ\n\ndấu hiệu nhận biết: Hộp đỏ"

    assert (
        build_context_header(title, ["Thông tin chi tiết"])
        == "Panadol Extra GSK\n> Thông tin chi tiết"
    )


def test_detect_terms_orders_by_first_match_and_matches_aliases() -> None:
    text = (
        "PARACETAMOL\n> Chống chỉ định\n\n"
        "Người thiếu men G6PD, người đang dùng NSAIDs; theo dõi ADR."
    )

    assert detect_terms(text, GLOSSARY) == [
        G6PD_ANNOTATION,
        NSAID_ANNOTATION,
        ADR_ANNOTATION,
    ]


def test_detect_terms_respects_case_and_vietnamese_word_boundaries() -> None:
    cmax = GlossaryEntry(
        term="CMAX", case_sensitive=False, english_expansions=["peak concentration"]
    )
    text = "adr, ADRs, NSAIDđ và nsaids không khớp; Cmax đạt sau 1 giờ."

    assert detect_terms(text, [*GLOSSARY, cmax]) == [
        TermAnnotation(term="CMAX", vi=[], en=["peak concentration"])
    ]


def test_detect_terms_limit_counts_matches_without_expansions() -> None:
    glossary = [
        GlossaryEntry(
            term=f"XT{index}",
            case_sensitive=True,
            english_expansions=[] if index < 2 else [f"expansion {index}"],
            aliases=[f"xt-{index}"],
        )
        for index in range(10)
    ]
    text = " ".join(f"XT{index}" for index in range(10))

    assert [annotation.term for annotation in detect_terms(text, glossary)] == [
        "XT2",
        "XT3",
        "XT4",
        "XT5",
        "XT6",
        "XT7",
    ]


def test_detect_terms_normalizes_expansions() -> None:
    inr = GlossaryEntry(
        term=" INR ",
        case_sensitive=True,
        vietnamese_expansions=[
            "tỷ số  chuẩn hóa\nquốc tế",
            "Tỷ số chuẩn hóa quốc tế",
            " ",
        ],
        english_expansions=["International Normalized Ratio"],
    )

    assert detect_terms("Theo dõi INR hằng tuần.", [inr]) == [
        TermAnnotation(
            term="INR",
            vi=["tỷ số chuẩn hóa quốc tế"],
            en=["International Normalized Ratio"],
        )
    ]


def test_mapping_for_section_compacts_the_record() -> None:
    assert mapping_for_section(LEAFLET_SECTION, [PANADOL]) == ColloquialMapping(
        key="panadol-extra-gsk-150-vien-11440",
        aliases=["Panadol đỏ", "Panadol extra đỏ", "Panadol hộp đỏ"],
        visual_sign="Hộp màu đỏ, vỉ thuốc màu đỏ",
        product_names=["Panadol Extra GSK giảm đau, hạ sốt", "Panadol Extra GSK"],
    )
    assert (
        mapping_for_section("drug:paracetamol:lieu-luong-va-cach-dung", [PANADOL])
        is None
    )


def test_mapping_for_section_keeps_product_names_without_a_curated_key() -> None:
    hapacol = "leaflet:thuoc-giam-dau-ha-sot:hapacol-250-dhg"
    efferalgan = "leaflet:thuoc-giam-dau-ha-sot:efferalgan-500mg"
    records = [
        ColloquialMappingRecord(
            key="",
            product_names=["Hapacol 250 DHG", "hapacol 250 dhg"],
            section_keys=[hapacol],
        ),
        ColloquialMappingRecord(key="", section_keys=[efferalgan]),
    ]

    assert mapping_for_section(hapacol, records) == ColloquialMapping(
        product_names=["Hapacol 250 DHG"]
    )
    assert mapping_for_section(efferalgan, records) is None


def test_compose_embedding_text_orders_header_colloquial_chunk_and_terms() -> None:
    chunk = "Panadol đỏ chứa paracetamol 500 mg. Thận trọng khi dùng cùng NSAID."
    mapping = mapping_for_section(LEAFLET_SECTION, [PANADOL])
    terms = detect_terms(f"{LEAFLET_HEADER}\n\n{chunk}", GLOSSARY)

    assert compose_embedding_text(
        context_header=LEAFLET_HEADER, chunk_text=chunk, colloquial=mapping, terms=terms
    ) == (
        f"{LEAFLET_HEADER}\n\n"
        "Tên gọi khác: Panadol extra đỏ, Panadol hộp đỏ\n"
        "Dấu hiệu nhận biết: Hộp màu đỏ, vỉ thuốc màu đỏ\n\n"
        f"{chunk}\n\n"
        "Thuật ngữ: NSAID = thuốc chống viêm không steroid; "
        "thuốc kháng viêm không steroid; Nonsteroidal Anti-inflammatory Drug"
    )


def test_compose_embedding_text_without_header_colloquial_or_terms() -> None:
    assert (
        compose_embedding_text(
            context_header="", chunk_text="Hạ sốt.", colloquial=None, terms=[]
        )
        == "Hạ sốt."
    )


def test_compose_embedding_text_skips_visible_names_and_existing_term_block() -> None:
    mapping = ColloquialMapping(aliases=["Panadol đỏ"], visual_sign="hộp màu đỏ")
    chunk = (
        "Panadol đỏ có HỘP MÀU ĐỎ.\n\n"
        "Thuật ngữ: G6PD = glucose-6-phosphate dehydrogenase"
    )

    assert (
        compose_embedding_text(
            context_header="PHỤ LỤC 1. THUẬT NGỮ",
            chunk_text=chunk,
            colloquial=mapping,
            terms=[G6PD_ANNOTATION],
        )
        == f"PHỤ LỤC 1. THUẬT NGỮ\n\n{chunk}"
    )


def test_compose_embedding_text_dedupes_expansions_and_repeated_terms() -> None:
    terms = [
        G6PD_ANNOTATION,
        ADR_ANNOTATION,
        TermAnnotation(term="ADR", vi=["phản ứng phụ"]),
    ]

    text = compose_embedding_text(
        context_header="PARACETAMOL\n> Chống chỉ định",
        chunk_text="Thiếu men G6PD; ADR trên gan.",
        colloquial=None,
        terms=terms,
    )

    assert text.endswith(
        "\n\nThuật ngữ: G6PD = glucose-6-phosphate dehydrogenase | "
        "ADR = tác dụng không mong muốn; phản ứng có hại của thuốc; "
        "Adverse Drug Reactions"
    )
