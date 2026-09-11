#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from corpus_pipeline.config.chunking import DEFAULT_CHUNK_MAX_CHARS
from corpus_pipeline.config.paths import (
    CANONICAL_INTERIM_DIR,
    DOCLING_INTERIM_DIR,
    PROJECT_ROOT,
    RAG_FINAL_DIR,
    RAG_INTERIM_DIR,
)
from corpus_pipeline.corpus.processing.preprocess_rag_corpus import (
    SectionRecord,
    chunk_sections,
    repair_targeted_punctuation_artifacts,
    split_long_text,
)

DEFAULT_SECTIONS = RAG_INTERIM_DIR / "sections.jsonl"
DEFAULT_CHUNKS = RAG_INTERIM_DIR / "chunks.jsonl"
DEFAULT_TABLES = DOCLING_INTERIM_DIR / "corpus_pipeline.corpus.tables.curated.jsonl"
DEFAULT_TABLE_OVERRIDES = DOCLING_INTERIM_DIR / "table_duplicate_overrides.json"
DEFAULT_CANONICAL_DIR = CANONICAL_INTERIM_DIR
DEFAULT_FINAL_DIR = RAG_FINAL_DIR
FULL_SECTION_MAX_CHARS = 16000
BRAND_INDEX_SECTION_ID = (
    "general:muc-luc-tra-cuu-biet-duoc-va-hoat-chat:bang-tra-cuu-biet-duoc"
)
ATC_SECTION_ID = (
    "general:phu-luc-3-danh-muc-thuoc-phan-loai-theo-ma-atc:bang-phan-loai-atc"
)
IV_INFUSION_SECTION_ID = (
    "general:phu-luc-2-pha-them-thuoc-tiem-vao-dich-truyen-tinh-mach:"
    "7-cac-thuoc-dua-vao-bang-duong-truyen-tinh-mach"
)
APPENDIX_LIST_SECTION_IDS = {ATC_SECTION_ID, IV_INFUSION_SECTION_ID}
CHUNK_ROLES = {"prose", "table", "index_entry", "appendix_list"}
BRAND_INDEX_ENTRY_RE = re.compile(r"^- \*\*[^*]+\*\*: .+")
TARGETED_SHORT_PART_MERGE_PAIRS = {
    (
        "drug:acid-boric:thoi-ky-cho-con-bu",
        "drug:acid-boric:thoi-ky-cho-con-bu:part-002",
    ),
    (
        "drug:alverin-citrat:thoi-ky-mang-thai",
        "drug:alverin-citrat:thoi-ky-mang-thai:part-002",
    ),
    (
        "drug:cefadroxil:do-on-dinh-va-bao-quan",
        "drug:cefadroxil:do-on-dinh-va-bao-quan:part-002",
    ),
    ("drug:cinarizin:thoi-ky-mang-thai", "drug:cinarizin:thoi-ky-mang-thai:part-002"),
    (
        "drug:diclofenac:do-on-dinh-va-bao-quan",
        "drug:diclofenac:do-on-dinh-va-bao-quan:part-002",
    ),
    (
        "drug:dinatri-calci-edetat:thoi-ky-mang-thai",
        "drug:dinatri-calci-edetat:thoi-ky-mang-thai:part-002",
    ),
    (
        "drug:disulfiram:thoi-ky-cho-con-bu",
        "drug:disulfiram:thoi-ky-cho-con-bu:part-002",
    ),
    ("drug:gliclazid:thoi-ky-cho-con-bu", "drug:gliclazid:thoi-ky-cho-con-bu:part-002"),
    (
        "drug:methylprednisolon:do-on-dinh-va-bao-quan",
        "drug:methylprednisolon:do-on-dinh-va-bao-quan:part-002",
    ),
    (
        "drug:oxytetracyclin:do-on-dinh-va-bao-quan:part-002",
        "drug:oxytetracyclin:do-on-dinh-va-bao-quan:part-003",
    ),
    (
        "drug:pancuronium:do-on-dinh-va-bao-quan",
        "drug:pancuronium:do-on-dinh-va-bao-quan:part-002",
    ),
    ("drug:thiopental:chong-chi-dinh", "drug:thiopental:chong-chi-dinh:part-002"),
    (
        "drug:tramadol-hydroclorid:do-on-dinh-va-bao-quan",
        "drug:tramadol-hydroclorid:do-on-dinh-va-bao-quan:part-002",
    ),
    (
        "drug:voriconazol:do-on-dinh-va-bao-quan:part-002",
        "drug:voriconazol:do-on-dinh-va-bao-quan:part-003",
    ),
    ("drug:baclofen:than-trong", "drug:baclofen:than-trong:part-002"),
    (
        "drug:cefamandol:lieu-luong-va-cach-dung",
        "drug:cefamandol:lieu-luong-va-cach-dung:part-002",
    ),
    (
        "drug:cefapirin-natri:lieu-luong-va-cach-dung",
        "drug:cefapirin-natri:lieu-luong-va-cach-dung:part-002",
    ),
    (
        "drug:cefradin:lieu-luong-va-cach-dung",
        "drug:cefradin:lieu-luong-va-cach-dung:part-002",
    ),
    ("drug:cloramphenicol:chi-dinh", "drug:cloramphenicol:chi-dinh:part-002"),
    ("drug:sulpirid:chi-dinh", "drug:sulpirid:chi-dinh:part-002"),
    (
        "drug:thuoc-tuong-tu-hormon-giai-phong-gonadotropin:chi-dinh",
        "drug:thuoc-tuong-tu-hormon-giai-phong-gonadotropin:chi-dinh:part-002",
    ),
}

NRTI_CHI_DINH_PART_002 = (
    "general:su-dung-hop-ly-thuoc-khang-hiv-cho-nguoi-benh-hiv-aids:"
    "a-dieu-tri-cho-nguoi-lon-nhiem-hiv:i-dieu-tri-bang-thuoc-khang-hiv-dieu-tri-arv:"
    "nrti:chi-dinh:part-002"
)
NRTI_CHI_DINH_PART_003 = (
    "general:su-dung-hop-ly-thuoc-khang-hiv-cho-nguoi-benh-hiv-aids:"
    "a-dieu-tri-cho-nguoi-lon-nhiem-hiv:i-dieu-tri-bang-thuoc-khang-hiv-dieu-tri-arv:"
    "nrti:chi-dinh:part-003"
)
NRTI_CHONG_CHI_DINH = (
    "general:su-dung-hop-ly-thuoc-khang-hiv-cho-nguoi-benh-hiv-aids:"
    "a-dieu-tri-cho-nguoi-lon-nhiem-hiv:i-dieu-tri-bang-thuoc-khang-hiv-dieu-tri-arv:"
    "nrti:chong-chi-dinh"
)
NRTI_DOSAGE_PART_002 = (
    "general:su-dung-hop-ly-thuoc-khang-hiv-cho-nguoi-benh-hiv-aids:"
    "a-dieu-tri-cho-nguoi-lon-nhiem-hiv:i-dieu-tri-bang-thuoc-khang-hiv-dieu-tri-arv:"
    "nrti:lieu-luong-va-cach-dung:part-002"
)
NRTI_DOSAGE_PART_003 = (
    "general:su-dung-hop-ly-thuoc-khang-hiv-cho-nguoi-benh-hiv-aids:"
    "a-dieu-tri-cho-nguoi-lon-nhiem-hiv:i-dieu-tri-bang-thuoc-khang-hiv-dieu-tri-arv:"
    "nrti:lieu-luong-va-cach-dung:part-003"
)
HIV_PNMT_CHI_DINH_PART_002 = (
    "general:su-dung-hop-ly-thuoc-khang-hiv-cho-nguoi-benh-hiv-aids:"
    "a-dieu-tri-cho-nguoi-lon-nhiem-hiv:ii-dieu-tri-arv-o-phu-nu-mang-thai-va-du-phong:"
    "chi-dinh:part-002"
)
HIV_PNMT_THOI_KY_MANG_THAI = (
    "general:su-dung-hop-ly-thuoc-khang-hiv-cho-nguoi-benh-hiv-aids:"
    "a-dieu-tri-cho-nguoi-lon-nhiem-hiv:ii-dieu-tri-arv-o-phu-nu-mang-thai-va-du-phong:"
    "thoi-ky-mang-thai"
)

PDF_VERIFIED_TEXT_OVERRIDES = {
    "drug:acebutolol:than-trong:part-002": "Thận trọng khi sử dụng cùng các thuốc gây mê (xem thêm phần\nThận trọng).",
    "drug:cac-chat-uc-che-hmg-coa-reductase:thoi-ky-mang-thai": (
        "Vì các statin làm giảm tổng hợp cholesterol và có thể cả nhiều chất\n"
        "khác có hoạt tính sinh học dẫn xuất từ cholesterol, nên thuốc có thể\n"
        "gây hại cho thai nhi nếu dùng cho người mang thai. Vì vậy chống\n"
        "chỉ định dùng statin trong thời kỳ mang thai."
    ),
    "drug:calci-lactat:thoi-ky-mang-thai": (
        "Thuốc sử dụng được cho phụ nữ có thai theo nhu cầu hàng ngày\n"
        "của đối tượng này (xem thêm về nhu cầu hàng ngày trong mục\n"
        "Dược lý và cơ chế tác dụng)"
    ),
    "drug:pantoprazol:thoi-ky-mang-thai": (
        "Chưa có nghiên cứu đầy đủ khi dùng pantoprazol trên người trong\n"
        "thời kỳ mang thai. Chỉ dùng pantoprazol khi thật cần thiết trong\n"
        "thời kỳ mang thai."
    ),
    "drug:penicilamin:thoi-ky-mang-thai": (
        "Penicilamin có thể qua nhau thai và tác động đến mô colagen trong\n"
        "thai, gây một số tai biến da. Đã thấy có hiện tượng quái thai sọ mặt.\n"
        "Trong thời kỳ mang thai, nếu thật cần thiết, dùng penicilamin với\n"
        "liều thấp nhất.\n"
        "Có một vài quan điểm khác nhau về sự dùng thuốc trong thời kỳ\n"
        "mang thai. Có tác giả đề nghị có thể dùng thuốc trong thời kỳ mang\n"
        "thai trong bệnh Wilson, nhưng trong bệnh viêm khớp dạng thấp thì\n"
        "chống chỉ định. Có tác giả đề nghị trong bệnh Wilson cũng chống\n"
        "chỉ định ở 3 tháng đầu mang thai."
    ),
    "drug:tolbutamid:than-trong:part-004": (
        "Thận trọng khi dùng tolbutamid kết hợp với một số thuốc khác\n"
        "như rifampicin, corticosteroid, cimetidin, rượu, cafein (xem mục\n"
        "Tương tác thuốc).\n"
        "Vẫn phải duy trì chế độ ăn kiêng phù hợp trong thời gian dùng thuốc."
    ),
    NRTI_CHI_DINH_PART_003: "Sử dụng một trong hai phác đồ này khi người bệnh có\nchống chỉ định với TDF.",
    "drug:gatifloxacin:thoi-ky-mang-thai": (
        "Chưa có nghiên cứu đầy đủ và có kiểm soát ở phụ nữ mang thai. "
        "Chỉ nên dùng gatifloxacin trong thời kỳ mang thai khi lợi ích mong đợi vượt trội so với nguy cơ trên thai nhi."
    ),
    "drug:gatifloxacin:thoi-ky-cho-con-bu": (
        "Do chưa biết thuốc có phân bố vào sữa mẹ khi dùng trên người hay không, "
        "nên thận trọng khi dùng gatifloxacin cho phụ nữ cho con bú."
    ),
    "drug:gatifloxacin:huong-dan-cach-xu-tri-adr": (
        "Cần ngừng thuốc ngay khi xuất hiện ban da hoặc bất kỳ dấu hiệu nào của phản ứng quá mẫn, "
        "có phản ứng bất lợi trên TKTW, viêm đau hoặc đứt gân. Cần giám sát người bệnh để phát hiện "
        "viêm đại tràng màng giả và có các biện pháp xử trí thích hợp khi xuất hiện ỉa chảy trong khi đang dùng."
    ),
    "drug:gatifloxacin:tuong-ky": (
        "Không được trộn lẫn gatifloxacin với các thuốc khác hoặc truyền vào cùng một đường truyền."
    ),
    "drug:gatifloxacin:tuong-tac-thuoc": (
        "Tránh sử dụng đồng thời với các antacid có chứa nhôm hoặc magnesi, do đó "
        "làm giảm hấp thu gatifloxacin. Nên uống các thuốc này ít nhất 4 giờ trước hoặc 2 giờ sau khi dùng gatifloxacin.\n"
        "Các antacid có chứa calci và sữa không gây ra tương tác dược động.\n"
        "Sắt, các chế phẩm multivitamin và khoáng chất, sucralfat, didanosin "
        "(dạng viên có đệm để nhai được, hoặc hòa tan hoặc dạng bột trộn "
        "với antacid dành cho trẻ em): Có tương tác dược động học làm giảm hấp thu gatifloxacin. "
        "Nên uống các thuốc này ít nhất 4 giờ trước hoặc 2 giờ sau khi dùng gatifloxacin.\n"
        "Warfarin: Mặc dù hiện nay chưa có báo cáo về tương tác xảy ra giữa gatifloxacin và warfarin, "
        "tuy nhiên quinolon có khả năng làm tăng tác dụng của warfarin và dẫn xuất, cần giám sát "
        "chặt chẽ thời gian prothrombin và các xét nghiệm đông máu thích hợp khác khi sử dụng các thuốc này đồng thời với các kháng sinh nhóm quinolon.\n"
        "Probenecid: Tương tác dược động học làm kéo dài nửa đời thải trừ của gatifloxacin.\n"
        "Digoxin: Tương tác dược động học làm tăng nồng độ và độc tính của digoxin. Cần giám sát biểu hiện ngộ độc digoxin, "
        "khi xảy ra các biểu hiện này, phải định lượng nồng độ digoxin trong huyết tương và hiệu chỉnh liều digoxin cho thích hợp.\n"
        "NSAID: Có khả năng xảy ra tương tác dược lực học, làm tăng nguy cơ kích thích TKTW, co giật.\n"
        "Các thuốc có khả năng kéo dài khoảng QT: Có khả năng xảy ra tương tác dược lực học, cộng tác dụng kéo dài khoảng QT.\n"
        "Cimetidin, glyburid, midazolam, theophylin, các thuốc chuyển hóa qua microsom gan: Không có tương tác có ý nghĩa lâm sàng.\n"
        "Đường huyết: Dùng đồng thời với các thuốc làm thay đổi nồng độ glucose máu sẽ làm tăng nguy cơ rối loạn glucose huyết."
    ),
}

PDF_VERIFIED_REMOVE_SECTION_IDS = {
    NRTI_CHI_DINH_PART_002,
    NRTI_CHONG_CHI_DINH,
    "drug:cac-chat-uc-che-hmg-coa-reductase:thoi-ky-mang-thai:part-002",
    "drug:cac-chat-uc-che-hmg-coa-reductase:chi-dinh:part-002",
    "drug:calci-lactat:thoi-ky-mang-thai:part-002",
    "drug:cetirizin-hydroclorid:chong-chi-dinh:part-002",
    "drug:pantoprazol:thoi-ky-mang-thai:part-002",
    "drug:penicilamin:chong-chi-dinh:part-002",
    "drug:penicilamin:chi-dinh:part-002",
    "drug:tolbutamid:tuong-tac-thuoc",
}

PDF_VERIFIED_APPEND_LINES = {
    "drug:cac-chat-uc-che-hmg-coa-reductase:chong-chi-dinh": "Thời kỳ mang thai hoặc cho con bú.",
    "drug:acid-aminocaproic:thoi-ky-cho-con-bu": "thận trọng.",
    "drug:amphotericin-b:thoi-ky-cho-con-bu": "thận trọng.",
    "drug:calcipotriol:thoi-ky-cho-con-bu": "thời kỳ cho con bú.",
    "drug:clorambucil:thoi-ky-mang-thai": "chỉ định.",
    "drug:donepezil-hydroclorid:thoi-ky-mang-thai": "thời kỳ mang thai.",
    "drug:hydralazin:thoi-ky-cho-con-bu": "thời kỳ cho con bú.",
    "drug:methotrexat:thoi-ky-mang-thai": "thời kỳ mang thai.",
    "drug:nhua-podophylum:thoi-ky-mang-thai": "thời kỳ mang thai.",
    "drug:testosteron:thoi-ky-cho-con-bu": "thời kỳ cho con bú.",
}

PDF_VERIFIED_CONTINUATION_MERGE_PAIRS = {
    ("drug:acid-ioxaglic:chi-dinh", "drug:acid-ioxaglic:chi-dinh:part-002"),
    (
        "drug:acid-ethacrynic:thoi-ky-cho-con-bu",
        "drug:acid-ethacrynic:chi-dinh:part-002",
    ),
    (
        "drug:alteplase:do-on-dinh-va-bao-quan",
        "drug:alteplase:do-on-dinh-va-bao-quan:part-002",
    ),
    ("drug:bari-sulfat:than-trong", "drug:bari-sulfat:than-trong:part-002"),
    (
        "drug:cac-chat-uc-che-hmg-coa-reductase:thoi-ky-cho-con-bu",
        "drug:cac-chat-uc-che-hmg-coa-reductase:chi-dinh:part-003",
    ),
    ("drug:carvedilol:thoi-ky-mang-thai", "drug:carvedilol:thoi-ky-mang-thai:part-002"),
    ("drug:cefadroxil:thoi-ky-cho-con-bu", "drug:cefadroxil:than-trong:part-003"),
    ("drug:cefalexin:thoi-ky-mang-thai", "drug:cefalexin:than-trong:part-002"),
    ("drug:cefazolin:qua-lieu-va-xu-tri", "drug:cefazolin:thong-tin-chung:part-002"),
    ("drug:cefpirom:thoi-ky-cho-con-bu", "drug:cefpirom:than-trong:part-002"),
    ("drug:ceftazidim:than-trong", "drug:ceftazidim:than-trong:part-002"),
    (
        "drug:ciprofloxacin:thoi-ky-mang-thai",
        "drug:ciprofloxacin:thoi-ky-mang-thai:part-002",
    ),
    ("drug:clofibrat:thoi-ky-mang-thai", "drug:clofibrat:chi-dinh:part-002"),
    (
        "drug:crotamiton:duoc-ly-va-co-che-tac-dung",
        "drug:crotamiton:thong-tin-chung:part-002",
    ),
    (
        "drug:cyclopentolat-hydroclorid:thoi-ky-cho-con-bu",
        "drug:cyclopentolat-hydroclorid:than-trong:part-003",
    ),
    (
        "drug:dextropropoxyphen:thoi-ky-cho-con-bu",
        "drug:dextropropoxyphen:than-trong:part-002",
    ),
    (
        "drug:dihydroergotamin:thoi-ky-mang-thai",
        "drug:dihydroergotamin:chi-dinh:part-002",
    ),
    ("drug:dimenhydrinat:tuong-tac-thuoc", "drug:dimenhydrinat:than-trong:part-002"),
    ("drug:disulfiram:thoi-ky-mang-thai", "drug:disulfiram:thoi-ky-mang-thai:part-002"),
    (
        "drug:esmolol-hydroclorid:thoi-ky-cho-con-bu",
        "drug:esmolol-hydroclorid:than-trong:part-002",
    ),
    ("drug:galantamin:than-trong", "drug:galantamin:than-trong:part-002"),
    (
        "drug:globulin-mien-dich-khang-viem-gan-b:thoi-ky-mang-thai",
        "drug:globulin-mien-dich-khang-viem-gan-b:chong-chi-dinh:part-002",
    ),
    ("drug:guanethidin:tuong-tac-thuoc", "drug:guanethidin:chi-dinh:part-002"),
    ("drug:hydroxyzin:thoi-ky-mang-thai", "drug:hydroxyzin:chi-dinh:part-002"),
    ("drug:imatinib:chi-dinh", "drug:imatinib:chi-dinh:part-002"),
    ("drug:iobitridol:chi-dinh", "drug:iobitridol:chi-dinh:part-002"),
    ("drug:iobitridol:chi-dinh", "drug:iobitridol:chi-dinh:part-003"),
    ("drug:iobitridol:chi-dinh", "drug:iobitridol:chi-dinh:part-004"),
    ("drug:iobitridol:chi-dinh", "drug:iobitridol:chi-dinh:part-005"),
    ("drug:kali-clorid:than-trong", "drug:kali-clorid:chi-dinh:part-002"),
    ("drug:levodopa:tuong-tac-thuoc", "drug:levodopa:chong-chi-dinh:part-002"),
    (
        "drug:levomepromazin:tuong-tac-thuoc",
        "drug:levomepromazin:chong-chi-dinh:part-002",
    ),
    ("drug:mebendazol:thoi-ky-cho-con-bu", "drug:mebendazol:than-trong:part-002"),
    ("drug:metoprolol:chong-chi-dinh", "drug:metoprolol:chi-dinh:part-002"),
    ("drug:milrinon:thoi-ky-cho-con-bu", "drug:milrinon:than-trong:part-003"),
    (
        "drug:natri-bicarbonat:chi-dinh",
        "drug:natri-bicarbonat:tac-dung-khong-mong-muon-adr",
    ),
    ("drug:nifedipin:tuong-tac-thuoc", "drug:nifedipin:tuong-tac-thuoc:part-002"),
    (
        "drug:orciprenalin-sulfat:thoi-ky-cho-con-bu",
        "drug:orciprenalin-sulfat:than-trong:part-002",
    ),
    (
        "drug:oxacilin-natri:do-on-dinh-va-bao-quan",
        "drug:oxacilin-natri:do-on-dinh-va-bao-quan:part-002",
    ),
    ("drug:pioglitazon:thoi-ky-cho-con-bu", "drug:pioglitazon:chi-dinh:part-002"),
    (
        "drug:prednisolon:thoi-ky-mang-thai",
        "drug:prednisolon:thoi-ky-mang-thai:part-002",
    ),
    (
        "drug:promethazin-hydroclorid:thoi-ky-mang-thai",
        "drug:promethazin-hydroclorid:thoi-ky-mang-thai:part-002",
    ),
    (
        "drug:rifampicin:do-on-dinh-va-bao-quan",
        "drug:rifampicin:do-on-dinh-va-bao-quan:part-002",
    ),
    ("drug:rocuronium-bromid:than-trong", "drug:rocuronium-bromid:than-trong:part-002"),
    ("drug:rosiglitazon:chi-dinh", "drug:rosiglitazon:chi-dinh:part-002"),
    ("drug:spiramycin:thoi-ky-cho-con-bu", "drug:spiramycin:than-trong:part-003"),
    (
        "drug:streptokinase:thoi-ky-cho-con-bu",
        "drug:streptokinase:thoi-ky-cho-con-bu:part-002",
    ),
    (
        "drug:suxamethonium-clorid:do-on-dinh-va-bao-quan",
        "drug:suxamethonium-clorid:do-on-dinh-va-bao-quan:part-002",
    ),
    (
        "drug:temozolomid:do-on-dinh-va-bao-quan",
        "drug:temozolomid:do-on-dinh-va-bao-quan:part-002",
    ),
    (
        "drug:terbutalin-sulfat:thoi-ky-mang-thai",
        "drug:terbutalin-sulfat:thoi-ky-mang-thai:part-002",
    ),
    ("drug:thioridazin:thoi-ky-mang-thai", "drug:thioridazin:than-trong:part-006"),
    ("drug:tim-gentian:thoi-ky-cho-con-bu", "drug:tim-gentian:than-trong:part-002"),
    (
        "drug:vac-xin-bcg:do-on-dinh-va-bao-quan",
        "drug:vac-xin-bcg:do-on-dinh-va-bao-quan:part-002",
    ),
    ("drug:warfarin:chi-dinh", "drug:warfarin:chi-dinh:part-002"),
}

PDF_VERIFIED_CONTINUATION_PREFIXES = {
    "drug:galantamin:than-trong:part-002": "thận trọng galantamin trên các đối tượng sau:",
}

PDF_VERIFIED_CONTINUATION_CLEANUPS = {
    "drug:cefazolin:thong-tin-chung:part-002": ("Loại thuốc: loại thuốc", "loại thuốc"),
    "drug:crotamiton:thong-tin-chung:part-002": (
        "Loại thuốc: loại thuốc",
        "loại thuốc",
    ),
}


@dataclass
class CanonicalBlock:
    block_id: str
    section_index: int
    section_id: str
    title: str
    section: str
    content_type: str
    source_type: str
    page_start: int | None
    page_end: int | None
    text: str | None
    markdown: str | None
    source_refs: list[str]
    quality_flags: list[str]
    table_id: str | None = None


@dataclass
class CanonicalSection:
    id: str
    content_type: str
    title: str
    section: str
    text: str
    source: str
    context_path: list[str]
    context_header: str
    start_page: int | None
    end_page: int | None
    warnings: list[str]
    block_ids: list[str]
    table_ids: list[str]
    source_mix: list[str]
    quality_flags: list[str]
    hydrate_strategy: str
    section_char_count: int


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def clean_unicode_text(text: str) -> str:
    if not isinstance(text, str):
        return text
    import unicodedata

    # Normalize to NFC to merge combining marks
    text = unicodedata.normalize("NFC", text)
    replacements = {
        # Quotes
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        # Ellipsis
        "\u2026": "...",
        # Dashes / Hyphens
        "\u2013": "-",
        "\u2014": "-",
        "\u2011": "-",
        "\u2212": "-",
        # Spaces / Invisible characters
        "\u00a0": " ",
        "\u200b": "",
        "\u00ad": "",
        "\u200e": "",
        "\u200f": "",
        "\u2003": " ",
        # Bullets / Symbols
        "\u2022": "*",
        "\u2122": "(TM)",
        # Degree Celsius
        "\u2103": "°C",
        # Cyrillic lookalikes (homoglyphs) to Latin
        "\u0421": "C",  # Capital ES -> C
        "\u0430": "a",  # Small A -> a
        "\u0445": "x",  # Small HA -> x
        "\u0410": "A",  # Capital A -> A
        "\u0423": "Y",  # Capital U -> Y
        "\u0420": "P",  # Capital ER -> P
        "\u0422": "T",  # Capital TE -> T
        "\u0415": "E",  # Capital IE -> E
        # Greek lookalikes to Latin
        "\u039c": "M",  # Capital MU -> M
        "\u0392": "B",  # Capital BETA -> B
        # Dotless i
        "\u0131": "i",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def clean_unicode_value(val: Any) -> Any:
    if isinstance(val, str):
        return clean_unicode_text(val)
    elif isinstance(val, dict):
        return {k: clean_unicode_value(v) for k, v in val.items()}
    elif isinstance(val, list):
        return [clean_unicode_value(v) for v in val]
    return val


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_payload = clean_unicode_value(payload)
    path.write_text(
        json.dumps(cleaned_payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def write_jsonl_records(path: Path, records: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_records = [clean_unicode_value(asdict(record)) for record in records]
    path.write_text(
        "".join(
            json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
            for r in cleaned_records
        ),
        encoding="utf-8",
    )


def write_jsonl_dicts(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_records = [clean_unicode_value(record) for record in records]
    path.write_text(
        "".join(
            json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
            for r in cleaned_records
        ),
        encoding="utf-8",
    )


def load_sections(path: Path) -> list[SectionRecord]:
    records: list[SectionRecord] = []
    for item in read_jsonl(path):
        records.append(
            SectionRecord(
                id=item["id"],
                content_type=item.get("content_type", "drug_monograph"),
                title=item.get("title", ""),
                section=item.get("section", ""),
                text=clean_unicode_text(item.get("text", "")),
                source=item.get(
                    "source",
                    "Dược thư Quốc gia Việt Nam (Xuất bản lần thứ 2) – Nhà xuất bản Y học, Hà Nội, 2018",
                ),
                context_path=list(item.get("context_path") or []),
                context_header=item.get("context_header", ""),
                start_page=item.get("start_page"),
                end_page=item.get("end_page"),
                warnings=list(item.get("warnings") or []),
            )
        )
    return records


def merge_targeted_short_part_sections(
    sections: list[SectionRecord],
) -> tuple[list[SectionRecord], dict[str, Any]]:
    by_id = {section.id: index for index, section in enumerate(sections)}
    remove_ids: set[str] = set()
    merge_events: list[dict[str, Any]] = []

    for target_id, removed_id in sorted(TARGETED_SHORT_PART_MERGE_PAIRS):
        target_index = by_id.get(target_id)
        removed_index = by_id.get(removed_id)
        if target_index is None or removed_index is None:
            continue
        target = sections[target_index]
        removed = sections[removed_index]
        if target.title != removed.title or target.section != removed.section:
            continue

        target.text = "\n".join(
            part.strip() for part in (target.text, removed.text) if part.strip()
        )
        target.end_page = (
            max(
                page for page in (target.end_page, removed.end_page) if page is not None
            )
            if (target.end_page or removed.end_page)
            else None
        )
        target.start_page = (
            min(
                page
                for page in (target.start_page, removed.start_page)
                if page is not None
            )
            if (target.start_page or removed.start_page)
            else None
        )
        target.warnings = [*target.warnings, *removed.warnings]
        remove_ids.add(removed_id)
        merge_events.append(
            {
                "target_section_id": target_id,
                "removed_section_id": removed_id,
                "title": target.title,
                "section": target.section,
                "merged_text_length": len(target.text),
            }
        )

    merged_sections = [section for section in sections if section.id not in remove_ids]
    return merged_sections, {
        "targeted_short_part_merge_count": len(merge_events),
        "targeted_short_part_merge_events": merge_events,
    }


def prepend_once(text: str, prefix: str) -> str:
    text = text.strip()
    prefix = prefix.strip()
    if not prefix or text.startswith(prefix):
        return text
    return "\n".join(part for part in (prefix, text) if part)


def append_once(text: str, suffix: str) -> str:
    text = text.strip()
    suffix = suffix.strip()
    if not suffix or suffix in text:
        return text
    return "\n".join(part for part in (text, suffix) if part)


def apply_pdf_verified_section_fixes(
    sections: list[SectionRecord],
) -> tuple[list[SectionRecord], dict[str, Any]]:
    by_id = {section.id: section for section in sections}
    remove_ids: set[str] = set()
    fix_events: list[dict[str, str]] = []

    for section_id, text in PDF_VERIFIED_TEXT_OVERRIDES.items():
        section = by_id.get(section_id)
        if section is None:
            continue
        if section.text != text:
            section.text = text
            fix_events.append({"section_id": section_id, "action": "override_text"})

    for section_id, suffix in PDF_VERIFIED_APPEND_LINES.items():
        section = by_id.get(section_id)
        if section is None:
            continue
        before = section.text
        section.text = append_once(section.text, suffix)
        if section.text != before:
            fix_events.append(
                {"section_id": section_id, "action": "append_pdf_verified_line"}
            )

    for target_id, removed_id in sorted(PDF_VERIFIED_CONTINUATION_MERGE_PAIRS):
        target = by_id.get(target_id)
        removed = by_id.get(removed_id)
        if target is None or removed is None:
            continue
        continuation = removed.text.strip()
        cleanup = PDF_VERIFIED_CONTINUATION_CLEANUPS.get(removed_id)
        if cleanup is not None:
            old, new = cleanup
            continuation = continuation.replace(old, new, 1)
        prefix = PDF_VERIFIED_CONTINUATION_PREFIXES.get(removed_id)
        if prefix:
            continuation = prepend_once(continuation, prefix)
        target.text = "\n".join(
            part.strip() for part in (target.text, continuation) if part.strip()
        )
        target.end_page = (
            max(
                page for page in (target.end_page, removed.end_page) if page is not None
            )
            if (target.end_page or removed.end_page)
            else None
        )
        target.start_page = (
            min(
                page
                for page in (target.start_page, removed.start_page)
                if page is not None
            )
            if (target.start_page or removed.start_page)
            else None
        )
        target.warnings = [*target.warnings, *removed.warnings]
        remove_ids.add(removed_id)
        fix_events.append(
            {"section_id": target_id, "action": "merge_pdf_verified_continuation"}
        )

    source = by_id.get(HIV_PNMT_THOI_KY_MANG_THAI)
    target = by_id.get(HIV_PNMT_CHI_DINH_PART_002)
    if source is not None and target is not None:
        target.text = "\n".join(
            part.strip() for part in (target.text, source.text) if part.strip()
        )
        remove_ids.add(HIV_PNMT_THOI_KY_MANG_THAI)
        fix_events.append(
            {
                "section_id": HIV_PNMT_CHI_DINH_PART_002,
                "action": "merge_misclassified_continuation",
            }
        )

    noisy_table = by_id.get(NRTI_CHI_DINH_PART_002)
    dosage = by_id.get(NRTI_DOSAGE_PART_002)
    if (
        noisy_table is not None
        and dosage is not None
        and "3.1.2. Phác đồ TDF + 3TC + NVP" in noisy_table.text
    ):
        dosage.text = prepend_once(dosage.text, "3.1.2. Phác đồ TDF + 3TC + NVP")
        fix_events.append(
            {
                "section_id": NRTI_DOSAGE_PART_002,
                "action": "transfer_heading_from_table_noise",
            }
        )

    wrong_contraindication = by_id.get(NRTI_CHONG_CHI_DINH)
    dosage = by_id.get(NRTI_DOSAGE_PART_003)
    if (
        wrong_contraindication is not None
        and dosage is not None
        and "3.2.1. Phác đồ AZT + 3TC + EFV" in wrong_contraindication.text
    ):
        dosage.text = prepend_once(dosage.text, "3.2.1. Phác đồ AZT + 3TC + EFV")
        fix_events.append(
            {
                "section_id": NRTI_DOSAGE_PART_003,
                "action": "transfer_heading_from_misclassified_section",
            }
        )

    # Move Capecitabine displaced NCIC tables from 'Tên thương mại' to 'Liều lượng và cách dùng'
    capecitabin_commercial = by_id.get("drug:capecitabin:ten-thuong-mai")
    capecitabin_dosage = by_id.get("drug:capecitabin:lieu-luong-va-cach-dung")
    if capecitabin_commercial is not None and capecitabin_dosage is not None:
        target_text = "Mức độ theo NCIC"
        idx = capecitabin_commercial.text.find(target_text)
        if idx != -1:
            table_text = capecitabin_commercial.text[idx:]
            capecitabin_commercial.text = capecitabin_commercial.text[:idx].strip()
            capecitabin_dosage.text = (
                capecitabin_dosage.text + "\n\n" + table_text.strip()
            )
            fix_events.append(
                {
                    "section_id": "drug:capecitabin:lieu-luong-va-cach-dung",
                    "action": "move_displaced_tables",
                }
            )

    # Apply manual text replacements for cut-off cleanups
    text_replacements = {
        "drug:ciprofibrat:duoc-ly-va-co-che-tac-dung": [
            ("kháng vitamin K (xem mục", "kháng vitamin K (xem mục Tương tác thuốc).")
        ],
        "drug:nhua-podophylum:lieu-luong-va-cach-dung": [
            (
                "khuyến cáo sử dụng, xem thêm mục",
                "khuyến cáo sử dụng, xem thêm mục Chống chỉ định và Thận trọng.",
            )
        ],
        "drug:tolbutamid:than-trong": [
            ("rượu, cafein (xem mục", "rượu, cafein (xem mục Tương tác thuốc).")
        ],
        "drug:acebutolol:tuong-tac-thuoc": [
            ("\nThận trọng khi sử dụng cùng các thuốc gây mê (xem thêm phần", "")
        ],
        "general:ke-don-thuoc:noi-dung": [("đối tương đặc biệt", "đối tượng đặc biệt")],
        "general:su-dung-hop-ly-thuoc-khang-dong-kinh:noi-dung": [
            ("bắt dầu", "bắt đầu")
        ],
        "drug:pantoprazol:lieu-luong-va-cach-dung": [
            ("loét dạ dầy lành tính", "loét dạ dày lành tính")
        ],
        "drug:ceftriaxon:duoc-ly-va-co-che-tac-dung": [
            ("aeruginosanhạy", "aeruginosa nhạy")
        ],
        "drug:thuoc-phien-opiat-opioid:duoc-ly-va-co-che-tac-dung": [
            ("б (sigma)", "σ (sigma)")
        ],
        "drug:quinin:qua-lieu-va-xu-tri": [("Pḥòng", "Phòng")],
        "drug:ritonavir:tac-dung-khong-mong-muon-adr": [("ADṚ", "ADR")],
        "drug:amiodaron:tuong-tac-thuoc": [("IA,,", "IA,")],
        "drug:cefalotin:duoc-ly-va-co-che-tac-dung": [("spp....", "spp...")],
        "drug:didanosin:duoc-ly-va-co-che-tac-dung": [("A62V....", "A62V...")],
        "drug:nifedipin:duoc-ly-va-co-che-tac-dung": [("kéo dài....", "kéo dài...")],
        "drug:retinol-vitamin-a:qua-lieu-va-xu-tri": [("ỉa chảy....", "ỉa chảy...")],
        "drug:sertralin:than-trong": [("cáu giận....", "cáu giận...")],
        "drug:spironolacton:qua-lieu-va-xu-tri": [
            ("Kayexalate....)", "Kayexalate...)")
        ],
        "drug:tolbutamid:chong-chi-dinh": [("lớn....,", "lớn...,")],
        "drug:pilocarpin:duoc-ly-va-co-che-tac-dung": [("hô hấp...Vì", "hô hấp... Vì")],
        (
            "general:su-dung-hop-ly-thuoc-khang-hiv-cho-nguoi-benh-hiv-aids:"
            "a-dieu-tri-cho-nguoi-lon-nhiem-hiv:i-dieu-tri-bang-thuoc-khang-hiv-dieu-tri-arv:nrti"
        ): [("PNMTvàtrẻ", "PNMT và trẻ"), ("ThayATV", "Thay ATV")],
        (
            "general:su-dung-hop-ly-thuoc-khang-hiv-cho-nguoi-benh-hiv-aids:"
            "a-dieu-tri-cho-nguoi-lon-nhiem-hiv:iii-du-phong-sau-phoi-nhiem-hiv"
        ): [("điều trịARVvà", "điều trị ARV và")],
        "drug:nevirapin:huong-dan-cach-xu-tri-adr": [("phẳn ứng", "phản ứng")],
        "general:huong-dan-su-dung-duoc-thu-quoc-gia-viet-nam:noi-dung": [
            (
                (
                    "trăm và lớn hơn 1 phần nghìn số người dùng thuốc, loại hiếm gặp\n"
                    "là ADR xảy ra dưới 1 phần nghìn số người dùng thuốc."
                ),
                (
                    "12. Tác dụng không mong muốn (Adverse drug reactions: ADR): Bao gồm\n"
                    "những phản ứng phụ và các phản ứng đối nghịch có hại. Trong phần này\n"
                    "các ADR được chia làm ba loại theo phân loại của Tổ chức y tế thế giới:\n"
                    "Loại thường gặp là ADR xảy ra trên 1 phần trăm số người dùng thuốc,\n"
                    "loại ít gặp là ADR xảy ra dưới 1 phần trăm và lớn hơn 1 phần nghìn\n"
                    "số người dùng thuốc, loại hiếm gặp là ADR xảy ra dưới 1 phần nghìn\n"
                    "số người dùng thuốc."
                ),
            ),
        ],
        "general:su-dung-hop-ly-cac-thuoc-dieu-tri-benh-hen-phe-quan:noi-dung": [
            (
                "Các thuốc điều trị dự phòng kiển soát hen",
                "Các thuốc điều trị dự phòng kiểm soát hen",
            ),
            ("v.v...Các", "v.v... Các"),
        ],
    }
    for section_id, reps in text_replacements.items():
        section = by_id.get(section_id)
        if section is not None:
            before = section.text
            for old, new in reps:
                section.text = section.text.replace(old, new)
            if section.text != before:
                fix_events.append(
                    {"section_id": section_id, "action": "replace_text_cleanup"}
                )

    remove_ids.update(
        section_id
        for section_id in PDF_VERIFIED_REMOVE_SECTION_IDS
        if section_id in by_id
    )
    fixed_sections = [section for section in sections if section.id not in remove_ids]
    for section_id in sorted(remove_ids):
        fix_events.append(
            {
                "section_id": section_id,
                "action": "remove_pdf_verified_noise_or_continuation",
            }
        )

    return fixed_sections, {
        "pdf_verified_section_fix_count": len(fix_events),
        "pdf_verified_section_fix_events": fix_events,
    }


def validate_tables(tables: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for table in tables:
        table_id = table["table_id"]
        if table_id in seen:
            raise ValueError(f"Duplicate curated table_id: {table_id}")
        seen.add(table_id)
        page_start = table.get("page_start")
        page_end = table.get("page_end")
        if (
            page_start is not None
            and page_end is not None
            and int(page_start) > int(page_end)
        ):
            raise ValueError(
                f"Invalid page range for table {table_id}: {page_start}-{page_end}"
            )


def normalize_table_markdown(markdown: str) -> str:
    normalized = re.sub(r"[\uE000-\uF8FF]", "", markdown)
    normalized = re.sub(r"\*{2,}", "", normalized)
    normalized = normalized.replace("nông độ", "nồng độ")
    normalized = repair_targeted_punctuation_artifacts(normalized)
    normalized = re.sub(r"[ \t]+\|", " |", normalized)
    return normalized.strip()


def load_table_overrides(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Table overrides must be a JSON object: {path}")
    return {str(table_id): dict(value) for table_id, value in payload.items()}


def apply_table_overrides(
    tables: list[dict[str, Any]], overrides: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    updated: list[dict[str, Any]] = []
    for table in tables:
        table_copy = dict(table)
        table_id = table_copy.get("table_id")
        override = overrides.get(table_id) if isinstance(table_id, str) else None
        if override:
            table_copy["_manual_override"] = override
            if override.get("target_section_id"):
                table_copy["section_id"] = override["target_section_id"]

        # Programmatic overrides for Capecitabine NCIC tables displaced by PDF layout
        if table_copy.get("table_id") in (
            "curated-table-0310-001",
            "curated-table-0310-002",
        ):
            table_copy["section_id"] = "drug:capecitabin:lieu-luong-va-cach-dung"

        updated.append(table_copy)
    return updated


def table_is_rag_eligible(table: dict[str, Any]) -> bool:
    return (
        table.get("table_type") == "clinical_table"
        and bool(table.get("section_id"))
        and bool(str(table.get("markdown") or "").strip())
    )


def section_overlaps_table(section: SectionRecord, table: dict[str, Any]) -> bool:
    section_start = section.start_page or 0
    section_end = section.end_page or section_start
    table_start = int(table.get("page_start") or 0)
    table_end = int(table.get("page_end") or table_start)
    return section_start <= table_end and table_start <= section_end


def choose_section_index(
    sections: list[SectionRecord], table: dict[str, Any]
) -> int | None:
    context_candidate = choose_context_section_index(sections, table)
    if context_candidate is not None:
        return context_candidate

    table_section_id = str(table.get("section_id") or "")
    table_base_section_id = base_section_id(table_section_id)
    candidates = [
        index
        for index, section in enumerate(sections)
        if section.id == table_section_id
        or base_section_id(section.id) == table_section_id
        or (table_base_section_id is not None and section.id == table_base_section_id)
    ]
    if not candidates:
        return None
    overlapping = [
        index for index in candidates if section_overlaps_table(sections[index], table)
    ]
    if overlapping and table_text_overlap_mapping_matches(table):
        return max(
            overlapping,
            key=lambda index: (
                table_section_overlap_score(sections[index], table),
                -index,
            ),
        )
    if overlapping:
        return min(
            overlapping,
            key=lambda index: (
                (sections[index].end_page or 0) - (sections[index].start_page or 0),
                index,
            ),
        )
    return candidates[0]


def normalize_text(value: str | None) -> str:
    text = re.sub(r"\s+", " ", (value or "").strip().lower())
    decomposed = unicodedata.normalize("NFD", text)
    without_marks = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return without_marks.replace("đ", "d")


def normalize_label(value: str | None) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]+", " ", normalize_text(value))).strip()


def section_id_prefix(section_id: str | None) -> str | None:
    if not section_id or ":" not in section_id:
        return None
    return section_id.rsplit(":", 1)[0]


def base_section_id(section_id: str | None) -> str | None:
    if not section_id:
        return None
    return re.sub(r":part-\d{3}$", "", section_id)


def context_heading_matches_section(
    section: SectionRecord, table: dict[str, Any]
) -> bool:
    context = normalize_label(table.get("context_heading"))
    labels = normalize_label(" ".join([section.section, *section.context_path]))
    return bool(context) and (context in labels or labels in context)


def choose_context_section_index(
    sections: list[SectionRecord], table: dict[str, Any]
) -> int | None:
    method = str(table.get("section_match_method") or "")
    confidence = float(table.get("section_match_confidence") or 0)
    if method != "page_range" and confidence >= 0.7:
        return None

    prefix = section_id_prefix(table.get("section_id"))
    if not prefix:
        return None

    candidates = [
        index
        for index, section in enumerate(sections)
        if (base_section_id(section.id) or section.id).startswith(f"{prefix}:")
        and section_overlaps_table(section, table)
        and context_heading_matches_section(section, table)
    ]
    if not candidates:
        return None

    return min(
        candidates,
        key=lambda index: (
            (sections[index].end_page or 0) - (sections[index].start_page or 0),
            index,
        ),
    )


def table_cells(markdown: str) -> list[str]:
    cells: list[str] = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or "---" in stripped:
            continue
        for cell in stripped.strip("|").split("|"):
            value = cell.strip()
            if value:
                cells.append(value)
    return cells


def table_header_cells(markdown: str) -> list[str]:
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and "---" not in stripped:
            return [
                cell.strip() for cell in stripped.strip("|").split("|") if cell.strip()
            ]
    return []


def matching_token_count(text: str, tokens: list[str]) -> int:
    normalized = normalize_text(text)
    return sum(
        1
        for token in tokens
        if normalize_text(token) and normalize_text(token) in normalized
    )


def preview_text(value: str, limit: int = 600) -> str:
    return re.sub(r"\s+", " ", value.strip())[:limit]


def duplicate_overlap_metrics(
    section_text: str, table: dict[str, Any]
) -> dict[str, int]:
    markdown = str(table.get("markdown") or "")
    headers = table_header_cells(markdown)
    cells = table_cells(markdown)[len(headers) :]
    return {
        "header_matches": matching_token_count(section_text, headers),
        "cell_matches": matching_token_count(section_text, cells[:12]),
        "header_count": len(headers),
        "sampled_cell_count": len(cells[:12]),
    }


def table_section_overlap_score(section: SectionRecord, table: dict[str, Any]) -> int:
    markdown = str(table.get("markdown") or "")
    headers = table_header_cells(markdown)
    cells = table_cells(markdown)[len(headers) :]
    return matching_token_count(section.text, headers) * 2 + matching_token_count(
        section.text, cells[:20]
    )


def has_meaningful_flattened_evidence(metrics: dict[str, int]) -> bool:
    return (
        metrics["cell_matches"] >= 2
        or metrics["header_matches"] >= 2
        or (metrics["header_matches"] >= 1 and metrics["cell_matches"] >= 1)
    )


def header_like_line_exists(section_text: str, table: dict[str, Any]) -> bool:
    headers = table_header_cells(str(table.get("markdown") or ""))
    return any(
        matching_token_count(line, headers) >= 2
        and len(line) <= 140
        and "." not in line
        for line in section_text.splitlines()
    )


def duplicate_review_reason(
    skip_reason: str, section_text: str, table: dict[str, Any]
) -> str:
    if skip_reason == "low_overlap_confidence" and not header_like_line_exists(
        section_text, table
    ):
        return "partial_overlap_without_header_line"
    return skip_reason


def resolve_duplicate_skip(
    *,
    section: SectionRecord,
    section_text: str,
    table: dict[str, Any],
    skip: dict[str, Any],
) -> tuple[str, dict[str, Any] | None]:
    # In Docling flow, we resolve all skips as resolved_kept because keeping clean markdown tables
    # inside the section text is preferred for complete RAG context.
    return "resolved_kept", None


def write_duplicate_review_markdown(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Table Duplicate Review", ""]
    if not records:
        lines.extend(["No table duplicate review items.", ""])
    for index, record in enumerate(records, start=1):
        lines.extend(
            [
                f"## {index}. {record['table_id']}",
                "",
                f"- Section: `{record['section_id']}`",
                f"- Title: {record.get('title') or ''}",
                f"- Section label: {record.get('section') or ''}",
                f"- Pages: {record.get('page_start')} - {record.get('page_end')}",
                f"- Reason: `{record.get('reason')}`",
                f"- Overlap: `{record.get('overlap_metrics')}`",
                "",
                "Section text preview:",
                "",
                "```text",
                record.get("section_text_preview") or "",
                "```",
                "",
                "Table markdown preview:",
                "",
                "```markdown",
                record.get("table_markdown_preview") or "",
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def remove_flattened_captioned_table_text(
    section_text: str, table: dict[str, Any]
) -> tuple[str, dict[str, Any] | None]:
    caption = table.get("canonical_caption") or table.get("caption")
    if not caption or normalize_text(caption) not in normalize_text(section_text):
        return section_text, None

    markdown = str(table.get("markdown") or "")
    headers = table_header_cells(markdown)
    cells = table_cells(markdown)[len(headers) :]

    def get_word_tokens(text_list: list[str]) -> set[str]:
        words = set()
        for text in text_list:
            normalized = normalize_text(text)
            for w in re.findall(r"[a-z0-9à-ỹ]{3,}", normalized):
                words.add(w)
        return words

    header_words = get_word_tokens(headers)
    cell_words = get_word_tokens(cells[:15])
    all_tokens = header_words | cell_words

    if not all_tokens:
        return section_text, None

    def match_word_token_count(line_text: str) -> int:
        normalized = normalize_text(line_text)
        words_in_text = set(re.findall(r"[a-z0-9à-ỹ]{3,}", normalized))
        return len(all_tokens & words_in_text)

    if match_word_token_count(section_text) < 4:
        return section_text, None

    lines = section_text.splitlines()
    caption_index = None
    normalized_caption = normalize_text(caption)

    # Try exact match first
    for index, line in enumerate(lines):
        if normalized_caption in normalize_text(line):
            caption_index = index
            break

    # Try prefix matching if not found (e.g. if the caption is split across lines)
    if caption_index is None:
        prefix_match = re.match(r"^(bang\s+\d+)", normalized_caption)
        if prefix_match:
            prefix = prefix_match.group(1)
            for index, line in enumerate(lines):
                norm_line = normalize_text(line)
                if prefix in norm_line:
                    words_in_line = set(re.findall(r"[a-z0-9à-ỹ]{3,}", norm_line))
                    words_in_caption = set(
                        re.findall(r"[a-z0-9à-ỹ]{3,}", normalized_caption)
                    )
                    overlap = words_in_caption & words_in_line
                    if len(overlap) >= min(3, len(words_in_caption)):
                        caption_index = index
                        break

    if caption_index is None:
        return section_text, None

    remove_end = caption_index + 1
    consecutive_zero = 0
    for index in range(caption_index + 1, min(len(lines), caption_index + 100)):
        line_stripped = re.sub(r"^#+\s*", "", lines[index].strip())

        # Stop if we hit a section heading pattern (e.g. 4.2.3. or 5.)
        if re.match(r"^(?:\d+(?:\.\d+)*|[A-ZĐ]|[IVXLCDM]+)\.\s+", line_stripped):
            break

        # Stop if we hit another table caption pattern
        if re.match(r"^Bảng\s+\d+", line_stripped):
            break

        match_count = match_word_token_count(lines[index])
        if match_count >= 1:
            remove_end = index + 1
            consecutive_zero = 0
        else:
            consecutive_zero += 1
            if consecutive_zero > 2:  # Allow up to 2 consecutive lines with 0 matches
                break

    removed_lines = lines[caption_index:remove_end]
    new_lines = lines[:caption_index] + lines[remove_end:]
    return "\n".join(line for line in new_lines if line.strip()), {
        "table_id": table["table_id"],
        "remove_method": "caption_header_cell_overlap",
        "confidence": 0.9,
        "removed_text_preview": " ".join(removed_lines)[:240],
        "page_start": table.get("page_start"),
        "page_end": table.get("page_end"),
    }


def context_matches_section(section: SectionRecord, table: dict[str, Any]) -> bool:
    return context_heading_matches_section(section, table)


def table_text_overlap_mapping_matches(table: dict[str, Any]) -> bool:
    return (
        str(table.get("section_match_method") or "") == "table_text_overlap"
        and float(table.get("section_match_confidence") or 0) >= 0.8
    )


def line_is_short_header_fragment(line: str, headers: list[str]) -> bool:
    return (
        matching_token_count(line, headers) >= 1
        and len(line) <= 160
        and "." not in line
    )


def find_no_caption_table_range(
    lines: list[str], table: dict[str, Any]
) -> tuple[int | None, int | None, str | None]:
    markdown = str(table.get("markdown") or "")
    headers = table_header_cells(markdown)
    cells = table_cells(markdown)[len(headers) :]
    tokens = headers + cells[:12]

    for index, line in enumerate(lines):
        if (
            matching_token_count(line, headers) >= 2
            and len(line) <= 160
            and "." not in line
        ):
            end = index + 1
            while end < len(lines) and matching_token_count(lines[end], tokens) > 0:
                end += 1
            if end > index + 1:
                return index, end, "header_cell_overlap"

        if not line_is_short_header_fragment(line, headers):
            continue

        header_end = index
        matched_headers: set[str] = set()
        while header_end < len(lines) and line_is_short_header_fragment(
            lines[header_end], headers
        ):
            normalized_line = normalize_text(lines[header_end])
            for header in headers:
                normalized_header = normalize_text(header)
                if normalized_header and normalized_header in normalized_line:
                    matched_headers.add(normalized_header)
            header_end += 1

        if len(matched_headers) < min(2, len(headers)):
            continue

        end = header_end
        row_count = 0
        while end < len(lines) and matching_token_count(lines[end], tokens) > 0:
            row_count += 1
            end += 1

        if row_count > 0:
            return index, end, "split_header_cell_overlap"

    return None, None, None


def apply_manual_line_range_override(
    section_text: str, table: dict[str, Any]
) -> tuple[str, dict[str, Any] | None]:
    override = table.get("_manual_override") or {}
    ranges = override.get("remove_line_ranges") or []
    if not ranges:
        return section_text, None

    lines = section_text.splitlines()
    normalized_ranges: list[tuple[int, int]] = []
    for item in ranges:
        start = int(item["start"])
        end = int(item["end"])
        if start < 0 or end < start or end > len(lines):
            # Warning suppressed – invalid range skipped
            # skip this invalid range
            continue
        normalized_ranges.append((start, end))
    if not normalized_ranges:
        return section_text, None

    removed_lines: list[str] = []
    kept_lines = list(lines)
    for start, end in sorted(normalized_ranges, reverse=True):
        removed_lines[0:0] = kept_lines[start:end]
        del kept_lines[start:end]

    return "\n".join(line for line in kept_lines if line.strip()), {
        "table_id": table["table_id"],
        "remove_method": "manual_line_range_override",
        "confidence": 1.0,
        "removed_text_preview": " ".join(removed_lines)[:240],
        "page_start": table.get("page_start"),
        "page_end": table.get("page_end"),
        "note": override.get("note", ""),
    }


def remove_flattened_no_caption_table_text(
    section: SectionRecord,
    section_text: str,
    table: dict[str, Any],
) -> tuple[str, dict[str, Any] | None, dict[str, Any] | None]:
    if not section_overlaps_table(section, table) or not (
        context_matches_section(section, table)
        or table_text_overlap_mapping_matches(table)
    ):
        return (
            section_text,
            None,
            {
                "table_id": table["table_id"],
                "reason": "context_or_page_mismatch",
                "action": "inserted_without_removal",
            },
        )

    markdown = str(table.get("markdown") or "")
    headers = table_header_cells(markdown)
    cells = table_cells(markdown)[len(headers) :]
    if (
        matching_token_count(section_text, headers) < 2
        or matching_token_count(section_text, cells[:8]) < 2
    ):
        return (
            section_text,
            None,
            {
                "table_id": table["table_id"],
                "reason": "low_overlap_confidence",
                "action": "inserted_without_removal",
            },
        )

    lines = section_text.splitlines()
    header_index, remove_end, remove_method = find_no_caption_table_range(lines, table)
    if header_index is None:
        return (
            section_text,
            None,
            {
                "table_id": table["table_id"],
                "reason": "low_overlap_confidence",
                "action": "inserted_without_removal",
            },
        )

    removed_lines = lines[header_index:remove_end]
    new_lines = lines[:header_index] + lines[remove_end:]
    return (
        "\n".join(line for line in new_lines if line.strip()),
        {
            "table_id": table["table_id"],
            "remove_method": remove_method,
            "confidence": 0.86,
            "removed_text_preview": " ".join(removed_lines)[:240],
            "page_start": table.get("page_start"),
            "page_end": table.get("page_end"),
        },
        None,
    )


def block_text(block: CanonicalBlock) -> str:
    if block.content_type == "table":
        return block.markdown or ""
    return block.text or ""


def hydrate_strategy_for_section(section: SectionRecord | CanonicalSection) -> str:
    if section.id == BRAND_INDEX_SECTION_ID:
        return "search_only"
    if len(section.text) > FULL_SECTION_MAX_CHARS:
        return "chunk_window"
    return "full_section"


def build_canonical_blocks(
    sections: list[SectionRecord], tables: list[dict[str, Any]]
) -> tuple[list[CanonicalBlock], dict[str, Any]]:
    by_section_index: dict[int, list[dict[str, Any]]] = {}
    skip_reasons: dict[str, int] = {}
    for table in tables:
        if not table_is_rag_eligible(table):
            skip_reasons["not_rag_eligible"] = (
                skip_reasons.get("not_rag_eligible", 0) + 1
            )
            continue
        section_index = choose_section_index(sections, table)
        if section_index is None:
            skip_reasons["section_not_found"] = (
                skip_reasons.get("section_not_found", 0) + 1
            )
            continue
        by_section_index.setdefault(section_index, []).append(table)

    blocks: list[CanonicalBlock] = []
    block_counter = 1
    replacement_events: list[dict[str, Any]] = []
    replacement_skips: list[dict[str, Any]] = []
    duplicate_review_items: list[dict[str, Any]] = []
    duplicate_resolution = {
        "manual_replaced": 0,
        "resolved_replaced": 0,
        "resolved_kept": 0,
        "needs_manual_review": 0,
    }
    manual_override_count = 0

    for index, section in enumerate(sections):
        section_text = section.text
        section_tables = by_section_index.get(index, [])
        table_flags: dict[str, list[str]] = {}

        for table in section_tables:
            updated_text, manual_event = apply_manual_line_range_override(
                section_text, table
            )
            if manual_event:
                section_text = updated_text
                manual_event["section_id"] = section.id
                replacement_events.append(manual_event)
                duplicate_resolution["manual_replaced"] += 1
                manual_override_count += 1
                continue

            updated_text, captioned_event = remove_flattened_captioned_table_text(
                section_text, table
            )
            if captioned_event:
                section_text = updated_text
                captioned_event["section_id"] = section.id
                replacement_events.append(captioned_event)
                continue

            if not table.get("canonical_caption") and not table.get("caption"):
                updated_text, no_caption_event, skip = (
                    remove_flattened_no_caption_table_text(section, section_text, table)
                )
                if no_caption_event:
                    section_text = updated_text
                    no_caption_event["section_id"] = section.id
                    replacement_events.append(no_caption_event)
                elif skip:
                    skip["section_id"] = section.id
                    resolution, review_item = resolve_duplicate_skip(
                        section=section,
                        section_text=section_text,
                        table=table,
                        skip=skip,
                    )
                    skip["resolution"] = resolution
                    replacement_skips.append(skip)
                    duplicate_resolution[resolution] += 1
                    if review_item:
                        duplicate_review_items.append(review_item)
                        table_flags.setdefault(table["table_id"], []).append(
                            "needs_manual_duplicate_review"
                        )

        section_text = repair_targeted_punctuation_artifacts(section_text)
        blocks.append(
            CanonicalBlock(
                block_id=f"block-{block_counter:06d}",
                section_index=index,
                section_id=section.id,
                title=section.title,
                section=section.section,
                content_type="paragraph",
                source_type="pymupdf_text",
                page_start=section.start_page,
                page_end=section.end_page,
                text=section_text,
                markdown=None,
                source_refs=[f"sections.jsonl:{index}:{section.id}"],
                quality_flags=[],
            )
        )
        block_counter += 1

        for table in section_tables:
            source_refs = list(table.get("source_table_ids") or [table["table_id"]])
            quality_flags = sorted(
                set(
                    list(table.get("quality_flags") or [])
                    + table_flags.get(table["table_id"], [])
                )
            )
            blocks.append(
                CanonicalBlock(
                    block_id=f"block-{block_counter:06d}",
                    section_index=index,
                    section_id=section.id,
                    title=section.title,
                    section=section.section,
                    content_type="table",
                    source_type="docling_table",
                    page_start=table.get("page_start"),
                    page_end=table.get("page_end"),
                    text=None,
                    markdown=normalize_table_markdown(str(table["markdown"])),
                    source_refs=source_refs,
                    quality_flags=quality_flags,
                    table_id=table["table_id"],
                )
            )
            block_counter += 1

    duplicate_resolution["resolved_replaced"] = (
        len(replacement_events) - duplicate_resolution["manual_replaced"]
    )
    audit = {
        "block_count": len(blocks),
        "table_block_count": sum(
            1 for block in blocks if block.content_type == "table"
        ),
        "skip_reasons": skip_reasons,
        "replacement_count": len(replacement_events),
        "replacement_events": replacement_events,
        "replacement_skips": replacement_skips,
        "duplicate_resolution": duplicate_resolution,
        "duplicate_review_count": len(duplicate_review_items),
        "duplicate_review_items": duplicate_review_items,
        "manual_override_count": manual_override_count,
    }
    return blocks, audit


def build_canonical_sections(
    sections: list[SectionRecord], blocks: list[CanonicalBlock]
) -> list[CanonicalSection]:
    blocks_by_index: dict[int, list[CanonicalBlock]] = {}
    for block in blocks:
        blocks_by_index.setdefault(block.section_index, []).append(block)

    canonical_sections: list[CanonicalSection] = []
    for index, section in enumerate(sections):
        section_blocks = blocks_by_index.get(index, [])
        text_parts = [
            block_text(block) for block in section_blocks if block_text(block).strip()
        ]
        table_ids = [
            block.table_id
            for block in section_blocks
            if block.content_type == "table" and block.table_id
        ]
        source_mix = sorted({block.source_type for block in section_blocks})
        quality_flags = sorted(
            {flag for block in section_blocks for flag in block.quality_flags}
        )
        text = "\n\n".join(text_parts)
        canonical_sections.append(
            CanonicalSection(
                id=section.id,
                content_type=section.content_type,
                title=section.title,
                section=section.section,
                text=text,
                source=section.source,
                context_path=list(section.context_path),
                context_header=section.context_header,
                start_page=section.start_page,
                end_page=section.end_page,
                warnings=list(section.warnings),
                block_ids=[block.block_id for block in section_blocks],
                table_ids=table_ids,
                source_mix=source_mix,
                quality_flags=quality_flags,
                hydrate_strategy=hydrate_strategy_for_section(
                    SectionRecord(
                        id=section.id,
                        content_type=section.content_type,
                        title=section.title,
                        section=section.section,
                        text=text,
                    )
                ),
                section_char_count=len(text),
            )
        )
    return canonical_sections


def canonical_sections_to_section_records(
    sections: list[CanonicalSection],
) -> list[SectionRecord]:
    return [
        SectionRecord(
            id=section.id,
            content_type=section.content_type,
            title=section.title,
            section=section.section,
            text=section.text,
            source=section.source,
            context_path=list(section.context_path),
            context_header=section.context_header,
            start_page=section.start_page,
            end_page=section.end_page,
            warnings=list(section.warnings),
        )
        for section in sections
    ]


def is_markdown_separator_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped.startswith("|") or "-" not in stripped:
        return False
    return stripped.replace("|", "").replace("-", "").replace(":", "").strip() == ""


def split_table_markdown(markdown: str, max_chars: int) -> list[str]:
    text = markdown.strip()
    if len(text) <= max_chars:
        return [text]

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 3:
        return split_long_text(text, max_chars)

    header = lines[0]
    separator = lines[1] if is_markdown_separator_line(lines[1]) else ""
    if not separator:
        return split_long_text(text, max_chars)

    prefix = [header, separator]
    rows = lines[2:]
    chunks: list[str] = []
    current_rows: list[str] = []

    for row in rows:
        candidate_rows = [*current_rows, row]
        candidate = "\n".join(prefix + candidate_rows)
        if current_rows and len(candidate) > max_chars:
            chunks.append("\n".join(prefix + current_rows))
            current_rows = [row]
            continue
        current_rows = candidate_rows

    if current_rows:
        chunks.append("\n".join(prefix + current_rows))

    return chunks or [text]


def split_lines_without_breaking_entries(text: str, max_chars: int) -> list[str]:
    lines = [line.rstrip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return []

    chunks: list[str] = []
    current: list[str] = []
    for line in lines:
        candidate = "\n".join([*current, line]) if current else line
        if current and len(candidate) > max_chars:
            next_lines = [line]
            while current and next_lines[0][:1].islower():
                next_lines.insert(0, current.pop())
            if current:
                chunks.append("\n".join(current))
            current = next_lines
            continue
        current.append(line)

    if current:
        chunks.append("\n".join(current))
    return chunks


def split_brand_index_text(text: str, max_chars: int) -> list[str]:
    parts = split_lines_without_breaking_entries(text, max_chars)
    if not parts:
        return []
    return parts


def chunk_role_for_section(section: CanonicalSection, block: CanonicalBlock) -> str:
    if block.content_type == "table":
        return "table"
    if section.id == BRAND_INDEX_SECTION_ID:
        return "index_entry"
    if section.id in APPENDIX_LIST_SECTION_IDS:
        return "appendix_list"
    return "prose"


def build_final_chunk_records(
    canonical_sections: list[CanonicalSection],
    blocks: list[CanonicalBlock],
    *,
    max_chars: int,
) -> list[dict[str, Any]]:
    blocks_by_section_id: dict[str, list[CanonicalBlock]] = {}
    for block in blocks:
        blocks_by_section_id.setdefault(block.section_id, []).append(block)

    records: list[dict[str, Any]] = []
    for section in canonical_sections:
        context_path = section.context_path or [section.section]
        context_header = section.context_header
        chunk_index = 1
        section_blocks = blocks_by_section_id.get(section.id, [])
        for block in section_blocks:
            raw_text = block_text(block).strip()
            if not raw_text:
                continue

            chunk_role = chunk_role_for_section(section, block)
            if chunk_role == "table":
                parts = split_table_markdown(raw_text, max_chars)
                table_chunk_count = len(parts)
            elif chunk_role == "index_entry":
                parts = split_brand_index_text(raw_text, max_chars)
                table_chunk_count = 0
            elif chunk_role == "appendix_list":
                parts = split_lines_without_breaking_entries(raw_text, max_chars)
                table_chunk_count = 0
            else:
                parts = split_long_text(raw_text, max_chars)
                table_chunk_count = 0

            for part_index, part in enumerate(parts, start=1):
                payload: dict[str, Any] = {
                    "id": f"{section.id}:chunk-{chunk_index:03d}",
                    "section_id": section.id,
                    "content_type": section.content_type,
                    "title": section.title,
                    "section": section.section,
                    "text": part,
                    "source": section.source,
                    "context_path": context_path,
                    "context_header": context_header,
                    "chunk_index": chunk_index,
                    "start_page": block.page_start
                    if block.page_start is not None
                    else section.start_page,
                    "end_page": block.page_end
                    if block.page_end is not None
                    else section.end_page,
                    "warnings": list(section.warnings),
                    "hydrate_strategy": section.hydrate_strategy,
                    "section_char_count": section.section_char_count,
                    "chunk_content_type": block.content_type,
                    "chunk_role": chunk_role,
                    "source_block_id": block.block_id,
                }
                if block.content_type == "table":
                    payload.update(
                        {
                            "table_id": block.table_id,
                            "table_chunk_index": part_index,
                            "table_chunk_count": table_chunk_count,
                        }
                    )
                records.append(payload)
                chunk_index += 1

        if chunk_index == 1:
            fallback = SectionRecord(
                id=section.id,
                content_type=section.content_type,
                title=section.title,
                section=section.section,
                text=section.text,
                source=section.source,
                context_path=context_path,
                context_header=context_header,
                start_page=section.start_page,
                end_page=section.end_page,
                warnings=list(section.warnings),
            )
            records.extend(
                annotate_chunks_for_hydration(
                    chunk_sections([fallback], max_chars=max_chars), [section]
                )
            )

    return records


def annotate_chunks_for_hydration(
    chunks: list[Any], sections: list[CanonicalSection]
) -> list[dict[str, Any]]:
    sections_by_id = {section.id: section for section in sections}
    output: list[dict[str, Any]] = []
    for chunk in chunks:
        payload = asdict(chunk)
        section = sections_by_id.get(str(payload.get("section_id") or ""))
        if section:
            payload["hydrate_strategy"] = section.hydrate_strategy
            payload["section_char_count"] = section.section_char_count
        output.append(payload)
    return output


def manifest_path(path: Path) -> str:
    resolved_path = path.resolve()
    try:
        return str(resolved_path.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def manifest_payload(
    *,
    sections_path: Path,
    tables_path: Path,
    table_overrides_path: Path | None,
    canonical_dir: Path,
    final_dir: Path,
) -> dict[str, Any]:
    return {
        "corpus": "duoc-thu-quoc-gia-viet-nam",
        "pipeline_version": "canonical-v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "inputs": {
            "sections": manifest_path(sections_path),
            "tables": manifest_path(tables_path),
            "table_overrides": manifest_path(table_overrides_path)
            if table_overrides_path
            else None,
        },
        "outputs": {
            "canonical_blocks": manifest_path(canonical_dir / "blocks.jsonl"),
            "canonical_sections": manifest_path(canonical_dir / "sections.jsonl"),
            "rag_sections": manifest_path(final_dir / "sections.jsonl"),
            "rag_chunks": manifest_path(final_dir / "chunks.jsonl"),
        },
    }


def process_canonical_rag(
    *,
    sections_path: Path = DEFAULT_SECTIONS,
    chunks_path: Path = DEFAULT_CHUNKS,
    tables_path: Path = DEFAULT_TABLES,
    table_overrides_path: Path | None = DEFAULT_TABLE_OVERRIDES,
    canonical_dir: Path = DEFAULT_CANONICAL_DIR,
    final_dir: Path = DEFAULT_FINAL_DIR,
    max_chars: int = DEFAULT_CHUNK_MAX_CHARS,
) -> dict[str, Path]:
    sections_path = Path(sections_path)
    chunks_path = Path(chunks_path)
    tables_path = Path(tables_path)
    table_overrides_path = (
        Path(table_overrides_path) if table_overrides_path is not None else None
    )
    canonical_dir = Path(canonical_dir)
    final_dir = Path(final_dir)

    sections = load_sections(sections_path)
    sections, targeted_merge_audit = merge_targeted_short_part_sections(sections)
    sections, pdf_verified_fix_audit = apply_pdf_verified_section_fixes(sections)
    original_chunk_count = len(read_jsonl(chunks_path))
    tables = [clean_unicode_value(t) for t in read_jsonl(tables_path)]
    validate_tables(tables)
    table_overrides = load_table_overrides(table_overrides_path)
    tables = apply_table_overrides(tables, table_overrides)

    blocks, canonical_audit = build_canonical_blocks(sections, tables)
    canonical_sections = build_canonical_sections(sections, blocks)
    final_chunk_records = build_final_chunk_records(
        canonical_sections, blocks, max_chars=max_chars
    )

    canonical_blocks_path = canonical_dir / "blocks.jsonl"
    canonical_sections_path = canonical_dir / "sections.jsonl"
    canonical_manifest_path = canonical_dir / "manifest.json"
    canonical_audit_path = canonical_dir / "audit.json"
    duplicate_review_jsonl_path = canonical_dir / "table_duplicate_review.jsonl"
    duplicate_review_md_path = canonical_dir / "table_duplicate_review.md"
    rag_sections_path = final_dir / "sections.jsonl"
    rag_chunks_path = final_dir / "chunks.jsonl"
    rag_manifest_path = final_dir / "manifest.json"
    rag_audit_path = final_dir / "audit.json"

    write_jsonl_records(canonical_blocks_path, blocks)
    write_jsonl_records(canonical_sections_path, canonical_sections)

    manifest = manifest_payload(
        sections_path=sections_path,
        tables_path=tables_path,
        table_overrides_path=table_overrides_path,
        canonical_dir=canonical_dir,
        final_dir=final_dir,
    )
    canonical_audit.update(
        {
            "canonical_section_count": len(canonical_sections),
            "input_section_count": len(sections),
            "input_chunk_count": original_chunk_count,
            **targeted_merge_audit,
            **pdf_verified_fix_audit,
        }
    )
    write_json(canonical_manifest_path, manifest)
    write_json(canonical_audit_path, canonical_audit)
    write_jsonl_dicts(
        duplicate_review_jsonl_path, canonical_audit["duplicate_review_items"]
    )
    write_duplicate_review_markdown(
        duplicate_review_md_path, canonical_audit["duplicate_review_items"]
    )

    write_jsonl_records(rag_sections_path, canonical_sections)
    write_jsonl_dicts(rag_chunks_path, final_chunk_records)
    write_json(rag_manifest_path, manifest)
    write_json(
        rag_audit_path,
        {
            "section_count": len(canonical_sections),
            "chunk_count": len(final_chunk_records),
            "source": "interim-canonical",
            "table_block_count": canonical_audit["table_block_count"],
            "replacement_count": canonical_audit["replacement_count"],
            "duplicate_resolution": canonical_audit["duplicate_resolution"],
            "duplicate_review_count": canonical_audit["duplicate_review_count"],
            "targeted_short_part_merge_count": targeted_merge_audit[
                "targeted_short_part_merge_count"
            ],
            "pdf_verified_section_fix_count": pdf_verified_fix_audit[
                "pdf_verified_section_fix_count"
            ],
        },
    )

    return {
        "canonical_blocks": canonical_blocks_path,
        "canonical_sections": canonical_sections_path,
        "canonical_manifest": canonical_manifest_path,
        "canonical_audit": canonical_audit_path,
        "duplicate_review_jsonl": duplicate_review_jsonl_path,
        "duplicate_review_md": duplicate_review_md_path,
        "rag_sections": rag_sections_path,
        "rag_chunks": rag_chunks_path,
        "rag_manifest": rag_manifest_path,
        "rag_audit": rag_audit_path,
    }
