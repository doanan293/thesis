#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

from seed_pipeline.config.paths import RESOURCES_DIR, TEXT_INTERIM_DIR

DEFAULT_INPUT = TEXT_INTERIM_DIR / "full.md"
DEFAULT_OUTPUT = TEXT_INTERIM_DIR / "full.cleaned.md"
DEFAULT_SYLLABLES = RESOURCES_DIR / "vietnamese_valid_syllables.json"


# Load Vietnamese syllables database for spacing validation
def _normalize_tone(s: str) -> str:
    replacements = {
        "óa": "oá",
        "òa": "oà",
        "ỏa": "oả",
        "õa": "oã",
        "ọa": "oạ",
        "úy": "uý",
        "uỳ": "uỳ",
        "ủy": "uỷ",
        "ũy": "uỹ",
        "ụy": "uỵ",
        "óe": "oé",
        "òe": "oè",
        "ỏe": "oẻ",
        "õe": "oẽ",
        "ọe": "oẹ",
        "hủy": "huỷ",
        "thủy": "thuỷ",
        "tủy": "tuỷ",
        "tùy": "tuỳ",
        "tụy": "tuỵ",
        "lũy": "luỹ",
        "hóa": "hoá",
        "hòa": "hoà",
        "thỏa": "thoả",
        "tỏa": "toả",
        "dọa": "doạ",
        "khỏe": "khoẻ",
        "thùy": "thuỳ",
        "nhùy": "nhuỳ",
        "khuyếch": "khuếch",
    }
    res = s.lower()
    for k, v in replacements.items():
        res = res.replace(k, v)
    return res


NORM_SYLLABLES_DB = set()
try:
    _syllables_file = DEFAULT_SYLLABLES
    if _syllables_file.exists():
        import json

        with open(_syllables_file, encoding="utf-8") as _f:
            _raw_syllables = json.load(_f)
            NORM_SYLLABLES_DB = {_normalize_tone(w) for w in _raw_syllables}
except Exception:
    pass

ALLOWED_ACRONYMS = {
    "adr",
    "nsaid",
    "nsaids",
    "atc",
    "dna",
    "rna",
    "hiv",
    "gaba",
    "mao",
    "av",
    "bp",
    "ct",
    "mri",
    "vzv",
    "hsv",
    "ebv",
    "cmv",
    "hbv",
    "hcv",
    "ldl",
    "hdl",
    "cox",
    "gi",
    "iv",
    "im",
}


def is_valid_word(word: str) -> bool:
    if not NORM_SYLLABLES_DB:
        return True  # Fallback if DB not loaded
    w_norm = _normalize_tone(word.lower())
    return w_norm in NORM_SYLLABLES_DB or w_norm in ALLOWED_ACRONYMS


OCR_TEXT_REPAIRS = {
    "đạị": "đại",
    "thảỉ": "thải",
    "gỉảm": "giảm",
    "giảỉ": "giải",
    "nhĩthất": "nhĩ thất",
    "D****ạ****ng thu****ố****c": "Dạng thuốc",
    "D****ạ****ng thuốc": "Dạng thuốc",
    "hà****m lượ****ng": "hàm lượng",
    "hàm lượ****ng": "hàm lượng",
    "Li****ều lượ****ng": "Liều lượng",
    "Liều lư ợng": "Liều lượng",
    "Liề u lượng": "Liều lượng",
    "Lo****ạ****i thu****ố****c": "Loại thuốc",
    "Tên chung qu****ố****c t****ế": "Tên chung quốc tế",
    "Tác d****ụ****ng không mong mu****ố****n": "Tác dụng không mong muốn",
    "Th****ờ****i k****ỳ": "Thời kỳ",
    "Tư ơng": "Tương",
    "Dư ợc": "Dược",
    "D ược": "Dược",
    "H ướng": "Hướng",
    "th ượng": "thượng",
    "Stevens Johnson": "Stevens-Johnson",
    "Steven Johnson": "Stevens-Johnson",
    "y ế u": "yếu",
    "huyế táp": "huyết áp",
    "diề u t": "điều trị",
    "Nguời": "Người",
    "nguời": "người",
    "huyếp áp": "huyết áp",
    "duới": "dưới",
    "mũit = ị 8 xy gàng n hà liều gổn (tiều chuổi bo ) và am go cr mit = 800 ị (4 x 1 600 microgam)": "mũi vào buổi tối (tổng liều mỗi ngày 800 microgam)",
    "O pium": "Opium",
    "O pioid": "Opioid",
    "O pizoic": "Opizoic",
    "O ral": "Oral",
    "ru ộ t": "ruột",
    "ho ặ c": "hoặc",
    "ru ộ": "ruột",
    "tho ặ c": "hoặc",
    "ru ộ tho ặ c": "ruột hoặc",
    "T ắc": "Tắc",
    "Ch ống": "Chống",
    "lo ạ n": "loạn",
    "làm ột": "là một",
    "cót ác": "có tác",
    "Ytế": "Y tế",
    "ruộthoặc": "ruột hoặc",
    "vớiCMV": "với CMV",
    "mgngày": "mg ngày",
    "TăngALAT": "Tăng ALAT",
    "TăngASAT": "Tăng ASAT",
    "lạiAPTT": "lại APTT",
    "hoăc": "hoặc",
    "đươc": "được",
    "xẫm": "sẫm",
    "khuẫn": "khuẩn",
    "blôc": "blốc",
    "tiêp bắp": "tiêm bắp",
    "vòngn": "vòng",
    "liểu": "liều",
    "hoặt": "hoặc",
    "làm ộ tyếu tố": "là một yếu tố",
}

SECTION_HEADING_REPAIRS = {
    "Liều lượng, cách dùng": "Liều lượng và cách dùng",
    "Liều lượng cách dùng": "Liều lượng và cách dùng",
    "Liều lượng và cách sử dụng": "Liều lượng và cách dùng",
    "Cách dùng và liều lượng": "Liều lượng và cách dùng",
    "Liều dùng và cách dùng": "Liều lượng và cách dùng",
    "Bảo quản": "Độ ổn định và bảo quản",
}

SPLIT_WORDS_RAW = [
    # (part1, part2, keep_hyphen)
    ("beta", "adrenergic", True),
    ("beta", "lactamase", True),
    ("beta", "lactam", True),
    ("beta1", "adrenergic", True),
    ("beta2", "adrenergic", True),
    ("alpha", "adrenergic", True),
    ("Stevens", "Johnson", True),
    ("Steven", "Johnson", True),  # Map Steven-Johnson to Stevens-Johnson
    ("Epstein", "Barr", True),
    ("Guillain", "Barré", True),
    ("Wolff", "Parkinson", True),
    ("Jarisch", "Herxheimer", True),
    ("renin", "angiotensin", True),
    ("angiotensin", "aldosteron", True),
    ("glucose", "dependent", True),
    ("penicilin", "binding", True),
    ("trimethoprim", "sulfamethoxazol", True),
    ("LDL", "cholesterol", True),
    ("X", "quang", True),
    ("pyrime", "thamin", False),
    ("methylpred", "nisolon", False),
    ("methyl", "prednisolon", False),
    ("Hydroxoco", "balamin", False),
    ("methylco", "balamin", False),
    ("dimercapto", "sucinic", False),
    ("glucuronosyl", "transferase", False),
    ("acetylcholin", "esterase", False),
    ("acetylcholi", "nesterase", False),
    ("Pepto", "streptococcus", False),
    ("Peptos", "treptococcus", False),
    ("Strepto", "coccus", False),
    ("Enterobac", "teriaceae", False),
    ("fluoro", "quinolon", False),
    ("fluoroqui", "nolon", False),
    ("cloramphe", "nicol", False),
    ("Alopu", "rinol", False),
    ("doxoru", "bicin", False),
    ("hydro", "clorid", False),
    ("hydro", "xyartemether", False),
    ("hydroxy", "cilostazol", False),
    ("penicilloyl", "polylysin", False),
    ("sulfinpy", "razon", False),
    ("phenylpropano", "lamin", False),
    ("micro", "gam", False),
    ("bupi", "vacain", False),
    ("5", "fluorouridin", True),
    ("beta", "hemolyticus", True),
    ("isoxazolyl", "penicilin", False),
    ("Mycobac", "terium", False),
    ("fluorou", "racil", False),
    ("sar", "gramostim", False),
    ("7", "aminoclonazepam", True),
    ("oleando", "mycin", False),
    ("corticosteroid", "ICS", True),
    ("phenobar", "bital", False),
    ("anti", "HBs", True),
    ("HLA", "B", True),
    ("Fcε", "RI", True),
    ("gamma", "carboxyglutamic", True),
    ("gamma", "aminobutyric", True),
    ("propyl", "hydroxypentanoic", False),
    ("benzodi", "zepin", False),
    ("benzodi", "azepin", False),
    ("betalacta", "mase", False),
    ("monocyto", "genes", False),
    ("rota", "haler", False),
    ("p", "hydroxyphenyl", True),
    ("para", "aminophenol", True),
    ("para", "aminobenzoic", True),
    ("carbidopa", "levodopa", True),
    ("Spas", "Meyer", True),
    ("Bioment", "Bid", True),
    ("Simtas", "10", True),
    ("RNI", "2002", True),
    ("Candes", "artan", False),
    ("DH", "Captohasan", True),
    ("pms", "Cefadroxil", True),
    ("TV", "Droxil", True),
    ("Cebopim", "BCPP", True),
    ("Epo", "rocine", False),
    ("acetyl", "p", True),
    ("2", "hydroxy", True),
    ("Euro", "Cee", True),
    ("Henoch", "Schonlein", True),
    ("cis", "acitretin", True),
    ("α", "TTP", True),
    ("α", "tan", True),
    ("Usatonic", "Natural", True),
    ("Euphoric", "Azoric", True),
    ("Albex", "400", True),
    ("alpra", "zolam", False),
    ("hydroxyalpra", "zolam", False),
    ("DOT", "Directly", True),
    ("hydroxy", "ethoxymethyl", False),
    ("Mecefix", "B", True),
    ("Twice", "cef", True),
    ("TV", "Ceftri", True),
    ("lacta", "mase", False),
    ("Vudu", "cefuroxim", True),
    ("Zanimex", "Dobfar", True),
    ("Stugon", "pharimex", True),
    ("cloram", "phenicol", False),
    ("Cloromy", "cetin", False),
    ("Clorpro", "mazin", False),
    ("in", "fluenzae", False),
    ("hydroxocobalamin", "transcobalamin", True),
    ("hydroxoco", "balamin", False),
    ("hydroxo", "cobalamin", False),
    ("Cyclophos", "phamid", False),
    ("dom", "peridon", False),
    ("cyclophos", "phamid", False),
    ("4", "carboxamid", True),
    ("aminoimidazol", "4", True),
    ("dextro", "propoxyphen", False),
    ("Diclo", "Denk", True),
    ("deoxyadenosin", "5", True),
    ("dihydrodigoxigenin", "bisdigitoxosid", True),
    ("dihydrodi", "goxigenin", False),
    ("alpha", "adrenolytic", True),
    ("Ganong", "Levine", True),
    ("dimenhyd", "rinat", False),
    ("Neo", "Allerfar", True),
    ("diethyldithio", "carbamat", False),
    ("noradre", "nalin", False),
    ("nhĩ", "thất", True),
    ("thetaio", "taomicron", False),
    ("beta", "1a", True),
    ("TBF", "β", True),
    ("pms", "Imelazide", True),
    ("Lan", "30", True),
    ("Lufi", "500", True),
    ("DS", "Pro", True),
    ("pms", "Lopradium", True),
    ("Fenxicam", "M", True),
    ("NDC", "Meloxicam", True),
    ("Melo", "fort", False),
    ("Varicella", "zoster", True),
    ("β1", "adrenergic", True),
    ("anthrace", "nedion", False),
    ("EGFR", "tyrosin", True),
    ("Geofman", "Ethambutol", True),
    ("clorpro", "mazin", False),
    ("Epider", "mophyton", False),
    ("fluoroarabino", "furanosyladenin", False),
    ("glucocorti", "coid", False),
    ("5", "Fluorouracil", True),
    ("fluoromethyl", "anilin", False),
    ("corticos", "teroid", False),
    ("amino", "glycosid", False),
    ("aminogly", "cosid", False),
    ("gliben", "clamid", False),
    ("extravas", "cular", False),
    ("intravas", "cular", False),
    ("UDP", "glucuronosyl", True),
    ("O", "methyltransferase", True),
    ("S", "reductase", True),
    ("R", "SNO", True),
    ("disease", "modifying", True),
    ("methanol", "bisbenzonitril", False),
    ("1", "esterase", True),
    ("SN", "38", True),
]


def build_split_words_repairs(raw_list: list[tuple[str, str, bool]]) -> dict[str, str]:
    repairs: dict[str, str] = {}
    spaces = [" ", "\n", " \n", "\n ", "  ", "\n  "]
    for p1, p2, keep in raw_list:
        good = p1 + ("-" if keep else "") + p2
        # Special override cases
        if p1 in ("Stevens", "Steven") and p2 == "Johnson":
            good = "Stevens-Johnson"
        if p1 == "Alopu" and p2 == "rinol":
            good = "Allopurinol"

        # Generate variations with spaces/newlines around the hyphen
        for sp in spaces:
            repairs[f"{p1}-{sp}{p2}"] = good
            repairs[f"{p1}{sp}-{p2}"] = good
            for sp2 in spaces:
                repairs[f"{p1}{sp}-{sp2}{p2}"] = good
        if not keep:
            repairs[f"{p1}-{p2}"] = good
    return repairs


SPLIT_WORDS_REPAIRS = dict(
    sorted(
        build_split_words_repairs(SPLIT_WORDS_RAW).items(),
        key=lambda x: len(x[0]),
        reverse=True,
    )
)


def _is_emphasized_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("**") and stripped.endswith("**")


def _inner_emphasis_text(line: str) -> str:
    stripped = line.strip()
    return stripped[2:-2].strip()


def _has_unbalanced_parentheses(text: str) -> bool:
    return text.count("(") != text.count(")")


def clean_markdown_line(line: str) -> str:
    cleaned = line
    for bad, good in OCR_TEXT_REPAIRS.items():
        cleaned = cleaned.replace(bad, good)

    if _is_emphasized_line(cleaned):
        inner = _inner_emphasis_text(cleaned)
        inner_without_stars = inner.replace("*", "")
        if _has_unbalanced_parentheses(inner_without_stars):
            return inner_without_stars
        canonical = SECTION_HEADING_REPAIRS.get(inner_without_stars)
        if canonical:
            return f"**{canonical}**"
        if inner_without_stars != inner:
            return f"**{inner_without_stars}**"

    return cleaned


def repair_spacing(text: str) -> str:
    V_chars = "aàáảãạăắằẳẵặâấầẩẫậeèéẻẽẹêếềểễệiíìỉĩịoóòỏõọôồốổỗộơớờởỡợuúùủũụưứừửữựyýỳỷỹỵAÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬEÈÉẺẼẸÊẾỀỂỄỆIÍÌỈĨỊOÓÒỎÕỌÔỐỔỔỖỘƠỚỜỞỠỢUÚÙỦŨỤƯỨỪỬỮỰYÝỲỶỸỴ"
    L_chars = V_chars + "bcdđfghjklmnpqrstvwxzBCDĐFGHJKLMNPQRSTVWXZ"

    # 1. Merge split digraphs/trigraphs (consonants)
    repaired = re.sub(r"\b([Cc])\s+([Hh])\b", r"\1\2", text)
    repaired = re.sub(r"\b([Tt])\s+([Rr])\b", r"\1\2", repaired)
    repaired = re.sub(r"\b([Nn])\s+([Gg])\s+([Hh])\b", r"\1\2\3", repaired)
    repaired = re.sub(r"\b([Nn])\s+([Gg])\b", r"\1\2", repaired)
    repaired = re.sub(r"\b([Nn])\s+([Hh])\b", r"\1\2", repaired)
    repaired = re.sub(r"\b([Kk])\s+([Hh])\b", r"\1\2", repaired)
    repaired = re.sub(r"\b([Pp])\s+([Hh])\b", r"\1\2", repaired)
    repaired = re.sub(r"\b([Gg])\s+([Hh])\b", r"\1\2", repaired)
    repaired = re.sub(r"\b([Qq])\s+([Uu])\b", r"\1\2", repaired)
    repaired = re.sub(r"\b([Tt])\s+([Hh])\b", r"\1\2", repaired)

    # Define valid initial consonants
    consonants = [
        "ch",
        "gh",
        "kh",
        "ngh",
        "ng",
        "nh",
        "ph",
        "qu",
        "th",
        "tr",
        "b",
        "c",
        "d",
        "đ",
        "g",
        "h",
        "k",
        "l",
        "m",
        "n",
        "p",
        "r",
        "s",
        "t",
        "v",
        "x",
    ]
    consonants += [c.capitalize() for c in consonants] + [c.upper() for c in consonants]
    consonants_pattern = "|".join(sorted(consonants, key=len, reverse=True))

    # Define valid terminals
    terminals = ["ch", "ng", "nh", "c", "m", "n", "p", "t", "g"]
    terminals += [t.upper() for t in terminals]
    terminals_pattern = "|".join(sorted(terminals, key=len, reverse=True))

    pattern_c_v = re.compile(
        rf"(?<![{L_chars}])({consonants_pattern})\s+(([{V_chars}])[{L_chars}]*)"
    )
    pattern_v_t = re.compile(rf"\b([{L_chars}]*[{V_chars}])\s+({terminals_pattern})\b")

    def repl_v_t(match):
        w1 = match.group(1)
        t = match.group(2)
        # Enforce Case Compatibility Constraints
        is_compat = (
            (w1[-1].islower() and t.islower())
            or (w1[-1].isupper() and t.isupper())
            or (w1[-1].isupper() and t.islower())
        )
        if not is_compat:
            return match.group(0)

        merged = w1 + t
        if is_valid_word(merged):
            return merged
        return match.group(0)

    def repl_c_v(match):
        c = match.group(1)
        w2 = match.group(2)
        v = match.group(3)
        # Enforce Case Compatibility Constraints
        is_compat = (
            (c.islower() and v.islower())
            or (c.isupper() and v.isupper())
            or (c.isupper() and v.islower())
        )
        if not is_compat:
            return match.group(0)

        # Check if the single uppercase letter is preceded by a lowercase letter
        if c.isupper() and len(c) == 1:
            start_idx = match.start()
            prev_letter = ""
            for idx in range(start_idx - 1, -1, -1):
                char = repaired[idx]
                if char.isalpha():
                    prev_letter = char
                    break
                elif char not in " \t\n\r":
                    break
            if prev_letter.islower():
                return match.group(0)

        merged = c + w2
        if is_valid_word(merged):
            return merged
        return match.group(0)

    for _ in range(3):
        # 2. Merge consonant followed by vowel
        repaired = pattern_c_v.sub(repl_c_v, repaired)
        # 4. Merge vowel followed by terminal
        repaired = pattern_v_t.sub(repl_v_t, repaired)

    return repaired


def repair_split_syllables(text: str) -> str:
    PREFIXES = {
        "b",
        "c",
        "ch",
        "d",
        "đ",
        "g",
        "gi",
        "h",
        "k",
        "kh",
        "l",
        "m",
        "n",
        "nh",
        "p",
        "ph",
        "r",
        "s",
        "t",
        "th",
        "tr",
        "v",
        "x",
    }
    SUFFIXES = {"i", "y", "c", "ch", "m", "n", "ng", "nh", "p", "t"}
    _exclusions = {
        ("b", "ở"),
        ("d", "ở"),
        ("c", "ở"),
        ("t", "ở"),
        ("g", "ở"),
        ("v", "ở"),
        ("r", "a"),
        ("t", "y"),
        ("v", "i"),
        ("x", "a"),
        ("ph", "ở"),
        ("b", "ít"),
        ("c", "ổn"),
        ("t", "ức"),
        ("b", "u"),
        ("c", "ho"),
        ("c", "hủ"),
        ("v", "y"),
        ("pha", "i"),
        ("pha", "ii"),
        ("pha", "iii"),
        ("pha", "iv"),
        ("pha", "v"),
        ("pha", "i."),
        ("pha", "ii."),
        ("pha", "iii."),
        ("pha", "iv."),
        ("pha", "v."),
    }

    # Match all words, spaces, or single characters
    token_pat = re.compile(
        r"([a-zA-ZàáảãạăắằẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđĐ]+)|(\s+)|(.)"
    )
    tokens = [m.group(0) for m in token_pat.finditer(text)]

    # We define helper checks for string types
    word_pat = re.compile(
        r"^[a-zA-ZàáảãạăắằẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđĐ]+$"
    )
    space_pat = re.compile(r"^\s+$")

    # Run two passes to catch nested cases (still O(N))
    for _ in range(2):
        stack = []
        for tok in tokens:
            if (
                stack
                and len(stack) >= 2
                and word_pat.match(tok)
                and space_pat.match(stack[-1])
                and word_pat.match(stack[-2])
            ):
                w1 = stack[-2]
                stack[-1]
                w2 = tok
                w1_low, w2_low = w1.lower(), w2.lower()

                is_valid_merge = False
                if len(w2) != 1 or not w2.isupper():
                    combined = w1_low + w2_low
                    combined_norm = _normalize_tone(combined)
                    if (
                        combined_norm in NORM_SYLLABLES_DB
                        and (w1_low, w2_low) not in _exclusions
                    ):
                        w1_norm = _normalize_tone(w1_low)
                        w2_norm = _normalize_tone(w2_low)
                        w1_is_word = w1_norm in NORM_SYLLABLES_DB
                        w2_is_word = w2_norm in NORM_SYLLABLES_DB
                        if (
                            not w1_is_word
                            or not w2_is_word
                            or w1_low in PREFIXES
                            or w2_low in SUFFIXES
                        ):
                            is_valid_merge = True

                if is_valid_merge:
                    stack.pop()  # remove space
                    stack.pop()  # remove w1
                    stack.append(w1 + w2)
                else:
                    stack.append(tok)
            else:
                stack.append(tok)
        tokens = stack

    return "".join(tokens)


def split_stuck_numbers_and_words(text: str) -> str:
    V_chars = "aàáảãạăắằẳẵặâấầẩẫậeèéẻẽẹêềếểễệiíìỉĩịoóòỏõọôồốổỗộơớờởỡợuúùủũụưứừửữựyýỳỷỹỵAÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬEÈÉẺẼẸÊẾỀỂỄỆIÍÌỈĨỊOÓÒỎÕỌÔỐỔỔỖỘƠỚỜỞỠỢUÚÙỦŨỤƯỨỪỬỮỰYÝỲỶỸỴ"
    L_chars = V_chars + "bcdđfghjklmnpqrstvwxzBCDĐFGHJKLMNPQRSTVWXZ"

    def repl(match):
        num = match.group(1)
        word = match.group(2)
        word_low = word.lower()
        if word_low in NORM_SYLLABLES_DB or word_low in {
            "đvqt",
            "đv",
            "viên",
            "giờ",
            "lần",
            "ngày",
            "tháng",
            "năm",
            "tuổi",
        }:
            return num + " " + word
        return match.group(0)

    return re.sub(rf"(\d+)([{L_chars}]+)", repl, text)


def clean_markdown_text(text: str) -> str:
    import unicodedata

    nfc_text = unicodedata.normalize("NFC", text)
    normalized = nfc_text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("Ð", "Đ")
    normalized = re.sub(r"\bày\b", "ngày", normalized)
    normalized = re.sub(r"\bÀy\b", "Ngày", normalized)

    # Translate specific PUA characters to standard symbols
    pua_replacements = {
        "\uf0b3": "≥",
        "\uf0a3": "≤",
        "\uf061": "α",
        "\uf062": "β",
        "\uf067": "γ",
        "\uf0d2": "®",
        "\uf031": "1",
    }
    for bad, good in pua_replacements.items():
        normalized = normalized.replace(bad, good)

    # Clean duplicate words
    normalized = re.sub(r"\bcác\s+các\b", "các", normalized)

    # Strip DTQGVN header/footer marks (e.g. DTQGVN 2 99)
    normalized = re.sub(r"\bDTQGVN(?:\s+\d+)*\b", "", normalized)
    # Clean up double dots resulting from stripped headers/footers
    normalized = normalized.replace(". .", ".")

    # Fix dính chữ hệ thống (e.g. Cở, Cít, Cổn, Cuống, VZVở, 50Sở, MAO-Bở, HDLvà, Fcở)
    normalized = re.sub(r"\bCở\b", "C ở", normalized)
    normalized = re.sub(r"\bCít\b", "C ít", normalized)
    normalized = re.sub(r"\bCổn\b", "C ổn", normalized)
    normalized = re.sub(r"\bvitamin\s+Cuống\b", "vitamin C uống", normalized)
    normalized = re.sub(r"\bFcở\b", "Fc ở", normalized)
    normalized = re.sub(r"\bHDLvà\b", "HDL và", normalized)
    normalized = re.sub(r"\bVZVở\b", "VZV ở", normalized)
    normalized = re.sub(r"50S\s*ở\b", "50S ở", normalized)  # normalize existing ok form
    normalized = re.sub(r"50Sở", "50S ở", normalized)
    normalized = re.sub(
        r"MAO-B\s*ở\b", "MAO-B ở", normalized
    )  # normalize existing ok form
    normalized = re.sub(r"MAO-Bở", "MAO-B ở", normalized)
    normalized = re.sub(r"\bDuống\b", "D uống", normalized)
    normalized = re.sub(r"\bCapecitabincó\b", "Capecitabin có", normalized)
    normalized = re.sub(r"\bCiprofloxacincó\b", "Ciprofloxacin có", normalized)
    normalized = re.sub(r"\bcefpodoximlà\b", "cefpodoxim là", normalized)
    normalized = re.sub(r"\bchủngnhạy\b", "chủng nhạy", normalized)
    normalized = re.sub(r"\bdượcđộng\b", "dược động", normalized)
    normalized = re.sub(r"\bkhángdopamin\b", "kháng dopamin", normalized)
    normalized = re.sub(
        r"\bKlebsiellapneumoniae\b", "Klebsiella pneumoniae", normalized
    )
    normalized = re.sub(r"\bProteusmirabilis\b", "Proteus mirabilis", normalized)
    normalized = re.sub(r"\bSalmonellatyphi\b", "Salmonella typhi", normalized)
    normalized = re.sub(
        r"\bStreptococcuspneumoniae\b", "Streptococcus pneumoniae", normalized
    )
    normalized = re.sub(
        r"\btrimethoprimsulfamethoxazol\b", "trimethoprim sulfamethoxazol", normalized
    )
    normalized = re.sub(r"\bJarischHerxheimer\b", "Jarisch-Herxheimer", normalized)
    normalized = re.sub(r"\bHenochSchonlein\b", "Henoch-Schönlein", normalized)
    normalized = re.sub(
        r"\bWolffParkinson-White\b", "Wolff-Parkinson-White", normalized
    )
    normalized = re.sub(r"\bPfA\s+TP6\b", "PfATP6", normalized)
    # Fix more dính chữ
    normalized = re.sub(r"\bgiantác\b", "gian tác", normalized)
    normalized = re.sub(r"\bVắcxin\b", "Vắc xin", normalized)
    normalized = re.sub(r"thưòng", "thường", normalized)

    # Fix space-split syllables (consonant splits, vowel splits, multi-space splits)
    normalized = repair_split_syllables(normalized)

    # Fix missing spaces after punctuation (comma, period)
    V_ALL = "a-zA-ZàáảãạăằắẳẵặâấầẩẫậeèéẻẽẹêếềểễệiíìỉĩịoóòỏõọôốồổỗộơớờởỡợuúùủũụưứừửữựyýỳỷỹỵAÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬEÈÉẺẼẸÊẾỀỂỄỆIÍÌỈĨỊOÓÒỎÕỌÔỐỔỔỖỘƠỚỜỞỠỢUÚÙỦŨỤƯỨỪỬỮỰYÝỲỶỸỴĐđ"

    def repl_period(match):
        full = match.group(0)
        prefix = match.group(1).lower()
        suffix = match.group(2).lower()
        abbreviations = {
            "e.coli",
            "e.cloacae",
            "m.bovis",
            "m.tuberculosis",
            "m.kansasii",
            "m.avium",
            "p.aeruginosa",
            "s.aureus",
            "s.pneumoniae",
            "h.influenzae",
            "n.meningitidis",
            "c.difficile",
            "staph.aureus",
            "h.ducreyi",
            "c.freundii",
            "s.agalactiae",
            "m.pneumoniae",
            "c.diphtheriae",
            "s.viridans",
            "n.gonorrhoeae",
            "v.v",
            "i.v",
            "i.m",
            "e.g",
            "i.e",
            "a.m",
            "p.m",
            "a.h",
            "g.p",
            "co.ltd",
            "ph.d",
            "b.s",
            "y.p",
            "a.t.p",
            "s.r",
            "f.c",
            "i.o",
            "i.u",
            "y.y",
            "op.razol",
            "tv.fenofibrat",
            "tv.mephenesin",
            "mg.l",
            "l.h",
            "ml.min",
        }
        if full.lower() in abbreviations:
            return full
        if (len(prefix) == 1 and match.group(1).islower()) or (
            len(suffix) == 1 and match.group(2).islower()
        ):
            return full
        return match.group(1) + ". " + match.group(2)

    def repl_comma(match):
        full = match.group(0)
        if full.lower() in {"n,n", "o,p", "p,p"}:
            return full
        return match.group(1) + ", " + match.group(2)

    normalized = re.sub(rf"\b([{V_ALL}]+)\.([{V_ALL}]+)\b", repl_period, normalized)
    normalized = re.sub(rf"\b([{V_ALL}]+),([{V_ALL}]+)\b", repl_comma, normalized)

    # Pre-clean known OCR double-spacing or merging errors
    normalized = normalized.replace("khôngmong muốn", "không mong muốn")
    normalized = normalized.replace("nhómhuyết thanh", "nhóm huyết thanh")
    normalized = normalized.replace("nhómhuyết-thanh", "nhóm huyết thanh")
    normalized = normalized.replace("cơcủa", "cơ của")
    normalized = normalized.replace("tínhvà/hoặc", "tính và/hoặc")
    normalized = normalized.replace("l  ượng", "lượng")
    normalized = normalized.replace("D  ược", "Dược")
    normalized = normalized.replace("th  ượng", "thượng")

    # Split stuck acronyms
    normalized = re.sub(
        r"(?<=[a-zà-ỹ])(ALT|MAC|O|O2|NVP|EFV|TDF|LPV|AZT|CD4|HGB|HIV)\b",
        r" \1",
        normalized,
    )
    normalized = re.sub(
        r"\b(ALT|MAC|O|O2|NVP|EFV|TDF|LPV|AZT|CD4|HGB|HIV)(?=[a-zà-ỹ])",
        r"\1 ",
        normalized,
    )

    normalized = repair_spacing(normalized)

    # Post-repair fixes: patterns that repair_spacing may re-merge
    normalized = re.sub(r"50Sở", "50S ở", normalized)
    normalized = re.sub(r"MAO-Bở", "MAO-B ở", normalized)
    normalized = re.sub(r"\bgiantác\b", "gian tác", normalized)
    normalized = re.sub(r"thưòng", "thường", normalized)
    # OCR: spurious 'c' inserted before 'ở' after 'thuốc'
    normalized = re.sub(r"\bthuốc\s+cở\b", "thuốc ở", normalized)
    normalized = re.sub(r"\bthu ố cở\b", "thuốc ở", normalized)

    # Fix split ch characters (e.g. C hỉ -> Chỉ)
    V_chars = "áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ"
    normalized = re.sub(rf"\b([Cc])\s+h([{V_chars}])", r"\1h\2", normalized)

    # Fix merged ## headers (insert newline before ## if it's attached to the end of a line)
    normalized = re.sub(r"([^\n])##\s+", r"\1\n## ", normalized)

    # Normalize oC/o C/OC/O C to °C
    normalized = re.sub(r"(\d+)\s*[oO]\s*C\b", r"\1 °C", normalized)
    normalized = re.sub(r"(\d+)\s*[oO]\n\s*C\b", r"\1 °C", normalized)
    # Normalize unbalanced number ranges like 1- 2 or 1 -2 to 1 - 2
    normalized = re.sub(r"(\b\d+(?:,\d+)?)-\s+(\d+(?:,\d+)?\b)", r"\1 - \2", normalized)
    normalized = re.sub(r"(\b\d+(?:,\d+)?)\s+-(\d+(?:,\d+)?\b)", r"\1 - \2", normalized)
    # Split stuck numbers and words
    normalized = split_stuck_numbers_and_words(normalized)
    # Normalize split words
    for bad, good in SPLIT_WORDS_REPAIRS.items():
        normalized = normalized.replace(bad, good)

    raw_lines = normalized.split("\n")
    cleaned_lines = [clean_markdown_line(line) for line in raw_lines]

    # Quét hậu xử lý gộp tiêu đề xếp chồng (stacked headings)
    merged_lines = []
    i = 0
    while i < len(cleaned_lines):
        line = cleaned_lines[i]
        stripped = line.strip()
        # Nếu dòng hiện tại là tiêu đề
        if stripped.startswith("## ") and not stripped.startswith("## ("):
            # Tìm dòng tiêu đề ngoặc đơn tiếp theo (cho phép tối đa 1 dòng trống ở giữa)
            next_non_empty_idx = i + 1
            if (
                next_non_empty_idx < len(cleaned_lines)
                and cleaned_lines[next_non_empty_idx].strip() == ""
            ):
                next_non_empty_idx += 1

            if next_non_empty_idx < len(cleaned_lines):
                next_line = cleaned_lines[next_non_empty_idx].strip()
                if next_line.startswith("## (") and next_line.endswith(")"):
                    synonym = next_line[3:].strip()  # Lấy phần ngoặc đơn "(Synonym)"
                    line = f"{line} {synonym}"
                    i = next_non_empty_idx + 1
                    merged_lines.append(line)
                    continue
        merged_lines.append(line)
        i += 1

    return "\n".join(merged_lines)


def process_corpus(*, input_path: Path, output_path: Path) -> Path:
    text = input_path.read_text(encoding="utf-8")
    cleaned = clean_markdown_text(text)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(cleaned, encoding="utf-8")
    return output_path
