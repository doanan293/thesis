# SoICT 2026 Paper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Viết paper LNCS tiếng Anh trong `report/` cho SoICT 2026, build được ngay từ đầu, có placeholder cho kết quả E2E và rerank 8B chưa xong, và hoàn chỉnh trước 20/09/2026.

**Architecture:** Một project LaTeX (`report/main.tex` + mỗi mục một file trong `sections/`), build bằng script của `latex-document-skill` với LuaLaTeX. Số retrieval lấy từ `report.md` đã commit; bảng E2E do `pharma-lab e2e report` sinh ra ở dạng dùng thẳng cho paper, rồi copy vào `report/tables/`. Kết quả chưa có được đánh dấu `\pending{}`; cờ `\finaltrue` biến mọi `\pending` còn sót thành lỗi build.

**Tech Stack:** LuaLaTeX, `llncs.cls` + `splncs04.bst` (CTAN, do Springer duy trì), fontspec (Latin Modern), booktabs, TikZ, BibTeX; Python 3.12 + pytest cho thay đổi ở `pharma_lab.e2e.report`.

**Spec:** `docs/superpowers/specs/2026-09-17-soict-paper-design.md`

## Global Constraints

- Tiếng Anh; LNCS; tối đa 12 trang nội dung, references không tính; không đánh số trang; single-blind, tác giả để placeholder.
- Mục: Introduction, Related Work, Research Gap, Methodology, Experimental Results, Discussion, Conclusion. Không có mục "System", không số La Mã, không đặt tên hệ thống ("our agentic RAG pipeline").
- Tên cấu hình trong bài: *Full agent*, *One-step RAG*, *w/o judge--refine*, *w/o rephrase*, *w/o rerank*.
- Mọi việc LaTeX: nạp skill `latex-document-skill` (Skill tool) trước khi làm; văn xuôi là mặc định, bullet chỉ cho danh sách thật sự song song; hình rộng tối đa `0.85\textwidth` (sơ đồ pipeline được phép `\textwidth`); float `[tb]`; không `\newpage`; escape `<`, `>`, `%`, `_`.
- Không viết số nào không có trong file kết quả; không tạo trích dẫn từ trí nhớ; mọi mục bib được kiểm từ DOI/arXiv/DBLP.
- Git: chỉ stage đường dẫn cụ thể (`report/...`, `docs/superpowers/...`, file pharma-lab được nêu), không `git add -A`, không `--no-verify`. Commit message kết thúc bằng `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`.
- Không sửa lint/type bằng suppress; chạy `uv run pytest` của pharma-lab trước và sau khi sửa code.

**Build check** (dùng ở mọi task LaTeX, chạy từ thư mục gốc repo):

```bash
SCRATCH=/tmp/claude-1000/-home-andv-personal-thesis/0911b6d2-6dd9-41f4-ba18-f32abbe67b9a/scratchpad
LOG="$SCRATCH/paper-build.log"
bash .agents/skills/latex-document-skill/scripts/compile_latex.sh report/main.tex \
  --engine lualatex --use-latexmk --verbose --preview --preview-dir "$PWD/report/preview" \
  > "$LOG" 2>&1; echo "exit=$?"
LAST=$(grep -n "^Running 'lualatex" "$LOG" | tail -1 | cut -d: -f1)
grep -nE "^! |LaTeX Error:|Undefined control sequence" "$LOG" | head
tail -n +"$LAST" "$LOG" | grep -E "Citation .* undefined|Reference .* undefined|There were undefined" | sort -u
tail -n +"$LAST" "$LOG" | grep -c 'Overfull \\hbox'
pdfinfo report/main.pdf | grep Pages
```

`--preview-dir` phải là đường dẫn tuyệt đối vì script `cd` vào `report/`. Cảnh báo undefined chỉ xét ở lượt lualatex cuối (các lượt đầu luôn có). Script này được lưu ở `$SCRATCH/paper-check.sh`.

Kỳ vọng: `exit=0`, hai lệnh grep đầu rỗng (trừ khi task ghi rõ ngoại lệ), số Overfull bằng 0, rồi xem PNG trong `report/preview/` bằng Read.

---

### Task 1: Nhánh và khung LaTeX

**Files:**
- Create: `report/main.tex`, `report/llncs.cls`, `report/splncs04.bst`, `report/.gitignore`, `report/README.md`, `report/references.bib`
- Create: `report/sections/{abstract,introduction,related-work,research-gap,methodology,results,discussion,conclusion}.tex`
- Create: `report/tables/.gitkeep`, `report/figures/.gitkeep`
- Commit thêm: `docs/superpowers/specs/2026-09-17-soict-paper-design.md`, `docs/superpowers/plans/2026-09-17-soict-paper.md`

**Interfaces:**
- Produces: macro `\pending{text}`, cờ `\finaltrue`/`\finalfalse`; `\input{sections/<name>}`; thư mục `tables/`, `figures/`; bib `references.bib` với `\bibliographystyle{splncs04}`.

- [ ] **Step 1: Nạp skill** `latex-document-skill` bằng Skill tool; đọc `references/long-form-best-practices.md`. Bước hỏi enrichment đã trả lời trong spec §4.

- [ ] **Step 2: Làm trên `dev`.** Checkout chính ở `dev`, `dev` không thiếu commit nào của `main` (`git rev-list --left-right --count origin/main...origin/dev` có vế trái bằng 0). Không tạo nhánh hay worktree riêng.

- [ ] **Step 3: Lấy template Springer**

```bash
SCRATCH=/tmp/claude-1000/-home-andv-personal-thesis/0911b6d2-6dd9-41f4-ba18-f32abbe67b9a/scratchpad
curl -fsSL -o "$SCRATCH/llncs.zip" https://mirrors.ctan.org/macros/latex/contrib/llncs.zip
unzip -o -q "$SCRATCH/llncs.zip" -d "$SCRATCH/llncs"
find "$SCRATCH/llncs" -name 'llncs.cls' -o -name 'splncs04.bst' -o -name 'samplepaper.tex'
cp "$(find "$SCRATCH/llncs" -name llncs.cls)" "$(find "$SCRATCH/llncs" -name splncs04.bst)" report/
head -20 report/llncs.cls   # ghi lại version vào README
```

Đọc `samplepaper.tex` để giữ đúng cách viết `\title`, `\author`, `\institute`, `\orcidID`, `\keywords`.

- [ ] **Step 4: Viết `report/main.tex`**

```latex
\documentclass[runningheads]{llncs}
\usepackage{fontspec}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{xcolor}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,fit,calc}
\usepackage{enumitem}
\setlist[itemize]{nosep, leftmargin=*, topsep=2pt, partopsep=0pt}
\setlist[enumerate]{nosep, leftmargin=*, topsep=2pt, partopsep=0pt}
\usepackage{hyperref}
\renewcommand\UrlFont{\color{blue}\rmfamily}
\urlstyle{rm}

% \finaltrue turns every remaining placeholder into a build error.
\newif\iffinal
\finalfalse
\newcommand{\pending}[1]{%
  \iffinal\PackageError{paper}{Pending result: #1}{Fill in the result.}%
  \else\textcolor{red}{[#1]}\fi}

\begin{document}
\title{Benchmarking Retrieval and Agentic RAG for Vietnamese Drug Question Answering with Low-Cost LLMs}
\titlerunning{Retrieval and Agentic RAG for Vietnamese Drug QA}
\author{First Author\inst{1}\orcidID{0000-0000-0000-0000} \and
Second Author\inst{1}\orcidID{0000-0000-0000-0000}}
\authorrunning{F. Author and S. Author}
\institute{Affiliation, City, Country\\
\email{first.author@example.com}}
\maketitle

\input{sections/abstract}
\input{sections/introduction}
\input{sections/related-work}
\input{sections/research-gap}
\input{sections/methodology}
\input{sections/results}
\input{sections/discussion}
\input{sections/conclusion}

\bibliographystyle{splncs04}
\bibliography{references}
\end{document}
```

Tên bài là bản nháp; tác giả là placeholder người dùng tự điền.

- [ ] **Step 5: File mục nháp.** `sections/abstract.tex`:

```latex
\begin{abstract}
\pending{abstract}
\keywords{Retrieval-augmented generation \and Drug question answering \and Vietnamese \and Retrieval benchmark \and LLM-as-a-judge}
\end{abstract}
```

Mỗi file còn lại gồm đúng tiêu đề và một dòng placeholder, ví dụ `sections/introduction.tex`:

```latex
\section{Introduction}\label{sec:introduction}
\pending{introduction}
```

Tiêu đề và label: `Related Work`/`sec:related`, `Research Gap`/`sec:gap`, `Methodology`/`sec:method`, `Experimental Results`/`sec:results`, `Discussion`/`sec:discussion`, `Conclusion`/`sec:conclusion`. `references.bib` ban đầu gồm một mục thật để BibTeX chạy được:

```bibtex
@inproceedings{lewis2020rag,
  title     = {Retrieval-Augmented Generation for Knowledge-Intensive {NLP} Tasks},
  author    = {Lewis, Patrick and Perez, Ethan and Piktus, Aleksandra and Petroni, Fabio and Karpukhin, Vladimir and Goyal, Naman and K{\"u}ttler, Heinrich and Lewis, Mike and Yih, Wen-tau and Rockt{\"a}schel, Tim and Riedel, Sebastian and Kiela, Douwe},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {33},
  pages     = {9459--9474},
  year      = {2020}
}
```

Kiểm lại mục này trên DBLP ở Task 3; trong introduction tạm thêm `\cite{lewis2020rag}` để bibliography không rỗng.

- [ ] **Step 6: `report/.gitignore`, `README.md`**

```gitignore
*.aux
*.bbl
*.blg
*.fdb_latexmk
*.fls
*.log
*.out
*.synctex.gz
main.pdf
preview/
```

README (tiếng Việt): mục đích, nguồn template (CTAN `llncs`, version ở Step 3), lệnh build (Build check dòng `compile_latex.sh`), cách bật `\finaltrue`, cách cập nhật bảng E2E (Task 10 Step 1).

- [ ] **Step 7: Chạy Build check.** Kỳ vọng `exit=0`, PDF 2 trang, placeholder đỏ, không lỗi. Xem `report/preview/*.png`.

- [ ] **Step 8: Kiểm tiếng Việt.** Thêm tạm câu `Thuốc hạ sốt Paracetamol` vào introduction, build, chạy `pdftotext report/main.pdf - | grep -c 'Thuốc hạ sốt'` → `1`; xóa câu tạm.

- [ ] **Step 9: Commit**

```bash
git add report/main.tex report/llncs.cls report/splncs04.bst report/.gitignore report/README.md \
  report/references.bib report/sections report/tables/.gitkeep report/figures/.gitkeep \
  docs/superpowers/specs/2026-09-17-soict-paper-design.md docs/superpowers/plans/2026-09-17-soict-paper.md
git status --short report docs/superpowers   # không có file sinh ra
git commit -m "docs(paper): scaffold the SoICT 2026 LNCS paper"
```

---

### Task 2: Bảng E2E dùng thẳng cho paper

**Files:**
- Modify: `pharma-lab/src/pharma_lab/e2e/report.py` (`_cell`, `_write_tex`, `_main_tables`, `_ablation_tables`, `_calibration_tables`)
- Test: `pharma-lab/tests/e2e/test_report.py`

**Interfaces:**
- Produces: `METRIC_LABELS: Mapping[str, str]`, `CONFIG_LABELS: Mapping[str, str]`, `PAPER_MAIN_CONFIGS = ("full", "one-step")`; `_write_tex(path, caption, label, header, rows, align)`. File `main.tex` (label `tab:e2e-main`), `ablation.tex` (`tab:e2e-ablation`), `calibration.tex` (`tab:e2e-calibration`) dùng booktabs, `\small`, tên dễ đọc. CSV không đổi.

- [ ] **Step 1: Chạy test hiện tại**

Run: `cd pharma-lab && uv run pytest -q` → ghi lại số test pass.

- [ ] **Step 2: Viết test thất bại** — thay đoạn kiểm `main.tex` cuối `test_report.py` (sau `groups = ...`) bằng:

```python
    tex = (reports / "main.tex").read_text("utf-8")
    assert r"\label{tab:e2e-main}" in tex
    assert r"\toprule" in tex and r"\hline" not in tex
    assert "Nugget recall" in tex and "Truthfulness" in tex
    assert "Full agent & One-step RAG" in tex
    assert "latency" not in tex.lower()  # latency is secondary: CSV only
    assert "0.750 [" in tex
    calls_row = next(line for line in tex.splitlines() if line.startswith("LLM calls"))
    assert calls_row.startswith("LLM calls / turn & 4.0 [")
    ablation_tex = (reports / "ablation.tex").read_text("utf-8")
    assert r"\label{tab:e2e-ablation}" in ablation_tex
    assert "One-step RAG" in ablation_tex
    nugget_row = next(
        line for line in ablation_tex.splitlines() if line.startswith("Nugget recall")
    )
    assert nugget_row.startswith(r"Nugget recall & $-$0.333")
    assert "[" not in nugget_row  # CIs stay in ablation.csv
    calibration_tex = (reports / "calibration.tex").read_text("utf-8")
    assert r"\label{tab:e2e-calibration}" in calibration_tex
    assert "--" in calibration_tex
```

Và thêm test đơn vị:

```python
def test_every_primary_metric_and_config_has_a_paper_label() -> None:
    assert {m.name for m in METRICS if m.primary} <= set(METRIC_LABELS)
    assert {c.value for c in E2EConfig} == set(CONFIG_LABELS)
```

(import `METRICS`, `METRIC_LABELS`, `CONFIG_LABELS` từ `pharma_lab.e2e.report`, `E2EConfig` từ `pharma_lab.e2e.configs`).

- [ ] **Step 3: Chạy để thấy fail**

Run: `cd pharma-lab && uv run pytest tests/e2e/test_report.py -q` → FAIL (ImportError `METRIC_LABELS`).

- [ ] **Step 4: Cài đặt** trong `report.py`:

```python
METRIC_LABELS: Mapping[str, str] = {
    "truthfulness": "Truthfulness",
    "perfect_rate": "Perfect",
    "missing_rate": "Missing",
    "hallucination_rate": "Hallucination",
    "nugget_recall": "Nugget recall",
    "severe_harm_rate": "Severe harm",
    "faithfulness": "Faithfulness",
    "citation_recall": "Citation recall",
    "response_rate": "Response rate",
    "negative_rejection_rate": "Negative rejection",
    "redirect_accuracy": "Redirect accuracy",
    "injection_defence_rate": "Injection defence",
    "tokens_per_turn": "Tokens / turn",
    "llm_calls_per_turn": "LLM calls / turn",
}
CONFIG_LABELS: Mapping[str, str] = {
    E2EConfig.FULL.value: "Full agent",
    E2EConfig.ONE_STEP.value: "One-step RAG",
    E2EConfig.NO_JUDGE_REFINE.value: "w/o judge--refine",
    E2EConfig.NO_REPHRASE.value: "w/o rephrase",
    E2EConfig.NO_RERANK.value: "w/o rerank",
}
# The LNCS text block is 12.2 cm wide: the main table compares the two systems,
# and the ablation table carries the remaining configurations as deltas.
PAPER_MAIN_CONFIGS: tuple[str, ...] = (E2EConfig.FULL.value, E2EConfig.ONE_STEP.value)


def _number(value: float) -> str:
    text = (
        f"{value:.0f}"
        if abs(value) >= 100
        else (f"{value:.1f}" if abs(value) >= 1 else f"{value:.3f}")
    )
    return text.replace("-", "$-$", 1) if text.startswith("-") else text


def _cell(summary: Summary) -> str:
    if summary.n == 0:
        return "--"
    return (
        f"{_number(summary.mean)} "
        f"[{_number(summary.ci_low)}, {_number(summary.ci_high)}]"
    )


def _write_tex(
    path: Path,
    caption: str,
    label: str,
    header: Sequence[str],
    rows: Sequence[Sequence[str]],
) -> None:
    lines = [
        r"\begin{table}[tb]",
        r"\centering",
        rf"\caption{{{caption}}}\label{{{label}}}",
        r"\small",
        r"\begin{tabular}{l" + "r" * (len(header) - 1) + "}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
        *(" & ".join(row) + r" \\" for row in rows),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
```

Nếu `_number(1.0)` cho `1.0` và mean nhỏ hơn 1 cho 3 chữ số thì `LLM calls / turn & 4.0 [` và `0.750 [` khớp test. Caption viết sẵn ở dạng LaTeX (`95\%`, `$p<0.05$`) nên bỏ `_latex()` cho caption/header; giữ `_latex` cho giá trị lấy từ dữ liệu (tên metric trong calibration). Trong `_main_tables`: `tex_row = [METRIC_LABELS[metric.name]]`, chỉ lặp `PAPER_MAIN_CONFIGS` (lọc theo `configs` có mặt) khi dựng `tex_row`, CSV vẫn lặp mọi config; header `["Metric", *(CONFIG_LABELS[c] for c in main_configs)]`; caption `"End-to-end results of the full agent and one-step RAG: mean [95\\% CI] (Wilson for rates, percentile bootstrap otherwise)."`, label `tab:e2e-main`. Trong `_ablation_tables`: ô tex là `f"{_number(s.mean)}{mark}"` (hoặc `--` khi `s.n == 0`), `mark = "$^{*}$"`; header dùng `CONFIG_LABELS`; caption `"Paired change of each configuration against the full agent. $^{*}$: Holm-adjusted randomization $p<0.05$; confidence intervals are in the released CSV."`, label `tab:e2e-ablation`. Trong `_calibration_tables`: header `["Label", "Statistic", "$n$", "Value [95\\% CI]"]`, ô metric/statistic qua `_latex`, label `tab:e2e-calibration`, caption `"Agreement between the LLM judge and the blind calibration grader."`. Các lời gọi `_write_tex` khác (nếu có) truyền label tương ứng.

- [ ] **Step 5: Chạy test**

Run: `cd pharma-lab && uv run pytest tests/e2e/test_report.py -q` → PASS; rồi `uv run pytest -q` → cùng số pass như Step 1 cộng test mới.
Run (từ gốc repo): `uv run ruff check pharma-lab && uv run ruff format --check pharma-lab && uv run --directory pharma-lab pyrefly check` → sạch.

- [ ] **Step 6: Commit**

```bash
git add pharma-lab/src/pharma_lab/e2e/report.py pharma-lab/tests/e2e/test_report.py
git commit -m "feat(lab): emit paper-ready E2E LaTeX tables"
```

---

### Task 3: References đã kiểm chứng

**Files:**
- Modify: `report/references.bib`

**Interfaces:**
- Produces: các key bib dưới đây, dùng ở Task 4–11.

- [ ] **Step 1: Lấy BibTeX.** Với DOI: `bash .agents/skills/latex-document-skill/scripts/fetch_bibtex.sh <DOI>`. Với arXiv/DBLP: `curl -fsSL https://dblp.org/search/publ/api?q=<title>&format=bib1&h=1` hoặc `curl -fsSL https://arxiv.org/bibtex/<id>`. So tiêu đề trả về với tiêu đề cần tìm; ưu tiên bản đã xuất bản (DBLP) hơn arXiv. Key dạng `<họ tác giả đầu><năm><từ khóa>`.

| Key | Công trình | Định danh gợi ý (phải kiểm) |
| --- | --- | --- |
| `lewis2020rag` | Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | arXiv 2005.11401 / NeurIPS 2020 |
| `asai2024selfrag` | Self-RAG | arXiv 2310.11511 (ICLR 2024) |
| `yan2024crag` | Corrective Retrieval Augmented Generation | arXiv 2401.15884 |
| `jeong2024adaptiverag` | Adaptive-RAG | arXiv 2403.14403 (NAACL 2024) |
| `xiong2024medrag` | Benchmarking RAG for Medicine (MedRAG/MIRAGE) | arXiv 2402.13178 (Findings ACL 2024) |
| `mikkelsen2026` | Paper nền, JMIR Medical Informatics 2026 | tra DOI trên jmir.org |
| `robertson2009bm25` | The Probabilistic Relevance Framework: BM25 and Beyond | DOI 10.1561/1500000019 |
| `cormack2009rrf` | Reciprocal Rank Fusion | DOI 10.1145/1571941.1572114 |
| `thakur2021beir` | BEIR | arXiv 2104.08663 (NeurIPS D&B 2021) |
| `muennighoff2023mteb` | MTEB | arXiv 2210.07316 (EACL 2023) |
| `chen2024bgem3` | BGE M3-Embedding | arXiv 2402.03216 |
| `zhang2025qwen3emb` | Qwen3 Embedding (embedding + reranker) | arXiv 2506.05176 |
| `vera2025embeddinggemma` | EmbeddingGemma | tra arXiv |
| `es2024ragas` | RAGAS | arXiv 2309.15217 (EACL 2024 demo) |
| `saadfalcon2024ares` | ARES | arXiv 2311.09476 (NAACL 2024) |
| `gao2023alce` | Enabling LLMs to Generate Text with Citations (ALCE) | arXiv 2305.14627 (EMNLP 2023) |
| `yang2024cragbench` | CRAG – Comprehensive RAG Benchmark | arXiv 2406.04744 (NeurIPS 2024 D&B) |
| `pradeep2024nuggets` | TREC 2024 RAG Track nugget evaluation | tra arXiv |
| `chen2024rgb` | Benchmarking LLMs in RAG (RGB) | arXiv 2309.01431 (AAAI 2024) |
| `debenedetti2024agentdojo` | AgentDojo | arXiv 2406.13352 (NeurIPS 2024 D&B) |
| `chart2025` | CHART reporting guideline cho chatbot sức khỏe | tra DOI |
| `gallifant2025tripodllm` | TRIPOD-LLM | tra DOI (Nature Medicine 2025) |
| `angelopoulos2023ppi` | Prediction-Powered Inference | DOI 10.1126/science.adi6000 |
| `smucker2007significance` | Comparison of Statistical Significance Tests for IR Evaluation | DOI 10.1145/1321440.1321528 |
| `holm1979` | A Simple Sequentially Rejective Multiple Test Procedure | JSTOR 4615733 |
| `wilson1927` | Probable Inference, the Law of Succession, and Statistical Inference | DOI 10.1080/01621459.1927.10502953 |
| `gwet2008ac1` | Computing inter-rater reliability … high agreement | DOI 10.1348/000711006X126600 |
| `zheng2023judge` | Judging LLM-as-a-Judge with MT-Bench | arXiv 2306.05685 (NeurIPS 2023) |
| `urag2024` | URAG (SoICT 2024) | arXiv 2501.16276 |
| `vi-medical-qa-*` | 2–3 bộ QA/RAG y khoa tiếng Việt (ví dụ ViMQ, ViHealthQA, ViMedAQA) | tìm bằng WebSearch, chỉ giữ bài đã xuất bản |

Mục nào không tìm được bản xác thực thì bỏ và ghi vào báo cáo task, không tự điền.

- [ ] **Step 2: Kiểm file.** Tạm thêm `\nocite{*}` vào `main.tex` trước `\bibliographystyle`, chạy Build check: không lỗi BibTeX (`grep -n "^Warning--\|error message" "$SCRATCH/paper-build.log"` rỗng hoặc chỉ cảnh báo thiếu trường không bắt buộc, sửa nếu có). Xem trang references trong preview: tên riêng có dấu và chữ hoa (`{BM25}`, `{LLM}`) hiển thị đúng. Gỡ `\nocite{*}`.

- [ ] **Step 3: Commit**

```bash
git add report/references.bib
git commit -m "docs(paper): add verified references"
```

---

### Task 4: Fig. 1 — sơ đồ pipeline

**Files:**
- Create: `report/figures/pipeline.tex`
- Modify: `report/sections/introduction.tex`

**Interfaces:**
- Produces: `\label{fig:pipeline}`; `\input{figures/pipeline}` trong introduction.

Nguồn sự thật: `backend/src/pharma_agent/application/chat/` (các node guard, rephrase, search, judge, refine, answer) và `docs/superpowers/specs/2026-09-16-e2e-golden-evaluation-design.md`. Đọc code node trước để sơ đồ đúng các nhánh (blocked, redirect, no_retrieval, grounded, partial, abstain).

- [ ] **Step 1: Viết `figures/pipeline.tex`** dạng `figure*`-không-cần (LNCS một cột) `\begin{figure}[tb]\centering\begin{tikzpicture}[node distance=...]` với các node: *Guard* (regex + LLM) → *Rephrase* → *Hybrid search* (dense Qwen3-Embedding-4B + BM25, RRF $k{=}2$, top 30) → *Rerank* (Qwen3-Reranker-4B, top 8) → *Judge* → (*Refine* quay về search, tối đa 3 vòng) → *Answer* có trích dẫn; nhánh ra từ Guard tới *Blocked / Redirect*. Dưới sơ đồ một dải chú thích "cheap LLM per role". Mỗi node `rounded corners`, chữ `\footnotesize`, rộng tổng không quá `\textwidth`. Caption: "Overview of the agentic RAG pipeline. Dashed boxes are removed in the ablation configurations." Các node rephrase, judge/refine, rerank vẽ nét đứt.
- [ ] **Step 2:** Trong introduction, thay placeholder bằng `\input{figures/pipeline}` + `\pending{introduction}` và câu `Fig.~\ref{fig:pipeline} gives an overview.`
- [ ] **Step 3: Build check**, xem PNG: không mũi tên chồng chữ, không tràn lề. Sửa đến khi sạch.
- [ ] **Step 4: Commit** `git add report/figures/pipeline.tex report/sections/introduction.tex && git commit -m "docs(paper): draw the pipeline figure"`

---

### Task 5: Methodology — dữ liệu, pipeline, bộ đánh giá (4.1–4.3)

**Files:**
- Modify: `report/sections/methodology.tex`
- Create: `report/tables/corpus.tex` (`tab:corpus`), `report/tables/eval-sets.tex` (`tab:eval-sets`)

**Interfaces:**
- Consumes: `fig:pipeline`, key bib Task 3.
- Produces: `\label{sec:corpus}`, `sec:pipeline`, `sec:eval-sets`; các bảng trên.

Nguồn: `git show fc75422:report/report.md` (mục 2–4, 6), `pharma-lab/docs/guides/evaluation.md`, `pharma-lab/docs/guides/e2e-golden-authoring.md`, `pharma-lab/data/evaluation/e2e/golden_e2e.manifest.json`, `backend/.env.example` (model theo role). Đối chiếu mọi số với manifest hiện tại; số nào khác bản report cũ thì theo manifest.

- [ ] **Step 1: 4.1 Corpus** (văn xuôi, khoảng 0,9 trang): hai nguồn (Dược thư Quốc gia Việt Nam, ấn bản 2, 1.668 trang, 689 chuyên luận thuốc; 2.406 tờ hướng dẫn sử dụng thuốc từ 38 nhóm thuốc, mô tả nguồn là "drug package leaflets", không nêu tên website); trích xuất (PyMuPDF, Docling cho bảng, curate bảng nối trang); chia section/chunk (tối đa 3.000 ký tự, bảng tách theo hàng lặp header); enrichment tất định không dùng LLM (context header, 71 thuật ngữ glossary, 2.409 colloquial mapping). Bảng `tables/corpus.tex` (booktabs, `\small`): theo loại tài liệu — section, chunk, ký tự TB/chunk; dòng tổng 3.120 tài liệu, 12.326 section, 24.955 chunk.
- [ ] **Step 2: 4.2 Pipeline** (khoảng 0,8 trang): mô tả từng bước theo Fig.~\ref{fig:pipeline}; lý do thiết kế cho LLM rẻ (mỗi bước một lời gọi nhỏ có structured output; routing, ngân sách và chống lặp nằm trong code); ngân sách mỗi lượt (tối đa 3 vòng search, 10 lời gọi LLM, 40.000 token, 24.000 ký tự evidence); retrieval production (Qdrant dense + BM25 phía server, RRF k=2, prefetch 50, 30 ứng viên, rerank giữ top 8, hydrate section); model theo role (guard/rephrase `gemini-3.1-flash-lite`, answer `gemini-3.8-flash-high` — lấy đúng từ `run.json` của run `e2e-v1`), LangGraph. Công thức RRF:

```latex
\begin{equation}
\mathrm{RRF}(d)=\sum_{r\in R}\frac{1}{k+\mathrm{rank}_r(d)},\qquad k=2.
\end{equation}
```

- [ ] **Step 3: 4.3 Bộ đánh giá** (khoảng 0,6 trang): 10.000 câu retrieval sinh tất định theo 6 nhóm (formulary 5.000, brand\_product\_qa 2.500, chunk\_level 1.000, patient\_natural 500, noisy\_confuser 500, multi\_intent 500; 500 câu cần nhiều section), nhãn section/chunk, cửa sổ 3 chunk; bộ golden E2E 500 câu (420 answerable, 50 multi-turn, 10 unanswerable, 10 out-of-scope, 10 injection) lấy mẫu phân tầng từ bộ 10k, soạn ý chính (nugget) và câu trả lời tham chiếu, đóng băng bằng sha256. Bảng `tables/eval-sets.tex`: hai khối (Retrieval / End-to-end), cột Group, #, What it tests. Tên nhóm hiển thị bằng chữ thường có khoảng trắng (không dấu gạch dưới).
- [ ] **Step 4: Build check**, xem PNG; methodology 4.1–4.3 không quá 2,3 trang.
- [ ] **Step 5: Commit** `git add report/sections/methodology.tex report/tables/corpus.tex report/tables/eval-sets.tex && git commit -m "docs(paper): describe corpus, pipeline and evaluation sets"`

---

### Task 6: Methodology — thiết kế thí nghiệm (4.4)

**Files:**
- Modify: `report/sections/methodology.tex`
- Create: `report/tables/e2e-configs.tex` (`tab:e2e-configs`)

**Interfaces:**
- Produces: `\label{sec:design}`, `eq:ndcg`, `eq:crag`, `tab:e2e-configs`.

Nguồn: bảng metric và mục Thống kê/Hiệu chỉnh của `pharma-lab/docs/guides/e2e-evaluation.md`; `pharma-lab/docs/guides/evaluation.md` (Hit@k, nDCG@10, MRR, Complete-evidence rate); `judge.json` của run (`gemini-3.1-flash-lite`).

- [ ] **Step 1: Retrieval protocol** (văn xuôi): các họ phương pháp (BM25, 6 dense, hybrid RRF k=60 và k=2, 4 reranker trên top 30), đều chạy qua đúng `RetrievalService` của backend trên Postgres/Qdrant; metric kèm công thức nDCG@10 gain nhị phân theo đơn vị evidence:

```latex
\begin{equation}\label{eq:ndcg}
\mathrm{nDCG@}10=\frac{1}{\mathrm{IDCG@}10}\sum_{i=1}^{10}\frac{g_i}{\log_2(i+1)},\quad g_i\in\{0,1\}
\end{equation}
```

- [ ] **Step 2: Cấu hình E2E** — `tables/e2e-configs.tex`: cột Configuration | Rephrase | Judge--refine | Rerank, dùng `\checkmark`/`--`, năm dòng theo Global Constraints. Một câu: One-step RAG dùng cùng retriever, prompt trả lời và ngân sách evidence.
- [ ] **Step 3: Metric E2E** (bold-label paragraphs, không bullet): *Correctness* — CRAG truthfulness với công thức

```latex
\begin{equation}\label{eq:crag}
\mathrm{Truthfulness}=\frac{1}{N}\sum_{j=1}^{N}s_j,\quad s_j=\begin{cases}1&\text{perfect}\\0.5&\text{acceptable}\\0&\text{missing}\\-1&\text{incorrect}\end{cases}
\end{equation}
```

cùng nugget recall (TREC RAG) và severe harm (CHART, TRIPOD-LLM); *Grounding* — faithfulness (RAGAS), citation recall (ALCE); *Behaviour* — response rate, negative rejection (RGB), redirect accuracy, injection defence / attack success (AgentDojo); *Cost* — token và số lời gọi LLM mỗi lượt, latency chỉ tham khảo. Mỗi metric trích dẫn key tương ứng.
- [ ] **Step 4: Judge, hiệu chỉnh, thống kê**: judge `gemini-3.1-flash-lite` khác model trả lời; judge có output cấu trúc từng quyết định nhỏ; 100 câu ngẫu nhiên (50 full, 50 one-step) chấm mù, báo percent agreement, Cohen's κ, Gwet's AC1, Spearman ρ, ngưỡng 0,6; PPI cho metric dựa trên judge; Wilson CI cho tỉ lệ, bootstrap percentile 10.000 lần cho trung bình; ablation dùng hiệu có cặp, kiểm định hoán vị dấu và Holm. Ghi rõ ai chấm hiệu chỉnh đúng theo thực tế (Task 10 Step 2 xác nhận); đến lúc đó để `\pending{calibration grader}`.
- [ ] **Step 5: Build check**; toàn bộ Methodology ≤ 3,5 trang.
- [ ] **Step 6: Commit** `git add report/sections/methodology.tex report/tables/e2e-configs.tex && git commit -m "docs(paper): describe the experimental design"`

---

### Task 7: Related Work và Research Gap

**Files:**
- Modify: `report/sections/related-work.tex`, `report/sections/research-gap.tex`

- [ ] **Step 1: Related Work** (≤ 1,2 trang, bốn đoạn có `\paragraph{}`): *RAG and agentic RAG* (lewis2020rag, asai2024selfrag, yan2024crag, jeong2024adaptiverag); *Medical RAG* (xiong2024medrag, mikkelsen2026 và các bài y khoa khác đã kiểm); *Retrieval benchmarks and models* (robertson2009bm25, cormack2009rrf, thakur2021beir, muennighoff2023mteb, chen2024bgem3, zhang2025qwen3emb, vera2025embeddinggemma); *Evaluating RAG systems* (es2024ragas, saadfalcon2024ares, gao2023alce, yang2024cragbench, pradeep2024nuggets, chen2024rgb, debenedetti2024agentdojo, zheng2023judge, angelopoulos2023ppi); *Vietnamese QA and RAG* (urag2024, các bài y khoa tiếng Việt). Mỗi câu nói điều bài đó làm, không khen chung chung. Chỉ nói điều đã đọc được trong abstract/bài; mở abstract bằng WebFetch nếu chưa chắc.
- [ ] **Step 2: Research Gap** (≤ 0,4 trang): một câu dẫn và danh sách `itemize` 4 ý theo spec §3, mỗi ý nối với một đóng góp của bài.
- [ ] **Step 3: Build check**; không có citation undefined.
- [ ] **Step 4: Commit** `git add report/sections/related-work.tex report/sections/research-gap.tex && git commit -m "docs(paper): write related work and research gap"`

---

### Task 8: Experimental Results — retrieval (5.1)

**Files:**
- Modify: `report/sections/results.tex`
- Create: `report/tables/retrieval.tex` (`tab:retrieval`)

**Interfaces:**
- Produces: `\label{sec:results-retrieval}`, `tab:retrieval`.

- [ ] **Step 1: Lấy số.** Đọc `report.md` của từng run dưới `pharma-lab/data/evaluation/runs/*/reports/**/top30-window3/` (baseline và rerank) và ghi lại Hit@3, Hit@10, nDCG@10, MRR@10, Complete-evidence rate@10 (mục Multi-required). Kiểm `manifest.json` cùng thư mục có `metrics_contract_version` 2; run nào không có (ví dụ `dense-text-embedding-3-large-k30`) thì bỏ khỏi bảng. Throughput reranker (cặp/giây, T4) từ `pharma-lab/data/cache/kaggle_profiles/rerank/*.json`:

```bash
for f in pharma-lab/data/cache/kaggle_profiles/rerank/*.json; do
  jq -r --arg f "$(basename "$f")" '[.measurements[] | select(.status=="ok") | .items/.elapsed_seconds] | max | "\($f) \(.*10|round/10)"' "$f"
done
```

Số chiều embedding: 768 (EmbeddingGemma-300M), 1.024 (BGE-M3, Qwen3-Embedding-0.6B), 2.560 (4B), 4.096 (8B).

- [ ] **Step 2: `tables/retrieval.tex`** — `\small`, booktabs, cột Method | Dim / pairs s$^{-1}$ | Hit@3 | Hit@10 | nDCG@10 | MRR@10 | CER@10; ba khối có `\midrule`: Sparse & dense; Hybrid (qwen3-4B + BM25, RRF $k{=}60$, $k{=}2$); Hybrid $k{=}2$ + reranker (BGE-v2-M3, Qwen3-0.6B, Qwen3-4B, Qwen3-8B). Hit dạng phần trăm một chữ số thập phân, nDCG/MRR ba chữ số; in đậm giá trị tốt nhất mỗi cột. Dòng Qwen3-8B: `\pending{8B}` ở mọi ô số cho đến khi job 8B xong (Task 10 Step 3). Caption nêu 10.000 câu, top 30, cửa sổ 3 chunk, CER = complete-evidence rate trên 500 câu nhiều section. Nếu bảng rộng hơn `\textwidth`, rút gọn tên model (ví dụ `Qwen3-Emb-4B`) chứ không dùng `\resizebox`.
- [ ] **Step 3: Văn bản 5.1** (chỉ báo kết quả, không diễn giải nguyên nhân): so sánh dense với BM25, hybrid k=2 với k=60, tác động của reranker lên Hit@3/nDCG@10/MRR@10 và lên CER@10; một câu theo nhóm khó nhất (`noisy_confuser`) lấy từ bảng breakdown của cùng report.
- [ ] **Step 4: Build check.**
- [ ] **Step 5: Commit** `git add report/sections/results.tex report/tables/retrieval.tex && git commit -m "docs(paper): report retrieval benchmark results"`

---

### Task 9: Introduction và bản nháp Abstract

**Files:**
- Modify: `report/sections/introduction.tex`, `report/sections/abstract.tex`

- [ ] **Step 1: Introduction** (≤ 1,3 trang gồm Fig. 1): bối cảnh hỏi đáp thuốc tiếng Việt và rủi ro trả lời sai; hạn chế chi phí (chỉ LLM rẻ); ba câu hỏi nghiên cứu RQ1–RQ3 theo spec §3 (viết thành `enumerate` ngắn có nhãn **RQ1**…); đóng góp (3 ý, mỗi ý một số chính — số E2E để `\pending{...}`); câu cuối dẫn đường các mục bằng `\ref`.
- [ ] **Step 2: Abstract** (150–250 từ): bối cảnh, phương pháp, số retrieval chính (lấy từ Table 4), số E2E `\pending{full vs one-step truthfulness}`, `\pending{injection defence}`, kết luận một câu.
- [ ] **Step 3: Build check**; tổng trang hiện tại ghi vào báo cáo task.
- [ ] **Step 4: Commit** `git add report/sections/introduction.tex report/sections/abstract.tex && git commit -m "docs(paper): draft introduction and abstract"`

---

### Task 10: Đưa kết quả E2E, hiệu chỉnh và 8B vào bài

**Chặn bởi:** cả 5 cấu hình `e2e-v1` đã chạy và chấm xong, `calibration score` đã chạy, job rerank 8B đã có `report.md` với contract 2.

**Files:**
- Create: `report/tables/e2e-main.tex`, `report/tables/e2e-ablation.tex`, `report/tables/e2e-calibration.tex` (copy nguyên từ report)
- Modify: `report/tables/retrieval.tex`, `report/sections/results.tex`, `report/sections/methodology.tex`

- [ ] **Step 1: Sinh và copy bảng** (lệnh đã có trong `pharma-lab/docs/guides/e2e-evaluation.md`):

```bash
cd pharma-lab && uv run pharma-lab e2e report --run e2e-v1 && cd ..
R=pharma-lab/data/evaluation/e2e/runs/e2e-v1/reports
cp "$R/main.tex" report/tables/e2e-main.tex
cp "$R/ablation.tex" report/tables/e2e-ablation.tex
cp "$R/calibration.tex" report/tables/e2e-calibration.tex
```

Nếu `e2e report` có tùy chọn `--evaluation` cho file nhãn relevance, dùng đúng như guide. Không sửa tay file đã copy; cần đổi định dạng thì sửa `report.py` (TDD như Task 2) rồi sinh lại.

- [ ] **Step 2: Methodology 4.4:** thay `\pending{calibration grader}` bằng mô tả đúng người/model đã chấm hiệu chỉnh (đọc `calibration/` của run và hỏi người dùng nếu không rõ). Thêm số lượt retry LLM mỗi cấu hình nếu `provenance.json` có.
- [ ] **Step 3: Dòng 8B** trong `tables/retrieval.tex`: điền từ `pharma-lab/data/evaluation/runs/hybrid-qwen4b-p50-k30-rrf2/reports/rerank/qwen3_reranker_8b_fp16/top30-window3/report.md` sau khi kiểm manifest contract 2; cập nhật in đậm nếu 8B tốt nhất.
- [ ] **Step 4: Results 5.2–5.4** (chỉ báo số): `\input{tables/e2e-main}` và `\input{tables/e2e-ablation}`; 5.2 so Full agent với One-step RAG trên truthfulness, nugget recall, hallucination, faithfulness, citation recall, kèm CI; ablation: cấu hình nào giảm có ý nghĩa sau Holm; 5.3 an toàn: negative rejection, redirect accuracy, injection defence/attack success, severe harm — ghi rõ n=10 cho từng nhóm và CI Wilson; 5.4 chi phí: token và lời gọi LLM mỗi lượt, latency p50/p95 từ `main.csv` (ghi rõ môi trường demo, reranker trên GPU Kaggle qua tunnel); 5.2 hoặc cuối 4.4: `\input{tables/e2e-calibration}` và các số PPI từ `ppi.csv`.
- [ ] **Step 5: Build check**; nếu vượt 12 trang nội dung (xem Task 12 Step 2) áp dụng thứ tự cắt của spec §4.
- [ ] **Step 6: Commit**

```bash
git add report/tables/e2e-main.tex report/tables/e2e-ablation.tex report/tables/e2e-calibration.tex \
  report/tables/retrieval.tex report/sections/results.tex report/sections/methodology.tex
git commit -m "docs(paper): report end-to-end and calibration results"
```

Commit report E2E (`pharma-lab/data/evaluation/e2e/runs/e2e-v1/reports/*.csv|tex|md|json`) và report 8B theo chính sách dữ liệu của repo, chỉ file đã được git theo dõi hoặc được `.gitignore` cho phép; kiểm `git status --short` trước khi commit.

---

### Task 11: Discussion, Conclusion, Abstract hoàn chỉnh

**Files:**
- Modify: `report/sections/discussion.tex`, `report/sections/conclusion.tex`, `report/sections/abstract.tex`, `report/sections/introduction.tex`

- [ ] **Step 1: Discussion** (≤ 1,3 trang): trả lời RQ1–RQ3 bằng số đã báo; đánh đổi rerank (thứ hạng tốt hơn nhưng CER@10 thấp hơn) và việc vòng judge–refine có bù được không (theo ablation); ý nghĩa với triển khai dùng LLM rẻ (chi phí/lượt). `\subsection{Limitations}`: judge là LLM (có hiệu chỉnh nhưng không có dược sĩ), nhóm an toàn chỉ 10 câu mỗi nhóm, câu hỏi retrieval sinh tất định, một nguồn tờ hướng dẫn, latency đo trên môi trường demo, benchmark tự xây. Không đưa kết luận vượt số liệu.
- [ ] **Step 2: Conclusion** (≤ 0,4 trang): tóm tắt ba đóng góp và hướng tiếp theo (chấm bởi chuyên gia, mở rộng nguồn, câu hỏi thật của người dùng). Nếu dữ liệu/code được công bố, để `\pending{release URL}` cho người dùng quyết định.
- [ ] **Step 3: Abstract và Introduction:** thay mọi `\pending` bằng số thật từ Table 5–6.
- [ ] **Step 4: Build check.**
- [ ] **Step 5: Commit** `git add report/sections/discussion.tex report/sections/conclusion.tex report/sections/abstract.tex report/sections/introduction.tex && git commit -m "docs(paper): write discussion and conclusion"`

---

### Task 12: Kiểm tra cuối trước khi nộp

**Files:**
- Modify: bất kỳ file trong `report/` cần sửa; `report/main.tex` (`\finaltrue`)

- [ ] **Step 1: Placeholder:** `grep -rn '\\pending{' report/sections report/tables report/figures` → chỉ còn các mục người dùng phải điền (tác giả không dùng `\pending`; release URL nếu người dùng chưa quyết). Hỏi người dùng các mục còn lại, rồi đặt `\finaltrue` và Build check → `exit=0`.
- [ ] **Step 2: Số trang:**

```bash
N=$(pdfinfo report/main.pdf | awk '/^Pages/{print $2}')
for p in $(seq 1 "$N"); do
  pdftotext -f "$p" -l "$p" report/main.pdf - | grep -qx 'References' && { echo "references start on page $p"; break; }
done
```

Nội dung chính phải kết thúc trong 12 trang: References bắt đầu ở trang ≤ 13, và nếu ở trang 13 thì trang 13 chỉ có references. Nếu vượt: chuyển `tab:e2e-calibration` xuống appendix sau references, rồi gộp `tab:eval-sets` vào văn bản, rồi rút gọn Related Work.
- [ ] **Step 3: Chất lượng:** `bash .agents/skills/latex-document-skill/scripts/latex_lint.sh report/main.tex` (sửa lỗi thật, không tắt rule); Build check không có undefined citation/reference và 0 Overfull; xem mọi trang PNG: bảng không tràn, hình đọc được, tên tiếng Việt đúng dấu, không ký tự lạ (`¡`, `¿`); `pdffonts report/main.pdf` không có font Type 3.
- [ ] **Step 4: Đối chiếu số:** mọi số trong abstract/introduction/discussion khớp bảng; mọi số trong bảng khớp file nguồn (spot-check từng bảng bằng cách mở file nguồn).
- [ ] **Step 5: Commit** `git add report && git status --short report` (không có `main.pdf`, `preview/`, file aux) rồi `git commit -m "docs(paper): finalize the SoICT 2026 submission"`; push `dev` và báo người dùng đường dẫn `report/main.pdf` để nộp EasyChair.
