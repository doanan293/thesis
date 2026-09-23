# Paper SoICT 2026

Paper tiếng Anh nộp SoICT 2026 (Springer CCIS, định dạng LNCS, tối đa 12 trang không tính references). Thiết kế: `docs/superpowers/specs/2026-09-17-soict-paper-design.md`.

## Template

`llncs.cls` và `splncs04.bst` lấy từ gói CTAN `llncs` do Springer duy trì (v2.26, 2025/02/25): <https://ctan.org/pkg/llncs>.

## Build

Chạy từ thư mục gốc repo, dùng script của skill `latex-document-skill`:

```bash
bash .agents/skills/latex-document-skill/scripts/compile_latex.sh \
  report/SoICT2026_Agentic_RAG_Vietnamese_Pharmaceutical_Documents.tex \
  --engine pdflatex --use-latexmk --preview --preview-dir preview
```

Kết quả là `report/SoICT2026_Agentic_RAG_Vietnamese_Pharmaceutical_Documents.pdf` (file `.tex` chính mang luôn tên bản nộp, nên PDF sinh ra dùng được để nộp ngay) và ảnh xem trước trong `report/preview/`. Bài dùng pdfLaTeX với bảng mã T5 như bản mẫu của Springer, nên Overleaf biên dịch được với compiler mặc định.

## Overleaf

Nén đúng các file nguồn rồi tải lên bằng "New Project → Upload Project":

```bash
cd report && zip -r SoICT2026_Agentic_RAG_Vietnamese_Pharmaceutical_Documents.zip \
  SoICT2026_Agentic_RAG_Vietnamese_Pharmaceutical_Documents.tex \
  llncs.cls splncs04.bst references.bib sections tables figures && cd ..
```

Overleaf biên dịch ngay với compiler mặc định (pdfLaTeX). Main document được nhận tự động vì chỉ có một file `.tex` ở gốc.

## Kết quả chưa có

Số chưa có được viết bằng `\pending{...}` (chữ đỏ). Trước khi nộp, đổi `\finalfalse` thành `\finaltrue` trong file `.tex` chính: build sẽ lỗi nếu còn `\pending`.

## Bảng end-to-end

`tables/e2e-*.tex` được sinh bởi `pharma-lab e2e report` và không sửa tay:

```bash
cd pharma-lab && uv run pharma-lab e2e report --run e2e-gemini && cd ..
R=pharma-lab/data/evaluation/e2e/runs/e2e-gemini/reports
cp "$R/main.tex" report/tables/e2e-main.tex
cp "$R/ablation.tex" report/tables/e2e-ablation.tex
cp "$R/calibration.tex" report/tables/e2e-calibration.tex
```

Muốn đổi định dạng bảng thì sửa `pharma-lab/src/pharma_lab/e2e/report.py` rồi sinh lại.
