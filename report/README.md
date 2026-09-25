# Paper SoICT 2026

Paper tiếng Anh nộp SoICT 2026 (Springer CCIS, định dạng LNCS, tối đa 12 trang không tính references).

## Template

`llncs.cls` và `splncs04.bst` lấy từ gói CTAN `llncs` do Springer duy trì (v2.26, 2025/02/25): <https://ctan.org/pkg/llncs>.

## Build

Cần TeX Live có pdfLaTeX, BibTeX và `latexmk`. Trên Ubuntu/Debian:

```bash
sudo apt install latexmk texlive-latex-recommended texlive-latex-extra \
  texlive-pictures texlive-lang-other zip
```

Mọi thiết lập build nằm trong `.latexmkrc`, nên chỉ cần chạy `latexmk` trong `report/`:

```bash
cd report
latexmk        # build PDF, tự chạy BibTeX và đủ số lượt pdflatex
latexmk -pvc   # build lại mỗi lần lưu file
latexmk -C     # xóa mọi file sinh ra, kể cả PDF
```

Kết quả là `report/SoICT2026_Agentic_RAG_Vietnamese_Pharmaceutical_Documents.pdf` (file `.tex` chính mang luôn tên bản nộp, nên PDF dùng để nộp ngay). Build thất bại nếu còn citation hoặc reference chưa định nghĩa, hay bất kỳ warning nào của LaTeX, class hoặc package. Không build bằng một lượt `pdflatex` đơn lẻ: lượt đó không chạy BibTeX nên mọi trích dẫn hiện thành `[?]`.

Trong VS Code, LaTeX Workshop dùng recipe mặc định `latexmk`, recipe này đọc `.latexmkrc` nên cho cùng kết quả. File `.tex` không có magic comment `% !TEX program`, vì magic comment khiến LaTeX Workshop bỏ qua recipe và chỉ chạy một lượt `pdflatex`.

Hook `report-build` trong `.pre-commit-config.yaml` ở gốc repo chạy `latexmk` mỗi khi commit đụng tới `report/`, nên bài luôn build sạch trước khi vào git.

## Overleaf

Nén đúng các file nguồn rồi tải lên bằng "New Project → Upload Project":

```bash
cd report && zip -r SoICT2026_Agentic_RAG_Vietnamese_Pharmaceutical_Documents.zip \
  SoICT2026_Agentic_RAG_Vietnamese_Pharmaceutical_Documents.tex \
  llncs.cls splncs04.bst references.bib sections tables figures && cd ..
```

Overleaf biên dịch ngay với compiler mặc định (pdfLaTeX) và tự chạy BibTeX. Main document được nhận tự động vì chỉ có một file `.tex` ở gốc.

## Bảng end-to-end

`tables/e2e-*.tex` dựng từ bảng mà `pharma-lab e2e report` sinh cho hai run `e2e-gemini` và `e2e-qwen35-9b`, rồi chỉnh tay cho paper: `e2e-main.tex` gộp hai setting vào một bảng. Số liệu gốc nằm trong `pharma-lab/data/evaluation/e2e/runs/<run>/reports/` (`main.csv`, `ablation.csv`, `calibration.csv`):

```bash
cd pharma-lab
uv run pharma-lab e2e report --run e2e-gemini
uv run pharma-lab e2e report --run e2e-qwen35-9b
cd ..
```

Kiểm tra số trong `report/tables/` với CSV sau mỗi lần sinh lại.
