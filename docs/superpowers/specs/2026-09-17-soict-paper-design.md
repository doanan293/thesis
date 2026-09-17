# Thiết kế paper SoICT 2026

Ngày: 2026-09-17.

## 1. Mục tiêu

Viết paper tiếng Anh bằng LaTeX trong `report/` để nộp SoICT 2026 (hạn full paper
20/09/2026). Paper viết dần trong lúc E2E và rerank 8B còn chạy; mọi số chưa có được
đánh dấu placeholder và điền khi kết quả xong.

Ba đóng góp, trình bày cân đối:

1. Corpus thuốc tiếng Việt (Dược thư + tờ hướng dẫn sử dụng) và benchmark retrieval
   10.000 câu, 6 nhóm.
2. Benchmark retrieval có hệ thống: BM25, dense (6 embedding), hybrid RRF, và
   reranker, đo cả chất lượng lẫn chi phí.
3. Agentic RAG dùng LLM rẻ, đánh giá end-to-end trên bộ golden 500 câu, có ablation,
   kiểm tra an toàn và hiệu chỉnh judge.

Ngoài phạm vi: bản luận văn tiếng Việt, slide, poster.

## 2. Định dạng hội nghị

- Springer CCIS, định dạng LNCS (`llncs.cls`, `splncs04.bst`), lấy từ template chính
  thức của Springer (bản Overleaf 2022-01-12). File class được copy vào `report/`, vì
  máy chưa cài.
- Tối đa 12 trang, không tính references. Appendix (nếu có) nằm sau references và
  không được dùng để vượt giới hạn nội dung chính.
- Single-blind: có tên tác giả. Tác giả, affiliation, email và ORCID để placeholder,
  người dùng tự điền.
- Không đánh số trang (class mặc định). Nộp qua EasyChair.
- Build bằng LuaLaTeX với `fontspec` (font mặc định Latin Modern, cùng dáng Computer
  Modern của LNCS, đủ dấu tiếng Việt), vì pdfLaTeX
  với T5 hiển thị và trích xuất tiếng Việt sai (đã kiểm chứng trong lần dựng `report/`
  trước). Tiếng Việt xuất hiện trong ví dụ câu hỏi và references.
- Mọi việc LaTeX tuân theo skill `.agents/skills/latex-document-skill`: nạp skill trước
  khi tạo file, compile bằng `scripts/compile_latex.sh --use-latexmk --preview`. Bước hỏi
  enrichment của skill đã được trả lời trong spec này (§4: hình TikZ, bảng booktabs,
  BibTeX).

## 3. Cấu trúc bài

Theo khung chuẩn người dùng đưa ra. Cập nhật 17/09: Research Gap là đoạn cuối của Related Work, các mục lớn theo skill `research-paper-writing`: Discussion là mục nhỏ cuối Experimental Results, bài kết bằng Conclusion. Class tự đánh số mục; không dùng
số La Mã và không đặt mục "System".

**Không đặt tên hệ thống.** Trong bài gọi là "our agentic RAG pipeline". Trong bảng,
các dòng mang tên cấu hình: *Full agent*, *One-step RAG*, *w/o judge–refine*,
*w/o rephrase*, *w/o rerank*.

| Mục | Nội dung | Trang |
| --- | --- | --- |
| Abstract + Keywords | Bối cảnh, phương pháp, số chính (có placeholder), 4–6 keyword | 0,3 |
| 1 Introduction | Bài toán hỏi đáp thuốc tiếng Việt; câu hỏi nghiên cứu (nêu bằng văn xuôi, không đánh nhãn RQ: retrieval nào tốt nhất theo chất lượng và chi phí, agentic RAG có hơn one-step không và bước nào đóng góp, hệ thống có an toàn không); thách thức, ý tưởng chính, tóm tắt thí nghiệm, đóng góp; không có hình hay chi tiết cài đặt | 1,3 |
| 2 Related Work | RAG và agentic RAG; retrieval cho QA y khoa; đánh giá RAG; QA y khoa tiếng Việt. Mỗi đoạn kết bằng hạn chế và điểm khác biệt của bài, không có đoạn research gap riêng | 1,4 |
| 3 Methodology | 3.1 Corpus (nguồn, làm sạch, chia section/chunk, enrichment). 3.2 Pipeline (Fig. 1, guard → rephrase → hybrid search + rerank → judge/refine → answer có trích dẫn; model theo role). 3.3 Bộ đánh giá (10k retrieval, 500 golden E2E, quota các nhóm). 3.4 Thiết kế thí nghiệm (bảng cấu hình, metric có công thức, judge + hiệu chỉnh, thống kê) | 3,5 |
| 4 Experimental Results | 4.1 Retrieval, 4.2 E2E chính và ablation (Δ, p Holm), 4.3 An toàn, 4.4 Chi phí và latency (chỉ báo số), 4.5 Discussion diễn giải kết quả | 3,3 |
| 5 Conclusion | Nhắc lại vấn đề và ý tưởng chính; bằng chứng chính; ý nghĩa thực tiễn; đoạn Limitations (câu hỏi sinh bằng luật, tham số chọn trên cùng benchmark, judge cùng nhà cung cấp, đáp án do LLM soạn, latency trên máy dev, một nguồn tờ HDSD); hướng tiếp theo | 0,8 |
| References | `splncs04`, không tính trang | – |

Tổng khoảng 11,7 trang, chừa 0,3 trang cho điều chỉnh.

## 4. Hình và bảng

- **Fig. 1:** sơ đồ pipeline bằng TikZ, đặt trong mục 3.2 (Methodology).
- **Table 1** (4.1): thống kê corpus: 3.120 tài liệu, 12.326 section, 24.955 chunk;
  Dược thư 1.668 trang (689 chuyên luận); 2.406 tờ HDSD.
- **Table 2** (4.3): hai bộ đánh giá theo nhóm câu hỏi.
- **Table 3** (4.4): cấu hình E2E (rephrase / judge–refine / rerank).
- **Table 4** (5.1): retrieval: Hit@3, Hit@10, nDCG@10, MRR@10, Complete-evidence
  rate@10, và chi phí (số chiều vector; throughput reranker, cặp/giây trên T4, lấy từ
  `pharma-lab/data/cache/kaggle_profiles/rerank/*.json`). Số lấy từ
  `pharma-lab/data/evaluation/runs/*/reports/**/report.md`. Dòng Qwen3-Reranker 8B để
  placeholder đến khi job 8B xong. Dòng text-embedding-3-large chỉ đưa vào nếu có số
  trên cùng nhãn gold hiện tại; nếu không thì bỏ.
- **Table 5–6** (5.2–5.3): E2E chính và ablation, lấy nguyên từ `main.tex` và
  `ablation.tex` do `pharma-lab e2e report` sinh ra. Khổ LNCS chỉ rộng 12,2 cm, nên
  `main.tex` chỉ gồm Full agent và One-step RAG (kèm CI), còn `ablation.tex` gồm Δ của
  bốn cấu hình còn lại so với Full agent (CI nằm trong CSV). Cả hai dùng booktabs, có
  `\label` và tên metric/cấu hình dễ đọc. Các file này được copy vào
  `report/tables/` và không sửa tay; nếu cần đổi định dạng thì sửa ở `report.py`.
- **Table 7** (4.4 hoặc 5.2): độ đồng thuận judge–người chấm (`calibration.tex`).

Nếu vượt trang: bỏ Table 7 vào appendix trước, rồi gộp Table 2 vào văn bản.

## 5. Placeholder cho kết quả chưa có

- Macro `\pending{mô tả}` in chữ đỏ, ví dụ `\pending{full truthfulness}`.
- Bảng chưa có dùng một file `tables/*.tex` giữ chỗ, có đúng số cột, ô ghi `\pending{}`.
- Trước khi nộp, `grep -rn '\\pending' report/` phải rỗng, và macro được định nghĩa để
  build lỗi nếu còn sót khi bật cờ `\finaltrue`.

## 6. Nguồn nội dung

- Phương pháp và metric: `docs/superpowers/specs/2026-09-16-e2e-golden-evaluation-design.md`,
  `pharma-lab/docs/guides/e2e-evaluation.md`, `pharma-lab/docs/guides/evaluation.md`.
- Corpus: bản progress report cũ (`report/report.md` trong lịch sử git) và các manifest
  dữ liệu.
- Kết quả: file report đã commit trong `pharma-lab/data/evaluation/`. Không viết số nào
  không có trong file kết quả.
- Paper nền: Mikkelsen 2026 (JMIR Medical Informatics). Định dạng tham khảo: URAG
  (SoICT 2024).

**References.** Mỗi mục trong `references.bib` phải được kiểm tra từ DOI, arXiv hoặc
DBLP; không tạo trích dẫn từ trí nhớ. Danh sách dự kiến gồm:
- RAG (Lewis 2020), Self-RAG, Corrective RAG, Adaptive-RAG;
- MedRAG/MIRAGE, Mikkelsen 2026;
- BM25, RRF (Cormack 2009), BEIR, MTEB, BGE-M3, Qwen3 Embedding/Reranker, EmbeddingGemma;
- RAGAS, ARES, ALCE, CRAG benchmark, TREC 2024 RAG (nugget), RGB, AgentDojo;
- CHART, TRIPOD-LLM, PPI (Angelopoulos 2023), Smucker 2007, Holm 1979, Wilson 1927,
  Gwet 2008;
- các công trình QA/RAG y khoa và pháp luật tiếng Việt (có URAG).

## 7. Bố cục file

```
report/
  main.tex            # preamble, \pending, \input các mục
  llncs.cls  splncs04.bst
  sections/{abstract,introduction,related-work,methodology,results,conclusion}.tex
  tables/             # bảng viết tay + bảng copy từ e2e report
  figures/pipeline.tex
  references.bib
  README.md           # lệnh build
```

Build: `bash .agents/skills/latex-document-skill/scripts/compile_latex.sh report/main.tex --engine lualatex --use-latexmk --preview --preview-dir report/preview`. File sinh ra (`*.aux`, `*.log`,
`main.pdf`, `preview/`, …) nằm trong `.gitignore`.

## 8. Quy trình và git

1. Dựng khung, build được, mọi mục có nội dung nháp và placeholder.
2. Viết các mục không phụ thuộc kết quả E2E trước: Related Work, Research Gap,
   Methodology, Results 5.1.
3. Khi E2E, hiệu chỉnh và 8B xong: copy bảng, điền số, viết 5.2–5.4, Discussion,
   Conclusion, Abstract.
4. Kiểm tra cuối: ≤ 12 trang nội dung, không còn `\pending`, không có warning
   undefined reference/citation, tên tiếng Việt hiển thị đúng.

Commit thẳng lên `dev` (giữ ngang `main`), chỉ stage đường dẫn `report/`, spec/plan này và
file pharma-lab được sửa.
