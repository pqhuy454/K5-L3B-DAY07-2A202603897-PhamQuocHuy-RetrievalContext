# Báo cáo Day 7

## Dữ liệu và benchmark

- Nguồn, quyền sử dụng, số trang và số chunk: Luật An ninh mạng số `24/2018/QH14` (`starter/data/24-2018-qh14.pdf` từ Day 6, văn bản quy phạm pháp luật công khai), 32 trang nội dung hợp lệ, 165 chunks theo Strategy `B` (`by_paragraph`, tối đa 500 ký tự/chunk, đã chuẩn hóa khoảng trắng để khớp `baseline.py`).
- Vector store, embedding model, phiên bản và số chiều: PostgreSQL 16.16 + `pgvector` 0.8.7 (`127.0.0.1:55433`, DB `day6_lab`, bảng `document_chunks`, `strategy='B'`), kết nối qua `pg8000`/`psycopg`. Embedding model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 chiều, chuẩn hóa L2, chạy local chi phí `$0.0`).
- Agent model, cách chạy và phân chia dev/test: Model `openai/gpt-4o-mini` qua OpenRouter API (`temperature=0.0`, hỗ trợ Tool Calling). Chia cố định 24 câu: **Dev (17 câu)** gồm 15 câu có đáp án (`q01`–`q15` từ Day 6) + 2 câu thiếu bằng chứng (`u01`, `u02`); **Test (7 câu)** gồm 5 câu có đáp án (`q16`–`q20` từ Day 6, trang tách biệt với `dev`) + 2 câu thiếu bằng chứng (`u03`, `u04`). Riêng `q20` là câu multi-evidence cần đủ 2 chunks (`p32_c02`, `p32_c03`).
- Cách xác định câu có/thiếu evidence và tiêu chí chấm: Câu có đáp án có `gold` (`source_id`, `page`, `evidence`) được kiểm chứng khớp nguyên văn trong `day6_chunks.json`. Câu thiếu bằng chứng (`u01`–`u04`) gần chủ đề an ninh mạng nhưng đã kiểm tra 0 kết quả trên toàn bộ 165 chunks (mức phạt tiền bằng đồng, sàn tiền điện tử, ISO 27001/GDPR, thuế suất ưu đãi). Tiêu chí chốt cấu hình trên `dev`: ưu tiên `Context Evidence Rate` & `Hit@1`, giảm từ chối nhầm trên câu có đáp án, giữ `Abstention đúng = 100%`, `errors = 0`.

## Retrieval và thử nghiệm

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline | `tfidf_b800`: Lexical TF-IDF, `top_k=3`, không filter, `budget=800` chars | Điểm chuẩn lexical offline | dev | 11/15 (73.3%) | 15/15 (100.0%) | 11/15 (73.3%) | 0.60 ms/câu / $0.0 |
| Phương pháp 2 | `dense_b800`: Dense `pgvector` Day 6, `top_k=3`, không filter, `budget=800` chars | Embedding đa ngôn ngữ bắt câu diễn đạt lại | dev | 11/15 (73.3%) | 12/15 (80.0%) | 11/15 (73.3%) | 95.54 ms/câu / $0.0 |
| Phương pháp 3 | `hybrid_rrf_b800`: Hybrid Dense (top-10) + TF-IDF (top-10) qua RRF (`k=60`), `top_k=3`, `budget=800` | Kết hợp ngữ nghĩa và từ khóa luật tăng `Hit@1` | dev | 12/15 (80.0%) | 13/15 (86.7%) | 12/15 (80.0%) | 97.67 ms/câu / $0.0 |
| Cải thiện 1 | `hybrid_rrf_b1600`: Hybrid RRF, `top_k=3`, tăng `budget=1600` chars | Giảm rớt chunk rank 2–3 do chật budget | dev | 12/15 (80.0%) | 13/15 (86.7%) | 13/15 (86.7%) | 97.27 ms/câu / $0.0 |
| Cải thiện 2 | `hybrid_rrf_filter_b1600`: Hybrid RRF + metadata `filter` (`section` khi query có sẵn), `budget=1600` | Lọc theo chương giúp đẩy đúng điều lên rank 1 | dev | 11/15 (73.3%) | 13/15 (86.7%) | 13/15 (86.7%) | 83.36 ms/câu / $0.0 |
| Cấu hình đã chọn | `hybrid_rrf_b1600`: Hybrid RRF, `top_k=3`, không filter, `budget=1600` chars | Đánh giá độc lập sau khi chốt | test | 4/5 (80.0%) | 5/5 (100.0%) | 4/5 (80.0%) | 107.34 ms/câu / $0.0 |
| Baseline | `tfidf_b800`: Lexical TF-IDF, `top_k=3`, không filter, `budget=800` chars | Đối chứng baseline trên test | test | 4/5 (80.0%) | 5/5 (100.0%) | 3/5 (60.0%) | 0.70 ms/câu / $0.0 |

Một cấu hình có thể đồng thời là phương pháp mới và cải thiện; giải thích rõ. Báo `n/N` và tỷ lệ, ghi đơn vị thời gian/chi phí, so sánh test sau khi chốt. Nêu hai lỗi dev, nguyên nhân, cách thử và lý do chọn cấu hình:
- **Lỗi dev 1 (Rớt chunk rank 2–3 do `budget=800`):** Ở `tfidf_b800`, `Hit@3` đạt `15/15` nhưng `Context Evidence Rate` chỉ đạt `11/15` (rớt `q01`, `q07`, `q09`, `q13`). Khi tăng `budget` lên `1600` ở `hybrid_rrf_b1600`, `q09` được giữ lại (`12/15 -> 13/15`).
- **Lỗi dev 2 (Từ khóa luật chính xác bị loãng trong Dense/RRF ở `q05`, `q12`; và `filter` làm tụt hạng `q13`):** Dense/Hybrid RRF làm rớt `q05`, `q12` khỏi top-3 dù TF-IDF tìm được, và bật `filter` làm `Hit@1` giảm từ `12/15` xuống `11/15` (`q13`). Vì vậy ta chốt `hybrid_rrf_b1600` (không ép filter) cho tầng retrieval, đồng thời ở tầng Agent cung cấp cả **2 tool** (`search_hybrid` + `search_keyword`) với tối đa 3 lượt tìm để Agent tự bù đắp khi một lượt tìm bị sót.

## Agent và đánh giá

- Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:
  - **Baseline (`agent_baseline`):** 1 tool `search_keyword` (TF-IDF), `max_searches=1`, `top_k=3`, `budget=800` ký tự.
  - **Cấu hình đã chọn (`agent_hybrid_multi`):** 2 tool `search_hybrid` (Dense pgvector + TF-IDF RRF) và `search_keyword` (TF-IDF), `max_searches=3`, `top_k=5`, `budget=2400` ký tự, lọc trùng theo `chunk_id`.
  - **Điều kiện dừng & Xử lý lỗi:** Agent dừng khi model trả JSON kết quả cuối (`answered` hoặc `abstain`) hoặc chạm giới hạn `max_searches`. Khi gặp lỗi mạng OpenRouter (`ConnectionResetError`/`ConnectTimeoutError` ở lần chạy `dev` đầu tiên), đã bổ sung retry tối đa 3 lần (exponential backoff 2s, 4s) giúp giảm lỗi về `0`.

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| `agent_baseline` / dev (n=17) | 17/17 | 1 lượt/câu (100%) | (Xem đối chiếu test; trên dev từ chối nhầm 3/15: `q07`, `q09`, `q12`) | — | 12/12 answered | 2/2 (100%) | 3/15 (20.0%) |
| `agent_hybrid_multi` / dev (n=17) | 17/17 | 1–3 lượt (1:7, 2:3, 3:7) | (Giải quyết được `q07`, `q09`, `q12`; chỉ còn `q04` từ chối) | — | 14/14 answered | 2/2 (100%) | 1/15 (6.7%) |
| `agent_baseline` / test (n=7) | 7/7 | 1 lượt/câu (7/7) | 0: 1, 1: 0, 2: 4 (TB: 1.60/2) | 5/5 (100.0%) | 4/4 answered | 2/2 (100%) | 1/5 (20.0%) |
| `agent_hybrid_multi` / test (n=7) | 7/7 | 1–3 lượt (1:4, 2:0, 3:3) | 0: 0, 1: 0, 2: 5 (TB: 2.00/2) | 6/7 (85.7%) | 5/5 answered | 2/2 (100%) | 0/5 (0.0%) |

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| `q16` | Có đáp án (`test`) | `search_hybrid` (1 lượt) -> có `p03_c02` (Context: 2) | Trả lời đúng "sự cố an ninh mạng" `[B_luat-an-ninh-mang-2018_p03_c02]` | Answer: 2, Citation: `[true]` — khớp khoản 13 Điều 2. |
| `q17` | Có đáp án (`test`) | `search_hybrid` (1 lượt) -> có `p12_c03` (Context: 2) | Trả lời đúng "giám sát an ninh mạng" `[B_luat-an-ninh-mang-2018_p12_c03]` | Answer: 2, Citation: `[true]` — khớp khoản 1 Điều 14. |
| `q18` | Có đáp án (`test`) | `search_hybrid` (1 lượt) -> có `p24_c01` và `p11_c05` (Context: 2) | Trả lời đúng "30 ngày", nhưng cite `[B_luat-an-ninh-mang-2018_p11_c05]` | Answer: 2, Citation: `[false]` — `p11_c05` thuộc Điều 13 (hệ thống quan trọng), không phải Điều 24 (`p24_c01`, hệ thống không thuộc Danh mục). |
| `q19` | Có đáp án (`test`) | `search_hybrid` (1 lượt) -> có `p30_c03`, `p14_c01` (Context: 2) | Trả lời đúng khoản 1 Điều 16; cite `p30_c03`, `p14_c01` | Answer: 2, Citation: `[true, true]` — cả 2 nguồn hỗ trợ trực tiếp. |
| `q20` | Có đáp án (`test`, multi-evidence) | `search_hybrid` + `search_keyword` (3 lượt) -> đủ `p32_c02`, `p32_c03` (Context: 2) | Trả lời đủ 01/01/2019 và 12 tháng (+ tối đa 12 tháng); cite `p32_c02`, `p32_c03` | Answer: 2, Citation: `[true, true]` — đủ cả 2 khoản của Điều 43. |
| `u03` | Thiếu bằng chứng (`test`) | `search_hybrid` + `search_keyword` (3 lượt) -> không có ISO 27001/GDPR | Từ chối: không có thông tin về ISO/IEC 27001 hay GDPR; `citations: []` | `abstention_correct: true` — từ chối trung thực, không suy diễn. |
| `u04` | Thiếu bằng chứng (`test`) | `search_hybrid` + `search_keyword` (3 lượt) -> không có thuế suất | Từ chối: không có thông tin về thuế suất ưu đãi/miễn thuế; `citations: []` | `abstention_correct: true` — từ chối trung thực, không suy diễn. |

- Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:
  - Dữ liệu & Benchmark: `data/day6_chunks.json`, `data/questions_retrieval.json`, `data/agent_cases.json` (xuất qua `python src/export_day6.py`).
  - Retrieval traces: `results/retrieval_dev/`, `results/retrieval_test/` (`python src/retrieval_benchmark.py --split dev|test`).
  - Agent traces & nhãn `test`: `results/agent_baseline_test/{traces.json,reviews.json}` (tổng chi phí `$0.00208`, `errors=0`), `results/agent_hybrid_multi_test/{traces.json,reviews.json}` (tổng chi phí `$0.00318`, `errors=0`). Tổng chi phí cả `dev` (`$0.00500` + `$0.00877`) và `test` là **`$0.01903`**.
- Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn:
  - **Trước/sau:** Trên `test`, `agent_hybrid_multi` nâng Grounded answer trung bình từ `1.60/2` lên **`2.00/2`**, xóa bỏ hoàn toàn từ chối nhầm (`1/5 -> 0/5` ở `q18`) và xử lý tốt câu multi-evidence `q20`, trong khi giữ `Abstention đúng = 2/2`.
  - **Ví dụ lỗi:** (1) Trên `dev`, `q04` bị `abstain` ở `agent_hybrid_multi` vì `p06_c02` bị cắt trước khoản 4 và top-3 lặp lại; (2) Trên `test`, `q18` trả lời đúng "30 ngày" nhưng trích nhầm `p11_c05` (Điều 13) thay vì `p24_c01` (Điều 24) do hai điều khoản có nội dung thời hạn 30 ngày giống nhau.
  - **Đánh đổi & giới hạn:** Multi-turn hybrid search (`max_searches=3`, `budget=2400`) tăng chất lượng trả lời và giảm từ chối nhầm, nhưng tăng chi phí API (~1.5×) và độ trễ (từ ~4–16s lên ~6–29s, cá biệt `u04` 89s khi gọi đủ 3 lượt). Chưa có Cross-Encoder reranker riêng để phân biệt điều khoản gần giống nhau như `p11_c05` vs `p24_c01`.

## Thử thách nâng cao (nếu thực hiện)

- Tài liệu, nguồn, quyền sử dụng, số trang và phiên bản: Báo cáo thường niên Công ty Cổ phần Sữa Việt Nam (Vinamilk) năm 2022 (`vinamilk_ar_2022.pdf`, 172 trang PDF, công bố công khai tại `https://www.vinamilk.com.vn/bao-cao-thuong-nien/bao-cao/2022/download-file/vinamilk_ar_2022.pdf`). Không phân phối lại file PDF gốc trên repo (chỉ lưu script tái tạo và index/trace).
- Cách xử lý text, bảng, ảnh, biểu đồ; phạm vi index và phần chưa xử lý:
  - Index toàn bộ **172/172 trang** (`0` trang lỗi, `0` trang dưới 50 ký tự). Dùng `pypdf` trích xuất text thành 647 text chunks (~650 ký tự, overlap 100), kết hợp `Windows.Data.Pdf` render **172/172 trang** sang PNG (`data/advanced_pages/page_001.png`..`page_172.png`) và dùng `openai/gpt-4o-mini` tạo 172 `visual_summary` chunks (tổng **819 chunks**, chi phí Vision index `$0.0899`). Embed bằng `paraphrase-multilingual-MiniLM-L12-v2` vào `pgvector` (`strategy='ADV_VNM2022'`).
  - Agent sử dụng **`google/gemini-2.5-flash`** (OpenRouter) với 3 tools: `search_hybrid` (`top_k=8`), `get_page_text(page)`, và **`view_page_image(page)`** để quan sát trực tiếp trang PDF gốc khi đọc bảng, biểu đồ, ảnh/sơ đồ.
- Cách chia benchmark, schema evidence, model và cấu hình đã chốt trên dev:
  - Chia cố định **48 câu** (`data/advanced_cases.json`): **Dev (18 câu)** = 15 có đáp án (3 Bảng, 3 Ảnh/sơ đồ, 3 Biểu đồ, 3 Multi-hop, 3 Text) + 3 thiếu bằng chứng; **Test (30 câu)** = 24 có đáp án (6 Bảng, 5 Ảnh/sơ đồ, 5 Biểu đồ, 4 Multi-hop, 4 Text) + 6 thiếu bằng chứng. Mỗi câu có `id`, `split`, `category`, `expected_answer`, `evidence` (`page`, `region`).
  - Chốt cấu hình trên `dev` với `google/gemini-2.5-flash` (`top_k=8`, `max_turns=8`, `max_images=4`, `errors=0`, chi phí `dev`: `$0.03043`).

| Nhóm test | Đạt / tổng | Tỷ lệ | Lỗi / nhận xét |
|---|---|---|---|
| Bảng | 5/6 | 83.3% | `adv_t01`–`adv_t05` đúng đơn vị VND/tỷ đồng; `adv_t06` xem nhầm trang 112 thay vì 109 rồi abstain. |
| Ảnh, sơ đồ | 4/5 | 80.0% | `adv_t07`, `adv_t08`, `adv_t10`, `adv_t11` đạt; `adv_t09` trả lời theo trang 13 thay vì sơ đồ Chuỗi giá trị trang 14. |
| Biểu đồ | 5/5 | 100.0% | `adv_t12`–`adv_t16` đọc đúng giá trị, % và khối lượng giao dịch trung bình (`adv_t14` cite bảng trang 16 cùng số liệu). |
| Multi-hop | 4/4 | 100.0% | `adv_t17`–`adv_t20` kết hợp đủ 2 trang, tính hiệu số và giải thích làm tròn chính xác. |
| Text | 3/4 | 75.0% | `adv_t21`–`adv_t23` đạt; `adv_t24` nhầm số trang PDF 65 với số trang in 67 nên abstain. |
| Thiếu bằng chứng | 6/6 | 100.0% | `adv_u04`–`adv_u09` đều từ chối chuẩn xác, không suy diễn. |
| **Toàn bộ** | **27/30** | **90.0%** | **Đạt ngưỡng $\ge 90\%$ (27/30); 0 lỗi hệ thống.** |

- Trace, nhãn và evidence từng câu; thời gian, chi phí, lỗi hệ thống:
  - Script: `src/advanced_scan.py`, `src/advanced_index.py`, `src/advanced_agent.py`.
  - Kết quả & nhãn `test`: `results/advanced_test/{traces.json, reviews.json}` (`errors=0`, chi phí chạy `test`: `$0.04859`; tổng chi phí toàn phần nâng cao gồm index + dev + test: `$0.16892`).
- Giới hạn, cách tái lập và đường dẫn benchmark:
  - Benchmark: `data/advanced_cases.json`; trang PNG & chunks: `data/advanced_pages/`, `data/advanced_chunks.json`.
  - Giới hạn: Chưa có bước tách/quy đổi tự động giữa số trang in trên footer tài liệu (ví dụ trang in 67) và chỉ số trang PDF (trang 65), dẫn tới lỗi ở `adv_t24`. Chưa crop từng vùng bảng/biểu đồ riêng mà gửi cả trang PNG cho Vision model.
  - Lệnh tái lập: `python src/advanced_index.py` rồi `python src/advanced_agent.py --split dev` và `python src/advanced_agent.py --split test`.
