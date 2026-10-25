# Project Charter - Tiểu luận giữa kỳ Nhập môn Học máy

**Issue:** I03 · **Owner:** TV3 - Nguyễn Hoàng Minh · **Milestone:** Phase 1 - Planning & Research  
**Trạng thái:** Bản cập nhật chờ nhóm review; tiến độ khẩn vì hạn nội bộ 20/10/2026 đã qua.

## Mục tiêu

Hoàn thành báo cáo Word/PDF, mã nguồn Python và slide cho tiểu luận giữa kỳ. Nội dung phải giải thích nguyên lý thuật toán, quy trình qua ví dụ tính từng bước, bài toán ứng dụng, mã nguồn thể hiện thuật toán và kết quả thực nghiệm. Mỗi thành viên đều tham gia thuyết trình; nhóm tối đa 3 thành viên.

## Phạm vi MVP

- Chủ đề nhóm đã thống nhất: Expectation-Maximization (EM), minh họa qua Gaussian Mixture Model (GMM). TV3 xác nhận kết quả modulo 8 hiện tại bằng 1; theo đề bài, số dư 1 là EM. Tuy nhiên Issue #1 trước đó ghi phép tính `0 + 1 + 2 = 3` và yêu cầu xác nhận bằng văn bản nếu giữ EM. Cần lưu phép tính modulo đã sửa hoặc xác nhận của giảng viên để giải quyết mâu thuẫn này.
- Kiểm chứng thuật toán trên dữ liệu tổng hợp 2D gồm 3 Gaussian; so sánh tham số học được với tham số đã biết và theo dõi log-likelihood.
- Ứng dụng trên dữ liệu Old Faithful (272 × 2); chọn số thành phần bằng BIC với K = 1,...,6 và nhiều lần khởi tạo theo thiết kế I04.
- Ví dụ tính tay một tập nhỏ 1D từ cột `eruptions`, K = 2, theo đầu việc I06.
- Tự cài đặt các bước EM cốt lõi bằng Python; `sklearn` chỉ dùng đối chiếu/kiểm chứng.

**Căn cứ trạng thái:** TV3 xác nhận modulo 8 = 1 và cả nhóm đã đồng ý phạm vi MVP EM/GMM. I04 đã được nhóm duyệt, merge vào `main`, pull về máy và tài liệu `docs/data-and-use-case.md` đã được cập nhật trạng thái “Đã duyệt”. Dependency #1 đã đóng nhưng nội dung của issue còn ghi kết quả modulo 8 = 3; đây là câu hỏi mở phải có bằng chứng giải quyết.

## Thành viên

| Vai trò | Họ tên |
|---|---|
| TV1 | Võ Thành Đạt |
| TV2 | Nguyễn Văn Tâm |
| TV3 | Nguyễn Hoàng Minh |

## Đầu ra, owner và reviewer

| Đầu ra | Owner | Reviewer | Trạng thái |
|---|---|---|---|
| Charter, outline, decision log và điều phối I03 | Nguyễn Hoàng Minh (TV3) | Võ Thành Đạt (TV1) - đề xuất | Chờ reviewer xác nhận |
| Cơ sở lý thuyết EM/GMM (`docs/theory-notes.md`) | Nguyễn Văn Tâm (TV2) | Nguyễn Hoàng Minh (TV3) - đề xuất | Đã có nội dung; chờ reviewer xác nhận |
| Dữ liệu và bài toán ứng dụng (`docs/data-and-use-case.md`) | Nguyễn Văn Tâm (TV2) | Võ Thành Đạt (TV1) - đề xuất từ vai trò tham vấn I04 | Đã duyệt; chờ xác nhận reviewer chính |
| Ví dụ tính tay 1D | Nguyễn Văn Tâm (TV2) | Võ Thành Đạt (TV1) - đề xuất | Theo I06; chờ reviewer xác nhận |
| Cài đặt EM, thí nghiệm và kết quả | Võ Thành Đạt (TV1) | Nguyễn Văn Tâm (TV2) - đề xuất | Cần ưu tiên hoàn thành; chờ reviewer xác nhận |
| Slide | Nguyễn Hoàng Minh (TV3) | Võ Thành Đạt (TV1) - đề xuất | Chờ reviewer xác nhận |
| Biên tập báo cáo và gói nộp | Nguyễn Hoàng Minh (TV3) - đề xuất | Nguyễn Văn Tâm (TV2) - đề xuất | Chờ nhóm xác nhận |

## RACI tạm thời

R = thực hiện, A = chịu trách nhiệm chốt, C = tham vấn, I = được cập nhật. Các vai trò reviewer ghi “đề xuất” cần được thành viên tương ứng xác nhận khi review PR.

| Công việc | Nguyễn Hoàng Minh (TV3) | Nguyễn Văn Tâm (TV2) | Võ Thành Đạt (TV1) | Giảng viên |
|---|---|---|---|---|
| Điều phối và hoàn thiện bộ tài liệu I03 | A/R | C | C | I |
| Theo dõi quyết định modulo 8 = 1, chủ đề EM | A/R | C | C | I |
| Dữ liệu/bài toán I04 | C | A/R | C | I |
| Lý thuyết EM/GMM | C | A/R | C | I |
| Ví dụ tính tay | C | A/R | R (review đề xuất) | I |
| Mã nguồn EM và kết quả | C | R (review đề xuất) | A/R | I |
| Slide, biên tập và đóng gói | A/R | C | C | I |
| Review nội bộ và đóng gói bài nộp | A | R | R | I |

Mỗi đầu ra cần một owner và ít nhất một reviewer khác owner. Các vai trò ghi “đề xuất” chỉ trở thành phân công chính thức sau khi thành viên liên quan xác nhận.

## Rủi ro và cách xử lý chậm tiến độ

| Rủi ro | Cách xử lý |
|---|---|
| Trễ các mốc nội bộ | Báo TV3 ngay khi có nguy cơ; xác định phần còn thiếu và thời gian thực tế; thu hẹp phần trình bày ngoài MVP; chuyển giao có người nhận và reviewer; cập nhật action log cùng nhóm. |
| EM cài đặt chưa ổn định hoặc kết quả không tái lập | Ưu tiên bản chạy tối thiểu trên dữ liệu tổng hợp; cố định seed/cấu hình; lưu log-likelihood và nhờ thành viên khác chạy/review. |
| Không kịp hoàn thiện báo cáo và slide | Chốt kết quả thí nghiệm trước, chia phần viết song song, đặt một người biên tập thống nhất; dành checkpoint riêng để rà file nộp. |
| Owner/reviewer chưa xác nhận đầy đủ | Dùng bảng phân công đề xuất trong charter; từng reviewer xác nhận trực tiếp trên PR để tạo bằng chứng review. |
| Kết quả modulo trong I03 mâu thuẫn Issue #1 | Ghi lại chữ số cuối MSSV và phép tính mới; nếu vẫn không ra 1, phải có xác nhận bằng văn bản của giảng viên cho đề tài EM trước khi coi dependency đã giải quyết. |

## Lịch tăng tốc đề xuất

Hạn nộp PDF theo đề bài là hết ngày 30/10/2026. Hạn hoàn thành I03 do nhóm cung cấp là 20/10/2026 và đã quá hạn. Checkpoint tiếp theo được thống nhất vào tối thứ Ba, 27/10/2026; giờ họp đề xuất là 20:00 và cần nhóm chốt. Lịch dưới đây là kế hoạch tăng tốc, trong đó các mốc ngoài checkpoint vẫn cần owner xác nhận:

| Mốc đề xuất | Đầu ra/checkpoint |
|---|---|
| 25/10 | Chốt owner/reviewer, kiểm tra dependency #1, khóa MVP và phân công |
| 26/10 | Bản EM chạy được, ví dụ tính tay và kết quả sơ bộ |
| Tối 27/10 (20:00 đề xuất) | Checkpoint: bản nháp đủ 7 chương, hình/bảng và slide sơ bộ |
| 28/10 | Review chéo, sửa nội dung và kiểm tra tái lập |
| 29/10 | Khóa Word/PDF, slide, mã nguồn và kiểm tra gói nộp |
| 30/10 | Nộp trên eLearning trước khi hết hạn |

## Điều kiện hoàn thành I03

Nhóm đã đồng ý phạm vi MVP. I03 hoàn thành khi charter và outline được cả nhóm review; các giả định/câu hỏi mở được đánh dấu; mọi phần có owner, đầu ra và reviewer; action items có owner, hạn và checkpoint; mâu thuẫn modulo với dependency #1 có bằng chứng giải quyết. Quy ước nhóm yêu cầu branch riêng và Pull Request trước khi merge vào `main`.
