# Đề cương báo cáo 7 chương - EM/GMM

**Trạng thái:** Bản cập nhật chờ nhóm review; cả nhóm đã đồng ý phạm vi MVP EM/GMM. TV3 xác nhận kết quả modulo 8 hiện tại bằng 1, nhưng Issue #1 cũ ghi modulo 8 bằng 3; cần lưu phép tính đã sửa hoặc xác nhận của giảng viên để đóng mâu thuẫn. Phương án dữ liệu I04 đã được duyệt và merge.

## Chương 1. Giới thiệu bài toán và mục tiêu

- Bài toán phân cụm dữ liệu chưa có nhãn và mục tiêu của báo cáo.
- Phạm vi: EM để ước lượng tham số GMM; đầu vào, đầu ra và giả định.
- Câu hỏi thực nghiệm: EM có học đúng trên dữ liệu sinh từ GMM không; kết quả thay đổi thế nào theo khởi tạo; Old Faithful được mô tả ra sao.
- Owner đề xuất: Nguyễn Hoàng Minh; reviewer đề xuất: Võ Thành Đạt.

## Chương 2. Cơ sở lý thuyết

- Gaussian, Gaussian mixture, biến ẩn và likelihood.
- Trọng số hỗn hợp, trung bình, covariance và responsibility.
- Dựa trên ghi chú hiện có tại `docs/theory-notes.md`.
- Owner: Nguyễn Văn Tâm; reviewer đề xuất: Nguyễn Hoàng Minh.

## Chương 3. Nguyên lý thuật toán EM

- Vì sao tối ưu likelihood trực tiếp khó khi nhãn thành phần là ẩn.
- E-step tính responsibility; M-step cập nhật trọng số, trung bình và covariance.
- Khởi tạo, lặp, điều kiện dừng, local optimum và covariance suy biến.
- Owner: Nguyễn Văn Tâm; reviewer đề xuất: Nguyễn Hoàng Minh.

## Chương 4. Quy trình và ví dụ tính từng bước

- Dùng tập nhỏ 1D lấy từ `eruptions` của Old Faithful, K = 2 (theo kế hoạch I04/I06).
- Nêu dữ liệu, tham số khởi tạo và quy tắc làm tròn.
- Tính responsibility ở E-step, cập nhật tham số ở M-step và diễn giải ít nhất một vòng đầy đủ.
- Owner: Nguyễn Văn Tâm; reviewer đề xuất: Võ Thành Đạt. Cần chốt các điểm dữ liệu cụ thể.

## Chương 5. Bài toán ứng dụng và thiết kế thực nghiệm

- Hai bộ dữ liệu: dữ liệu tổng hợp 2D (300 điểm, 3 Gaussian) để kiểm chứng; Old Faithful (272 × 2) làm ứng dụng.
- Tiền xử lý theo I04; mô tả đầu vào/đầu ra và nguồn dữ liệu.
- Với Old Faithful, thử K = 1,...,6 và chọn BIC nhỏ nhất qua nhiều khởi tạo.
- So sánh với `sklearn.mixture.GaussianMixture` chỉ để kiểm chứng; không dùng nhãn để chọn K.
- Owner dữ liệu/bài toán: Nguyễn Văn Tâm (TV2); reviewer đề xuất: Võ Thành Đạt (TV1).

## Chương 6. Cài đặt Python, kết quả và nhận xét

- Giải thích cài đặt EM cốt lõi, cấu trúc chương trình và cách chạy.
- Báo cáo log-likelihood theo vòng lặp và kiểm tra không giảm.
- Dữ liệu tổng hợp: đối chiếu tham số thật với tham số học được; báo cáo ảnh hưởng khởi tạo.
- Old Faithful: bảng BIC, K được chọn, tham số và biểu đồ; diễn giải cụm thận trọng.
- Owner: Võ Thành Đạt; reviewer đề xuất: Nguyễn Văn Tâm.

## Chương 7. Kết luận và hướng phát triển

- Trả lời câu hỏi thực nghiệm và tổng kết mức độ đạt mục tiêu.
- Hạn chế của giả định GMM, dữ liệu và cách đánh giá.
- Hướng cải tiến.
- Owner đề xuất: Nguyễn Hoàng Minh; reviewer đề xuất: Nguyễn Văn Tâm.

## Tài liệu tham khảo và phụ lục

Đặt sau 7 chương. Chuẩn hóa trích dẫn từ `docs/theory-notes.md` và `docs/data-and-use-case.md`. Phụ lục có thể chứa phép tính đầy đủ, hướng dẫn chạy và bảng kết quả bổ sung.

## Phân công cần chốt

| Phần | Owner | Reviewer | Đầu ra |
|---|---|---|---|
| Chương 1 | Nguyễn Hoàng Minh - đề xuất | Võ Thành Đạt - đề xuất | Bối cảnh, mục tiêu, phạm vi |
| Chương 2-3 | Nguyễn Văn Tâm | Nguyễn Hoàng Minh - đề xuất | Lý thuyết và EM |
| Chương 4 | Nguyễn Văn Tâm | Võ Thành Đạt - đề xuất | Ví dụ tính tay 1D |
| Chương 5 | Nguyễn Văn Tâm | Võ Thành Đạt - đề xuất | Thiết kế dữ liệu/thực nghiệm |
| Chương 6 | Võ Thành Đạt | Nguyễn Văn Tâm - đề xuất | Code, kết quả, biểu đồ |
| Chương 7 | Nguyễn Hoàng Minh - đề xuất | Nguyễn Văn Tâm - đề xuất | Kết luận, hạn chế |
| Slide | Nguyễn Hoàng Minh | Võ Thành Đạt - đề xuất | Slide hoàn chỉnh |
| Biên tập báo cáo, gói nộp | Nguyễn Hoàng Minh - đề xuất | Nguyễn Văn Tâm - đề xuất | Bộ file nộp hoàn chỉnh |
