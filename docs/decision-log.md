# Decision Log - Issue I03

**Trạng thái:** Bản cập nhật chờ nhóm review. Nguồn: Issue I03, PDF đề bài, repo `main` và xác nhận trực tiếp của TV3 trong cuộc trao đổi này.

## Quyết định và căn cứ

| ID | Nội dung | Nguồn | Trạng thái |
|---|---|---|---|
| D-01 | TV3 xác nhận kết quả modulo 8 hiện tại bằng 1 và nhóm chọn EM. Issue #1 trước đó ghi `0 + 1 + 2 = 3`, tương ứng LDA. | TV3 và GitHub Issue #1 | Mâu thuẫn mở: cần phép tính/MSSV đã sửa hoặc xác nhận bằng văn bản của giảng viên |
| D-02 | Phạm vi minh họa EM bằng GMM. | Issue I03 và repo | Phạm vi dự kiến; nhóm review cùng charter |
| D-03 | Dùng dữ liệu tổng hợp 2D để kiểm chứng và Old Faithful làm ứng dụng; chọn K bằng BIC. | I04 đã duyệt/merge theo TV3; `docs/data-and-use-case.md` ghi “Đã duyệt” | Đã xác nhận |
| D-04 | Báo cáo phải có nguyên lý, quy trình với ví dụ tính bước, bài toán ứng dụng, mã nguồn và kết quả. | PDF đề bài | Ràng buộc đã xác nhận |
| D-05 | Nộp Word/PDF, mã nguồn Python và slide; hạn eLearning hết 30/10/2026. | PDF đề bài | Ràng buộc đã xác nhận |
| D-06 | Hạn nội bộ I03 ban đầu là 20/10/2026. | Issue I03 | Đã quá hạn; cần lịch tăng tốc mới |
| D-07 | TV1 là Võ Thành Đạt, TV2 là Nguyễn Văn Tâm, TV3 là Nguyễn Hoàng Minh. | TV3 cung cấp | Đã xác nhận |
| D-08 | Võ Thành Đạt phụ trách code EM; Nguyễn Văn Tâm phụ trách ví dụ tính tay; Nguyễn Hoàng Minh phụ trách slide. | TV3 cung cấp | Đã xác nhận |
| D-09 | Cả nhóm đã đồng ý phạm vi MVP EM/GMM. | TV3 xác nhận | Đã xác nhận trong trao đổi; approval/comment trên PR là bằng chứng review cuối |
| D-10 | Checkpoint tiếp theo vào tối thứ Ba 27/10/2026; 20:00 là giờ đề xuất. | TV3 cung cấp | Đã chốt ngày/buổi; chưa chốt giờ cụ thể |

## Câu hỏi còn mở

| ID | Câu hỏi | Owner | Trạng thái |
|---|---|---|---|
| Q-01 | Issue #1 đã đóng nhưng ghi modulo 8 = 3 và yêu cầu xác nhận bằng văn bản nếu giữ EM. Phép tính modulo 8 = 1 hiện tại hoặc xác nhận của giảng viên được lưu ở đâu? | TV3 | Mở; đang chặn việc coi dependency đã giải quyết |
| Q-02 | Các reviewer và owner đề xuất cho Chương 1, Chương 7, biên tập/gói nộp đã được thành viên tương ứng xác nhận chưa? | TV3 điều phối, nhóm chốt trên PR | Mở |
| Q-03 | Giờ checkpoint tối 27/10/2026 có chốt là 20:00 không? | TV3 đề xuất, cả nhóm chốt | Mở một phần |
| Q-04 | Điểm dữ liệu, khởi tạo và quy tắc làm tròn cho ví dụ tính tay là gì? | Nguyễn Văn Tâm | Mở |

## Action items khẩn cấp

Các ngày dưới đây là đề xuất do hạn nộp 30/10 đang gần; nhóm cần xác nhận ngay trong checkpoint kế tiếp.

| ID | Việc | Owner | Hạn đề xuất | Checkpoint/đầu ra |
|---|---|---|---|---|
| A-01 | Lưu phép tính MSSV cho modulo 8 = 1 hoặc xác nhận bằng văn bản của giảng viên; liên kết bằng chứng vào decision log. | TV3 | Khẩn cấp, trước checkpoint 27/10 | Giải quyết mâu thuẫn với Issue #1 |
| A-02 | Xác nhận reviewer đề xuất và phạm vi MVP trên PR. | TV3 + cả nhóm | 25/10 | Có approval/comment của thành viên |
| A-03 | Hoàn tất bản chạy EM tối thiểu và kết quả sơ bộ. | Võ Thành Đạt | 26/10 | Nguyễn Văn Tâm review đề xuất |
| A-04 | Hoàn tất ví dụ tính tay. | Nguyễn Văn Tâm | 26/10 | Võ Thành Đạt review đề xuất |
| A-05 | Hoàn thiện bản nháp báo cáo 7 chương và slide sơ bộ. | Nguyễn Hoàng Minh điều phối; các owner đóng góp | 27/10 | Nhóm review nội dung |
| A-06 | Review chéo, kiểm tra tái lập, khóa Word/PDF/Python/slide. | Cả nhóm | 28-29/10 | Checklist gói nộp hoàn tất |
| A-07 | Nộp trên eLearning trước khi hết hạn. | TV3 điều phối; cả nhóm kiểm tra | 30/10 | Xác nhận nộp thành công |

## Lịch sử cập nhật

| Ngày | Cập nhật | Người ghi |
|---|---|---|
| 25/10/2026 | Ghi nhận modulo 8 = 1, I04 đã duyệt/merge và tình trạng tiến độ trễ theo xác nhận của TV3; cập nhật đề cương và lịch đề xuất. | Codex theo thông tin TV3 cung cấp |
| 25/10/2026 | Điền tên thành viên và owner code/ví dụ/slide; thêm reviewer luân phiên dưới dạng đề xuất chờ nhóm xác nhận. | Codex theo thông tin TV3 cung cấp |
| 25/10/2026 | Đối chiếu dependency #1: issue đã đóng nhưng ghi modulo 8 = 3; ghi nhận nhóm đồng ý MVP và checkpoint tối 27/10. | Codex theo Issue #1 và thông tin TV3 cung cấp |
