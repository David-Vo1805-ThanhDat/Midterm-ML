# Dữ liệu và bài toán ứng dụng (I04)

> Trạng thái: **Chờ nhóm duyệt** · Owner: TV2 · Tham vấn: TV1 · Issue: #4

## 1. Quyết định

Nhóm dùng **hai bộ dữ liệu** với hai vai trò khác nhau:

| Vai trò | Dữ liệu | Lý do |
|---|---|---|
| **Kiểm chứng thuật toán** | Dữ liệu tổng hợp 2D (3 Gaussian) | Biết tham số thật → so sánh được π, μ, Σ học được với đáp án; vẽ trực tiếp |
| **Bài toán ứng dụng** | Old Faithful (272 × 2) | Dữ liệu thật, 2 chiều, nhỏ; là ví dụ GMM kinh điển trong Bishop §9.2 (khớp phần lý thuyết I05) |

### So sánh các phương án đã xét

| Tiêu chí | Tổng hợp 2D | Old Faithful | Iris |
|---|---|---|---|
| Kích thước | 300 × 2 (tùy chỉnh) | 272 × 2 | 150 × 4 |
| Có đáp án thật | Có (tham số + nhãn) | Không | Có nhãn loài |
| Trực quan hóa | Trực tiếp | Trực tiếp | Cần PCA / chọn 2 cột |
| Tái lập | Seed | File CSV + checksum | Có sẵn trong sklearn |
| Nguy cơ | Quá "đẹp", không phản ánh dữ liệu thật | Dữ liệu bị làm tròn | Dễ diễn giải sai "cụm = loài" |

Iris **không chọn** vì 4 chiều khó giải thích trong thời gian còn lại và dễ diễn giải quá mức.

## 2. Mô tả dữ liệu

### 2.1 Dữ liệu tổng hợp 2D

- **Sinh bằng:** `src/data.py::make_synthetic()` · `seed = 42` · `n = 300`
- **Đặc trưng:** `x1`, `x2` (không có đơn vị)
- **Tham số thật:**

| k | π_k | μ_k | Σ_k | Hình dạng |
|---|---|---|---|---|
| 1 | 0.5 | (0, 0) | [[1.0, 0.0], [0.0, 1.0]] | Tròn |
| 2 | 0.3 | (4, 4) | [[1.5, 0.8], [0.8, 1.0]] | Nghiêng, tương quan dương |
| 3 | 0.2 | (−3, 4) | [[0.5, 0.0], [0.0, 2.0]] | Dẹt theo trục dọc |

- Với seed 42, số điểm thực tế mỗi thành phần: **155 / 88 / 57**.
- Ba thành phần có hình dạng covariance khác nhau, nên đây là trường hợp **K-means làm không tốt còn GMM làm được**.

### 2.2 Old Faithful

| Mục | Nội dung |
|---|---|
| Mô tả | Thời gian phun và thời gian chờ đến lần phun kế tiếp của mạch nước Old Faithful, Yellowstone, Mỹ |
| Số quan sát | 272 |
| Đặc trưng | `eruptions`: thời gian phun (phút), 1.6 – 5.1 · `waiting`: thời gian chờ (phút), 43 – 96 |
| Nguồn gốc | Härdle, W. (1991). *Smoothing Techniques with Implementation in S*. Springer. Phiên bản đầy đủ hơn: Azzalini & Bowman (1990), *Applied Statistics* 39, 357–365, doi:10.2307/2347385 |
| Nơi lấy | Bộ `datasets` của R, qua Rdatasets: <https://vincentarelbundock.github.io/Rdatasets/csv/datasets/faithful.csv> |
| Giấy phép | Phân phối kèm R (gói `datasets`, GPL-2 \| GPL-3). Đây là số liệu đo đạc thực tế, dùng cho mục đích học thuật có trích nguồn. **NEED CONFIRMATION:** TV3 ghi lại vào I20 (quản lý nguồn) |
| Bản lưu trong repo | `data/faithful.csv` · SHA-256 `5043db1e2c51c8e8fd67e0868c768ae589770cc76ad0ac0c5b7afd1fca31fc57` |
| Lệnh tải lại | `python -c "from src.data import download_faithful; download_faithful()"` (tự kiểm tra checksum) |

## 3. Tiền xử lý

| Bước | Tổng hợp 2D | Old Faithful |
|---|---|---|
| Bỏ cột | — | `rownames` |
| Giá trị thiếu | Không có | Không có (đã kiểm tra trong `tests/test_data.py`) |
| Chuẩn hóa | Không (đã cùng thang đo) | **z-score từng cột** (`standardize()`) |

**Vì sao chuẩn hóa Old Faithful?** Độ lệch chuẩn của `waiting` (13.57 phút) gấp khoảng 12 lần của `eruptions` (1.14 phút). Về lý thuyết, với covariance đầy đủ, đổi thang đo tuyến tính chỉ làm log-likelihood lệch một hằng số, nên nghiệm tối ưu tương đương. Tuy vậy, chuẩn hóa vẫn giúp (1) bước khởi tạo dùng khoảng cách Euclid không bị cột `waiting` lấn át, (2) ma trận covariance có điều kiện số tốt hơn. Khi vẽ hình và báo cáo kết quả, nhóm đổi ngược về phút bằng `mean` và `std` trả về.

## 4. Câu hỏi thực nghiệm

**Q1. EM tự cài đặt có học đúng không?** (tổng hợp 2D)
So sánh π, μ, Σ học được với tham số thật. Kiểm tra log-likelihood **không giảm** qua các vòng lặp. Đối chiếu kết quả với `sklearn.mixture.GaussianMixture`, chỉ để kiểm chứng.

**Q2. Kết quả có ổn định khi đổi khởi tạo không?** (cả hai bộ)
Chạy nhiều seed khởi tạo, so sánh log-likelihood cuối và tham số để quan sát hiện tượng local optimum.

**Q3. Các lần phun của Old Faithful có tách thành các "chế độ" khác nhau không?** (ứng dụng)
GMM chọn bao nhiêu thành phần, mỗi thành phần có thời gian phun và thời gian chờ trung bình bao nhiêu, và hai biến tương quan thế nào bên trong từng thành phần.

## 5. Cách chọn K, không dùng nhãn thật

- Chạy EM với **K = 1, …, 6**, mỗi K chạy nhiều lần khởi tạo và giữ lần có log-likelihood cao nhất.
- Chọn K theo **BIC** = −2·ℓ̂ + p·ln(n), với số tham số p = (K − 1) + K·d + K·d(d+1)/2 (d = 2 ⇒ p = 6K − 1). Chọn K có BIC nhỏ nhất.
- ℓ̂ lấy từ EM tự cài đặt, **không** dùng nhãn.
- **Nhãn thật của dữ liệu tổng hợp chỉ dùng sau khi huấn luyện**, để đánh giá (ví dụ Adjusted Rand Index ở I13) và kiểm tra BIC có chọn đúng K = 3 hay không.
- Old Faithful không có nhãn. Kỳ vọng (theo Bishop §9.2) là K = 2, nhưng **nhóm sẽ báo cáo đúng giá trị BIC chọn ra**, không ép theo kỳ vọng.

## 6. Vì sao dùng GMM cho bài toán này

- Biểu đồ phân tán Old Faithful có **hai đám điểm hình elip, nghiêng**, nên mô hình hỗn hợp Gaussian với covariance đầy đủ là giả định hợp lý.
- GMM cho **phân cụm mềm** (responsibility γ_nk), nên các điểm nằm giữa hai đám được biểu diễn là "không chắc chắn" thay vì bị gán cứng như K-means.
- Không có nhãn: biến ẩn z (điểm thuộc thành phần nào) đúng là thứ EM được thiết kế để xử lý.

## 7. Giới hạn và nguy cơ diễn giải quá mức

1. **Cụm không phải là nhóm "có thật".** Một thành phần Gaussian chỉ là cách mô tả mật độ dữ liệu. Không được kết luận "mạch nước có 2 cơ chế vật lý" chỉ từ kết quả phân cụm.
2. **Giả định Gaussian.** Nếu một cụm thật có hình dạng lệch hoặc cong, GMM có thể dùng *nhiều* Gaussian để mô tả nó, làm K bị ước lượng lớn hơn số nhóm thật.
3. **Dữ liệu bị làm tròn.** Theo tài liệu của R, `eruptions` bị làm tròn mạnh, các bội số của 5 giây xuất hiện nhiều bất thường. Các giá trị trùng nhau có thể kéo một thành phần co lại (covariance gần suy biến).
4. **Old Faithful không độc lập theo thời gian.** Các lần phun liên tiếp có phụ thuộc, trong khi GMM coi các quan sát là i.i.d. Đây là đơn giản hóa.
5. **Local optimum và khởi tạo.** EM chỉ hội tụ đến điểm dừng, nên kết quả có thể khác nhau theo seed. Nhóm phải báo cáo số lần khởi tạo và seed.
6. **BIC chỉ là tiêu chí xấp xỉ**, đặc biệt khi n nhỏ (272). Nếu K = 2 và K = 3 có BIC gần nhau, nhóm nêu rõ điều đó thay vì khẳng định chắc chắn.
7. **Dữ liệu tổng hợp quá lý tưởng**: được sinh đúng từ mô hình GMM, nên kết quả tốt trên đó không chứng minh GMM tốt trên dữ liệu thật.

## 8. Cách tái tạo

```bash
pip install -r requirements.txt
python -m src.data      # in thống kê tóm tắt của cả hai bộ dữ liệu
pytest                  # kiểm tra seed, shape, checksum, chuẩn hóa
```

Kết quả mong đợi: Synthetic `(300, 2)` với `[155 88 57]` điểm mỗi thành phần; Old Faithful `(272, 2)`.

## 9. Việc chuyển sang issue sau

- **I06:** issue yêu cầu dữ liệu 1D nhỏ, K = 2, nên có thể lấy 4–6 giá trị của riêng cột `eruptions` (Old Faithful) để tính tay một vòng E/M.
- **I10:** cố định seed khởi tạo EM và số lần khởi tạo; cập nhật README.
- **I13:** tính ARI trên dữ liệu tổng hợp; viết phần giới hạn dựa trên mục 7.
- **I20:** ghi nguồn Härdle (1991), Azzalini & Bowman (1990), Rdatasets.
