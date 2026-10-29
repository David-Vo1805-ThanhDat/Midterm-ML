# Dữ liệu và bài toán ứng dụng (I04)

> Trạng thái: **Đã duyệt** · Owner: Võ Thành Đạt · Tham vấn: Nguyễn Văn Tâm · Issue: #4 · Cập nhật kết quả cuối: 29/10/2026

## 1. Bài toán, mục tiêu, đầu vào và đầu ra

**Bài toán.** Cho một tập điểm dữ liệu **không có nhãn**, tìm xem dữ liệu được tạo thành từ bao nhiêu nhóm, mỗi nhóm nằm ở đâu, rộng hẹp và nghiêng thế nào, và mỗi điểm thuộc nhóm nào với xác suất bao nhiêu. Nhóm mô hình hóa dữ liệu bằng **Gaussian Mixture Model (GMM)**: mật độ của dữ liệu là tổng có trọng số của K phân phối Gaussian. Vì không biết điểm nào sinh từ Gaussian nào (biến ẩn), nhóm ước lượng tham số bằng **thuật toán EM** tự cài đặt bằng NumPy.

**Mục tiêu.**

1. Kiểm chứng EM tự cài đặt học **đúng** tham số trên dữ liệu đã biết đáp án (câu hỏi Q1).
2. Đo mức **phụ thuộc vào khởi tạo** của EM (Q2).
3. Áp dụng cho dữ liệu thật Old Faithful: chọn số thành phần K và **mô tả các kiểu phun** (Q3).

**Đầu vào.**

| Đầu vào | Dạng | Ý nghĩa |
|---|---|---|
| Ma trận dữ liệu `X` | N × D số thực (ở đây D = 2) | Tổng hợp: 300 × 2; Old Faithful: 272 × 2 (thời gian phun, thời gian chờ, tính bằng phút), chuẩn hóa z-score trước khi fit |
| Số thành phần `K` | Số nguyên, 1 ≤ K ≤ N | Thử K = 1, …, 6 rồi chọn bằng BIC |
| Tham số chạy | `src/config.py` | Seed khởi tạo, số lần khởi tạo, số vòng tối đa, ngưỡng dừng `tol`, `reg_covar` |

Nhãn thật của dữ liệu tổng hợp **không** phải đầu vào; chỉ dùng sau khi huấn luyện để chấm điểm.

**Đầu ra.**

| Đầu ra | Dạng | Ý nghĩa | Lưu ở |
|---|---|---|---|
| Trọng số π | K số, tổng bằng 1 | Tỷ lệ dữ liệu thuộc mỗi thành phần | `experiment-results.json` |
| Trung bình μ | K × D | Tâm của mỗi thành phần | `experiment-results.json` |
| Covariance Σ | K × D × D | Độ rộng, hình dạng, độ nghiêng của mỗi thành phần | `experiment-results.json` |
| Responsibility r_nk | N × K, mỗi hàng tổng bằng 1 | Xác suất điểm n thuộc thành phần k (phân cụm mềm) | `synthetic-labels.csv`, `faithful-labels.csv` |
| Nhãn dự đoán | N số nguyên | Thành phần có r_nk lớn nhất (phân cụm cứng) | như trên |
| Log-likelihood ℓ theo vòng | Dãy số | Mức khớp dữ liệu; dùng để kiểm tra hội tụ | `experiment-results.json` |
| BIC theo K | Bảng | Chọn số thành phần | `bic.csv` |

**Cách áp dụng EM để giải bài toán.**

| Bước | Việc làm | Code |
|---|---|---|
| 1 | Đọc dữ liệu, kiểm tra checksum, chuẩn hóa z-score (Old Faithful) | `src/data.py` |
| 2 | Với mỗi K = 1, …, 6: khởi tạo ngẫu nhiên nhiều lần, chạy EM (E-step tính r_nk, M-step cập nhật π, μ, Σ) đến khi hội tụ, giữ lần có ℓ cao nhất | `src/em_gmm.py::fit_gmm` |
| 3 | Tính BIC = −2ℓ + p ln N với p = 6K − 1, chọn K có BIC nhỏ nhất | `src/em_gmm.py::bic`, `src/run_experiments.py` |
| 4 | Fit lại với K đã chọn, lấy r_nk và nhãn dự đoán cho từng điểm | `src/em_gmm.py::predict` |
| 5 | Đổi μ, Σ từ thang chuẩn hóa về phút: μ = z·std + mean, Σ = Σ_z ⊙ (std stdᵀ) | `src/run_experiments.py::experiment_q3` |
| 6 | Đánh giá: ARI và sai số tham số (dữ liệu tổng hợp); vẽ hình | `src/run_experiments.py`, `src/make_figures.py` |

## 2. Quyết định

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

## 3. Mô tả dữ liệu

### 3.1 Dữ liệu tổng hợp 2D

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

### 3.2 Old Faithful

| Mục | Nội dung |
|---|---|
| Mô tả | Thời gian phun và thời gian chờ đến lần phun kế tiếp của mạch nước phun (geyser) Old Faithful, Yellowstone, Mỹ |
| Số quan sát | 272 |
| Đặc trưng | `eruptions`: thời gian phun (phút), 1.6 – 5.1 · `waiting`: thời gian chờ (phút), 43 – 96 |
| Nguồn gốc | Härdle, W. (1991). *Smoothing Techniques with Implementation in S*. Springer. Phiên bản đầy đủ hơn: Azzalini & Bowman (1990), *Applied Statistics* 39, 357–365, doi:10.2307/2347385 |
| Nơi lấy | Bộ `datasets` của R, qua Rdatasets: <https://vincentarelbundock.github.io/Rdatasets/csv/datasets/faithful.csv> |
| Giấy phép | Phân phối kèm R (gói `datasets`, GPL-2 \| GPL-3). Đây là số liệu đo đạc thực tế, dùng cho mục đích học thuật có trích nguồn |
| Bản lưu trong repo | `data/faithful.csv` · SHA-256 `5043db1e2c51c8e8fd67e0868c768ae589770cc76ad0ac0c5b7afd1fca31fc57` |
| Lệnh tải lại | `python -m src.data --download` (tự kiểm tra checksum) |

## 4. Tiền xử lý

| Bước | Tổng hợp 2D | Old Faithful |
|---|---|---|
| Bỏ cột | — | `rownames` |
| Giá trị thiếu | Không có | Không có (đã kiểm tra trong `tests/test_data.py`) |
| Chuẩn hóa | Không (đã cùng thang đo) | **z-score từng cột** (`standardize()`) |

**Vì sao chuẩn hóa Old Faithful?** Độ lệch chuẩn của `waiting` (13.57 phút) gấp khoảng 12 lần của `eruptions` (1.14 phút). Về lý thuyết, với covariance đầy đủ, đổi thang đo tuyến tính chỉ làm log-likelihood lệch một hằng số, nên nghiệm tối ưu tương đương. Tuy vậy, chuẩn hóa vẫn giúp (1) bước khởi tạo dùng khoảng cách Euclid không bị cột `waiting` lấn át, (2) ma trận covariance có điều kiện số tốt hơn. Khi vẽ hình và báo cáo kết quả, nhóm đổi ngược về phút bằng `mean` và `std` trả về.

## 5. Câu hỏi thực nghiệm

**Q1. EM tự cài đặt có học đúng không?** (tổng hợp 2D)
So sánh π, μ, Σ học được với tham số thật. Kiểm tra log-likelihood **không giảm** qua các vòng lặp. Đối chiếu kết quả với `sklearn.mixture.GaussianMixture`, chỉ để kiểm chứng.

**Q2. Kết quả có ổn định khi đổi khởi tạo không?** (cả hai bộ)
Chạy nhiều seed khởi tạo, so sánh log-likelihood cuối và tham số để quan sát hiện tượng local optimum.

**Q3. Các lần phun của Old Faithful có tách thành các "chế độ" khác nhau không?** (ứng dụng)
GMM chọn bao nhiêu thành phần, mỗi thành phần có thời gian phun và thời gian chờ trung bình bao nhiêu, và hai biến tương quan thế nào bên trong từng thành phần.

## 6. Cách chọn K, không dùng nhãn thật

- Chạy EM với **K = 1, …, 6**, mỗi K chạy **100 lần khởi tạo** (`N_INIT_BIC`) và giữ lần có log-likelihood cao nhất. K lớn dễ kẹt ở local optimum nên cần nhiều khởi tạo hơn phép chạy chính (10 lần); lý do ghi ở decision-log D-11.
- Chọn K theo **BIC** = −2·ℓ̂ + p·ln(n), với số tham số p = (K − 1) + K·d + K·d(d+1)/2 (d = 2 ⇒ p = 6K − 1). Chọn K có BIC nhỏ nhất.
- ℓ̂ lấy từ EM tự cài đặt, **không** dùng nhãn.
- **Nhãn thật của dữ liệu tổng hợp chỉ dùng sau khi huấn luyện**, để đánh giá (Adjusted Rand Index) và kiểm tra BIC có chọn đúng K = 3 hay không.
- Old Faithful không có nhãn. Kỳ vọng (theo Bishop §9.2) là K = 2, nhưng **nhóm báo cáo đúng giá trị BIC chọn ra**, không ép theo kỳ vọng.

## 7. Vì sao dùng GMM cho bài toán này

- Biểu đồ phân tán Old Faithful có **hai đám điểm hình elip, nghiêng**, nên mô hình hỗn hợp Gaussian với covariance đầy đủ là giả định hợp lý.
- GMM cho **phân cụm mềm** (responsibility r_nk), nên các điểm nằm giữa hai đám được biểu diễn là "không chắc chắn" thay vì bị gán cứng như K-means.
- Không có nhãn: biến ẩn z (điểm thuộc thành phần nào) đúng là thứ EM được thiết kế để xử lý.

## 8. Giới hạn và nguy cơ diễn giải quá mức

1. **Cụm không phải là nhóm "có thật".** Một thành phần Gaussian chỉ là cách mô tả mật độ dữ liệu. Không được kết luận "mạch nước có 2 cơ chế vật lý" chỉ từ kết quả phân cụm.
2. **Giả định Gaussian.** Nếu một cụm thật có hình dạng lệch hoặc cong, GMM có thể dùng *nhiều* Gaussian để mô tả nó, làm K bị ước lượng lớn hơn số nhóm thật.
3. **Dữ liệu bị làm tròn.** Theo tài liệu của R, `eruptions` bị làm tròn mạnh, các bội số của 5 giây xuất hiện nhiều bất thường. Các giá trị trùng nhau có thể kéo một thành phần co lại (covariance gần suy biến).
4. **Old Faithful không độc lập theo thời gian.** Các lần phun liên tiếp có phụ thuộc, trong khi GMM coi các quan sát là i.i.d. Đây là đơn giản hóa.
5. **Local optimum và khởi tạo.** EM chỉ hội tụ đến điểm dừng, nên kết quả có thể khác nhau theo seed. Nhóm báo cáo số lần khởi tạo và seed.
6. **BIC chỉ là tiêu chí xấp xỉ**, đặc biệt khi n nhỏ (272). Thực tế K = 2 và K = 3 của Old Faithful chỉ chênh 1.988 điểm BIC (mục 9), nên nhóm nêu rõ điều đó thay vì khẳng định chắc chắn.
7. **Dữ liệu tổng hợp quá lý tưởng**: được sinh đúng từ mô hình GMM, nên kết quả tốt trên đó không chứng minh GMM tốt trên dữ liệu thật.

## 9. Kết quả và nhận xét

Mọi số dưới đây lấy từ `results/` (chạy lại bằng `python -m src.run_experiments`, cấu hình trong `src/config.py`).

### Q1. EM học đúng trên dữ liệu tổng hợp

| Chỉ số | Kết quả |
|---|---|
| Hội tụ | 19 vòng, ℓ = −1145.379; ℓ không giảm ở vòng nào |
| ARI so với nhãn thật | **0.977** (2/300 điểm khác nhãn thật sau khi ghép thành phần) |
| Sai số lớn nhất so với tham số thật | π: 0.015 · μ: 0.106 · Σ: 0.373 (chuẩn Frobenius) |
| BIC | Nhỏ nhất tại **K = 3** (2387.722); K = 4 cao hơn 4.991 |
| Đối chiếu scikit-learn | Cùng khởi tạo: ℓ/N lệch < 1e-5 (test `test_matches_sklearn_with_same_init`) |

**Nhận xét.** EM tự cài đặt tìm lại đúng ba thành phần và đúng số thành phần mà không dùng nhãn. Sai số của Σ lớn nhất vì covariance là đại lượng bậc hai ước lượng từ vài chục điểm (thành phần nhỏ nhất có 57 điểm); đây là biến động mẫu, không phải lỗi công thức. Thứ tự thành phần do EM đặt khác thứ tự thật (label switching), nên phải ghép thành phần trước khi so và dùng ARI thay cho độ chính xác.

### Q2. Phụ thuộc khởi tạo

| Dữ liệu | K | Số nghiệm khác nhau / 20 seed | Số seed đạt nghiệm tốt nhất |
|---|---|---|---|
| Tổng hợp | 3 (đúng) | 1 | 20/20 |
| Tổng hợp | 5 (thừa) | 13 | 1/20 |
| Old Faithful | 2 (được chọn) | 1 | 20/20 |
| Old Faithful | 4 (thừa) | 11 | 2/20 |

**Nhận xét.** Với K đúng, EM ổn định; với K thừa, EM kẹt ở nhiều local optimum. Hệ quả thực tế: ban đầu bước chọn K dùng 10 khởi tạo, khiến Old Faithful K = 3 dừng ở ℓ = −374.413 thay vì −369.637 và phóng đại khoảng cách BIC giữa K = 2 và K = 3 từ 1.988 lên 11.539. Nhóm đã tăng lên 100 khởi tạo cho bước chọn K (D-11). Ở K lớn, nghiệm tốt hơn đôi khi chứa một thành phần chỉ khoảng 3 điểm (ví dụ tổng hợp, K = 4); với thành phần gần suy biến như vậy, `reg_covar` làm M-step lệch rất nhỏ khỏi cực đại chính xác và ℓ có thể giảm cỡ 1e-4 ở vòng cuối. Mọi cảnh báo này được ghi lại trong `experiment-results.json`, không bị bỏ qua.

### Q3. Old Faithful có hai kiểu phun

| K | 1 | **2** | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| BIC | 1118.016 | **832.585** | 834.573 | 851.389 | 872.820 | 890.922 |

| Thành phần | π | Phun trung bình | Chờ trung bình | Độ lệch chuẩn (phun; chờ) | Tương quan trong thành phần |
|---|---|---|---|---|---|
| Phun dài, chờ dài | 0.644 | 4.290 phút | 79.968 phút | 0.412; 6.004 | 0.380 |
| Phun ngắn, chờ ngắn | 0.356 | 2.036 phút | 54.479 phút | 0.263; 5.805 | 0.285 |

**Nhận xét.**

- BIC chọn **K = 2**, nhưng chỉ hơn K = 3 khoảng **1.988**: đây là lựa chọn theo tiêu chí trong tập K đã thử, không phải bằng chứng chắc chắn có đúng hai nhóm.
- Hai thành phần cho thấy **lần phun càng dài thì thời gian chờ đến lần sau càng dài**. Ngay trong mỗi thành phần, hai biến vẫn tương quan dương (0.29–0.38), nên covariance đầy đủ hợp lý hơn giả định cụm tròn của K-means.
- Phân cụm rất tách bạch: 271/272 điểm có xác suất thuộc cụm của mình ≥ 0.9. Chỉ một điểm (phun 2.9 phút, chờ 63 phút) nằm giữa hai cụm, với xác suất khoảng 0.80 thuộc cụm phun ngắn.
- Old Faithful không có nhãn nên không tính được ARI; đánh giá dựa trên log-likelihood, BIC và tính hợp lý của tham số.

## 10. Cách tái tạo

```bash
pip install -r requirements-lock.txt
python -m src.data              # thống kê hai bộ dữ liệu
pytest                          # 41 test
python -m src.run_experiments   # kết quả vào results/ (khoảng 1 phút)
python -m src.make_figures      # 6 hình vào results/figures/
```

Kết quả mong đợi của `python -m src.data`: Synthetic `(300, 2)` với `[155 88 57]` điểm mỗi thành phần; Old Faithful `(272, 2)`.

## 11. Liên kết với các issue khác

- **I06:** ví dụ tính tay dùng dữ liệu 1D x = 0, 2, 5 với K = 2 (`docs/em-gmm-one-iteration.md`); code tái hiện đúng các con số đó.
- **I10, I16:** seed, số lần khởi tạo và lệnh chạy nằm trong `src/config.py` và `README.md`.
- **I13:** đánh giá chi tiết và phân tích giới hạn ở `docs/I13-evaluation-results.md`, `docs/I13-comparison-limitations.md`.
- **I20:** nguồn Härdle (1991), Azzalini & Bowman (1990), Rdatasets.
