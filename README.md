# Midterm-ML: Thuật toán EM cho Gaussian Mixture Model

Tiểu luận giữa kỳ môn Nhập môn Học máy. Nhóm **tự cài đặt thuật toán Expectation–Maximization (EM)** bằng NumPy để học Gaussian Mixture Model (GMM), kiểm chứng trên dữ liệu tổng hợp 2D và áp dụng cho dữ liệu Old Faithful.

| Thành viên | Phụ trách chính |
|---|---|
| Võ Thành Đạt | Dữ liệu (I04), thiết kế và cài đặt EM (I07–I09), thực nghiệm và hình (I10–I12), README (I16) |
| Nguyễn Văn Tâm | Lý thuyết EM/GMM (I05), ví dụ tính tay (I06), đánh giá (I13) |
| Nguyễn Hoàng Minh | Điều phối (I01–I03), báo cáo (I14), slide (I15), kiểm tra và nộp bài (I17–I19) |

## 1. Môi trường

Đã chạy và kiểm thử với:

| Thành phần | Phiên bản |
|---|---|
| Python | 3.12.6 (Windows 11) |
| numpy | 2.1.2 |
| matplotlib | 3.9.2 |
| scikit-learn | 1.9.1 (chỉ để đối chiếu) |
| pytest | 9.0.3 |

`requirements.txt` ghi phiên bản tối thiểu. Phiên bản thực tế của mỗi lần chạy thực nghiệm được lưu tự động trong file kết quả.

## 2. Cài đặt

```bash
git clone https://github.com/David-Vo1805-ThanhDat/Midterm-ML.git
cd Midterm-ML
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

Mọi lệnh bên dưới chạy từ **thư mục gốc của repo**. Code không dùng đường dẫn tuyệt đối của máy cá nhân.

## 3. Chạy

| Mục đích | Lệnh | Kết quả mong đợi |
|---|---|---|
| Kiểm tra dữ liệu | `python -m src.data` | Synthetic `(300, 2)` với `[155 88 57]` điểm mỗi thành phần; Old Faithful `(272, 2)` |
| Tải lại Old Faithful (không bắt buộc) | `python -m src.data --download` | Tải từ Rdatasets và kiểm tra SHA-256 |
| Chạy toàn bộ test | `pytest` | Tất cả test pass |
| Chạy toàn bộ thực nghiệm | `python -m src.run_experiments` | Khoảng 10 giây; in tóm tắt Q1–Q3 và ghi file vào `results/` |

**Kết quả thực nghiệm** (`results/`, tạo lại được bằng lệnh trên):

| File | Nội dung |
|---|---|
| `experiment-results.json` | Cấu hình, phiên bản thư viện, tham số cuối, log-likelihood theo vòng lặp, BIC, sai số so với tham số thật, cảnh báo và lỗi của từng lần chạy |
| `bic.csv` | BIC theo K cho cả hai bộ dữ liệu |
| `init-study.csv` | Log-likelihood cuối của 20 seed khởi tạo (câu hỏi Q2) |
| `synthetic-labels.csv`, `faithful-labels.csv` | Từng điểm dữ liệu kèm nhãn dự đoán và responsibility |

**Hình** (`python -m src.make_figures`, vẽ từ các file kết quả trên, không fit lại):

| File trong `results/figures/` | Nội dung |
|---|---|
| `fig1_data_before.png` | Hai bộ dữ liệu trước khi phân cụm |
| `fig2_synthetic_clusters.png` | Cụm dự đoán trên dữ liệu tổng hợp; Gaussian học được so với Gaussian thật |
| `fig3_faithful_clusters.png` | Cụm trên Old Faithful (đơn vị phút), khoanh các điểm không chắc chắn |
| `fig4_loglikelihood.png` | Log-likelihood theo vòng lặp |
| `fig5_bic.png` | BIC theo K cho hai bộ dữ liệu |
| `fig6_init_study.png` | Log-likelihood cuối theo seed khởi tạo (local optimum) |

Số thứ tự cụm do EM đặt tùy ý (label switching): “cụm 1” không có nghĩa đặc biệt gì.

## 4. Dữ liệu

| Bộ | Nguồn | Vai trò |
|---|---|---|
| Tổng hợp 2D | Sinh bằng `src/data.py::make_synthetic()`, seed 42, 3 Gaussian đã biết tham số | Kiểm chứng EM tìm lại đúng tham số |
| Old Faithful | Härdle (1991); bộ `datasets` của R qua Rdatasets; lưu tại `data/faithful.csv` (272 × 2) | Bài toán ứng dụng |

Nguồn, giấy phép, tiền xử lý (chuẩn hóa z-score) và giới hạn: xem `docs/data-and-use-case.md`.

## 5. Cấu hình thực nghiệm

Mọi tham số cố định nằm trong `src/config.py`:

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `EM_SEED` | 0 | Seed khởi tạo EM |
| `N_INIT` | 10 | Số lần khởi tạo; giữ lần có log-likelihood cao nhất |
| `MAX_ITER` | 500 | Số vòng lặp tối đa |
| `TOL` | 1e-6 | Dừng khi \|Δℓ\| / (1 + \|ℓ\|) < TOL |
| `REG_COVAR` | 1e-6 | Cộng vào đường chéo covariance để tránh suy biến |
| `K_RANGE` | 1…6 | Các giá trị K thử khi chọn bằng BIC |
| `INIT_STUDY_SEEDS` | 0…19 | Các seed dùng để khảo sát ảnh hưởng khởi tạo |

## 6. Dùng thư viện EM

```python
from src.data import load_faithful, standardize
from src.em_gmm import fit_gmm, predict, bic

X, mean, std = standardize(load_faithful())
result = fit_gmm(X, K=2, n_init=10, seed=0)

result.pi, result.mu, result.Sigma   # tham số học được
result.resp                          # responsibility r_nk, shape (N, K)
result.trace                         # log-likelihood sau mỗi vòng lặp
result.converged, result.n_iter
labels = predict(result, X)          # nhãn cứng: argmax của r_nk
score = bic(result, X)               # nhỏ hơn là tốt hơn
```

| Hàm | Mô tả |
|---|---|
| `fit_gmm(X, K, n_init, max_iter, tol, reg_covar, seed, init)` | Chạy EM; `init=(pi, mu, Sigma)` để tái hiện ví dụ tính tay |
| `e_step`, `m_step`, `log_gaussian`, `init_params` | Từng bước của thuật toán, dùng riêng được để minh họa |
| `predict`, `bic`, `n_parameters` | Hàm phụ cho thực nghiệm |

Thiết kế chi tiết: `docs/algorithm-spec.md`. Lý thuyết: `docs/theory-notes.md`.

**Giả định của code:** dữ liệu số thực, không có giá trị thiếu; covariance đầy đủ (full); K do người dùng chọn.

## 7. Vai trò của scikit-learn

`sklearn.mixture.GaussianMixture` **không** được dùng để fit trong kết quả chính. Nó chỉ xuất hiện trong test (`tests/test_em.py`) để kiểm chứng: với cùng khởi tạo, log-likelihood của EM tự cài đặt phải khớp sklearn.

## 8. Giới hạn đã biết

- Chỉ hỗ trợ covariance full; không có diagonal, tied hay spherical.
- Khởi tạo ngẫu nhiên có thể rơi vào local optimum; vì vậy dùng `n_init` > 1.
- `reg_covar` làm covariance lệch rất nhỏ khỏi nghiệm MLE chính xác.
- GMM giả định các quan sát độc lập và mỗi cụm có dạng Gaussian; xem thêm `docs/data-and-use-case.md` mục 7.

## 9. Cấu trúc thư mục

```
data/        dữ liệu Old Faithful (faithful.csv)
docs/        tài liệu: lý thuyết, thiết kế thuật toán, dữ liệu, quyết định nhóm
src/         data.py (dữ liệu), em_gmm.py (EM), config.py (cấu hình)
tests/       kiểm thử pytest
results/     kết quả thực nghiệm và hình
report/      báo cáo
slides/      slide thuyết trình
```
