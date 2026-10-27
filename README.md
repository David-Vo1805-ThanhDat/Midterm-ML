# Midterm-ML: Thuật toán EM cho Gaussian Mixture Model

Tiểu luận giữa kỳ môn Nhập môn Học máy. Nhóm **tự cài đặt thuật toán Expectation–Maximization (EM)** bằng NumPy để học Gaussian Mixture Model (GMM), kiểm chứng trên dữ liệu tổng hợp 2D và áp dụng cho dữ liệu Old Faithful.

| Thành viên | Phụ trách chính |
|---|---|
| Võ Thành Đạt | Dữ liệu (I04), thiết kế và cài đặt EM (I07–I09), thực nghiệm và hình (I10–I12), README (I16) |
| Nguyễn Văn Tâm | Lý thuyết EM/GMM (I05), ví dụ tính tay (I06), đánh giá (I13) |
| Nguyễn Hoàng Minh | Điều phối (I01–I03), báo cáo (I14), slide (I15), kiểm tra và nộp bài (I17–I19) |

## Chạy nhanh

Từ thư mục gốc của repo, sau khi cài đặt (mục 2):

```bash
pytest                           # kiểm thử: tất cả phải pass
python -m src.run_experiments    # chạy thực nghiệm, ghi vào results/ (khoảng 10 giây)
python -m src.make_figures       # vẽ 6 hình vào results/figures/
```

## 1. Môi trường

| Thành phần | Phiên bản đã dùng để tạo `results/` | Ghi chú |
|---|---|---|
| Python | 3.12.6 (Windows 11) | Cũng đã kiểm thử với Python 3.11 |
| numpy | 2.1.2 | |
| matplotlib | 3.9.2 | |
| scikit-learn | 1.9.1 | Chỉ để đối chiếu và tính ARI (mục 7) |
| pytest | 9.0.3 | |

Có hai file requirements:

| File | Dùng khi |
|---|---|
| `requirements.txt` | Cài bình thường; ghi phiên bản **tối thiểu** |
| `requirements-lock.txt` | Muốn dùng **đúng** phiên bản đã tạo `results/` |

Kết quả đã được kiểm tra là **giống hệt** khi cài bản mới hơn (numpy 2.5.4, matplotlib 3.11.2). Phiên bản thực tế của mỗi lần chạy được lưu tự động trong `results/experiment-results.json` (trường `environment`).

## 2. Cài đặt

```bash
git clone https://github.com/David-Vo1805-ThanhDat/Midterm-ML.git
cd Midterm-ML
python -m venv .venv
```

Kích hoạt môi trường ảo:

| Hệ điều hành | Lệnh |
|---|---|
| Windows (PowerShell) | `.venv\Scripts\Activate.ps1` |
| Windows (cmd) | `.venv\Scripts\activate.bat` |
| macOS / Linux | `source .venv/bin/activate` |

Sau đó cài thư viện:

```bash
pip install -r requirements.txt        # hoặc: pip install -r requirements-lock.txt
```

Mọi lệnh bên dưới chạy từ **thư mục gốc của repo** (thư mục chứa file README này). Code không dùng đường dẫn tuyệt đối của máy cá nhân và không cần mạng (dữ liệu có sẵn trong `data/`).

## 3. Chạy

| Mục đích | Lệnh | Kết quả mong đợi |
|---|---|---|
| Kiểm tra dữ liệu | `python -m src.data` | Synthetic `(300, 2)` với `[155 88 57]` điểm mỗi thành phần; Old Faithful `(272, 2)` |
| Tải lại Old Faithful (không bắt buộc, cần mạng) | `python -m src.data --download` | Tải từ Rdatasets và kiểm tra SHA-256 |
| Chạy toàn bộ test | `pytest` | `41 passed` |
| Chạy toàn bộ thực nghiệm | `python -m src.run_experiments` | Khoảng 10 giây; in tóm tắt bên dưới và ghi file vào `results/` |
| Vẽ hình | `python -m src.make_figures` | 6 file PNG trong `results/figures/` |

Output mong đợi của `python -m src.run_experiments`:

```
Q1 synthetic: ll=-1145.379, iters=19, ARI=0.977, BIC selects K=3
   max |mu - mu_true| = 0.106
Q2 synthetic K=3: 20 seeds, 1 distinct optima, 20 reach best (best=-1145.379, worst=-1145.379)
Q2 synthetic K=5: 20 seeds, 13 distinct optima, 1 reach best (best=-1127.170, worst=-1136.459)
Q2 faithful K=2: 20 seeds, 1 distinct optima, 20 reach best (best=-385.461, worst=-385.461)
Q2 faithful K=4: 20 seeds, 11 distinct optima, 2 reach best (best=-361.230, worst=-371.248)
Q3 faithful: BIC selects K=2; means (min) = [[4.29, 79.97], [2.04, 54.48]]
Saved to <thư mục repo>/results
```

### Kết quả thực nghiệm (`results/`)

Các file kết quả đã được commit sẵn. Chạy lại sẽ ghi đè bằng nội dung giống hệt, chỉ khác thời điểm chạy (`generated_at`) và thông tin môi trường (`environment`). Để tự kiểm tra: chạy lại rồi dùng `git diff results/`.

| File | Nội dung |
|---|---|
| `experiment-results.json` | Cấu hình, phiên bản thư viện, tham số cuối, log-likelihood theo vòng lặp, BIC, sai số so với tham số thật, cảnh báo và lỗi của từng lần chạy |
| `bic.csv` | BIC theo K cho cả hai bộ dữ liệu |
| `init-study.csv` | Log-likelihood cuối của từng seed khởi tạo (câu hỏi Q2) |
| `synthetic-labels.csv`, `faithful-labels.csv` | Từng điểm dữ liệu kèm nhãn dự đoán và responsibility |

### Hình (`results/figures/`)

`python -m src.make_figures` vẽ **từ các file kết quả trên, không fit lại**, nên hình luôn khớp số liệu.

| File | Nội dung |
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
| Old Faithful | Härdle (1991); bộ `datasets` của R qua Rdatasets; lưu tại `data/faithful.csv` (272 × 2), có kiểm tra SHA-256 | Bài toán ứng dụng |

Nguồn, giấy phép, tiền xử lý (chuẩn hóa z-score) và giới hạn: xem `docs/data-and-use-case.md`.

## 5. Cấu hình thực nghiệm

Mọi tham số cố định nằm trong `src/config.py`. Muốn thử giá trị khác thì sửa ở đó và ghi lý do vào `docs/decision-log.md`.

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `SYNTHETIC_SEED` | 42 | Seed sinh dữ liệu tổng hợp |
| `EM_SEED` | 0 | Seed khởi tạo EM |
| `N_INIT` | 10 | Số lần khởi tạo; giữ lần có log-likelihood cao nhất |
| `MAX_ITER` | 500 | Số vòng lặp tối đa |
| `TOL` | 1e-6 | Dừng khi \|Δℓ\| / (1 + \|ℓ\|) < TOL |
| `REG_COVAR` | 1e-6 | Cộng vào đường chéo covariance để tránh suy biến |
| `K_RANGE` | 1…6 | Các giá trị K thử khi chọn bằng BIC |
| `INIT_STUDY_SEEDS` | 0…19 | Các seed khảo sát ảnh hưởng khởi tạo (Q2) |
| `INIT_STUDY_K` | tổng hợp: 3, 5; Old Faithful: 2, 4 | Các K dùng cho Q2: K đúng/được chọn và K lớn hơn |

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

**Giả định của code:** dữ liệu số thực, không có giá trị thiếu; covariance đầy đủ (full); K do người dùng chọn. Đầu vào sai (rỗng, NaN, K không hợp lệ…) bị báo lỗi ngay, xem `docs/algorithm-spec.md` mục 6.

## 7. Vai trò của scikit-learn

`sklearn.mixture.GaussianMixture` **không** được dùng để fit trong kết quả chính; mọi lần fit đều dùng `src/em_gmm.py`. scikit-learn chỉ xuất hiện ở hai chỗ:

| Chỗ dùng | Mục đích |
|---|---|
| `tests/test_em.py` | Kiểm chứng: với cùng khởi tạo, log-likelihood của EM tự cài đặt phải khớp `GaussianMixture` |
| `src/run_experiments.py` | `adjusted_rand_score` để **đánh giá** độ khớp với nhãn thật của dữ liệu tổng hợp (không dùng khi huấn luyện) |

## 8. Giới hạn đã biết

- Chỉ hỗ trợ covariance full; không có diagonal, tied hay spherical.
- Khởi tạo ngẫu nhiên có thể rơi vào local optimum; vì vậy dùng `n_init` > 1 (Q2 cho thấy rõ điều này khi K lớn).
- `reg_covar` làm covariance lệch rất nhỏ khỏi nghiệm MLE chính xác.
- K phải chọn từ bên ngoài (bằng BIC); `fit_gmm` không tự chọn K.
- GMM giả định các quan sát độc lập và mỗi cụm có dạng Gaussian. Cụm tìm được là mô tả mật độ dữ liệu, không phải bằng chứng về nguyên nhân vật lý. Xem `docs/data-and-use-case.md` mục 7.

## 9. Xử lý lỗi thường gặp

| Hiện tượng | Nguyên nhân | Cách sửa |
|---|---|---|
| `ModuleNotFoundError: No module named 'src'` | Chạy lệnh ở sai thư mục | `cd` về thư mục gốc của repo (chứa `README.md`) rồi chạy lại |
| PowerShell báo `running scripts is disabled` khi kích hoạt venv | Chính sách bảo mật mặc định của Windows | Chạy `Set-ExecutionPolicy -Scope Process Bypass` rồi kích hoạt lại (chỉ áp dụng cho cửa sổ hiện tại) |
| `python` không tìm thấy (Windows) | Python chưa có trong PATH | Dùng `py` thay cho `python`, ví dụ `py -m venv .venv` |
| `python` không tìm thấy (macOS/Linux) | Hệ thống dùng tên `python3` | Dùng `python3` thay cho `python` |
| `ValueError: Checksum ... không khớp` | File `data/faithful.csv` bị đổi ký tự xuống dòng hoặc bị sửa | `git checkout -- data/faithful.csv` (repo có `.gitattributes` giữ nguyên file này) |
| Cảnh báo về font khi vẽ hình (macOS/Linux) | Máy không có font Arial | Không ảnh hưởng: hình tự dùng font DejaVu Sans |

## 10. Tài liệu

| File | Nội dung |
|---|---|
| `docs/theory-notes.md` | Lý thuyết GMM và EM, công thức, tính chất hội tụ |
| `docs/algorithm-spec.md` | Thiết kế thuật toán: ký hiệu, pseudocode, quyết định, xử lý suy biến, ví dụ số |
| `docs/data-and-use-case.md` | Dữ liệu, nguồn, giấy phép, câu hỏi thực nghiệm, giới hạn |
| `docs/project-charter.md`, `docs/outline.md`, `docs/decision-log.md` | Phạm vi, phân công, dàn ý báo cáo, các quyết định của nhóm |

## 11. Cấu trúc thư mục

```
data/        faithful.csv (Old Faithful)
docs/        tài liệu (mục 10)
src/
  data.py             sinh dữ liệu tổng hợp, đọc/tải Old Faithful, chuẩn hóa
  em_gmm.py           thuật toán EM tự cài đặt
  config.py           cấu hình cố định của thực nghiệm
  run_experiments.py  chạy Q1–Q3, ghi kết quả vào results/
  make_figures.py     vẽ hình từ results/
tests/       kiểm thử pytest
results/     kết quả thực nghiệm (JSON, CSV) và figures/
report/      báo cáo
slides/      slide thuyết trình
```
