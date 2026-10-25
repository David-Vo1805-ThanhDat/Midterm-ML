# Thiết kế thuật toán EM cho GMM (I07)

> Trạng thái: **Chờ TV1 duyệt toán** · Owner: TV2 · Issue: #7 · Triển khai ở I08 (`src/em_gmm.py`)

Tài liệu này đủ chi tiết để một thành viên bất kỳ cài đặt lại EM cho Gaussian Mixture Model (GMM) bằng NumPy mà không cần hỏi thêm.

## 1. Phạm vi

| Có hỗ trợ | Không hỗ trợ (ghi vào giới hạn, mục 9) |
|---|---|
| GMM với covariance **đầy đủ** (full), D chiều bất kỳ, K thành phần | Covariance diagonal / tied / spherical |
| Khởi tạo ngẫu nhiên có seed, chạy lại nhiều lần (`n_init`) | Khởi tạo bằng K-means |
| Dừng theo độ thay đổi log-likelihood | Dữ liệu thiếu (NaN), dữ liệu streaming |
| Tính BIC để chọn K (dùng ở I11) | Tự động chọn K bên trong `fit` |

Chỉ dùng **NumPy**. sklearn chỉ được dùng trong test hoặc notebook để đối chiếu.

## 2. Ký hiệu và shape

Tên biến trong code trùng với ký hiệu trong báo cáo (theo Bishop §9.2).

| Ký hiệu | Tên trong code | Shape | Ý nghĩa |
|---|---|---|---|
| $N$ | `N` | số nguyên | Số điểm dữ liệu |
| $D$ | `D` | số nguyên | Số chiều (Old Faithful: 2; ví dụ tính tay: 1) |
| $K$ | `K` | số nguyên | Số thành phần |
| $\mathbf{X}$ | `X` | `(N, D)` | Dữ liệu; mỗi hàng là một điểm $\mathbf{x}_n$ |
| $\pi_k$ | `pi` | `(K,)` | Trọng số; $\pi_k \ge 0$, $\sum_k \pi_k = 1$ |
| $\boldsymbol{\mu}_k$ | `mu` | `(K, D)` | Trung bình của thành phần $k$ |
| $\boldsymbol{\Sigma}_k$ | `Sigma` | `(K, D, D)` | Ma trận hiệp phương sai, đối xứng, xác định dương |
| $\gamma_{nk}$ | `gamma` | `(N, K)` | Responsibility: xác suất điểm $n$ thuộc thành phần $k$; mỗi hàng cộng lại bằng 1 |
| $N_k$ | `Nk` | `(K,)` | Số điểm “hiệu dụng” của thành phần $k$: $\sum_n \gamma_{nk}$ |
| $\ell$ | `log_likelihood` | số thực | Log-likelihood của toàn bộ dữ liệu |
| $\varepsilon$ | `reg_covar` | số thực | Số nhỏ cộng vào đường chéo $\boldsymbol{\Sigma}_k$ |

**Quy ước đầu vào:** nếu `X` là mảng 1 chiều shape `(N,)`, code tự đổi thành `(N, 1)`. Nhờ vậy ví dụ tính tay 1D của I06 chạy được bằng cùng một hàm.

## 3. Công thức

**Mật độ Gaussian, tính ở dạng log để tránh tràn số:**

$$\log \mathcal{N}(\mathbf{x}\mid\boldsymbol{\mu},\boldsymbol{\Sigma}) = -\tfrac12\Big[D\log(2\pi) + \log|\boldsymbol{\Sigma}| + (\mathbf{x}-\boldsymbol{\mu})^\top\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\Big]$$

**E-step: tính responsibility**

$$\gamma_{nk} = \frac{\pi_k\,\mathcal{N}(\mathbf{x}_n\mid\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k)}{\sum_{j=1}^{K}\pi_j\,\mathcal{N}(\mathbf{x}_n\mid\boldsymbol{\mu}_j,\boldsymbol{\Sigma}_j)}$$

**M-step: cập nhật tham số bằng các γ vừa tính**

$$N_k = \sum_{n=1}^{N}\gamma_{nk},\qquad \pi_k = \frac{N_k}{N},\qquad \boldsymbol{\mu}_k = \frac{1}{N_k}\sum_{n=1}^{N}\gamma_{nk}\,\mathbf{x}_n$$

$$\boldsymbol{\Sigma}_k = \frac{1}{N_k}\sum_{n=1}^{N}\gamma_{nk}(\mathbf{x}_n-\boldsymbol{\mu}_k)(\mathbf{x}_n-\boldsymbol{\mu}_k)^\top + \varepsilon\mathbf{I}$$

($\boldsymbol{\mu}_k$ trong công thức $\boldsymbol{\Sigma}_k$ là giá trị **mới** vừa cập nhật.)

**Log-likelihood**

$$\ell = \sum_{n=1}^{N}\log\sum_{k=1}^{K}\pi_k\,\mathcal{N}(\mathbf{x}_n\mid\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k)$$

**BIC** (dùng ở I11 để chọn K), với số tham số $p = (K-1) + KD + K\frac{D(D+1)}{2}$:

$$\mathrm{BIC} = -2\ell + p\log N$$

## 4. Pseudocode

### 4.1 `log_gaussian(X, mu_k, Sigma_k)` → `(N,)`

```
L      ← cholesky(Sigma_k)                    # Sigma_k = L Lᵀ; lỗi nếu không xác định dương
Z      ← solve(L, (X − mu_k)ᵀ)                # shape (D, N)
maha   ← tổng theo cột của Z²                 # (x−μ)ᵀ Σ⁻¹ (x−μ) cho từng điểm, shape (N,)
logdet ← 2 · Σ log(diag(L))                   # log|Σ_k|
return −½ · (D·log(2π) + logdet + maha)
```

Dùng Cholesky thay cho `inv` và `det`: chính xác hơn, không bao giờ ra `log` của số âm, và báo lỗi ngay khi Σ không hợp lệ.

### 4.2 `e_step(X, pi, mu, Sigma)` → `(gamma, log_likelihood)`

```
for k in 0..K−1:
    log_r[:, k] ← log(pi[k]) + log_gaussian(X, mu[k], Sigma[k])     # shape (N, K)
m       ← max theo hàng của log_r                                   # (N,)
log_px  ← m + log( Σ_k exp(log_r[:, k] − m) )                      # log-sum-exp, (N,)
gamma   ← exp(log_r − log_px[:, None])                              # mỗi hàng cộng lại = 1
return gamma, Σ_n log_px[n]
```

**Vì sao cần log-sum-exp?** Với điểm ở xa, $\mathcal{N}(\cdot)$ có thể nhỏ hơn $10^{-308}$ và bị làm tròn thành 0, dẫn tới phép chia 0/0. Trừ đi giá trị lớn nhất `m` trước khi lấy `exp` thì số hạng lớn nhất luôn bằng $e^0 = 1$, nên không bao giờ bị tràn dưới.

### 4.3 `m_step(X, gamma, reg_covar)` → `(pi, mu, Sigma)`

```
Nk    ← tổng theo cột của gamma + 10·eps_máy      # (K,); cộng eps để không chia cho 0
pi    ← Nk / N
mu    ← (gammaᵀ @ X) / Nk[:, None]                 # (K, D)
for k in 0..K−1:
    d        ← X − mu[k]                           # (N, D)
    Sigma[k] ← (gamma[:, k, None] · d)ᵀ @ d / Nk[k] + reg_covar · I_D
return pi, mu, Sigma
```

### 4.4 `init_params(X, K, rng, reg_covar)` → `(pi, mu, Sigma)`

```
mu       ← K hàng khác nhau của X, chọn ngẫu nhiên bằng rng (không lặp)
Sigma[k] ← covariance của toàn bộ X (chia N) + reg_covar · I_D,   với mọi k
pi       ← [1/K, …, 1/K]
```

### 4.5 `fit_gmm(X, K, n_init=1, max_iter=200, tol=1e-6, reg_covar=1e-6, seed=None)` → `GMMResult`

```
X ← kiểm tra và chuẩn hóa đầu vào (mục 6); nếu 1 chiều thì đổi thành (N, 1)
rng ← default_rng(seed)
best ← None
lặp n_init lần:
    pi, mu, Sigma ← init_params(X, K, rng, reg_covar)
    gamma, ll ← e_step(X, pi, mu, Sigma)
    trace ← [ll]; converged ← False
    for t in 1..max_iter:
        pi, mu, Sigma ← m_step(X, gamma, reg_covar)       # M trước …
        gamma, ll     ← e_step(X, pi, mu, Sigma)          # … rồi E, để gamma và ll khớp tham số mới
        trace.append(ll)
        nếu ll < trace[−2] − 1e-8·|trace[−2]|: cảnh báo "log-likelihood giảm"   # dấu hiệu có lỗi
        nếu |ll − trace[−2]| / N < tol: converged ← True; break
    nếu không converged: cảnh báo "chưa hội tụ sau max_iter vòng"
    nếu best là None hoặc ll > best.log_likelihood: best ← kết quả lần chạy này
return best
```

### 4.6 Kết quả trả về: `GMMResult` (dataclass)

| Trường | Kiểu | Ghi chú |
|---|---|---|
| `pi`, `mu`, `Sigma` | ndarray | Tham số cuối, shape như mục 2 |
| `gamma` | `(N, K)` | Responsibility khớp với tham số cuối |
| `log_likelihood` | float | $\ell$ cuối cùng |
| `trace` | list[float] | $\ell$ sau mỗi vòng, phần tử 0 là trước vòng 1 (dùng để vẽ ở I12) |
| `n_iter` | int | Số vòng đã chạy |
| `converged` | bool | Có đạt tiêu chí dừng không |

Hàm phụ: `predict(result, X)` → nhãn `argmax` của `gamma` (để đánh giá ARI ở I13); `bic(result, X)` → BIC theo mục 3.

## 5. Các quyết định thiết kế

| Vấn đề | Quyết định | Lý do | Phương án đã cân nhắc |
|---|---|---|---|
| Loại covariance | **Full** | Old Faithful và bộ dữ liệu tổng hợp có cụm elip **nghiêng**, cần số hạng ngoài đường chéo | Diagonal: đơn giản hơn nhưng không mô tả được độ nghiêng |
| Khởi tạo μ | **K điểm dữ liệu ngẫu nhiên** | Dễ giải thích, dễ tái lập bằng seed, không cần cài thêm K-means | K-means (ổn định hơn, nhưng thêm một thuật toán phải cài và giải thích) |
| Khởi tạo Σ, π | Σ = covariance toàn bộ dữ liệu; π = 1/K | Bắt đầu “rộng”, mọi điểm đều có γ đáng kể, tránh thành phần rỗng ngay vòng đầu | Σ = I (sai thang đo nếu dữ liệu chưa chuẩn hóa) |
| Thứ tự trong vòng lặp | E-step đầu tiên, sau đó lặp M → E | `gamma` và `log_likelihood` trả về luôn khớp với tham số cuối | — |
| Ổn định số | `reg_covar = 1e-6` cộng vào đường chéo Σ; log-sum-exp; Cholesky; `Nk + 10·eps` | Gợi ý trong issue; cùng giá trị mặc định với sklearn nên dễ đối chiếu | Lớn hơn (1e-3) thì làm sai lệch Σ của dữ liệu nhỏ |
| Tiêu chí dừng | $\lvert\ell_t-\ell_{t-1}\rvert / N <$ `tol = 1e-6`, tối đa `max_iter = 200` | Chia cho N để `tol` không phụ thuộc kích thước dữ liệu | So sánh thay đổi tham số (phức tạp hơn, cần chuẩn) |
| Chạy nhiều lần | `n_init` lần, giữ lần có $\ell$ cao nhất; mặc định 1 | Giảm rủi ro local optimum; phục vụ Q2 của I04 | — |
| Seed | Một `rng = default_rng(seed)` dùng cho mọi lần khởi tạo | Cùng seed thì cùng kết quả | — |

**Đã kiểm chứng bằng bản cài thử** (trong thư mục nháp, không commit): trên dữ liệu tổng hợp của I04 (K = 3), cùng khởi tạo với sklearn, log-likelihood trung bình mỗi điểm khớp sklearn đến sai số $4\times10^{-7}$, hội tụ sau 11 vòng, và $\ell$ tăng ở mọi vòng.

## 6. Kiểm tra đầu vào

Mọi lỗi được báo **trước** khi chạy EM, kèm thông báo rõ ràng.

| Trường hợp | Hành vi |
|---|---|
| `X` không phải số hoặc không chuyển được thành mảng float | `TypeError` |
| `X` có số chiều khác 1 hoặc 2 | `ValueError` |
| `X` rỗng (N = 0 hoặc D = 0) | `ValueError` |
| `X` có NaN hoặc ±inf | `ValueError` |
| `K` không phải số nguyên, hoặc `K < 1` | `ValueError` |
| `K > N` (không đủ điểm để chọn K tâm khác nhau) | `ValueError` |
| `max_iter < 1`, `n_init < 1`, `tol <= 0`, `reg_covar < 0` | `ValueError` |

## 7. Thành phần có phương sai quá nhỏ (covariance suy biến)

**Hiện tượng:** nếu một thành phần chỉ “ôm” một điểm (hoặc nhiều điểm trùng nhau, ví dụ do dữ liệu Old Faithful bị làm tròn), thì $\boldsymbol{\Sigma}_k \to 0$. Khi đó $\mathcal{N}(\mathbf{x}_n\mid\cdot)\to\infty$ và $\ell\to\infty$. Đây là nghiệm **vô nghĩa** nhưng lại có likelihood “tốt nhất” (Bishop §9.2.1).

| Tình huống | Cách xử lý |
|---|---|
| Σ_k sắp suy biến | `reg_covar` giữ trị riêng nhỏ nhất ≥ $\varepsilon$, nên Σ luôn xác định dương và $\ell$ bị chặn trên |
| Cholesky vẫn thất bại | Báo `RuntimeError` ghi rõ thành phần k, gợi ý tăng `reg_covar` |
| Thành phần gần rỗng ($N_k < 1$, tức chưa tới một điểm hiệu dụng) | Cảnh báo `warnings.warn`, vẫn chạy tiếp; `n_init > 1` giúp chọn lần chạy khác tốt hơn |
| Chia cho $N_k = 0$ | Không xảy ra vì $N_k$ được cộng `10·eps` |

**Hệ quả cần ghi vào báo cáo:** vì có $\varepsilon$, M-step không còn là cực đại chính xác, nên $\ell$ về lý thuyết có thể giảm cực nhỏ. Do đó kiểm tra “ℓ không giảm” dùng sai số tương đối $10^{-8}$.

## 8. Ví dụ số kiểm chứng (1D, K = 2), TV2 tính độc lập cho I06

Dữ liệu: 6 giá trị đầu tiên của cột `eruptions` (Old Faithful), **chưa chuẩn hóa**, $\varepsilon = 0$.
Khởi tạo: $\pi = (0.5,\ 0.5)$, $\mu = (2,\ 4)$, $\sigma^2 = (1,\ 1)$.

**E-step:**

| $x_n$ | 3.600 | 1.800 | 3.333 | 2.283 | 4.533 | 2.883 |
|---|---|---|---|---|---|---|
| $\gamma_{n1}$ | 0.2315 | 0.9168 | 0.3394 | 0.8075 | 0.0445 | 0.5582 |
| $\gamma_{n2}$ | 0.7685 | 0.0832 | 0.6606 | 0.1925 | 0.9555 | 0.4418 |

Kiểm tra nhanh điểm đầu: $\mathcal{N}(3.6\mid2,1)\propto e^{-1.28}=0.278$, $\mathcal{N}(3.6\mid4,1)\propto e^{-0.08}=0.923$, nên $\gamma_{11} = 0.278/(0.278+0.923) = 0.2315$.

**M-step:**

| | Thành phần 1 | Thành phần 2 |
|---|---|---|
| $N_k$ | 2.898 | 3.102 |
| $\pi_k$ | 0.4830 | 0.5170 |
| $\mu_k$ | 2.5085 | 3.5984 |
| $\sigma_k^2$ | 0.4377 | 0.5510 |

**Log-likelihood:** trước vòng lặp $\ell = -8.9599$; sau một vòng $\ell = -7.7070$ (tăng, đúng tính chất của EM).

Khi TV1 hoàn thành I06: nếu dùng cùng dữ liệu và cùng khởi tạo, các số trên phải khớp đến 4 chữ số thập phân. Test ở I09 sẽ kiểm tra code I08 ra đúng các số này.

## 9. Giới hạn và những gì không hỗ trợ

1. Chỉ có covariance full; với D lớn, số tham số tăng theo $D^2$ (Old Faithful D = 2 nên không ảnh hưởng).
2. Khởi tạo ngẫu nhiên có thể rơi vào local optimum; cần `n_init > 1` khi chạy thực nghiệm.
3. Không xử lý dữ liệu thiếu; NaN bị từ chối ngay.
4. `reg_covar` làm Σ lệch rất nhỏ khỏi nghiệm MLE chính xác.
5. K phải chọn từ bên ngoài (bằng BIC ở I11); `fit_gmm` không tự chọn K.
6. Không đảm bảo thứ tự thành phần: thành phần 1 của lần chạy này có thể là thành phần 2 của lần khác (label switching). Khi so với tham số thật, phải ghép cặp theo μ gần nhất.

## 10. Bàn giao cho I08 và I09

**I08** cài đúng các hàm ở mục 4 trong `src/em_gmm.py`.

**I09** kiểm tra, ánh xạ theo checklist của issue:

| Checklist I09 | Test |
|---|---|
| Tổng γ mỗi điểm bằng 1 | `gamma.sum(axis=1) ≈ 1` |
| π không âm, tổng bằng 1; Σ hữu hạn và hợp lệ | `pi >= 0`, `pi.sum() ≈ 1`; Σ đối xứng, Cholesky thành công |
| Shape, dữ liệu rỗng hoặc sai kiểu, K không hợp lệ | Từng dòng ở mục 6 |
| Likelihood trace và điều kiện dừng | `trace` không giảm (sai số mục 7); `converged` đúng |
| Covariance suy biến | Dữ liệu có nhiều điểm trùng nhau: không crash, Σ vẫn xác định dương |
| Khớp ví dụ tính tay | Số ở mục 8 |
| Khớp sklearn | Cùng khởi tạo thì $\ell/N$ lệch < $10^{-5}$ |
