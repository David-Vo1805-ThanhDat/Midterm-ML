"""Thuật toán EM cho Gaussian Mixture Model, tự cài đặt bằng NumPy (I08).

Thiết kế, ký hiệu và lý do của từng quyết định: docs/algorithm-spec.md (I07).
Công thức: docs/theory-notes.md (I05), theo Bishop (2006) §9.2–9.4.

Ánh xạ ký hiệu báo cáo -> tên biến trong code:
    pi_k -> pi (K,)   mu_k -> mu (K, D)   Sigma_k -> Sigma (K, D, D)
    r_nk -> resp (N, K)   N_k -> Nk (K,)   lambda -> reg_covar

Ví dụ:
    from src.em_gmm import fit_gmm
    result = fit_gmm(X, K=2, n_init=10, seed=0)
    result.mu, result.log_likelihood, result.trace
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

# Cộng vào N_k để không bao giờ chia cho 0 (thành phần rỗng).
_EPS = 10 * np.finfo(float).eps


@dataclass
class GMMResult:
    """Kết quả của fit_gmm (lần khởi tạo tốt nhất)."""

    pi: np.ndarray            # (K,)      mixing weights
    mu: np.ndarray            # (K, D)    trung bình
    Sigma: np.ndarray         # (K, D, D) covariance
    resp: np.ndarray          # (N, K)    responsibility, khớp với tham số cuối
    log_likelihood: float     # ℓ cuối cùng
    trace: list = field(default_factory=list)                # ℓ sau mỗi vòng (phần tử 0: trước vòng 1)
    n_iter: int = 0
    converged: bool = False
    all_log_likelihoods: list = field(default_factory=list)  # ℓ cuối của từng lần khởi tạo


# ---------------------------------------------------------------------------
# Kiểm tra đầu vào (spec mục 6)
# ---------------------------------------------------------------------------
def _check_X(X) -> np.ndarray:
    try:
        X = np.asarray(X, dtype=float)
    except (TypeError, ValueError) as e:
        raise TypeError(f"X phải là mảng số thực: {e}") from None
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2:
        raise ValueError(f"X phải có shape (N,) hoặc (N, D), nhận được ndim={X.ndim}")
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError(f"X rỗng, shape={X.shape}")
    if not np.all(np.isfinite(X)):
        raise ValueError("X chứa NaN hoặc vô cực")
    return X


def _check_K(K, N: int) -> int:
    if isinstance(K, bool) or not isinstance(K, (int, np.integer)):
        raise ValueError(f"K phải là số nguyên, nhận được {K!r}")
    if K < 1:
        raise ValueError(f"K phải >= 1, nhận được {K}")
    if K > N:
        raise ValueError(f"K={K} lớn hơn số điểm N={N}")
    return int(K)


def _check_init(init, K: int, D: int):
    pi, mu, Sigma = (np.asarray(a, dtype=float) for a in init)
    mu = mu.reshape(K, D)
    Sigma = Sigma.reshape(K, D, D)
    if pi.shape != (K,) or np.any(pi < 0) or not np.isclose(pi.sum(), 1.0):
        raise ValueError("init: pi phải có shape (K,), không âm và tổng bằng 1")
    return pi, mu, Sigma


# ---------------------------------------------------------------------------
# Các bước của thuật toán (spec mục 4)
# ---------------------------------------------------------------------------
def log_gaussian(X: np.ndarray, mu_k: np.ndarray, Sigma_k: np.ndarray) -> np.ndarray:
    """log N(x_n | mu_k, Sigma_k) cho mọi điểm, shape (N,). Dùng Cholesky thay cho inv/det."""
    D = X.shape[1]
    L = np.linalg.cholesky(Sigma_k)                 # Sigma_k = L L^T
    Z = np.linalg.solve(L, (X - mu_k).T)            # (D, N)
    maha = np.sum(Z ** 2, axis=0)                   # (x-mu)^T Sigma^-1 (x-mu)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))       # log|Sigma_k|
    return -0.5 * (D * np.log(2 * np.pi) + logdet + maha)


def e_step(X, pi, mu, Sigma):
    """E-step: trả về (resp, log_likelihood). Dùng log-sum-exp để không bị tràn số."""
    K = len(pi)
    log_r = np.empty((X.shape[0], K))
    for k in range(K):
        try:
            log_r[:, k] = np.log(pi[k]) + log_gaussian(X, mu[k], Sigma[k])
        except np.linalg.LinAlgError:
            raise RuntimeError(
                f"Covariance của thành phần {k} không xác định dương; hãy tăng reg_covar"
            ) from None
    m = log_r.max(axis=1, keepdims=True)
    log_px = m[:, 0] + np.log(np.exp(log_r - m).sum(axis=1))   # log p(x_n)
    resp = np.exp(log_r - log_px[:, None])
    return resp, float(log_px.sum())


def m_step(X, resp, reg_covar):
    """M-step: cập nhật (pi, mu, Sigma) từ responsibility."""
    N, D = X.shape
    Nk = resp.sum(axis=0) + _EPS
    if np.any(Nk < 1.0):
        warnings.warn(f"Có thành phần gần rỗng (N_k < 1): {np.round(Nk, 4)}", stacklevel=3)
    pi = Nk / Nk.sum()
    mu = (resp.T @ X) / Nk[:, None]
    Sigma = np.empty((len(Nk), D, D))
    for k in range(len(Nk)):
        d = X - mu[k]
        Sigma[k] = (resp[:, k, None] * d).T @ d / Nk[k] + reg_covar * np.eye(D)
    return pi, mu, Sigma


def init_params(X, K, rng, reg_covar):
    """Khởi tạo: mu = K điểm dữ liệu ngẫu nhiên, Sigma = covariance toàn bộ X, pi = 1/K."""
    N, D = X.shape
    mu = X[rng.choice(N, size=K, replace=False)].copy()
    cov = np.atleast_2d(np.cov(X, rowvar=False, bias=True)) + reg_covar * np.eye(D)
    Sigma = np.repeat(cov[None], K, axis=0)
    pi = np.full(K, 1.0 / K)
    return pi, mu, Sigma


# ---------------------------------------------------------------------------
# Vòng lặp chính
# ---------------------------------------------------------------------------
def _run_once(X, pi, mu, Sigma, max_iter, tol, reg_covar) -> GMMResult:
    resp, ll = e_step(X, pi, mu, Sigma)
    trace = [ll]
    converged = False
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        pi, mu, Sigma = m_step(X, resp, reg_covar)
        resp, ll = e_step(X, pi, mu, Sigma)
        prev = trace[-1]
        trace.append(ll)
        if ll < prev - 1e-8 * (1 + abs(prev)):
            warnings.warn(f"Log-likelihood giảm ở vòng {n_iter}: {prev:.6f} -> {ll:.6f}", stacklevel=3)
        if abs(ll - prev) / (1 + abs(prev)) < tol:
            converged = True
            break
    return GMMResult(pi, mu, Sigma, resp, ll, trace, n_iter, converged)


def fit_gmm(X, K, *, n_init=1, max_iter=200, tol=1e-6, reg_covar=1e-6, seed=None, init=None) -> GMMResult:
    """Học GMM bằng EM. Chạy n_init lần khởi tạo, trả về lần có log-likelihood cao nhất.

    init: (pi, mu, Sigma) cho trước, dùng để tái hiện ví dụ tính tay; khi có init thì n_init = 1.
    """
    X = _check_X(X)
    N, D = X.shape
    K = _check_K(K, N)
    if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError(f"max_iter phải là số nguyên >= 1, nhận được {max_iter!r}")
    if isinstance(n_init, bool) or not isinstance(n_init, (int, np.integer)) or n_init < 1:
        raise ValueError(f"n_init phải là số nguyên >= 1, nhận được {n_init!r}")
    if not tol > 0:
        raise ValueError(f"tol phải > 0, nhận được {tol!r}")
    if not reg_covar >= 0:
        raise ValueError(f"reg_covar phải >= 0, nhận được {reg_covar!r}")

    rng = np.random.default_rng(seed)
    best = None
    all_ll = []
    for _ in range(1 if init is not None else n_init):
        start = _check_init(init, K, D) if init is not None else init_params(X, K, rng, reg_covar)
        result = _run_once(X, *start, max_iter, tol, reg_covar)
        all_ll.append(result.log_likelihood)
        if best is None or result.log_likelihood > best.log_likelihood:
            best = result
    if not best.converged:
        warnings.warn(f"EM chưa hội tụ sau max_iter={max_iter} vòng", stacklevel=2)
    best.all_log_likelihoods = all_ll
    return best


# ---------------------------------------------------------------------------
# Hàm phụ cho thực nghiệm
# ---------------------------------------------------------------------------
def predict(result: GMMResult, X) -> np.ndarray:
    """Nhãn cứng: thành phần có responsibility lớn nhất cho mỗi điểm."""
    resp, _ = e_step(_check_X(X), result.pi, result.mu, result.Sigma)
    return resp.argmax(axis=1)


def n_parameters(K: int, D: int) -> int:
    """Số tham số tự do của GMM covariance full: (K-1) + K*D + K*D*(D+1)/2."""
    return (K - 1) + K * D + K * D * (D + 1) // 2


def bic(result: GMMResult, X) -> float:
    """BIC = -2*log_likelihood + p*ln(N). Nhỏ hơn là tốt hơn."""
    N, D = _check_X(X).shape
    K = len(result.pi)
    return -2.0 * result.log_likelihood + n_parameters(K, D) * np.log(N)
