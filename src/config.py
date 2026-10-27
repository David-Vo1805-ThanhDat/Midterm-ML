"""Cấu hình cố định cho mọi lần chạy thực nghiệm (I10).

Mọi thành viên dùng đúng các giá trị này thì thu được cùng kết quả.
Muốn thử cấu hình khác, sửa ở đây và ghi lý do vào docs/decision-log.md.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# Dữ liệu (chi tiết ở docs/data-and-use-case.md)
SYNTHETIC_SEED = 42        # seed sinh dữ liệu tổng hợp; trùng src/data.py
SYNTHETIC_N = 300
SYNTHETIC_TRUE_K = 3       # chỉ dùng để đánh giá, không dùng khi huấn luyện

# Tham số EM (lý do ở docs/algorithm-spec.md mục 5)
EM_SEED = 0                # seed khởi tạo EM
N_INIT = 10                # số lần khởi tạo, giữ lần có log-likelihood cao nhất
MAX_ITER = 500             # 200 chưa đủ cho K lớn trên Old Faithful
TOL = 1e-6                 # |Δℓ| / (1 + |ℓ|) < TOL thì dừng
REG_COVAR = 1e-6           # cộng vào đường chéo covariance

# Chọn K bằng BIC
K_RANGE = list(range(1, 7))

# Câu hỏi Q2 (phụ thuộc khởi tạo): chạy riêng từng seed, mỗi seed một lần khởi tạo
INIT_STUDY_SEEDS = list(range(20))


def as_dict() -> dict:
    """Cấu hình dạng dict để lưu kèm kết quả."""
    return {
        "synthetic_seed": SYNTHETIC_SEED, "synthetic_n": SYNTHETIC_N,
        "synthetic_true_k": SYNTHETIC_TRUE_K, "em_seed": EM_SEED, "n_init": N_INIT,
        "max_iter": MAX_ITER, "tol": TOL, "reg_covar": REG_COVAR,
        "k_range": K_RANGE, "init_study_seeds": INIT_STUDY_SEEDS,
    }
