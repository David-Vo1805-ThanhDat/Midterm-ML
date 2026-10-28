"""Dữ liệu cho thực nghiệm EM/GMM (I04).

Hai nguồn dữ liệu:
- Dữ liệu tổng hợp 2D sinh từ 3 Gaussian đã biết tham số -> kiểm chứng EM.
- Old Faithful (272 x 2) -> bài toán ứng dụng thật, không có nhãn.

Nhãn thật của dữ liệu tổng hợp CHỈ dùng để đánh giá, không đưa vào huấn luyện.

Chạy thử:  python -m src.data              (in thống kê hai bộ dữ liệu)
Tải lại:   python -m src.data --download   (tải faithful.csv và kiểm tra SHA-256)
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ---------------------------------------------------------------------------
# Dữ liệu tổng hợp 2D
# ---------------------------------------------------------------------------
SYNTHETIC_SEED = 42
SYNTHETIC_N = 300

# Tham số thật (ground truth) của 3 thành phần Gaussian.
TRUE_WEIGHTS = np.array([0.5, 0.3, 0.2])
TRUE_MEANS = np.array([
    [0.0, 0.0],
    [4.0, 4.0],
    [-3.0, 4.0],
])
TRUE_COVS = np.array([
    [[1.0, 0.0], [0.0, 1.0]],     # tròn
    [[1.5, 0.8], [0.8, 1.0]],     # nghiêng, tương quan dương
    [[0.5, 0.0], [0.0, 2.0]],     # dẹt theo trục dọc
])


def make_synthetic(n: int = SYNTHETIC_N, seed: int = SYNTHETIC_SEED):
    """Sinh n điểm từ hỗn hợp 3 Gaussian.

    Trả về (X, z): X có shape (n, 2); z là nhãn thành phần thật, chỉ để đánh giá.
    """
    rng = np.random.default_rng(seed)
    z = rng.choice(len(TRUE_WEIGHTS), size=n, p=TRUE_WEIGHTS)
    X = np.empty((n, 2))
    for k in range(len(TRUE_WEIGHTS)):
        idx = z == k
        X[idx] = rng.multivariate_normal(TRUE_MEANS[k], TRUE_COVS[k], size=idx.sum())
    return X, z


# ---------------------------------------------------------------------------
# Old Faithful
# ---------------------------------------------------------------------------
FAITHFUL_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/faithful.csv"
FAITHFUL_PATH = DATA_DIR / "faithful.csv"
FAITHFUL_SHA256 = "5043db1e2c51c8e8fd67e0868c768ae589770cc76ad0ac0c5b7afd1fca31fc57"
FAITHFUL_FEATURES = ["eruptions", "waiting"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download_faithful(path: Path = FAITHFUL_PATH) -> Path:
    """Tải lại faithful.csv từ Rdatasets và kiểm tra checksum."""
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(FAITHFUL_URL, path)
    if _sha256(path) != FAITHFUL_SHA256:
        raise ValueError(f"Checksum của {path} không khớp; nguồn có thể đã thay đổi.")
    return path


def load_faithful(path: Path = FAITHFUL_PATH) -> np.ndarray:
    """Đọc Old Faithful, trả về mảng (272, 2): [eruptions, waiting] tính bằng phút."""
    if not path.exists():
        download_faithful(path)
    # Bỏ cột rownames, giữ eruptions và waiting.
    return np.genfromtxt(path, delimiter=",", skip_header=1, usecols=(1, 2))


# ---------------------------------------------------------------------------
# Tiền xử lý
# ---------------------------------------------------------------------------
def standardize(X: np.ndarray):
    """Chuẩn hóa z-score từng cột. Trả về (X_std, mean, std) để đổi ngược khi vẽ."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / std, mean, std


def _summary(name: str, X: np.ndarray) -> None:
    print(f"{name}: shape={X.shape}")
    print(f"  min ={X.min(axis=0)}")
    print(f"  max ={X.max(axis=0)}")
    print(f"  mean={X.mean(axis=0).round(3)}")


if __name__ == "__main__":
    import sys

    if "--download" in sys.argv:
        print(f"Downloaded and verified: {download_faithful()}")

    X_syn, z_syn = make_synthetic()
    _summary("Synthetic 2D", X_syn)
    print(f"  points per component: {np.bincount(z_syn)}")

    X_of = load_faithful()
    _summary("Old Faithful", X_of)
    X_of_std, _, _ = standardize(X_of)
    _summary("Old Faithful (standardized)", X_of_std)
