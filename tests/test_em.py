"""Kiểm thử EM cho GMM (I09). Mỗi nhóm test ứng với một dòng checklist của issue #9.

Số liệu kỳ vọng đều có căn cứ: ví dụ tính tay ở docs/algorithm-spec.md mục 8,
tham số thật của dữ liệu tổng hợp ở src/data.py, hoặc sklearn tính trực tiếp trong test.
"""

import warnings

import numpy as np
import pytest
from sklearn.mixture import GaussianMixture

from src.data import TRUE_MEANS, load_faithful, make_synthetic, standardize
from src.em_gmm import (
    bic,
    e_step,
    fit_gmm,
    init_params,
    log_gaussian,
    m_step,
    n_parameters,
    predict,
)

HAND_X = [3.600, 1.800, 3.333, 2.283, 4.533, 2.883]
HAND_INIT = ([0.5, 0.5], [2.0, 4.0], [1.0, 1.0])


@pytest.fixture(scope="module")
def synthetic():
    X, z = make_synthetic()
    return X, z


@pytest.fixture(scope="module")
def synthetic_fit(synthetic):
    X, _ = synthetic
    return fit_gmm(X, 3, n_init=5, seed=0)


# ---------------------------------------------------------------------------
# 1. Responsibility: mỗi hàng cộng lại bằng 1
# ---------------------------------------------------------------------------
def test_resp_rows_sum_to_one(synthetic, synthetic_fit):
    X, _ = synthetic
    r = synthetic_fit
    assert r.resp.shape == (300, 3)
    np.testing.assert_allclose(r.resp.sum(axis=1), 1.0, atol=1e-10)
    assert np.all((r.resp >= 0) & (r.resp <= 1))


def test_e_step_no_underflow_for_far_points():
    # Điểm cách tâm 1000 độ lệch chuẩn: mật độ = 0 nếu tính trực tiếp, log-sum-exp vẫn đúng.
    X = np.array([[1000.0], [-1000.0]])
    resp, ll = e_step(X, np.array([0.5, 0.5]), np.array([[0.0], [1.0]]),
                      np.array([[[1.0]], [[1.0]]]))
    assert np.all(np.isfinite(resp)) and np.isfinite(ll)
    np.testing.assert_allclose(resp.sum(axis=1), 1.0)
    assert resp[0, 1] > 0.99 and resp[1, 0] > 0.99


# ---------------------------------------------------------------------------
# 2. Tham số hợp lệ: pi không âm, tổng 1; Sigma đối xứng, xác định dương
# ---------------------------------------------------------------------------
def test_parameters_are_valid(synthetic_fit):
    r = synthetic_fit
    assert np.all(r.pi >= 0)
    assert r.pi.sum() == pytest.approx(1.0)
    assert np.all(np.isfinite(r.mu)) and np.all(np.isfinite(r.Sigma))
    for S in r.Sigma:
        np.testing.assert_allclose(S, S.T)
        assert np.all(np.linalg.eigvalsh(S) > 0)


def test_log_gaussian_matches_direct_formula():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(5, 2))
    mu = np.array([0.3, -0.2])
    S = np.array([[1.5, 0.4], [0.4, 0.8]])
    d = X - mu
    direct = -0.5 * (2 * np.log(2 * np.pi) + np.log(np.linalg.det(S))
                     + np.einsum("ni,ij,nj->n", d, np.linalg.inv(S), d))
    np.testing.assert_allclose(log_gaussian(X, mu, S), direct)


# ---------------------------------------------------------------------------
# 3. Kiểm tra đầu vào: shape, dữ liệu rỗng / sai kiểu, K không hợp lệ
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("X, err", [
    (np.empty((0, 2)), ValueError),
    (np.empty((5, 0)), ValueError),
    (np.zeros((2, 2, 2)), ValueError),
    ([[1.0, np.nan], [0.0, 1.0]], ValueError),
    ([[1.0, np.inf], [0.0, 1.0]], ValueError),
    ([["a", "b"], ["c", "d"]], TypeError),
])
def test_invalid_X(X, err):
    with pytest.raises(err):
        fit_gmm(X, 1)


@pytest.mark.parametrize("K", [0, -1, 2.5, "2", True, 11])
def test_invalid_K(K):
    with pytest.raises(ValueError):
        fit_gmm(np.arange(10.0), K)


@pytest.mark.parametrize("kwargs", [
    {"max_iter": 0}, {"n_init": 0}, {"tol": 0}, {"tol": -1}, {"reg_covar": -1e-6},
])
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        fit_gmm(np.arange(10.0), 2, **kwargs)


def test_invalid_init():
    with pytest.raises(ValueError):
        fit_gmm(HAND_X, 2, init=([0.7, 0.7], [2, 4], [1, 1]))


def test_1d_input_is_reshaped():
    r = fit_gmm(np.arange(20.0), 2, seed=0)
    assert r.mu.shape == (2, 1) and r.Sigma.shape == (2, 1, 1)


# ---------------------------------------------------------------------------
# 4. Likelihood trace và điều kiện dừng
# ---------------------------------------------------------------------------
def test_trace_never_decreases(synthetic_fit):
    t = np.array(synthetic_fit.trace)
    assert len(t) == synthetic_fit.n_iter + 1
    assert np.all(np.diff(t) >= -1e-8 * (1 + np.abs(t[:-1])))


def test_converges_and_stops(synthetic_fit):
    r = synthetic_fit
    assert r.converged and r.n_iter < 200
    last, prev = r.trace[-1], r.trace[-2]
    assert abs(last - prev) / (1 + abs(prev)) < 1e-6


def test_warns_when_not_converged(synthetic):
    X, _ = synthetic
    with pytest.warns(UserWarning, match="chưa hội tụ"):
        r = fit_gmm(X, 3, max_iter=2, seed=0)
    assert not r.converged and r.n_iter == 2


def test_same_seed_same_result(synthetic):
    X, _ = synthetic
    a = fit_gmm(X, 3, n_init=3, seed=7)
    b = fit_gmm(X, 3, n_init=3, seed=7)
    np.testing.assert_array_equal(a.mu, b.mu)
    assert a.all_log_likelihoods == b.all_log_likelihoods


def test_n_init_keeps_best(synthetic):
    X, _ = synthetic
    r = fit_gmm(X, 3, n_init=5, seed=0)
    assert len(r.all_log_likelihoods) == 5
    assert r.log_likelihood == max(r.all_log_likelihoods)


# ---------------------------------------------------------------------------
# 5. Covariance suy biến
# ---------------------------------------------------------------------------
def test_duplicate_points_do_not_crash():
    # 10 điểm trùng nhau + 10 điểm khác: một thành phần có thể co về đúng một điểm.
    X = np.vstack([np.zeros((10, 2)), np.random.default_rng(0).normal(5, 1, (10, 2))])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = fit_gmm(X, 2, seed=0)
    assert np.isfinite(r.log_likelihood)
    for S in r.Sigma:
        assert np.all(np.linalg.eigvalsh(S) >= 1e-6 * 0.999)   # bị chặn dưới bởi reg_covar


def test_singular_covariance_without_reg_raises():
    # reg_covar = 0 và mọi điểm của một thành phần trùng nhau -> Sigma = 0 -> báo lỗi rõ ràng.
    X = np.array([[0.0, 0.0]] * 5 + [[3.0, 1.0], [4.0, 2.0], [3.5, 0.5]])
    init = ([0.5, 0.5], [[0.0, 0.0], [3.5, 1.2]], [np.eye(2) * 1e-3, np.eye(2)])
    with pytest.raises(RuntimeError, match="reg_covar"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_gmm(X, 2, init=init, reg_covar=0.0, max_iter=5)


def test_m_step_empty_component_is_finite():
    X = np.arange(6.0)[:, None]
    resp = np.column_stack([np.ones(6), np.zeros(6)])     # thành phần 2 không nhận điểm nào
    with pytest.warns(UserWarning, match="gần rỗng"):
        pi, mu, Sigma = m_step(X, resp, 1e-6)
    assert np.all(np.isfinite(mu)) and np.all(np.isfinite(Sigma))


# ---------------------------------------------------------------------------
# 6. Khớp ví dụ tính tay (docs/algorithm-spec.md mục 8)
# ---------------------------------------------------------------------------
def test_matches_hand_calculation():
    with pytest.warns(UserWarning, match="chưa hội tụ"):
        r = fit_gmm(HAND_X, 2, init=HAND_INIT, max_iter=1, reg_covar=0.0)
    np.testing.assert_allclose(r.trace[0], -8.9599, atol=1e-4)
    np.testing.assert_allclose(r.pi, [0.4830, 0.5170], atol=1e-4)
    np.testing.assert_allclose(r.mu.ravel(), [2.5085, 3.5984], atol=1e-4)
    np.testing.assert_allclose(r.Sigma.ravel(), [0.4377, 0.5510], atol=1e-4)
    np.testing.assert_allclose(r.trace[1], -7.7070, atol=1e-4)


def test_hand_calculation_responsibilities():
    resp, _ = e_step(np.array(HAND_X)[:, None], np.array([0.5, 0.5]),
                     np.array([[2.0], [4.0]]), np.array([[[1.0]], [[1.0]]]))
    np.testing.assert_allclose(resp[:, 0], [0.2315, 0.9168, 0.3394, 0.8075, 0.0445, 0.5582], atol=1e-4)


# ---------------------------------------------------------------------------
# 7. Đối chiếu sklearn và tham số thật
# ---------------------------------------------------------------------------
def test_matches_sklearn_with_same_init(synthetic):
    X, _ = synthetic
    pi0, mu0, S0 = init_params(X, 3, np.random.default_rng(0), 1e-6)
    ours = fit_gmm(X, 3, init=(pi0, mu0, S0), tol=1e-10, max_iter=1000)
    gm = GaussianMixture(3, covariance_type="full", reg_covar=1e-6, tol=1e-10, max_iter=1000,
                         weights_init=pi0, means_init=mu0,
                         precisions_init=np.linalg.inv(S0)).fit(X)
    assert ours.log_likelihood / len(X) == pytest.approx(gm.score(X), abs=1e-5)


def test_recovers_true_means(synthetic_fit):
    found = synthetic_fit.mu
    for true_mu in TRUE_MEANS:          # ghép cặp theo mu gần nhất (label switching)
        assert np.min(np.linalg.norm(found - true_mu, axis=1)) < 0.3


def test_predict_matches_resp(synthetic, synthetic_fit):
    X, _ = synthetic
    np.testing.assert_array_equal(predict(synthetic_fit, X), synthetic_fit.resp.argmax(axis=1))


# ---------------------------------------------------------------------------
# 8. BIC
# ---------------------------------------------------------------------------
def test_n_parameters():
    assert n_parameters(2, 2) == 11 and n_parameters(3, 2) == 17 and n_parameters(1, 1) == 2


def test_bic_matches_sklearn_formula():
    F, _, _ = standardize(load_faithful())
    r = fit_gmm(F, 2, n_init=5, seed=0)
    expected = -2 * r.log_likelihood + 11 * np.log(len(F))
    assert bic(r, F) == pytest.approx(expected)
