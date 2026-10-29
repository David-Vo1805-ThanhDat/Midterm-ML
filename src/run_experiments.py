"""Chạy toàn bộ thực nghiệm và lưu kết quả (I11).

    python -m src.run_experiments

Trả lời ba câu hỏi ở docs/data-and-use-case.md mục 5:
    Q1. EM tự cài đặt có học đúng không?            (dữ liệu tổng hợp, so với tham số thật)
    Q2. Kết quả có phụ thuộc khởi tạo không?        (nhiều seed, mỗi seed một lần khởi tạo)
    Q3. Old Faithful có tách thành các "kiểu" không? (chọn K bằng BIC)

Đầu ra trong results/:
    experiment-results.json   cấu hình, phiên bản thư viện, tham số, trace, BIC, lỗi/cảnh báo
    synthetic-labels.csv      x1, x2, nhãn thật, nhãn dự đoán, responsibility
    faithful-labels.csv       eruptions, waiting, nhãn dự đoán, responsibility
    bic.csv, init-study.csv   bảng tóm tắt để đưa vào báo cáo
"""

from __future__ import annotations

import csv
import itertools
import json
import platform
import warnings
from datetime import datetime, timezone

import matplotlib
import numpy as np
import sklearn
from sklearn.metrics import adjusted_rand_score   # chỉ dùng để đánh giá, không dùng để fit

from src import config
from src.data import (FAITHFUL_FEATURES, FAITHFUL_SHA256, TRUE_COVS, TRUE_MEANS, TRUE_WEIGHTS,
                      load_faithful, make_synthetic, standardize)
from src.em_gmm import bic, fit_gmm, predict


def _fit(X, K, **kw):
    """Gọi fit_gmm, ghi lại cảnh báo và lỗi thay vì bỏ qua (yêu cầu của I11)."""
    opts = dict(n_init=config.N_INIT, max_iter=config.MAX_ITER, tol=config.TOL,
                reg_covar=config.REG_COVAR, seed=config.EM_SEED)
    opts.update(kw)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = fit_gmm(X, K, **opts)
            error = None
        except Exception as e:   # ghi lại lần chạy thất bại, không loại bỏ
            result, error = None, f"{type(e).__name__}: {e}"
    return result, error, sorted({str(w.message) for w in caught})


def _params(result) -> dict:
    return {"pi": result.pi.tolist(), "mu": result.mu.tolist(), "Sigma": result.Sigma.tolist(),
            "log_likelihood": result.log_likelihood, "n_iter": result.n_iter,
            "converged": result.converged, "trace": result.trace,
            "all_log_likelihoods": result.all_log_likelihoods}


def _match_to_truth(mu):
    """Ghép thành phần học được với thành phần thật (label switching): thử mọi hoán vị."""
    K = len(TRUE_MEANS)
    return min(itertools.permutations(range(K)),
               key=lambda p: sum(np.linalg.norm(mu[p[k]] - TRUE_MEANS[k]) for k in range(K)))


def _bic_sweep(X):
    rows = []
    for K in config.K_RANGE:
        result, error, warns = _fit(X, K, n_init=config.N_INIT_BIC)
        rows.append({"K": K, "bic": None if result is None else bic(result, X),
                     "log_likelihood": None if result is None else result.log_likelihood,
                     "n_iter": None if result is None else result.n_iter,
                     "converged": None if result is None else result.converged,
                     "error": error, "warnings": warns})
    ok = [r for r in rows if r["bic"] is not None]
    best = min(ok, key=lambda r: r["bic"])["K"]
    return rows, best


def experiment_q1():
    X, z = make_synthetic(n=config.SYNTHETIC_N, seed=config.SYNTHETIC_SEED)
    result, error, warns = _fit(X, config.SYNTHETIC_TRUE_K)
    perm = _match_to_truth(result.mu)
    pred = predict(result, X)
    errors = {
        "pi_abs": [abs(result.pi[perm[k]] - TRUE_WEIGHTS[k]) for k in range(3)],
        "mu_dist": [float(np.linalg.norm(result.mu[perm[k]] - TRUE_MEANS[k])) for k in range(3)],
        "Sigma_fro": [float(np.linalg.norm(result.Sigma[perm[k]] - TRUE_COVS[k])) for k in range(3)],
    }
    bic_rows, best_k = _bic_sweep(X)
    return {
        "data": {"name": "synthetic", "n": len(X), "seed": config.SYNTHETIC_SEED,
                 "true_counts": np.bincount(z).tolist(),
                 "true_params": {"pi": TRUE_WEIGHTS.tolist(), "mu": TRUE_MEANS.tolist(),
                                 "Sigma": TRUE_COVS.tolist()}},
        "K": config.SYNTHETIC_TRUE_K, "fit": _params(result), "error": error, "warnings": warns,
        "matching": list(perm), "errors_vs_truth": errors,
        "ari_vs_true_labels": adjusted_rand_score(z, pred),
        "bic": bic_rows, "bic_selected_k": best_k,
    }, X, z, result


def experiment_q2():
    X_syn, _ = make_synthetic(n=config.SYNTHETIC_N, seed=config.SYNTHETIC_SEED)
    X_of, _, _ = standardize(load_faithful())
    out = []
    for name, X in [("synthetic", X_syn), ("faithful", X_of)]:
        for K in config.INIT_STUDY_K[name]:
            rows = []
            for s in config.INIT_STUDY_SEEDS:
                result, error, warns = _fit(X, K, n_init=1, seed=s)
                rows.append({"seed": s, "log_likelihood": None if result is None else result.log_likelihood,
                             "n_iter": None if result is None else result.n_iter,
                             "converged": None if result is None else result.converged,
                             "error": error, "warnings": warns})
            lls = np.array([r["log_likelihood"] for r in rows if r["log_likelihood"] is not None])
            out.append({"dataset": name, "K": K, "runs": rows,
                        "best": float(lls.max()), "worst": float(lls.min()),
                        "n_distinct_optima": int(len(np.unique(np.round(lls, 2)))),
                        "n_reaching_best": int(np.sum(lls > lls.max() - 0.01))})
    return out


def experiment_q3():
    raw = load_faithful()
    X, mean, std = standardize(raw)
    bic_rows, best_k = _bic_sweep(X)
    result, error, warns = _fit(X, best_k)
    # Đổi tham số về đơn vị phút: x = z*std + mean
    mu_min = result.mu * std + mean
    Sigma_min = result.Sigma * np.outer(std, std)
    corr = [float(S[0, 1] / np.sqrt(S[0, 0] * S[1, 1])) for S in Sigma_min]
    return {
        "data": {"name": "faithful", "n": len(X), "features": FAITHFUL_FEATURES,
                 "sha256": FAITHFUL_SHA256, "preprocessing": "z-score",
                 "mean": mean.tolist(), "std": std.tolist()},
        "bic": bic_rows, "bic_selected_k": best_k,
        "fit_standardized": _params(result), "error": error, "warnings": warns,
        "params_minutes": {"pi": result.pi.tolist(), "mu": mu_min.tolist(),
                           "Sigma": Sigma_min.tolist(), "corr_within_component": corr},
    }, raw, X, result


def _write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    q1, X_syn, z_syn, r_syn = experiment_q1()
    q2 = experiment_q2()
    q3, raw_of, X_of, r_of = experiment_q3()

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": config.as_dict(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__,
                        "platform": platform.platform()},
        "note": "sklearn chỉ dùng adjusted_rand_score để đánh giá; mọi lần fit dùng src/em_gmm.py",
        "q1_synthetic": q1, "q2_initialization": q2, "q3_faithful": q3,
    }
    with open(config.RESULTS_DIR / "experiment-results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    pred = predict(r_syn, X_syn)
    _write_csv(config.RESULTS_DIR / "synthetic-labels.csv",
               ["x1", "x2", "true_label", "pred_label", "resp_0", "resp_1", "resp_2"],
               [[*x, t, p, *r] for x, t, p, r in zip(X_syn, z_syn, pred, r_syn.resp)])
    pred = predict(r_of, X_of)
    K = r_of.resp.shape[1]
    _write_csv(config.RESULTS_DIR / "faithful-labels.csv",
               ["eruptions", "waiting", "pred_label"] + [f"resp_{k}" for k in range(K)],
               [[*x, p, *r] for x, p, r in zip(raw_of, pred, r_of.resp)])
    _write_csv(config.RESULTS_DIR / "bic.csv",
               ["dataset", "K", "bic", "log_likelihood", "n_iter", "converged"],
               [[name, r["K"], r["bic"], r["log_likelihood"], r["n_iter"], r["converged"]]
                for name, q in [("synthetic", q1), ("faithful", q3)] for r in q["bic"]])
    _write_csv(config.RESULTS_DIR / "init-study.csv",
               ["dataset", "K", "seed", "log_likelihood", "n_iter", "converged"],
               [[q["dataset"], q["K"], r["seed"], r["log_likelihood"], r["n_iter"], r["converged"]]
                for q in q2 for r in q["runs"]])

    # Tóm tắt ra màn hình (ASCII để console Windows in được)
    print("Q1 synthetic: ll=%.3f, iters=%d, ARI=%.3f, BIC selects K=%d"
          % (r_syn.log_likelihood, r_syn.n_iter, q1["ari_vs_true_labels"], q1["bic_selected_k"]))
    print("   max |mu - mu_true| = %.3f" % max(q1["errors_vs_truth"]["mu_dist"]))
    for q in q2:
        print("Q2 %s K=%d: %d seeds, %d distinct optima, %d reach best (best=%.3f, worst=%.3f)"
              % (q["dataset"], q["K"], len(q["runs"]), q["n_distinct_optima"],
                 q["n_reaching_best"], q["best"], q["worst"]))
    print("Q3 faithful: BIC selects K=%d; means (min) = %s"
          % (q3["bic_selected_k"], np.round(q3["params_minutes"]["mu"], 2).tolist()))
    print("Saved to", config.RESULTS_DIR)


if __name__ == "__main__":
    main()
