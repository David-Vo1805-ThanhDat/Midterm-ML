"""Vẽ hình cho báo cáo và demo từ kết quả đã lưu (I12).

    python -m src.run_experiments   # nếu chưa có results/experiment-results.json
    python -m src.make_figures

Mọi số liệu lấy từ results/ (không fit lại), nên hình luôn khớp với kết quả lưu.
Hình xuất vào results/figures/ dạng PNG.
"""

from __future__ import annotations

import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

from src import config  # noqa: E402

# Bảng màu phân loại đã kiểm tra cho người mù màu (3 màu đầu); xám cho dữ liệu chưa phân cụm.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
GRAY, INK, INK2 = "#8a8985", "#1b1b1a", "#52514e"
plt.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"], "font.size": 10,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
})

SYN_SOURCE = "Dữ liệu tổng hợp: 300 điểm từ 3 Gaussian, seed sinh dữ liệu = {seed}"
OF_SOURCE = "Old Faithful (Härdle 1991; R datasets), 272 lần phun"


def _load():
    with open(config.RESULTS_DIR / "experiment-results.json", encoding="utf-8") as f:
        res = json.load(f)

    def read(name):
        with open(config.RESULTS_DIR / name, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}

    return res, read("synthetic-labels.csv"), read("faithful-labels.csv")


def _ellipse(ax, mu, cov, color, ls="-", lw=1.8, label=None):
    """Đường elip bao 2 độ lệch chuẩn của một Gaussian."""
    vals, vecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    w, h = 4 * np.sqrt(vals[::-1])
    ax.add_patch(Ellipse(mu, w, h, angle=angle, fill=False, color=color, ls=ls, lw=lw, label=label))


def _footer(fig, text):
    fig.text(0.01, -0.02, text, fontsize=8, color=INK2, ha="left", va="top")


def _save(fig, name):
    path = config.FIGURES_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print("saved", path.relative_to(config.ROOT))


def fig_data(res, syn, of):
    """Hình 1: dữ liệu trước khi phân cụm (chưa tô màu theo cụm)."""
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    axs[0].scatter(syn["x1"], syn["x2"], s=12, color=GRAY, alpha=0.8, edgecolor="white", lw=0.3)
    axs[0].set(xlabel="x1", ylabel="x2", title="Dữ liệu tổng hợp (trước phân cụm)")
    axs[0].set_aspect("equal")
    axs[1].scatter(of["eruptions"], of["waiting"], s=12, color=GRAY, alpha=0.8, edgecolor="white", lw=0.3)
    axs[1].set(xlabel="Thời gian phun, eruptions (phút)", ylabel="Thời gian chờ, waiting (phút)",
               title="Old Faithful (trước phân cụm)")
    for ax in axs:
        ax.title.set_ha("left"); ax.title.set_x(0)
    _footer(fig, SYN_SOURCE.format(seed=res["config"]["synthetic_seed"]) + "  ·  " + OF_SOURCE)
    _save(fig, "fig1_data_before.png")


def fig_synthetic(res, syn):
    """Hình 2: dữ liệu tổng hợp tô màu theo cụm dự đoán, elip học được so với elip thật."""
    q1 = res["q1_synthetic"]
    fit, true = q1["fit"], q1["data"]["true_params"]
    fig, ax = plt.subplots(figsize=(7.6, 5))
    pred = syn["pred_label"].astype(int)
    for k in range(q1["K"]):
        m = pred == k
        ax.scatter(syn["x1"][m], syn["x2"][m], s=13, color=COLORS[k], alpha=0.75,
                   edgecolor="white", lw=0.3, label=f"Cụm dự đoán {k + 1} ({m.sum()} điểm)")
        _ellipse(ax, fit["mu"][k], np.array(fit["Sigma"][k]), COLORS[k])
        ax.plot(*fit["mu"][k], marker="X", ms=9, color=COLORS[k], mec="white")
    for k in range(len(true["mu"])):
        _ellipse(ax, true["mu"][k], np.array(true["Sigma"][k]), INK, ls="--", lw=1.1,
                 label="Gaussian thật (2σ)" if k == 0 else None)
        ax.plot(*true["mu"][k], marker="+", ms=11, mew=2, color=INK)
    ax.plot([], [], color=INK2, lw=1.8, label="Gaussian EM học được (2σ)")
    ax.set(xlabel="x1", ylabel="x2")
    ax.set_title(f"EM trên dữ liệu tổng hợp: K = {q1['K']}, ARI = {q1['ari_vs_true_labels']:.3f}", loc="left")
    ax.set_aspect("equal")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8.5)
    _footer(fig, SYN_SOURCE.format(seed=res["config"]["synthetic_seed"])
            + f"  ·  seed EM = {res['config']['em_seed']}, n_init = {res['config']['n_init']}"
            + "  ·  Số thứ tự cụm do EM đặt tùy ý; X: tâm học được, +: tâm thật")
    _save(fig, "fig2_synthetic_clusters.png")


def fig_faithful(res, of):
    """Hình 3: Old Faithful tô màu theo cụm dự đoán, khoanh các điểm không chắc chắn."""
    q3 = res["q3_faithful"]
    p = q3["params_minutes"]
    K = q3["bic_selected_k"]
    fig, ax = plt.subplots(figsize=(7.6, 5))
    pred = of["pred_label"].astype(int)
    resp_max = np.max(np.column_stack([of[f"resp_{k}"] for k in range(K)]), axis=1)
    for k in range(K):
        m = pred == k
        ax.scatter(of["eruptions"][m], of["waiting"][m], s=14, color=COLORS[k], alpha=0.8,
                   edgecolor="white", lw=0.3,
                   label=f"Cụm {k + 1}: π = {p['pi'][k]:.2f}, tâm ({p['mu'][k][0]:.2f}; {p['mu'][k][1]:.1f}) phút")
        _ellipse(ax, p["mu"][k], np.array(p["Sigma"][k]), COLORS[k])
    unsure = resp_max < 0.9
    ax.scatter(of["eruptions"][unsure], of["waiting"][unsure], s=70, facecolor="none",
               edgecolor=INK, lw=1.1, label=f"Không chắc chắn: max r_nk < 0.9 ({unsure.sum()} điểm)")
    ax.set(xlabel="Thời gian phun, eruptions (phút)", ylabel="Thời gian chờ, waiting (phút)")
    ax.set_title(f"EM trên Old Faithful: K = {K} (chọn bằng BIC)", loc="left")
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    _footer(fig, OF_SOURCE + "  ·  fit trên dữ liệu chuẩn hóa z-score, đổi về phút để vẽ"
            + f"  ·  seed EM = {res['config']['em_seed']}, n_init = {res['config']['n_init']}"
            + "  ·  Elip: 2σ. Cụm là mô tả mật độ, không phải nhóm vật lý đã được chứng minh")
    _save(fig, "fig3_faithful_clusters.png")


def fig_loglik(res):
    """Hình 4: log-likelihood theo vòng lặp."""
    runs = [("Dữ liệu tổng hợp, K = 3", res["q1_synthetic"]["fit"]),
            (f"Old Faithful, K = {res['q3_faithful']['bic_selected_k']}", res["q3_faithful"]["fit_standardized"])]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, (title, fit) in zip(axs, runs):
        t = fit["trace"]
        ax.plot(range(len(t)), t, color=COLORS[0], lw=2, marker="o", ms=4, mec="white")
        ax.set(xlabel="Vòng lặp (0 = trước vòng đầu tiên)", ylabel="Log-likelihood ℓ")
        ax.set_title(f"{title}: hội tụ sau {fit['n_iter']} vòng", loc="left")
        ax.text(0.97, 0.06, f"ℓ cuối = {t[-1]:.2f}", transform=ax.transAxes, ha="right",
                fontsize=9, color=INK)
    _footer(fig, f"Lần khởi tạo tốt nhất trong n_init = {res['config']['n_init']}, seed EM = "
            f"{res['config']['em_seed']}  ·  Dừng khi |Δℓ|/(1+|ℓ|) < {res['config']['tol']:g}"
            "  ·  ℓ không giảm ở mọi vòng (tính chất của EM)")
    _save(fig, "fig4_loglikelihood.png")


def fig_bic(res):
    """Hình 5: BIC theo K."""
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, (title, q) in zip(axs, [("Dữ liệu tổng hợp", res["q1_synthetic"]),
                                     ("Old Faithful (chuẩn hóa)", res["q3_faithful"])]):
        K = [r["K"] for r in q["bic"]]
        b = [r["bic"] for r in q["bic"]]
        best = q["bic_selected_k"]
        ax.plot(K, b, color=COLORS[0], lw=2, marker="o", ms=7, mec="white")
        ax.plot(best, b[K.index(best)], marker="o", ms=11, color=COLORS[1], mec="white")
        ax.set(xlabel="Số thành phần K", ylabel="BIC (nhỏ hơn là tốt hơn)", xticks=K)
        ax.set_title(f"{title}\nBIC nhỏ nhất ở K = {best}", loc="left")
    _footer(fig, f"Mỗi K: n_init = {res['config']['n_init_bic']}, seed EM = {res['config']['em_seed']}"
            "  ·  BIC = −2ℓ + p·ln N, p = (K−1) + KD + KD(D+1)/2  ·  Không dùng nhãn thật để chọn K")
    _save(fig, "fig5_bic.png")


def fig_init(res):
    """Hình 6: log-likelihood cuối theo seed khởi tạo (câu hỏi Q2)."""
    groups = res["q2_initialization"]
    fig, axs = plt.subplots(1, len(groups), figsize=(12, 3.4), sharey=False)
    for ax, g in zip(axs, groups):
        ll = np.array([r["log_likelihood"] for r in g["runs"]])
        seeds = [r["seed"] for r in g["runs"]]
        best = ll >= g["best"] - 0.01
        ax.scatter(np.array(seeds)[~best], ll[~best], s=28, color=GRAY, label="local optimum")
        ax.scatter(np.array(seeds)[best], ll[best], s=34, color=COLORS[1], label="tốt nhất")
        name = "Tổng hợp" if g["dataset"] == "synthetic" else "Old Faithful"
        ax.set_title(f"{name}, K = {g['K']}\n{g['n_distinct_optima']} nghiệm khác nhau", loc="left", fontsize=10)
        ax.set(xlabel="Seed khởi tạo")
        ax.set_xticks([0, 5, 10, 15, 19])
        # Trục y tối thiểu 2 đơn vị log-likelihood: sai khác cỡ 1e-3 (cùng một nghiệm) không bị phóng to.
        span = max(g["best"] - g["worst"], 2.0)
        ax.set_ylim(g["best"] - 1.1 * span, g["best"] + 0.15 * span)
        ax.ticklabel_format(axis="y", useOffset=False, style="plain")
    axs[0].set_ylabel("Log-likelihood cuối")
    axs[-1].legend(frameon=False, fontsize=8, loc="upper right")
    _footer(fig, "Mỗi điểm: một lần chạy EM với n_init = 1. Trục y khác nhau giữa các ô; nghiệm được coi là "
            "giống nhau nếu ℓ chênh < 0.01")
    _save(fig, "fig6_init_study.png")


def main():
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    res, syn, of = _load()
    fig_data(res, syn, of)
    fig_synthetic(res, syn)
    fig_faithful(res, of)
    fig_loglik(res)
    fig_bic(res)
    fig_init(res)


if __name__ == "__main__":
    main()
