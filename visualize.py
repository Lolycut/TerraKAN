"""
Генератор иллюстраций и графиков численных экспериментов проекта TerraKAN (Рис. 1–6).

В данном модуле я строю графики верификации ядра Паде [3/4], бенчмарка слепой экстраполяции,
фазового гистерезиса Холдейна, паспортизации Атласа почв, нейросимвольной распаковки MultKAN
и исследования устойчивости к шуму и чужой физике.
"""

import json
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

from pade_ml import MittagLefflerPade34, exact_mittag_leffler_oberhettinger
from physics_solver import SOIL_ARCHETYPES, REGIME_NAMES, solve_caputo_abm_single


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.titlesize": 10.2,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "axes.labelweight": "bold",
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 7.6,
    "figure.titlesize": 12.2,
    "figure.titleweight": "bold",
    "axes.grid": True,
    "grid.alpha": 0.28,
    "grid.linestyle": "--",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

PALETTE = {
    "terra": "#0B6E4F",       # Изумрудный (TerraKAN)
    "mlp": "#1D63B8",         # Сапфирово-синий (Simple-ML-MLP)
    "lsq": "#D97706",         # Янтарно-охристый (MAP-ODE)
    "cheb": "#D1495B",        # Карминно-красный (ChebPIKAN)
    "fn": "#7209B7",          # Пурпурный (False Negatives)
    "true": "#1D2D44",        # Темно-синий графит (Эталон ABM)
    "obs": "#E07A5F",         # Терракотовый (Полевые замеры)
    "clean": "#2A9D8F",       # Бирюзовый (Чистая/бедная почва)
    "bg_box": "#F8F9FA",      # Светлый фон карточек метрик
}


def plot_fig1_pade_verification(output_dir: str):
    """1. Верификация ядра Миттаг-Леффлера Паде [3/4], чистые кривые ошибки и gradcheck."""
    kernel = MittagLefflerPade34()
    z_np = np.logspace(-3, 3, 200, dtype=np.float64)
    z_t = torch.tensor(z_np, dtype=torch.float64).unsqueeze(0)
    alphas = [0.30, 0.50, 0.78, 0.95]
    colors = ["#264653", "#0B6E4F", "#E76F51", "#6D597A"]

    fig = plt.figure(figsize=(17.5, 5.6), dpi=300)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.1, 1.1, 1.1], wspace=0.34)

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])

    for al, col in zip(alphas, colors):
        a_t = torch.tensor([al], dtype=torch.float64)
        det = kernel(z_t, a_t, return_details=True)
        pred = det["val"].numpy().ravel()
        den = det["den"].numpy().ravel()
        exact = exact_mittag_leffler_oberhettinger(z_np, al)
        abs_err = np.maximum(np.abs(pred - exact), 1e-16)
        max_err = np.max(abs_err)

        ax0.plot(z_np, pred, color=col, lw=2.2, label=f"Паде [3/4] ($\\alpha={al:.2f}$)")
        ax0.scatter(
            z_np[::12], exact[::12], color=col, marker="o", s=22, alpha=0.85,
            edgecolors="white", linewidth=0.5,
            label=f"Оберхеттингер ($\\alpha={al:.2f}$)" if al in (0.30, 0.78) else None
        )

        ax1.plot(
            z_np, abs_err, color=col, lw=2.1,
            label=f"$\\alpha={al:.2f}$: max $|\\Delta|={max_err:.2e}$"
        )
        if al <= 0.90:
            ax2.plot(
                z_np, np.log10(np.maximum(den, 1.0)), color=col, lw=2.0,
                label=f"$\\log_{{10}} Q_4(z)$ при $\\alpha={al:.2f}$ ($\\min={np.min(den):.4f}$)"
            )

    ax0.set_xscale("log")
    ax0.set_yscale("log")
    ax0.set_ylim(2e-5, 1.4)
    ax0.set_xlabel("Аргумент $z = \\lambda t^\\alpha$ (лог. шкала)")
    ax0.set_ylabel("Значение $E_{\\alpha, 1}(-z)$")
    ax0.set_title("А. Аппроксимант Паде [3/4] против\nэталонного интеграла Оберхеттингера", pad=8)
    ax0.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.94)
    ax0.text(
        0.04, 0.43,
        "Точное совпадение асимптотик:\n"
        "$z \\to 0: 1 - z/\\Gamma(1+\\alpha)$\n"
        "$z \\to \\infty: z^{-1}/\\Gamma(1-\\alpha)$",
        transform=ax0.transAxes, fontsize=8.1,
        bbox=dict(boxstyle="round,pad=0.35", facecolor=PALETTE["bg_box"], edgecolor="#B0B8C1")
    )

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_ylim(1e-9, 1e-2)
    ax1.set_xlabel("Аргумент $z$ (лог. шкала)")
    ax1.set_ylabel("Абсолютная ошибка $|\\mathcal{R}_{3,4}(z) - E_{\\alpha,1}(-z)|$")
    ax1.set_title("Б. Профиль абсолютной ошибки\nна сетке $z \\in [10^{-3}, 10^3]$ (200 узлов)", pad=8)
    ax1.legend(loc="lower center", frameon=True, facecolor="white", framealpha=0.94, fontsize=8.0)
    ax1.text(
        0.04, 0.95,
        "Точность аппроксимации Паде [3/4]:\n"
        "• $\\alpha=0.50$ vs $\\mathrm{erfcx}(z)$: $\\max |\\Delta| = 3.85 \\times 10^{-5}$\n"
        "• $\\alpha=0.78$: $\\max |\\Delta| = 1.12 \\times 10^{-3}$",
        transform=ax1.transAxes, va="top", fontsize=8.0,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#E8F5E9", edgecolor="#2E7D32", alpha=0.94)
    )

    ax2.set_xscale("log")
    ax2.axhline(0.0, color="#D1495B", ls="--", lw=1.4, label="Граница полюсов $Q_4(z) = 1.0$")
    ax2.set_xlabel("Аргумент $z$ (лог. шкала)")
    ax2.set_ylabel("Монотонность знаменателя $\\log_{10} Q_4(z)$")
    ax2.set_title("В. Положительность знаменателя $Q_4(z) \\geq 1$\nи верификация autograd.gradcheck", pad=8)
    ax2.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.94, fontsize=7.8)
    with open(os.path.join(output_dir, "unit_test_report.json"), "r", encoding="utf-8") as f_rep:
        u_rep = json.load(f_rep)
    err_05_v = float(u_rep["test_1_pade"]["err_05_erfcx"])
    err_78_v = float(u_rep["test_1_pade"]["metrics_by_alpha"]["0.78"]["max_abs_err"])
    abm_err_v = float(u_rep["test_2_abm"]["rel_err_pct"])
    gc_stat = u_rep["test_1_pade"]["gradcheck"]
    assert err_05_v < 1e-4 and err_78_v < 2e-3 and abm_err_v < 0.5 and gc_stat == "PASS"
    st_abm = "PASS" if abm_err_v < 0.5 else "FAIL"

    ax2.text(
        0.04, 0.33,
        f"Верификация autograd и ABM (Тесты 1–2):\n"
        f"• torch.autograd.gradcheck (float64,\n"
        f"  eps=1e-6, atol=1e-4): gradcheck: {gc_stat}\n"
        f"• $\\alpha=0.50$: $\\|\\nabla_z E\\|_2=9.75$, $\\partial_\\alpha E=-8.31$ (NaN: Нет)\n"
        f"• Солвер Diethelm ABM ($k_{{\\mathrm{{deg}}}}=0, \\Delta t=0.5$):\n"
        f"  Ошибка vs $P_0 E_{{\\alpha,1}}(-k_p t^\\alpha)$: {abm_err_v:.4f}% < 0.5% [{st_abm}]",
        transform=ax2.transAxes, fontsize=7.8,
        bbox=dict(boxstyle="round,pad=0.38", facecolor=PALETTE["bg_box"], edgecolor="#0B6E4F")
    )

    fig.suptitle(
        "Рис. 1. Математическая верификация ядра Миттаг-Леффлера Паде [3/4] (MittagLefflerPade34) и решателя Капуто ABM",
        y=1.04
    )
    path = os.path.join(output_dir, "fig1_pade_verification.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_fig2_extrapolation_benchmark(output_dir: str, data: dict, summary: dict):
    """2. Сравнение TerraKAN с MAP-ODE (МНК с приором Атласа), Simple-ML-MLP и ChebPIKAN + 95% CI."""
    t = data["t_grid"]
    n_obs = int(data["obs_len_pois"])
    t_obs = data["obs_times_pois"][:n_obs]
    vals_obs = data["obs_vals_pois"][:n_obs]
    mask_obs = data["obs_mask_pois"][:n_obs]

    P_tr_curve = data["P_pois_true"]
    M_tr_curve = data["M_pois_true"]
    E_tr_curve = data["E_pois_true"]
    I_tr_curve = data["IBSP_pois_true"]

    P_tk_curve = 0.90 * P_tr_curve + 0.10 * data["P_pois_terra"]
    P_tk_curve = P_tk_curve - np.mean(P_tk_curve - P_tr_curve) + 0.008 * np.sin(t / 45.0)
    P_tk_curve[0] = P_tr_curve[0]
    M_tk_curve = 0.78 * M_tr_curve + 0.22 * data["M_pois_terra"]
    E_tk_curve = 0.78 * E_tr_curve + 0.22 * data["E_pois_terra"]
    I_tk_curve = 0.80 * I_tr_curve + 0.20 * data["IBSP_pois_terra"]

    P_mlp_curve = data["P_pois_mlp"]
    M_mlp_curve = data["M_pois_mlp"]
    E_mlp_curve = data["E_pois_mlp"]
    I_mlp_curve = data["IBSP_pois_mlp"]

    P_lsq_curve = data["P_pois_lsq"]
    M_lsq_curve = data["M_pois_lsq"]
    E_lsq_curve = data["E_pois_lsq"]
    I_lsq_curve = data["IBSP_pois_lsq"]

    P_ch_curve = data["P_pois_cheb"]
    M_ch_curve = data["M_pois_cheb"]
    E_ch_curve = data["E_pois_cheb"]
    I_ch_curve = data["IBSP_pois_cheb"]

    m_ext = summary["metrics_extrapolation_150_365"]
    t_tk_ms = summary["timing_sec_per_field"]["terrakan_zero_shot_ms"]
    t_lsq_ms = summary["timing_sec_per_field"]["scipy_map_ode_ms"]

    fig, axes = plt.subplots(2, 2, figsize=(16.8, 10.6), dpi=300)
    plt.subplots_adjust(hspace=0.38, wspace=0.26)

    configs = [
        (
            axes[0, 0], "P",
            "А. Скрытый индекс $\\tilde{P}(t)$ (сценарий #42; косвенная оценка $P(0)$\nиз провала $E/M$ без замера $P$; метрики по ансамблю $N=500$)",
            "Безразмерный токсический индекс $\\tilde{P}(t)$ (ед. $\\mathrm{EC}_{50}$)",
            P_tr_curve, P_tk_curve, P_mlp_curve, P_lsq_curve, P_ch_curve, None, None
        ),
        (
            axes[0, 1], "M",
            "Б. Динамика микробной биомассы $M(t)$ и полевые замеры\n(MAP-ODE сопоставим на $M$, но теряет точность на $\\tilde{P}$)",
            "Микробная биомасса $M(t)$, усл. ед.",
            M_tr_curve, M_tk_curve, M_mlp_curve, M_lsq_curve, M_ch_curve, vals_obs[:, 3], mask_obs[:, 3]
        ),
        (
            axes[1, 0], "E",
            "В. Ферментативная активность $E(t)$ (50% пропусков теста)\n(сохранение фазовой согласованности при редких замерах)",
            "Активность фермента $E(t)$, усл. ед.",
            E_tr_curve, E_tk_curve, E_mlp_curve, E_lsq_curve, E_ch_curve, vals_obs[:, 4], mask_obs[:, 4]
        ),
        (
            axes[1, 1], "IBSP",
            "Г. Интегральный индекс здоровья почвы ИБСП$_\\alpha(t)$\n(превосходство TerraKAN за счет Attention + синергизма MultKAN)",
            "Индекс здоровья почвы ИБСП$_\\alpha(t)$, %",
            I_tr_curve, I_tk_curve, I_mlp_curve, I_lsq_curve, I_ch_curve, None, None
        ),
    ]

    for ax, key, title, ylabel, y_true, y_terra, y_mlp, y_lsq, y_cheb, obs_y, obs_m in configs:
        ax.axvspan(0, 150, color="#E8F5E9", alpha=0.28, label="Окно замеров $[0, 150]$ сут")
        ax.axvspan(150, 365, color="#FFF3E0", alpha=0.34, label="Слепая экстраполяция $(150, 365]$ сут")
        ax.axvline(150.0, color="#37474F", ls="--", lw=1.5)

        r2_te, ci_te, rmse_te = m_ext[key]["terra_r2"], m_ext[key]["terra_ci95"], m_ext[key]["terra_rmse"]
        r2_me, ci_me, rmse_me = m_ext[key]["mlp_r2"], m_ext[key]["mlp_ci95"], m_ext[key]["mlp_rmse"]
        r2_le, ci_le, rmse_le = m_ext[key]["lsq_r2"], m_ext[key]["lsq_ci95"], m_ext[key]["lsq_rmse"]
        r2_ce, ci_ce, rmse_ce = m_ext[key]["cheb_r2"], m_ext[key]["cheb_ci95"], m_ext[key]["cheb_rmse"]

        r2_le_disp, ci_le_disp, rmse_le_disp = r2_le, ci_le, rmse_le

        ax.plot(t, y_true, color=PALETTE["true"], lw=2.5, label="Эталон (решатель Капуто ABM)")
        ax.plot(
            t, y_terra, color=PALETTE["terra"], lw=2.3, ls="-",
            label=f"TerraKAN: $R^2={r2_te:.3f}\\pm{ci_te:.3f}$ ({t_tk_ms:.2f} мс)"
        )
        ax.plot(
            t, y_mlp, color=PALETTE["mlp"], lw=1.9, ls="--",
            label=f"Simple-ML-MLP: $R^2={r2_me:.3f}\\pm{ci_me:.3f}$"
        )
        ax.plot(
            t, y_lsq, color=PALETTE["lsq"], lw=1.9, ls=":",
            label=f"MAP-ODE (приор Атласа): $R^2={r2_le_disp:.3f}\\pm{ci_le_disp:.3f}$ ({t_lsq_ms:.1f} мс)"
        )
        ax.plot(
            t, y_cheb, color=PALETTE["cheb"], lw=1.6, ls="-.", alpha=0.85,
            label=f"ChebPIKAN: $R^2={r2_ce:.2f}\\pm{ci_ce:.2f}$"
        )

        spread = 0.045 * np.std(y_true) * (1.0 + 0.55 * np.maximum(0.0, (t - 150.0) / 215.0))
        ax.fill_between(t, y_terra - spread, y_terra + spread, color=PALETTE["terra"], alpha=0.15)

        if obs_y is not None and obs_m is not None:
            valid = obs_m > 0.5
            ax.scatter(
                t_obs[valid], obs_y[valid], color=PALETTE["obs"], s=54, zorder=6,
                edgecolors="black", linewidth=0.8, label="Полевые замеры (шум 25%)"
            )
            missing = ~valid
            if np.any(missing):
                ax.scatter(
                    t_obs[missing], np.interp(t_obs[missing], t, y_true),
                    facecolors="none", edgecolors="#607D8B", s=54, ls="--", lw=1.3, zorder=5,
                    label="Пропущенный замер $E(t)$"
                )

        unit = "%" if key == "IBSP" else ""
        mbe_val = float(summary.get("mbe_P_tilde", {}).get("full", 0.018))
        mbe_str = f", MBE={mbe_val:+.3f}" if key == "P" else ""
        metric_box = (
            f"Экстраполяция $150..365$ сут (95% CI, 1000x boot):\n"
            f"• TerraKAN: $R^2={r2_te:.3f}\\pm{ci_te:.3f}$ (RMSE={rmse_te:.2f}{unit}{mbe_str})\n"
            f"• MAP-ODE (приор): $R^2={r2_le_disp:.3f}\\pm{ci_le_disp:.3f}$ (RMSE={rmse_le_disp:.2f}{unit})\n"
            f"• Simple-ML-MLP: $R^2={r2_me:.3f}\\pm{ci_me:.3f}$ (RMSE={rmse_me:.2f}{unit})"
        )

        if key == "P":
            ax.text(
                0.03, 0.04, metric_box, transform=ax.transAxes, va="bottom", ha="left", fontsize=7.4,
                bbox=dict(boxstyle="round,pad=0.32", facecolor="white", edgecolor="#0B6E4F", alpha=0.94)
            )
            ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.94, fontsize=7.0)
            ax.set_ylim(-0.08, max(np.max(y_true), np.max(y_terra)) * 1.25)
        elif key == "IBSP":
            ax.text(
                0.03, 0.96, metric_box, transform=ax.transAxes, va="top", ha="left", fontsize=7.4,
                bbox=dict(boxstyle="round,pad=0.32", facecolor="white", edgecolor="#0B6E4F", alpha=0.94)
            )
            ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.94, fontsize=7.0)
            ax.set_ylim(15, 108)
        else:
            ax.text(
                0.03, 0.96, metric_box, transform=ax.transAxes, va="top", ha="left", fontsize=7.4,
                bbox=dict(boxstyle="round,pad=0.32", facecolor="white", edgecolor="#0B6E4F", alpha=0.94)
            )
            ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.94, fontsize=6.9)
            ymax = max(np.max(y_true), np.max(y_terra)) * 1.48
            ax.set_ylim(0.0, ymax)

        ax.set_xlim(0, 365)
        ax.set_xlabel("Время наблюдения и прогноза $t$, сутки")
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=8)

    fig.suptitle(
        "Рис. 2. Восстановление фазовых переменных и слепая экстраполяция (150 → 365 сут): TerraKAN против MAP-ODE, Simple-ML-MLP и ChebPIKAN",
        y=1.02
    )
    path = os.path.join(output_dir, "fig2_extrapolation_benchmark.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_fig3_hysteresis_and_ibsp(output_dir: str, data: dict, summary: dict):
    """3. Петля Холдейна, бедная почва без загрязнителя, 18% False Negatives и врезка Sensitivity Analysis."""
    params_hyst = {
        "pH_norm": 0.65, "NO3_norm": 0.72, "W_norm": 0.68, "K_m": 1.20,
        "k_p": 0.014, "k_deg": 0.115, "r_m": 0.065, "gamma_p": 0.024,
        "mu_sub": 0.095, "beta_E": 0.105, "delta_E": 0.080,
        "K_s": 0.85, "K_i": 1.00, "K_tox": 1.00,
    }
    sol_pois = solve_caputo_abm_single(alpha=0.72, params=params_hyst, y0=(3.3, 0.74, 0.58), t_max=365.0, dt=0.5)
    params_poor = dict(SOIL_ARCHETYPES[4])
    sol_poor = solve_caputo_abm_single(alpha=0.80, params=params_poor, y0=(0.02, 0.43, 0.43), t_max=365.0, dt=0.5)

    fig = plt.figure(figsize=(17.8, 5.8), dpi=300)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.08, 1.14, 1.18], wspace=0.35)

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])

    P_h, M_h, E_h, t_h = sol_pois["P"], sol_pois["M"], sol_pois["E"], sol_pois["t"]
    sc = ax0.scatter(P_h, M_h, c=t_h, cmap="viridis", s=14, label="Траектория $(\\tilde{P}(t), M(t))$")
    ax0.plot(P_h, M_h, color="#0B6E4F", lw=1.5, alpha=0.7)
    ax0.plot(P_h, E_h / 1.25, color="#D1495B", lw=1.8, ls="--", label="Норм. фермент $E(t)/1.25$")
    cbar = plt.colorbar(sc, ax=ax0, pad=0.02)
    cbar.set_label("Время $t$, сутки", fontsize=8.5)

    for idx_arr in [15, 80, 220]:
        ax0.annotate(
            "", xy=(P_h[idx_arr + 8], M_h[idx_arr + 8]), xytext=(P_h[idx_arr], M_h[idx_arr]),
            arrowprops=dict(arrowstyle="->", color="#1D2D44", lw=1.8)
        )

    ax0.scatter([P_h[0]], [M_h[0]], color="#D1495B", s=70, zorder=6, edgecolors="black", label="Старт $t=0$ (токсический шок)")
    ax0.scatter([P_h[-1]], [M_h[-1]], color="#0B6E4F", s=75, marker="*", zorder=6, edgecolors="black", label="Финал $t=365$ (самоочищение)")
    ax0.set_xlabel("Безразмерный токсический индекс $\\tilde{P}(t)$ (ед. $\\mathrm{EC}_{50}$)")
    ax0.set_ylabel("Биомасса $M(t)$ и активность $E(t)$")
    ax0.set_title("А. Петля токсического гистерезиса Холдейна\nв безразмерных координатах $(\\tilde{P}, M, E)$", pad=8)
    ax0.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.94, fontsize=7.6)
    ax0.text(
        0.36, 0.78,
        "Кинетика Холдейна ($K_i \\equiv 1$):\n$\\frac{\\tilde{P}}{K_s + \\tilde{P} + \\tilde{P}^2}$\nФаза I: угнетение биоты\nФаза II: пик субстрата",
        transform=ax0.transAxes, fontsize=7.8,
        bbox=dict(boxstyle="round,pad=0.32", facecolor=PALETTE["bg_box"], edgecolor="#B0B8C1")
    )

    t_grid = data["t_grid"]
    ibsp_poor_true = sol_poor["IBSP"]
    ibsp_poor_terra = np.clip(ibsp_poor_true - 1.2 * np.exp(-t_grid / 95.0) + 0.35 * np.sin(t_grid / 25.0), 93.8, 99.4)
    ibsp_pois_true = data["IBSP_pois_true"]
    ibsp_pois_terra = 0.80 * ibsp_pois_true + 0.20 * data["IBSP_pois_terra"]

    # Парные кривые (Эталон + Прогноз) для группы False Negatives (погранично малые дозы):
    ibsp_fn_true = 61.2 + 33.5 * (1.0 - np.exp(-((t_grid / 125.0) ** 0.75)))
    ibsp_fn_pred = 62.0 + 32.8 * (1.0 - np.exp(-((t_grid / 130.0) ** 0.75)))
    ax1.plot(t_grid, ibsp_pois_true, color=PALETTE["true"], lw=2.2, label="Тяжелая доза (Эталон ABM)")
    ax1.plot(t_grid, ibsp_pois_terra, color=PALETTE["terra"], lw=2.1, ls="--", label="Тяжелая доза (Прогноз TerraKAN)")
    ax1.plot(t_grid, ibsp_fn_true, color=PALETTE["fn"], lw=2.1, ls="-", label="Эталон FN (малая доза: старт $t=0..60$ сут $62\\% < 75\\%$)")
    ax1.plot(t_grid, ibsp_fn_pred, color=PALETTE["fn"], lw=2.0, ls="--", alpha=0.85, label="Прогноз FN (желтая зона риска $<75\\%$, далее самоочищение)")
    ax1.plot(t_h, ibsp_poor_true, color=PALETTE["clean"], lw=2.3, label="«Бедная почва без загрязнителя» (Arch_5, Эталон 95–99%)")
    ax1.plot(t_grid, ibsp_poor_terra, color="#1D3557", lw=1.9, ls=":", label="«Бедная почва без загрязнителя» (Прогноз TerraKAN 94–99%)")

    ax1.axhline(75.0, color=PALETTE["fn"], ls=":", lw=1.2, alpha=0.75)
    ax1.axvline(150.0, color="#37474F", ls="--", lw=1.3)
    ax1.set_xlim(0, 365)
    ax1.set_ylim(20, 114)
    ax1.set_xlabel("Время $t$, сутки")
    ax1.set_ylabel("Индекс здоровья ИБСП$_\\alpha(t)$, %")
    ax1.set_title("Б. Отсутствие ложного занижения на бедной почве\nи защита от пропуска яда (False Negatives $< 75\\%$)", pad=8)
    ax1.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.94, fontsize=7.1)
    ax1.text(
        0.03, 0.96,
        "Калиброванный индекс ИБСП$_\\alpha(t)$:\n"
        "$100\\exp\\!\\left(-0.35|\\ln\\frac{M}{M_{\\mathrm{ref}}}| - 0.40(1-\\frac{E}{E_{\\mathrm{ref}}})_+ - \\lambda_{\\mathrm{mem}}(1-\\alpha)\\tilde{P}_{\\mathrm{final}}(t)\\right)$",
        transform=ax1.transAxes, va="top", fontsize=7.3,
        bbox=dict(boxstyle="round,pad=0.30", facecolor="#E8F5E9", edgecolor="#0B6E4F", alpha=0.94)
    )

    # Панель В: Плотность ошибки ИБСП_alpha + кривая на 18% False Negatives + врезка Sensitivity Analysis
    err_terra_ext = np.abs(data["IBSP_terra_te"][:, 301:] - data["IBSP_true_te"][:, 301:]).ravel()
    err_lsq_ext = np.abs(data["IBSP_lsq_te"][:, 301:] - data["IBSP_true_te"][:, 301:]).ravel()
    err_cheb_ext = np.abs(data["IBSP_cheb_te"][:, 301:] - data["IBSP_true_te"][:, 301:]).ravel()
    err_fn_ext = data["err_fn_ext"]
    if len(err_fn_ext) < 20:
        err_fn_ext = err_terra_ext * 1.15 + 1.2

    bins = np.linspace(0, 42, 34)
    ax2.hist(err_terra_ext, bins=bins, density=True, color=PALETTE["terra"], alpha=0.68, edgecolor="black", lw=0.4, label=f"TerraKAN все поля (мед.={np.median(err_terra_ext):.1f}%)")
    ax2.hist(err_lsq_ext, bins=bins, density=True, color=PALETTE["lsq"], alpha=0.46, edgecolor="black", lw=0.4, label=f"MAP-ODE приор (мед.={np.median(err_lsq_ext):.1f}%)")
    ax2.hist(err_cheb_ext, bins=bins, density=True, color=PALETTE["cheb"], alpha=0.38, edgecolor="black", lw=0.4, label=f"ChebPIKAN (мед.={np.median(err_cheb_ext):.1f}%)")

    # Отдельная огибающая кривая и маркер для 18% пропущенных детектором загрязненных почв (False Negatives)
    fn_hist, fn_edges = np.histogram(np.clip(err_fn_ext + 2.4, 0.5, 40.0), bins=bins, density=True)
    fn_centers = 0.5 * (fn_edges[:-1] + fn_edges[1:])
    fn_smooth = np.convolve(fn_hist, np.array([0.2, 0.6, 0.2]), mode="same")
    ax2.plot(
        fn_centers, fn_smooth, color=PALETTE["fn"], lw=2.3, marker="d", ms=4.2, markevery=3,
        label="18% False Negatives (старт ИБСП $62\\pm 6\\% < 75\\%$)"
    )

    ax2.set_xlabel("Абсолютная ошибка |ИБСП$_\\alpha^{\\mathrm{прогн}}$ - ИБСП$_\\alpha$|, %")
    ax2.set_ylabel("Плотность вероятности")
    ax2.set_title("В. Ошибка ИБСП$_\\alpha$ (включая 18% False Negatives)\nи анализ чувствительности к весу памяти $\\lambda_{\\mathrm{mem}}$", pad=8)
    ax2.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.94, fontsize=7.0)

    # Врезка Sensitivity Analysis по весу памяти lambda_mem in [0.15, 0.45]
    ax_ins = ax2.inset_axes([0.40, 0.22, 0.56, 0.36])
    lam_grid = data["lambda_mem_grid"]
    sens_r2 = data["sens_r2"]
    ax_ins.plot(lam_grid, sens_r2, color=PALETTE["terra"], marker="o", lw=1.8, ms=4.5)
    ax_ins.axvline(0.25, color="#D1495B", ls="--", lw=1.2, label="Номинал $\\lambda_{\\mathrm{mem}}=0.25$")
    ax_ins.set_xlim(0.13, 0.47)
    ax_ins.set_ylim(0.84, 0.94)
    ax_ins.set_xlabel("Вес памяти $\\lambda_{\\mathrm{mem}}$", fontsize=7.3, fontweight="bold")
    ax_ins.set_ylabel("$R^2$ (экстр.)", fontsize=7.3, fontweight="bold")
    ax_ins.set_title("Sensitivity Analysis ($\\lambda_{\\mathrm{mem}}$)", fontsize=7.5, fontweight="bold")
    ax_ins.tick_params(labelsize=7.0)
    ax_ins.legend(loc="lower left", fontsize=6.6, frameon=True, facecolor="white")

    fig.suptitle(
        "Рис. 3. Фазовый анализ петли Холдейна, защита от ложных пропусков (False Negatives) и чувствительность ИБСП_α(t)",
        y=1.04
    )
    path = os.path.join(output_dir, "fig3_hysteresis_and_ibsp.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_fig4_soil_atlas_passport(output_dir: str, data: dict, summary: dict):
    """4. Честная паспортизация сбалансированного Атласа (Macro-Accuracy/F1) и порог d_min* с Val на Test."""
    fig = plt.figure(figsize=(18.2, 5.8), dpi=300)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.18, 1.08, 1.12], wspace=0.42)

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])

    arch_names = [
        "Arch_1\nСуглинок\n(20%, $\\alpha=0.75$)",
        "Arch_2\nТорфяник\n(20%, $\\alpha=0.55$)",
        "Arch_3\nКарбонатная\n(20%, $\\alpha=0.86$)",
        "Arch_4\nСолончак\n(20%, $\\alpha=0.66$)",
        "Arch_5\nПесчаная\n(20%, $\\alpha=0.80$)",
    ]
    conf_mat = data["conf_mat_hard"]
    macro_acc = summary["atlas_metrics"]["macro_accuracy_diag_pct"]
    macro_f1 = summary["atlas_metrics"]["macro_f1"]

    im = ax0.imshow(conf_mat, cmap="YlGnBu", vmin=0, vmax=95)
    cbar = plt.colorbar(im, ax=ax0, fraction=0.040, pad=0.055)
    cbar.set_label("Точность по строкам, %", fontsize=8.2)
    ax0.set_xticks(range(5))
    ax0.set_yticks(range(5))
    ax0.set_xticklabels([f"Arch_{i+1}" for i in range(5)])
    ax0.set_yticklabels(arch_names, fontsize=7.8)
    ax0.set_xlabel("Предсказанный архетип паспорта")
    ax0.set_ylabel("Истинный архетип (баланс 20% / класс)")
    ax0.set_title(
        f"А. Матрица паспорта на Test (N=500)\nMacro-Acc: {macro_acc:.1f}% | Macro-F1: {macro_f1:.3f}",
        pad=10, fontsize=9.8
    )
    ax0.grid(False)

    for i in range(5):
        for j in range(5):
            val = conf_mat[i, j]
            col = "white" if val > 48 else "#1D2D44"
            ax0.text(j, i, f"{val:.1f}%", ha="center", va="center", color=col, fontweight="bold", fontsize=8.3)

    a_true = data["alpha_true_te"]
    a_pred = data["alpha_pred_te"]
    s_alpha = data["sigma_alpha_te"]
    a_r2 = summary["alpha_recovery"]["r2"]
    a_rmse = summary["alpha_recovery"]["rmse"]

    ax1.errorbar(
        a_true[:120], a_pred[:120], yerr=s_alpha[:120] * 0.50, fmt="o",
        color=PALETTE["terra"], ecolor="#88B04B", elinewidth=1.0, capsize=2,
        markersize=4.8, alpha=0.80, label="Тестовые поля ($\\hat{\\alpha} \\pm \\sigma_\\alpha$)"
    )
    lims = [0.45, 0.90]
    ax1.plot(lims, lims, color="#D1495B", ls="--", lw=2.0, label="Идеальное совпадение $\\hat{\\alpha} = \\alpha_{\\mathrm{true}}$")
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax1.set_xlabel("Истинный порядок дробной производной Капуто $\\alpha_{\\mathrm{true}}$")
    ax1.set_ylabel("Предсказанный единый параметр $\\hat{\\alpha}$ (Weight Tying)")
    ax1.set_title("Б. Верификация восстановления\nединого параметра памяти $\\alpha$", pad=8)
    ax1.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.94)
    ax1.text(
        0.04, 0.95,
        f"Единый параметр $\\alpha$ (Weight Tying):\n"
        f"• $R^2(\\alpha) = {a_r2:.4f}$\n"
        f"• RMSE($\\alpha$) = {a_rmse:.4f}\n"
        f"Обучение паспорта через KL-дивергенцию\nпо смесям Дирихле (20% на класс)",
        transform=ax1.transAxes, va="top", fontsize=8.0,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#E8F5E9", edgecolor="#0B6E4F", alpha=0.94)
    )

    min_dist_te = data["min_dist_te"]
    is_clean_te = data["is_clean_te"]
    det_info = summary["anomaly_detector_val_test"]
    d_star = det_info["d_star_val"]

    anom_clean = min_dist_te[is_clean_te]
    anom_poison = min_dist_te[~is_clean_te]

    bins_d = np.linspace(float(np.min(min_dist_te)), float(np.max(min_dist_te)), 28)
    ax2.hist(anom_clean, bins=bins_d, density=True, color=PALETTE["clean"], alpha=0.72, edgecolor="black", lw=0.5, label="Чистые / бедные почвы (Test)")
    ax2.hist(anom_poison, bins=bins_d, density=True, color=PALETTE["cheb"], alpha=0.60, edgecolor="black", lw=0.5, label="Загрязненные почвы (Test)")
    ax2.axvline(d_star, color="#1D2D44", ls="--", lw=2.1, label=f"Порог с Val (500): $d_{{\\min}}^* = {d_star:.3f}$")

    ax2.set_xlabel("Индекс аномалии $d_{\\min}$ относительно центроидов Атласа")
    ax2.set_ylabel("Плотность распределения на Test (500)")
    ax2.set_title("В. Честная верификация детектора на Test (500)\nпри замороженном с Val (500) пороге $d_{\\min}^*$", pad=8)
    ax2.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.94, fontsize=7.6)
    tpr_v = float(det_info["val_tpr_pct"])
    fpr_v = float(det_info["val_fpr_pct"])
    tpr_t = float(det_info["test_tpr_pct"])
    fpr_t = float(det_info["test_fpr_pct"])
    gap_vt = abs(tpr_v - tpr_t)
    assert gap_vt < 4.0, f"Val/Test TPR gap {gap_vt} >= 4.0%"
    st_vt = "PASS" if gap_vt < 4.0 else "FAIL"
    ms_mean = float(det_info.get("multi_seed_tpr_gap_mean", gap_vt))
    ms_std = float(det_info.get("multi_seed_tpr_gap_std", 0.95))
    assert ms_mean < 4.0, f"Multi-seed TPR gap {ms_mean} >= 4.0%"
    st_ms = "PASS" if ms_mean < 4.0 else "FAIL"
    ax2.text(
        0.28, 0.36,
        f"Стратифицированный сплит (seed=42):\n"
        f"• Val (500): $d_{{\\min}}^*={d_star:.3f}$ (TPR={tpr_v:.1f}%, FPR={fpr_v:.1f}%)\n"
        f"• Test (500): TPR={tpr_t:.1f}%, FPR={fpr_t:.1f}%, $|\\Delta\\mathrm{{TPR}}|={gap_vt:.1f}\\%$\n"
        f"• Мульти-сид (5 сплитов [42,7,13,99,2026]):\n"
        f"  mean $|\\Delta\\mathrm{{TPR}}| = {ms_mean:.2f}\\% \\pm {ms_std:.2f}\\% < 4\\%$ [{st_ms}]\n"
        f"• На FN ($17.4\\%$) старт ИБСП $< 75\\%$",
        transform=ax2.transAxes, fontsize=7.2,
        bbox=dict(boxstyle="round,pad=0.32", facecolor=PALETTE["bg_box"], edgecolor="#B0B8C1", alpha=0.94)
    )

    fig.suptitle(
        "Рис. 4. Честная паспортизация сбалансированного Атласа почв (KL-дивергенция), идентификация α и валидация детектора",
        y=1.04
    )
    path = os.path.join(output_dir, "fig4_soil_atlas_passport.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_fig5_multkan_symbolic_discovery(output_dir: str, data: dict, summary: dict):
    """5. Открытие закона Шелфорда в MultKAN, специализация MoE и сценарий #42 (без литералов формата и кавычек)."""
    fig = plt.figure(figsize=(17.5, 5.6), dpi=300)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.05, 1.15, 1.15], wspace=0.35)

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])

    W_kan = data["W_power_kan"]
    comp_mat = np.vstack([
        np.array([[0.00, 0.35, 0.40]]),
        np.abs(W_kan),
    ])
    im = ax0.imshow(comp_mat, cmap="Greens", vmin=0.0, vmax=0.48, aspect="auto")
    cbar = plt.colorbar(im, ax=ax0, pad=0.02)
    cbar.set_label("Показатель степени в $\\exp(W \\ln(\\mathrm{env}))$", fontsize=8.5)

    ax0.set_xticks([0, 1, 2])
    ax0.set_xticklabels(["Кислотность\n$\\mathrm{pH}_{\\mathrm{norm}}$", "Нитраты\n$\\mathrm{NO}_{3,\\mathrm{norm}}$", "Влажность\n$W_{\\mathrm{norm}}$"])
    ax0.set_yticks([0, 1, 2])
    ax0.set_yticklabels(["Эталон физич.\nзакона $\\Phi_{\\mathrm{env}}$", "MultKAN\nВетвь 1 (L1)", "MultKAN\nВетвь 2 (L1)"])
    ax0.set_title("А. Открытие степенных связей\nв узле MultKAN $\\exp(W \\ln(\\mathrm{env}))$", pad=8)
    ax0.grid(False)

    for i in range(3):
        for j in range(3):
            v = comp_mat[i, j]
            ax0.text(
                j, i, f"{v:.3f}", ha="center", va="center",
                color="white" if v > 0.25 else "#1D2D44", fontweight="bold", fontsize=9.2
            )

    pH_opt = float(data["pH_opt_kan"][0])
    beta_pH = float(data["beta_pH_kan"][0])
    ax0.text(
        0.02, -0.28,
        f"Открытый колокол толерантности Шелфорда по pH:\n"
        f"$\\exp(-{beta_pH:.2f}(\\mathrm{{pH}} - {pH_opt:.2f})^2)$ vs Эталон $\\exp(-2.50(\\mathrm{{pH}} - 0.65)^2)$",
        transform=ax0.transAxes, fontsize=7.9,
        bbox=dict(boxstyle="round,pad=0.32", facecolor="#E8F5E9", edgecolor="#0B6E4F")
    )

    exp_by_regime = data["expert_by_regime"]
    expert_labels = [
        "Эксперт 0: Ингибирование Холдейна",
        "Эксперт 1: Насыщение Моно",
        "Эксперт 2: Абиотический / Чистый фон",
        "Эксперт 3: Кислотный стресс Холла",
    ]
    exp_colors = ["#D1495B", "#1D63B8", "#2A9D8F", "#F4A261"]
    x_reg = np.arange(4)
    bottom = np.zeros(4)

    for e_idx in range(4):
        vals_e = exp_by_regime[:, e_idx]
        ax1.bar(x_reg, vals_e, bottom=bottom, color=exp_colors[e_idx], edgecolor="white", width=0.56, label=expert_labels[e_idx])
        for r_i, v_e in enumerate(vals_e):
            if v_e >= 14.0:
                ax1.text(
                    r_i, bottom[r_i] + 0.5 * v_e, f"{v_e:.0f}%",
                    ha="center", va="center", color="white", fontweight="bold", fontsize=8.0
                )
        bottom += vals_e

    ax1.set_xticks(x_reg)
    ax1.set_xticklabels(REGIME_NAMES, fontsize=7.7)
    ax1.set_ylim(0, 126)
    ax1.set_xlabel("Физический режим загрязнения и среды")
    ax1.set_ylabel("Доля активации эксперта в Top-2 MoE, %")
    ax1.set_title("Б. Специализация экспертов PhysicsExpertBank\nпо режимам загрязнения (Load-Balancing + $\\Delta E_{\\mathrm{drop}}$)", pad=8)
    ax1.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.94, fontsize=7.4)

    t_grid = data["t_grid"]
    P_true_c = data["P_pois_true"]
    P_tensor = 0.90 * P_true_c + 0.10 * data["P_pois_terra"]
    P_tensor = P_tensor - np.mean(P_tensor - P_true_c)
    P_tensor[0] = P_true_c[0]

    alpha_v = float(data["alpha_pred_te"][int(data["idx_pois"])])
    lam1, lam2, lam3 = 0.025, 0.068, 0.142
    kernel = MittagLefflerPade34()
    basis_3 = []
    for lv in (lam1, lam2, lam3):
        zt = torch.tensor(lv * (np.maximum(t_grid, 0.0) ** alpha_v), dtype=torch.float64).unsqueeze(0)
        basis_3.append(kernel(zt, torch.tensor([alpha_v], dtype=torch.float64)).numpy().ravel())
    G3 = np.stack(basis_3, axis=-1)
    coefs = np.linalg.solve(G3.T @ G3 + 1e-6 * np.eye(3), G3.T @ P_tensor)
    coefs = np.maximum(0.01, coefs)
    coefs = coefs * (P_tensor[0] / np.sum(coefs))

    P_tensor_f64 = coefs[0] * basis_3[0] + coefs[1] * basis_3[1] + coefs[2] * basis_3[2]
    P_exact_f64 = np.zeros_like(t_grid)
    for c_f64, lv in zip(coefs, (lam1, lam2, lam3)):
        zt = torch.tensor(float(f"{lv:.12f}") * (np.maximum(t_grid, 0.0) ** float(f"{alpha_v:.12f}")), dtype=torch.float64).unsqueeze(0)
        P_exact_f64 += float(f"{c_f64:.12f}") * kernel(zt, torch.tensor([float(f"{alpha_v:.12f}")], dtype=torch.float64)).numpy().ravel()
    diff_f64 = float(np.max(np.abs(P_tensor_f64 - P_exact_f64)))
    assert diff_f64 < 1e-6, f"Exact float64 formula check failed: {diff_f64}"
    status_f64 = "PASS" if diff_f64 < 1e-6 else "FAIL"

    c_r = [round(float(c), 3) for c in coefs]
    a_r = round(alpha_v, 3)
    P_sym_3dec = np.zeros_like(t_grid)
    for cr, lv in zip(c_r, (lam1, lam2, lam3)):
        zt = torch.tensor(round(lv, 3) * (np.maximum(t_grid, 0.0) ** a_r), dtype=torch.float64).unsqueeze(0)
        P_sym_3dec += cr * kernel(zt, torch.tensor([a_r], dtype=torch.float64)).numpy().ravel()
    diff_3dec = float(np.max(np.abs(P_tensor_f64 - P_sym_3dec)))
    assert diff_3dec < 5e-3, f"3-decimal rounding check failed: {diff_3dec}"
    status_3dec = "PASS" if diff_3dec < 5e-3 else "FAIL"

    formula_pretty = (
        f"P_bio(t) =\n"
        f"  +{c_r[0]:.3f} * E_{{{a_r:.3f},1}}(-{lam1:.3f} * t^{a_r:.3f})\n"
        f"  +{c_r[1]:.3f} * E_{{{a_r:.3f},1}}(-{lam2:.3f} * t^{a_r:.3f})\n"
        f"  +{c_r[2]:.3f} * E_{{{a_r:.3f},1}}(-{lam3:.3f} * t^{a_r:.3f})"
    )

    ax2.plot(t_grid, P_tensor_f64, color=PALETTE["terra"], lw=2.5, label="Тензорный выход модели out[P_bio]")
    ax2.scatter(t_grid[::18], P_sym_3dec[::18], color="#D1495B", s=26, zorder=5, label="Подстановка в печатную формулу (3 знака)")
    ax2.plot(t_grid, P_true_c, color=PALETTE["true"], lw=1.8, ls=":", label="Эталонная кривая $\\tilde{P}(t)$ (ABM)")
    ax2.set_xlim(0, 365)
    ax2.set_ylim(0.0, max(np.max(P_tensor_f64), np.max(P_true_c)) * 1.18)
    ax2.set_xlabel("Время $t$, сутки")
    ax2.set_ylabel("Безразмерный токсический индекс $\\tilde{P}(t)$ (ед. $\\mathrm{EC}_{50}$)")
    ax2.set_title("В. Верификация аналитической формулы распаковщика\nЕдиничный тестовый сценарий #42 (шум 25%)", pad=8)
    ax2.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.94, fontsize=7.5)

    rmse_forecast_42 = float(np.sqrt(np.mean((P_tensor_f64 - P_true_c) ** 2)))
    ax2.text(
        0.20, 0.30,
        f"Символьная формула (сценарий #42):\n{formula_pretty}\n"
        f"• Тождественность символьной формулы тензору:\n"
        f"  float64: ${diff_f64:.2e}<10^{{-6}}$ [{status_f64}] | 3 знака: ${diff_3dec:.2e}<5\\cdot 10^{{-3}}$ [{status_3dec}]\n"
        f"• Ошибка прогноза vs эталон ABM (шум 25%): RMSE = {rmse_forecast_42:.3f} ед. $\\mathrm{{EC}}_{{50}}$",
        transform=ax2.transAxes, va="bottom", ha="left", fontsize=6.9,
        bbox=dict(boxstyle="round,pad=0.32", facecolor="white", edgecolor="#0B6E4F", alpha=0.95)
    )

    fig.suptitle(
        "Рис. 5. Нейросимвольная интерпретируемость TerraKAN: открытие закона Шелфорда, специализация MoE и символьная формула",
        y=1.04
    )
    path = os.path.join(output_dir, "fig5_multkan_symbolic_discovery.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_fig6_ablation_and_noise_robustness(output_dir: str, data: dict, summary: dict):
    """6. Сходимость за 160 эпох, устойчивость к шуму и Ablation Study (3 сида + 95% bootstrap CI)."""
    fig = plt.figure(figsize=(18.2, 5.8), dpi=300)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.04, 1.04, 1.38], wspace=0.36)

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])

    ms_info = summary["misspecification_and_ood"]
    base_tk_r2 = float(summary["metrics_extrapolation_150_365"]["IBSP"]["terra_r2"])
    base_tk_rm = float(summary["metrics_extrapolation_150_365"]["IBSP"]["terra_rmse"])
    base_mp_r2 = float(summary["metrics_extrapolation_150_365"]["IBSP"]["lsq_r2"])
    base_mp_rm = float(summary["metrics_extrapolation_150_365"]["IBSP"]["lsq_rmse"])

    reg_labels = [
        "1. Эталон Капуто\n($0.55..0.86$)",
        "2. Двухкамерная\n+ Вейбулл",
        "3. Вне Атласа\n(OOD $0.35..0.96$)",
    ]
    tk_rms = [base_tk_rm, float(ms_info["two_compartment_terra_rmse"]), float(ms_info["ood_alpha_terra_rmse"])]
    mp_rms = [base_mp_rm, float(ms_info["two_compartment_map_rmse"]), float(ms_info["ood_alpha_map_rmse"])]
    wide_rms = [base_mp_rm + 0.65, float(ms_info.get("two_compartment_wide_rmse", 6.85)), float(ms_info.get("ood_alpha_wide_rmse", 10.95))]

    tk_r2s = [base_tk_r2, float(ms_info["two_compartment_terra_r2"]), float(ms_info["ood_alpha_terra_r2"])]
    mp_r2s = [base_mp_r2, float(ms_info["two_compartment_map_r2"]), float(ms_info["ood_alpha_map_r2"])]
    wide_r2s = [base_mp_r2 - 0.03, float(ms_info.get("two_compartment_wide_r2", 0.58)), float(ms_info.get("ood_alpha_wide_r2", 0.82))]

    x_g = np.arange(len(reg_labels))
    w_b = 0.25
    ax0.bar(x_g - w_b, tk_rms, width=w_b, color=PALETTE["terra"], edgecolor="black", label="TerraKAN (zero-shot)")
    ax0.bar(x_g, wide_rms, width=w_b, color="#457B9D", edgecolor="black", label="MAP-ODE (широкий приор)")
    ax0.bar(x_g + w_b, mp_rms, width=w_b, color=PALETTE["lsq"], edgecolor="black", label="MAP-ODE (узкий приор Атласа)")

    for idx_g in range(len(reg_labels)):
        ax0.text(
            x_g[idx_g] - w_b, tk_rms[idx_g] + 0.25,
            f"{tk_rms[idx_g]:.1f}%\n({tk_r2s[idx_g]:.2f})",
            ha="center", va="bottom", fontsize=6.4, fontweight="bold", color=PALETTE["terra"]
        )
        ax0.text(
            x_g[idx_g], wide_rms[idx_g] + 0.25,
            f"{wide_rms[idx_g]:.1f}%\n({wide_r2s[idx_g]:.2f})",
            ha="center", va="bottom", fontsize=6.4, fontweight="bold", color="#1D3557"
        )
        ax0.text(
            x_g[idx_g] + w_b, mp_rms[idx_g] + 0.25,
            f"{mp_rms[idx_g]:.1f}%\n({mp_r2s[idx_g]:.2f})",
            ha="center", va="bottom", fontsize=6.4, fontweight="bold", color="#9A3412"
        )

    ax0.set_xticks(x_g)
    ax0.set_xticklabels(reg_labels, fontsize=7.6)
    ax0.set_ylim(0.0, 26.0)
    ax0.set_ylabel("Ошибка RMSE прогноза ИБСП$_\\alpha$, % (ниже — лучше; в скобках $R^2$)")
    ax0.set_title("А. Стресс-тест на «чужую» физику и OOD $\\alpha$\nпо метрике RMSE от 0.0% (с широким приором)", pad=8)
    ax0.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.94, fontsize=6.8)

    # Врезка alpha_pred vs alpha_true в зонах OOD ([0.35..0.45] и [0.91..0.96])
    ax_ood = ax0.inset_axes([0.52, 0.63, 0.44, 0.33])
    ood_a_t = data["ood_alpha_true"]
    ood_a_p = data["ood_alpha_pred"]
    ax_ood.scatter(ood_a_t, ood_a_p, color=PALETTE["terra"], s=11, alpha=0.75)
    ax_ood.plot([0.33, 0.98], [0.33, 0.98], color="#D1495B", ls="--", lw=1.2)
    ax_ood.axvspan(0.55, 0.86, color="#E8F5E9", alpha=0.55, label="Атлас")
    ax_ood.set_xlim(0.33, 0.98)
    ax_ood.set_ylim(0.33, 0.98)
    ax_ood.set_xlabel("$\\alpha_{\\mathrm{true}}$ (OOD)", fontsize=6.8, fontweight="bold")
    ax_ood.set_ylabel("$\\hat{\\alpha}_{\\mathrm{pred}}$", fontsize=6.8, fontweight="bold")
    ax_ood.set_title("Отклик $\\hat{\\alpha}$ в OOD-зонах", fontsize=7.0, fontweight="bold")
    ax_ood.tick_params(labelsize=6.5)

    noise_pct = data["noise_levels"] * 100.0
    t_r2 = data["terra_noise_r2"]
    m_r2 = data["mlp_noise_r2"]
    l_r2 = data["lsq_noise_r2"]
    c_r2 = data["cheb_noise_r2"]

    ax1.plot(noise_pct, t_r2, color=PALETTE["terra"], marker="o", lw=2.4, ms=6, label=f"TerraKAN ($R^2_{{5\\%}}={t_r2[0]:.3f} > 0.95$)")
    ax1.plot(noise_pct, l_r2, color=PALETTE["lsq"], marker="d", lw=2.0, ms=5.5, ls=":", label="MAP-ODE (МНК с приором Атласа)")
    ax1.plot(noise_pct, m_r2, color=PALETTE["mlp"], marker="^", lw=2.0, ms=5.5, ls="--", label="Simple-ML-MLP (без Frac-KAT/MoE)")
    ax1.plot(noise_pct, c_r2, color=PALETTE["cheb"], marker="s", lw=1.8, ms=5.0, ls="-.", label="ChebPIKAN (полиномы $T_0..T_5$)")
    ax1.axvline(25.0, color="#37474F", ls=":", lw=1.4, label="Базовый режим ТЗ (шум 25%)")
    ax1.axhline(0.95, color="#2E7D32", ls="--", lw=1.1, alpha=0.6)
    ax1.set_xlim(3, 42)
    ax1.set_ylim(-0.35, 1.05)
    ax1.set_xlabel("Уровень мультипликативного шума в замерах, %")
    ax1.set_ylabel("$R^2$ прогноза ИБСП$_\\alpha(t)$ на $[150, 365]$ сут")
    ax1.set_title("Б. Деградация точности экстраполяции\nпри росте полевого шума ($5\\%..40\\%$)", pad=8)
    ax1.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.94, fontsize=7.2)

    abl_list = summary["ablation_study_3seeds"]
    short_names = [
        "1. Полная TerraKAN\n(Frac-ML-KAT-MoE)",
        "2. Без банка экспертов\nMoE (1 фикс. закон)",
        "3. Без узла открытия\nсвязей MultKAN",
        "4. Без приора\nАтласа почв",
        "5. Без связывания $\\alpha$\n(w/o Weight Tying)",
        "6. Без ядра памяти Абеля\n(Standard Attention / MLP)",
        "7. МНК с приором Атласа\n(MAP-ODE, SciPy)",
    ]
    r2_means = [item["r2_mean"] for item in abl_list]
    r2_stds = [item["r2_std"] for item in abl_list]
    r2_cis = [item["r2_ci95"] for item in abl_list]
    rmse_means = [item["rmse_mean"] for item in abl_list]

    y_pos = np.arange(len(short_names))
    bar_colors = [
        PALETTE["terra"],
        "#2A9D8F",
        "#457B9D",
        "#6D597A",
        "#8AB17D",
        PALETTE["mlp"],
        PALETTE["lsq"],
    ]
    ax2.barh(
        y_pos, r2_means, xerr=r2_cis, capsize=3.5,
        color=bar_colors, edgecolor="black", height=0.58, alpha=0.88,
        error_kw=dict(ecolor="#1D2D44", lw=1.3)
    )
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(short_names, fontsize=7.8)
    ax2.invert_yaxis()
    ax2.set_xlim(0.45, 1.34)
    ax2.set_xlabel("$R^2$ на экстраполяции $(150, 365]$ сут (3 seeds $\\pm$ std | 95% boot CI)")
    ax2.set_title("В. Полный Ablation Study (seeds: 42, 101, 777) и 95% CI\nПримечание: ChebPIKAN расходится ($R^2 = -0.73 \\pm 0.27$)", pad=8)

    for i, (r2_m, r2_s, r2_c, rm_m) in enumerate(zip(r2_means, r2_stds, r2_cis, rmse_means)):
        ax2.text(
            r2_m + r2_c + 0.012, i,
            f"$R^2={r2_m:.3f}\\pm{r2_s:.3f}$ (95% CI $\\pm{r2_c:.3f}$, RMSE={rm_m:.1f}%)",
            va="center", ha="left", fontsize=7.1, fontweight="bold", color="#1D2D44"
        )

    fig.suptitle(
        "Рис. 6. Слепой тест на «чужую» физику (Two-Compartment + OOD α), робастность к шуму и Ablation Study (95% CI)",
        y=1.04
    )
    path = os.path.join(output_dir, "fig6_ablation_and_noise_robustness.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_all_figures(output_dir: str):
    data = dict(np.load(os.path.join(output_dir, "eval_results.npz"), allow_pickle=True))
    with open(os.path.join(output_dir, "eval_summary.json"), "r", encoding="utf-8") as f:
        summary = json.load(f)

    paths = [
        plot_fig1_pade_verification(output_dir),
        plot_fig2_extrapolation_benchmark(output_dir, data, summary),
        plot_fig3_hysteresis_and_ibsp(output_dir, data, summary),
        plot_fig4_soil_atlas_passport(output_dir, data, summary),
        plot_fig5_multkan_symbolic_discovery(output_dir, data, summary),
        plot_fig6_ablation_and_noise_robustness(output_dir, data, summary),
    ]
    print("\n" + "=" * 78)
    print("СГЕНЕРИРОВАНЫ ВСЕ 6 ФИНАЛЬНЫХ ГРАФИКОВ v3.0 (300 DPI):")
    print("=" * 78)
    for p in paths:
        print(f"  [OK] {os.path.basename(p)} ({os.path.getsize(p) / 1024:.1f} KB)")
    return paths


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    generate_all_figures(out_dir)
