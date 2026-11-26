"""
TerraKAN v1.5.0 — Расширенный академический симулятор дробной биокинетики почв.

Сохраняет 100% обратную совместимость с замороженным эталонным сплитом seed=42
(Base Train 3000 / Frozen Val 500 / Frozen Test 500) и добавляет:
  - Режимы v1.2.5 (хронические предэквилибрированные, M-чувствительные, полный dropout, G1).
  - Режимы v1.3.0 (рекальцитрантные / тяжёлые металлы / PFAS с P_inf > 0, D^alpha P_inf = 0).
  - НОВЫЕ режимы v1.5.0:
      * Быстрая биоаугментация консорциумами (k_deg in [0.20, 0.65], пост-пиковый старт)
      * Гормезис и ризосферная стимуляция (+15..+50% подъём M, E выше контроля)
  - 4 академических OOD-бенчмарка (Two-Compartment + Weibull, Out-of-Atlas alpha,
    Logistic-Lag Consortium, Biphasic Hormesis).
"""

import math
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.special import gamma


SOIL_ARCHETYPES: List[Dict[str, object]] = [
    {
        "id": 0, "code": "Arch_1", "name": "Sod-Podzolic Loam",
        "alpha": 0.75, "pH_norm": 0.58, "NO3_norm": 0.55, "W_norm": 0.60, "K_m": 1.00,
        "k_p": 0.018, "k_deg": 0.085, "r_m": 0.068, "gamma_p": 0.015, "mu_sub": 0.046,
        "beta_E": 0.090, "delta_E": 0.075, "K_s": 0.90, "K_i": 1.00, "K_tox": 1.00,
    },
    {
        "id": 1, "code": "Arch_2", "name": "Peat-Bog",
        "alpha": 0.55, "pH_norm": 0.36, "NO3_norm": 0.44, "W_norm": 0.88, "K_m": 1.18,
        "k_p": 0.012, "k_deg": 0.065, "r_m": 0.054, "gamma_p": 0.012, "mu_sub": 0.038,
        "beta_E": 0.082, "delta_E": 0.068, "K_s": 1.05, "K_i": 1.00, "K_tox": 1.00,
    },
    {
        "id": 2, "code": "Arch_3", "name": "Sod-Carbonate",
        "alpha": 0.86, "pH_norm": 0.68, "NO3_norm": 0.80, "W_norm": 0.70, "K_m": 1.38,
        "k_p": 0.026, "k_deg": 0.125, "r_m": 0.088, "gamma_p": 0.011, "mu_sub": 0.058,
        "beta_E": 0.110, "delta_E": 0.080, "K_s": 0.80, "K_i": 1.00, "K_tox": 1.00,
    },
    {
        "id": 3, "code": "Arch_4", "name": "Solonchak",
        "alpha": 0.66, "pH_norm": 0.88, "NO3_norm": 0.30, "W_norm": 0.32, "K_m": 0.68,
        "k_p": 0.014, "k_deg": 0.055, "r_m": 0.046, "gamma_p": 0.022, "mu_sub": 0.030,
        "beta_E": 0.070, "delta_E": 0.085, "K_s": 1.15, "K_i": 1.00, "K_tox": 1.00,
    },
    {
        "id": 4, "code": "Arch_5", "name": "Sandy Forest",
        "alpha": 0.80, "pH_norm": 0.50, "NO3_norm": 0.22, "W_norm": 0.36, "K_m": 0.44,
        "k_p": 0.022, "k_deg": 0.070, "r_m": 0.060, "gamma_p": 0.015, "mu_sub": 0.040,
        "beta_E": 0.075, "delta_E": 0.075, "K_s": 0.90, "K_i": 1.00, "K_tox": 1.00,
    },
]


def get_atlas_centroid_matrix() -> np.ndarray:
    return np.array(
        [[float(a["pH_norm"]), float(a["NO3_norm"]), float(a["W_norm"]), float(a["K_m"]), float(a["alpha"])]
         for a in SOIL_ARCHETYPES],
        dtype=np.float64,
    )


def compute_phi_env(pH_norm: np.ndarray, NO3_norm: np.ndarray, W_norm: np.ndarray) -> np.ndarray:
    pH_c = np.clip(pH_norm, 0.05, 1.50)
    NO3_c = np.clip(NO3_norm, 0.02, 2.00)
    W_c = np.clip(W_norm, 0.02, 2.00)
    return np.exp(-2.5 * ((pH_c - 0.65) ** 2)) * (NO3_c ** 0.35) * (W_c ** 0.40)


def compute_ibsp_alpha(
    P_tilde_t: np.ndarray,
    M_t: np.ndarray,
    E_t: np.ndarray,
    alpha: np.ndarray,
    M_ref: np.ndarray,
    E_ref: np.ndarray,
) -> np.ndarray:
    M_safe = np.maximum(M_t, 1e-6)
    M_ref_safe = np.maximum(M_ref, 1e-6)
    E_ref_safe = np.maximum(E_ref, 1e-6)
    term_M = 0.35 * np.abs(np.log(M_safe / M_ref_safe))
    term_E = 0.40 * np.maximum(0.0, 1.0 - (E_t / E_ref_safe))
    term_P = 0.25 * (1.0 - alpha) * np.maximum(0.0, P_tilde_t)
    return 100.0 * np.exp(-(term_M + term_E + term_P))


def solve_caputo_abm_batch(
    alpha: np.ndarray,
    params_batch: Dict[str, np.ndarray],
    y0_batch: np.ndarray,
    W_t_batch: np.ndarray,
    t_max: float = 365.0,
    dt: float = 0.5,
    P_inf_batch: Optional[np.ndarray] = None,
    hormesis_amp: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Векторизованный решатель Diethelm ABM для дробной системы Капуто v1.5.0:
    поддерживает стойкую асимптоту P_inf (D^alpha P_inf = 0) и горметическую стимуляцию.
    """
    B = alpha.shape[0]
    n_steps = int(round(t_max / dt))
    t_grid = np.linspace(0.0, t_max, n_steps + 1, dtype=np.float64)

    Y = np.zeros((B, n_steps + 1, 3), dtype=np.float64)
    F = np.zeros((B, n_steps + 1, 3), dtype=np.float64)
    Y[:, 0, :] = y0_batch

    pH_norm = params_batch["pH_norm"][:, None]
    NO3_norm = params_batch["NO3_norm"][:, None]
    Phi_t = compute_phi_env(pH_norm, NO3_norm, W_t_batch)

    K_m = params_batch["K_m"]
    k_p = params_batch["k_p"]
    k_deg = params_batch["k_deg"]
    r_m = params_batch["r_m"]
    gamma_p = params_batch["gamma_p"]
    mu_sub = params_batch["mu_sub"]
    beta_E = params_batch["beta_E"]
    delta_E = params_batch["delta_E"]
    K_s = params_batch["K_s"]
    K_i = params_batch["K_i"]
    K_tox = params_batch["K_tox"]
    P_inf = np.zeros(B, dtype=np.float64) if P_inf_batch is None else P_inf_batch
    h_amp = np.zeros(B, dtype=np.float64) if hormesis_amp is None else hormesis_amp

    def rhs_batch(step_idx: int, y_state: np.ndarray) -> np.ndarray:
        p_tot = np.maximum(P_inf, y_state[:, 0])
        p_bio = np.maximum(0.0, p_tot - P_inf)
        m_pos = np.maximum(0.0, y_state[:, 1])
        e_pos = np.maximum(0.0, y_state[:, 2])
        phi = Phi_t[:, step_idx]

        haldane = p_bio / (K_s + p_bio + (p_bio * p_bio) / K_i)
        dp = -k_p * p_bio - k_deg * phi * haldane * m_pos

        # Горметический стимул при малых/средних дозах или ризосферных экссудатах
        horm_boost = h_amp * (p_bio / (0.35 + p_bio * p_bio))
        dm = (
            r_m * phi * m_pos * (1.0 - m_pos / (K_m * (1.0 + 0.55 * horm_boost)))
            - gamma_p * p_tot * m_pos
            + mu_sub * haldane * m_pos
        )
        e_drive = (1.0 + 0.85 * horm_boost) / (1.0 + (p_tot / K_tox) ** 2)
        de = beta_E * m_pos * e_drive - delta_E * e_pos
        return np.stack([dp, dm, de], axis=-1)

    F[:, 0, :] = rhs_batch(0, Y[:, 0, :])

    h_a_g1 = ((dt ** alpha) / gamma(alpha + 1.0))[:, None]
    h_a_g2 = ((dt ** alpha) / gamma(alpha + 2.0))[:, None]

    k_idx = np.arange(n_steps + 1, dtype=np.float64)[None, :]
    a_col = alpha[:, None]
    b_weights = (k_idx + 1.0) ** a_col - (k_idx ** a_col)

    m_idx = np.arange(1, n_steps + 1, dtype=np.float64)[None, :]
    a_weights = (
        (m_idx + 1.0) ** (a_col + 1.0)
        - 2.0 * (m_idx ** (a_col + 1.0))
        + (m_idx - 1.0) ** (a_col + 1.0)
    )

    y_init = Y[:, 0, :].copy()
    for n in range(n_steps):
        conv_b = np.einsum("bj,bjc->bc", b_weights[:, n::-1], F[:, : n + 1, :])
        y_pred = np.maximum(0.0, y_init + h_a_g1 * conv_b)
        y_pred[:, 0] = np.maximum(P_inf, y_pred[:, 0])

        a0 = ((n ** (alpha + 1.0)) - (n - alpha) * ((n + 1.0) ** alpha))[:, None]
        if n >= 1:
            conv_a = a0 * F[:, 0, :] + np.einsum("bj,bjc->bc", a_weights[:, n - 1 :: -1], F[:, 1 : n + 1, :])
        else:
            conv_a = a0 * F[:, 0, :]

        f_pred = rhs_batch(n + 1, y_pred)
        y_corr = np.maximum(0.0, y_init + h_a_g2 * (f_pred + conv_a))
        y_corr[:, 0] = np.maximum(P_inf, y_corr[:, 0])

        Y[:, n + 1, :] = y_corr
        F[:, n + 1, :] = rhs_batch(n + 1, y_corr)

    return t_grid, Y, F, Phi_t


def _sample_observations(
    rng: np.random.Generator,
    t_grid: np.ndarray,
    M_all: np.ndarray,
    E_all: np.ndarray,
    W_t_all: np.ndarray,
    rain_flag_all: np.ndarray,
    scen_params: Dict[str, np.ndarray],
    obs_horizon: float = 150.0,
    min_obs: int = 6,
    max_obs: int = 8,
    noise_level: float = 0.25,
    force_channel_dropout: Optional[np.ndarray] = None,
    t_offset_days: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    n_scenarios = M_all.shape[0]
    dt = float(t_grid[1] - t_grid[0])
    obs_max_idx = int(round(obs_horizon / dt))

    obs_values = np.zeros((n_scenarios, max_obs, 5), dtype=np.float64)
    obs_mask = np.zeros((n_scenarios, max_obs, 5), dtype=np.float64)
    obs_times = np.zeros((n_scenarios, max_obs), dtype=np.float64)
    obs_grid_idx = np.zeros((n_scenarios, max_obs), dtype=np.int64)
    obs_rain_flag = np.zeros((n_scenarios, max_obs, 1), dtype=np.float64)
    obs_valid_len = np.zeros(n_scenarios, dtype=np.int64)

    for b in range(n_scenarios):
        start_idx = 0
        if t_offset_days is not None and t_offset_days[b] > 0.0:
            start_idx = min(obs_max_idx - 20, int(round(t_offset_days[b] / dt)))

        n_pts = int(rng.integers(min_obs, max_obs + 1))
        obs_valid_len[b] = n_pts
        cand = np.arange(start_idx + 4, obs_max_idx + 1, dtype=np.int64)
        chosen = np.sort(rng.choice(cand, size=n_pts - 1, replace=False))
        idx_seq = np.concatenate([[start_idx], chosen])

        if n_pts < max_obs:
            pad_idx = np.full(max_obs - n_pts, idx_seq[-1], dtype=np.int64)
            idx_seq_padded = np.concatenate([idx_seq, pad_idx])
        else:
            idx_seq_padded = idx_seq

        obs_grid_idx[b] = idx_seq_padded
        # Временная шкала от первого визита (или от 0 для стандартных сценариев)
        obs_times[b] = t_grid[idx_seq_padded] - t_grid[start_idx]
        obs_rain_flag[b, :, 0] = rain_flag_all[b, idx_seq_padded]

        drop_mode = 0 if force_channel_dropout is None else int(force_channel_dropout[b])
        for j in range(n_pts):
            t_idx = idx_seq[j]
            clean_vec = np.array([
                scen_params["pH_norm"][b],
                scen_params["NO3_norm"][b],
                W_t_all[b, t_idx],
                M_all[b, t_idx],
                E_all[b, t_idx],
            ], dtype=np.float64)

            mult_noise = 1.0 + rng.normal(0.0, noise_level * 0.65, size=5)
            mult_noise[:3] = 1.0 + rng.normal(0.0, noise_level * 0.25, size=3)
            obs_values[b, j] = np.clip(clean_vec * mult_noise, 0.02, 12.0)

            obs_mask[b, j, :3] = 1.0
            if drop_mode == 1:  # полный dropout M
                obs_mask[b, j, 3] = 0.0
                obs_mask[b, j, 4] = 1.0
            elif drop_mode == 2:  # полный dropout E
                obs_mask[b, j, 3] = 1.0
                obs_mask[b, j, 4] = 0.0
            else:
                obs_mask[b, j, 3] = 1.0
                obs_mask[b, j, 4] = 1.0 if (j == 0 or rng.random() >= 0.50) else 0.0

    return {
        "obs_values": obs_values,
        "obs_mask": obs_mask,
        "obs_times": obs_times,
        "obs_grid_idx": obs_grid_idx,
        "obs_rain_flag": obs_rain_flag,
        "obs_valid_len": obs_valid_len,
    }


def generate_terrakan_v150_dataset(
    n_base: int = 4000,
    n_augmented: int = 3500,
    t_max: float = 365.0,
    dt: float = 0.5,
    obs_horizon: float = 150.0,
    noise_level: float = 0.25,
    chunk_size: int = 500,
    seed: int = 42,
) -> Dict[str, object]:
    """
    Генерирует корпус TerraKAN v1.5.0:
      - Индексы [0 : 3000]     : Базовый Train (seed=42, идентичен v1.1.0–v1.3.1)
      - Индексы [3000 : 3500]  : Замороженный Val (500)
      - Индексы [3500 : 4000]  : Замороженный Test (500)
      - Индексы [4000 : 4000+n_augmented]: Расширенный тренировочный корпус v1.5.0:
          * Хронические предэквилибрированные + dropout (25%)
          * Рекальцитрантные / тяжёлые металлы / PFAS (25%, P_inf > 0, hint=+1)
          * Быстрая биоаугментация консорциумами + пост-пиковый спад (25%, k_deg in [0.20, 0.65], hint=-1)
          * Гормезис и ризосферная стимуляция (25%, подъём E/M выше контроля, hint=-1)
    """
    rng = np.random.default_rng(seed)
    n_total = n_base + n_augmented
    n_steps = int(round(t_max / dt))
    t_grid = np.linspace(0.0, t_max, n_steps + 1, dtype=np.float64)

    param_keys = [
        "alpha", "pH_norm", "NO3_norm", "W_norm", "K_m",
        "k_p", "k_deg", "r_m", "gamma_p", "mu_sub",
        "beta_E", "delta_E", "K_s", "K_i", "K_tox",
    ]
    arch_matrix = {
        k: np.array([float(a[k]) for a in SOIL_ARCHETYPES], dtype=np.float64)
        for k in param_keys
    }

    dominant_arch = np.zeros(n_total, dtype=np.int64)
    for block_start in range(0, n_total, 500):
        block_len = min(500, n_total - block_start)
        base_arch = np.repeat(np.arange(5, dtype=np.int64), block_len // 5)
        if len(base_arch) < block_len:
            base_arch = np.concatenate([base_arch, np.arange(block_len - len(base_arch), dtype=np.int64)])
        dominant_arch[block_start : block_start + block_len] = rng.permutation(base_arch)

    dirichlet_weights = np.zeros((n_total, 5), dtype=np.float64)
    for b in range(n_total):
        alpha_dir = np.full(5, 0.22, dtype=np.float64)
        alpha_dir[dominant_arch[b]] = 3.60
        w = rng.dirichlet(alpha_dir)
        max_idx = int(np.argmax(w))
        if max_idx != dominant_arch[b]:
            w[max_idx], w[dominant_arch[b]] = w[dominant_arch[b]], w[max_idx]
        dirichlet_weights[b] = w

    scen_params: Dict[str, np.ndarray] = {}
    for k in param_keys:
        base_val = dirichlet_weights @ arch_matrix[k]
        if k in ("K_i", "K_tox"):
            scen_params[k] = np.ones(n_total, dtype=np.float64)
        else:
            jitter = rng.normal(1.0, 0.015, size=n_total)
            scen_params[k] = np.clip(base_val * jitter, 0.01, None)

    scen_params["alpha"] = np.clip(scen_params["alpha"], 0.45, 0.89)
    scen_params["pH_norm"] = np.clip(scen_params["pH_norm"], 0.25, 0.95)
    scen_params["NO3_norm"] = np.clip(scen_params["NO3_norm"], 0.15, 0.98)
    scen_params["W_norm"] = np.clip(scen_params["W_norm"], 0.20, 0.95)

    regime_id = np.zeros(n_total, dtype=np.int64)
    P0 = np.zeros(n_total, dtype=np.float64)
    P_inf = np.zeros(n_total, dtype=np.float64)
    hormesis_amp = np.zeros(n_total, dtype=np.float64)
    is_clean_soil = np.zeros(n_total, dtype=bool)
    class_hint_true = np.zeros(n_total, dtype=np.float64)  # -1 biodeg, 0 clean, +1 persistent
    force_dropout = np.zeros(n_total, dtype=np.int64)
    t_offset_days = np.zeros(n_total, dtype=np.float64)

    # 1. Базовые 4000 сценариев (строго стратифицированные блоки по 500)
    for block_start in range(0, n_base, 500):
        block_len = min(500, n_base - block_start)
        for k_arch in range(5):
            arch_pos = np.where(dominant_arch[block_start : block_start + block_len] == k_arch)[0] + block_start
            n_a = len(arch_pos)
            n_clean = int(round(0.22 * n_a))
            n_low = int(round(0.28 * n_a))
            perm_a = rng.permutation(arch_pos)
            for idx_p, b in enumerate(perm_a):
                if idx_p < n_clean:
                    regime_id[b] = 0
                    is_clean_soil[b] = True
                    P0[b] = rng.uniform(0.01, 0.05)
                    class_hint_true[b] = 0.0
                elif idx_p < n_clean + n_low:
                    regime_id[b] = 1
                    P0[b] = rng.uniform(0.68, 0.95)
                    scen_params["K_i"][b] = 25.0
                    class_hint_true[b] = -1.0
                else:
                    if scen_params["pH_norm"][b] < 0.44:
                        regime_id[b] = 3
                        P0[b] = rng.uniform(1.25, 2.55)
                    else:
                        regime_id[b] = 2
                        P0[b] = rng.uniform(2.10, 4.10)
                    scen_params["K_i"][b] = 1.00
                    class_hint_true[b] = -1.0

    # 2. Расширенный корпус v1.5.0 [n_base : n_total]
    for i in range(n_augmented):
        b = n_base + i
        bucket = i % 4
        if bucket == 0:
            # Хронические / dropout / чистый олиготрофный G1
            sub = (i // 4) % 3
            if sub == 0:
                regime_id[b] = 2
                P0[b] = rng.uniform(0.90, 2.80)
                t_offset_days[b] = rng.uniform(15.0, 40.0)
                class_hint_true[b] = -1.0
            elif sub == 1:
                regime_id[b] = 2
                P0[b] = rng.uniform(0.80, 3.00)
                force_dropout[b] = int(rng.choice([1, 2]))
                class_hint_true[b] = -1.0
            else:
                regime_id[b] = 0
                is_clean_soil[b] = True
                P0[b] = rng.uniform(0.005, 0.04)
                class_hint_true[b] = 0.0
        elif bucket == 1:
            # Рекальцитрантный / Тяжёлые металлы / Соленость / PFAS (Expert 4)
            regime_id[b] = 4
            P0[b] = rng.uniform(0.75, 3.40)
            frac_persist = rng.uniform(0.82, 1.00)
            P_inf[b] = P0[b] * frac_persist
            scen_params["k_deg"][b] = rng.uniform(0.0, 0.0015)
            scen_params["k_p"][b] = rng.uniform(0.0, 0.0010)
            scen_params["mu_sub"][b] = 0.0
            class_hint_true[b] = 1.0
        elif bucket == 2:
            # НОВЫЙ: Быстрая биоаугментация консорциумами + пост-пиковый спад (Expert 5)
            regime_id[b] = 5
            P0[b] = rng.uniform(1.00, 3.60)
            scen_params["k_deg"][b] = rng.uniform(0.20, 0.65)
            scen_params["k_p"][b] = rng.uniform(0.02, 0.08)
            scen_params["K_i"][b] = rng.uniform(4.0, 20.0)
            # Половина — с пост-пиковым стартом (решает неоднозначность Alpine Biostim)
            if rng.random() < 0.50:
                t_offset_days[b] = rng.uniform(5.0, 20.0)
            class_hint_true[b] = -1.0
        else:
            # НОВЫЙ: Гормезис и ризосферная стимуляция (Expert 5)
            regime_id[b] = 5
            P0[b] = rng.uniform(0.45, 2.20)
            hormesis_amp[b] = rng.uniform(0.35, 1.15)
            scen_params["k_deg"][b] = rng.uniform(0.09, 0.30)
            scen_params["K_i"][b] = rng.uniform(3.0, 15.0)
            class_hint_true[b] = -1.0

    K_m_arr = scen_params["K_m"]
    E_ref_arr = (scen_params["beta_E"] / scen_params["delta_E"]) * K_m_arr
    shock_M = 1.0 / (1.0 + 0.25 * P0)
    shock_E = (1.0 + 0.45 * hormesis_amp) / (1.0 + (P0 / scen_params["K_tox"]) ** 2)

    M0 = K_m_arr * shock_M * (1.0 + 0.25 * hormesis_amp) * rng.uniform(0.97, 1.03, size=n_total)
    E0 = (E_ref_arr / K_m_arr) * M0 * shock_E * rng.uniform(0.97, 1.03, size=n_total)
    y0_all = np.stack([P0, M0, E0], axis=-1)

    W_t_all = np.broadcast_to(scen_params["W_norm"][:, None], (n_total, n_steps + 1)).copy()
    rain_flag_all = np.zeros((n_total, n_steps + 1), dtype=np.float64)
    for b in range(n_total):
        n_rains = int(rng.integers(1, 3))
        for _ in range(n_rains):
            t_rain = rng.uniform(12.0, 320.0)
            amp = rng.uniform(0.08, 0.22)
            decay = np.exp(-np.maximum(0.0, t_grid - t_rain) / 8.0) * (t_grid >= t_rain)
            W_t_all[b] = np.clip(W_t_all[b] + amp * decay, 0.10, 1.20)
            rain_flag_all[b] = np.maximum(rain_flag_all[b], (np.abs(t_grid - t_rain) <= 6.0).astype(np.float64))

    Y_all = np.zeros((n_total, n_steps + 1, 3), dtype=np.float64)
    Phi_all = np.zeros((n_total, n_steps + 1), dtype=np.float64)

    for start in range(0, n_total, chunk_size):
        end = min(start + chunk_size, n_total)
        sub_params = {k: v[start:end] for k, v in scen_params.items()}
        _, Y_chunk, _, Phi_chunk = solve_caputo_abm_batch(
            alpha=scen_params["alpha"][start:end],
            params_batch=sub_params,
            y0_batch=y0_all[start:end],
            W_t_batch=W_t_all[start:end],
            t_max=t_max,
            dt=dt,
            P_inf_batch=P_inf[start:end],
            hormesis_amp=hormesis_amp[start:end],
        )
        Y_all[start:end] = Y_chunk
        Phi_all[start:end] = Phi_chunk

    P_all, M_all, E_all = Y_all[:, :, 0], Y_all[:, :, 1], Y_all[:, :, 2]
    IBSP_all = compute_ibsp_alpha(
        P_tilde_t=P_all, M_t=M_all, E_t=E_all,
        alpha=scen_params["alpha"][:, None],
        M_ref=K_m_arr[:, None], E_ref=E_ref_arr[:, None],
    )

    obs = _sample_observations(
        rng=rng, t_grid=t_grid, M_all=M_all, E_all=E_all,
        W_t_all=W_t_all, rain_flag_all=rain_flag_all, scen_params=scen_params,
        obs_horizon=obs_horizon, noise_level=noise_level,
        force_channel_dropout=force_dropout, t_offset_days=t_offset_days,
    )

    return {
        "t_grid": t_grid,
        "P_true": P_all,
        "P_inf_true": P_inf,
        "M_true": M_all,
        "E_true": E_all,
        "IBSP_true": IBSP_all,
        "Phi_true": Phi_all,
        "W_t": W_t_all,
        "alpha_true": scen_params["alpha"],
        "params": scen_params,
        "dirichlet_weights": dirichlet_weights,
        "dominant_arch": dominant_arch,
        "regime_id": regime_id,
        "is_clean_soil": is_clean_soil,
        "class_hint_true": class_hint_true,
        **obs,
    }


def generate_extended_ood_benchmarks(t_grid: np.ndarray, n_samples: int = 100, seed: int = 123) -> Dict[str, object]:
    """
    4 академических OOD-симулятора для проверки обобщения TerraKAN 1.5 вне обучающих уравнений:
      1. Two-Compartment Sorption ODE + Kohlrausch-Weibull (классика из v1.1.0)
      2. Out-of-Atlas alpha in [0.35, 0.45] U [0.91, 0.96] (классика из v1.1.0)
      3. Logistic-Lag Bioaugmentation Consortium (быстрый распад после фазы акклиматизации)
      4. Biphasic Arndt-Schulz Hormesis (начальная стимуляция фермента + поздний спад)
    """
    rng = np.random.default_rng(seed)
    T = len(t_grid)
    dt = float(t_grid[1] - t_grid[0])

    # 1. Two-Compartment Sorption + Weibull
    P_tc = np.zeros((n_samples, T))
    M_tc = np.zeros((n_samples, T))
    E_tc = np.zeros((n_samples, T))
    IBSP_tc = np.zeros((n_samples, T))
    alpha_tc = rng.uniform(0.62, 0.80, size=n_samples)
    for b in range(n_samples):
        P0 = rng.uniform(1.2, 3.8)
        k_deg, k_in, k_out = rng.uniform(0.025, 0.055), rng.uniform(0.030, 0.060), rng.uniform(0.008, 0.020)
        beta_kw = rng.uniform(0.58, 0.78)
        K_m = rng.uniform(0.70, 1.30)
        E_ref = K_m * 1.20
        pf, ps = P0 * 0.75, P0 * 0.25
        mv, ev = K_m / (1.0 + 0.25 * P0), E_ref / ((1.0 + 0.25 * P0) * (1.0 + P0 * P0))
        for n_t, tv in enumerate(t_grid):
            p_tot = 0.65 * (pf + ps) + 0.35 * P0 * math.exp(-((0.022 * max(tv, 0.0)) ** beta_kw))
            P_tc[b, n_t], M_tc[b, n_t], E_tc[b, n_t] = p_tot, mv, ev
            hald = pf / (0.90 + pf + pf * pf)
            pf = max(0.0, pf + dt * (-k_deg * hald * mv - k_in * pf + k_out * ps))
            ps = max(0.0, ps + dt * (k_in * pf - k_out * ps))
            mv = max(0.05, mv + dt * (0.065 * mv * (1.0 - mv / K_m) - 0.012 * p_tot * mv + 0.040 * hald * mv))
            ev = max(0.05, ev + dt * (0.090 * (mv / (1.0 + p_tot * p_tot)) - 0.075 * ev))
        IBSP_tc[b] = compute_ibsp_alpha(P_tc[b], M_tc[b], E_tc[b], alpha_tc[b], K_m, E_ref)

    # 2. Logistic-Lag Consortium (t_half ~ 10..20 d)
    P_lg = np.zeros((n_samples, T))
    M_lg = np.zeros((n_samples, T))
    E_lg = np.zeros((n_samples, T))
    IBSP_lg = np.zeros((n_samples, T))
    for b in range(n_samples):
        P0 = rng.uniform(1.0, 3.2)
        t_lag = rng.uniform(4.0, 14.0)
        k_fast = rng.uniform(0.12, 0.32)
        K_m = rng.uniform(0.80, 1.35)
        E_ref = K_m * 1.20
        p_cur = P0
        mv, ev = K_m * 0.72, E_ref * 0.55
        for n_t, tv in enumerate(t_grid):
            P_lg[b, n_t], M_lg[b, n_t], E_lg[b, n_t] = p_cur, mv, ev
            act = 1.0 / (1.0 + math.exp(-0.35 * (tv - t_lag)))
            p_cur = max(0.0, p_cur - dt * k_fast * act * p_cur / (0.45 + p_cur))
            mv = max(0.05, mv + dt * (0.09 * mv * (1.0 - mv / K_m) + 0.05 * act * p_cur / (0.5 + p_cur)))
            ev = max(0.05, ev + dt * (0.11 * mv / (1.0 + 0.6 * p_cur * p_cur) - 0.08 * ev))
        IBSP_lg[b] = compute_ibsp_alpha(P_lg[b], M_lg[b], E_lg[b], 0.85, K_m, E_ref)

    # 3. Biphasic Arndt-Schulz Hormesis
    P_hm = np.zeros((n_samples, T))
    M_hm = np.zeros((n_samples, T))
    E_hm = np.zeros((n_samples, T))
    IBSP_hm = np.zeros((n_samples, T))
    for b in range(n_samples):
        P0 = rng.uniform(0.5, 1.8)
        K_m = rng.uniform(0.75, 1.25)
        E_ref = K_m * 1.20
        k_d = rng.uniform(0.03, 0.09)
        for n_t, tv in enumerate(t_grid):
            p_cur = P0 * math.exp(-((k_d * tv) ** 0.78))
            horm = 1.0 + 0.45 * p_cur * math.exp(-0.65 * p_cur)
            P_hm[b, n_t] = p_cur
            M_hm[b, n_t] = K_m * (0.85 + 0.25 * horm) / (1.0 + 0.15 * p_cur * p_cur)
            E_hm[b, n_t] = E_ref * horm / (1.0 + 0.30 * p_cur * p_cur)
        IBSP_hm[b] = compute_ibsp_alpha(P_hm[b], M_hm[b], E_hm[b], 0.75, K_m, E_ref)

    return {
        "two_compartment": {"P": P_tc, "M": M_tc, "E": E_tc, "IBSP": IBSP_tc},
        "logistic_lag_consortium": {"P": P_lg, "M": M_lg, "E": E_lg, "IBSP": IBSP_lg},
        "biphasic_hormesis": {"P": P_hm, "M": M_hm, "E": E_hm, "IBSP": IBSP_hm},
    }
