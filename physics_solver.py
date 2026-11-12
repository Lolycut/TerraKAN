"""
Численный решатель прямой задачи дробной биокинетики почв (предиктор-корректор
Адамса-Башфорта-Моултона для производной Капуто) и генератор эталонного Атласа почв.

В данном модуле я задаю 5 почвенных архетипов, уравнения динамики микробной биомассы M(t),
ферментативной активности E(t) и безразмерной нагрузки P_tilde(t) (в единицах EC50),
а также динамический индекс здоровья почвы с дробной памятью ИБСП_alpha(t).
"""

import math
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.special import gamma
import torch


SOIL_ARCHETYPES: List[Dict[str, object]] = [
    {
        "id": 0,
        "code": "Arch_1",
        "name": "Дерново-подзолистый суглинок",
        "desc": "Умеренная буферность, сбалансированный водно-воздушный режим",
        "alpha": 0.75,
        "pH_norm": 0.58,
        "NO3_norm": 0.55,
        "W_norm": 0.60,
        "K_m": 1.00,
        "k_p": 0.018,
        "k_deg": 0.085,
        "r_m": 0.068,
        "gamma_p": 0.015,
        "mu_sub": 0.046,
        "beta_E": 0.090,
        "delta_E": 0.075,
        "K_s": 0.90,
        "K_i": 1.00,
        "K_tox": 1.00,
    },
    {
        "id": 1,
        "code": "Arch_2",
        "name": "Торфяно-болотная",
        "desc": "Сильная память микропор (alpha=0.55), кислый pH, высокая влагоемкость",
        "alpha": 0.55,
        "pH_norm": 0.36,
        "NO3_norm": 0.44,
        "W_norm": 0.88,
        "K_m": 1.18,
        "k_p": 0.012,
        "k_deg": 0.065,
        "r_m": 0.054,
        "gamma_p": 0.012,
        "mu_sub": 0.038,
        "beta_E": 0.082,
        "delta_E": 0.068,
        "K_s": 1.05,
        "K_i": 1.00,
        "K_tox": 1.00,
    },
    {
        "id": 2,
        "code": "Arch_3",
        "name": "Дерново-карбонатная",
        "desc": "Высокая биогенность, быстрое самоочищение (alpha=0.86), оптимум pH",
        "alpha": 0.86,
        "pH_norm": 0.68,
        "NO3_norm": 0.80,
        "W_norm": 0.70,
        "K_m": 1.38,
        "k_p": 0.026,
        "k_deg": 0.125,
        "r_m": 0.088,
        "gamma_p": 0.011,
        "mu_sub": 0.058,
        "beta_E": 0.110,
        "delta_E": 0.080,
        "K_s": 0.80,
        "K_i": 1.00,
        "K_tox": 1.00,
    },
    {
        "id": 3,
        "code": "Arch_4",
        "name": "Техногенный солончак",
        "desc": "Высокий осмотический и токсический стресс (alpha=0.66), щелочной сдвиг",
        "alpha": 0.66,
        "pH_norm": 0.88,
        "NO3_norm": 0.30,
        "W_norm": 0.32,
        "K_m": 0.68,
        "k_p": 0.014,
        "k_deg": 0.055,
        "r_m": 0.046,
        "gamma_p": 0.022,
        "mu_sub": 0.030,
        "beta_E": 0.070,
        "delta_E": 0.085,
        "K_s": 1.15,
        "K_i": 1.00,
        "K_tox": 1.00,
    },
    {
        "id": 4,
        "code": "Arch_5",
        "name": "Песчаная лесная",
        "desc": "Низкая емкость биомассы K_m («бедная почва»), быстрый дренаж (alpha=0.80)",
        "alpha": 0.80,
        "pH_norm": 0.50,
        "NO3_norm": 0.22,
        "W_norm": 0.36,
        "K_m": 0.44,
        "k_p": 0.022,
        "k_deg": 0.070,
        "r_m": 0.060,
        "gamma_p": 0.015,
        "mu_sub": 0.040,
        "beta_E": 0.075,
        "delta_E": 0.075,
        "K_s": 0.90,
        "K_i": 1.00,
        "K_tox": 1.00,
    },
]

REGIME_NAMES: List[str] = [
    "Чистая почва\n($\\tilde{P}_0 < 0.1$)",
    "Малая доза Моно\n($\\tilde{P}_0 < 0.8$)",
    "Токсический шок\nХолдейна ($\\tilde{P}_0 > 2.0$)",
    "Кислый стресс\n($\\mathrm{pH} < 0.45$)",
]


def get_atlas_centroid_matrix() -> np.ndarray:
    """
    Возвращает матрицу эталонных признаков 5 архетипов почв размера (5, 5):
    [pH_norm, NO3_norm, W_norm, K_m, alpha].
    """
    rows = []
    for arch in SOIL_ARCHETYPES:
        rows.append([
            float(arch["pH_norm"]),
            float(arch["NO3_norm"]),
            float(arch["W_norm"]),
            float(arch["K_m"]),
            float(arch["alpha"]),
        ])
    return np.array(rows, dtype=np.float64)


def compute_phi_env(pH_norm: np.ndarray, NO3_norm: np.ndarray, W_norm: np.ndarray) -> np.ndarray:
    """
    Скрытый мультипликативный закон влияния косвенных факторов среды:
        Phi_env(pH, NO3, W) = exp(-2.5 * (pH_norm - 0.65)^2) * NO3_norm^0.35 * W_norm^0.40
    """
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
    P_0: Optional[np.ndarray] = None,
    eps: float = 0.25,
) -> np.ndarray:
    """
    Абсолютно калиброванный интегральный индекс здоровья почвы ИБСП_alpha(t):
        ИБСП_alpha(t) = 100% * exp(
            -0.35 * |ln(M(t) / M_ref)|
            -0.40 * max(0, 1 - E(t) / E_ref)
            -0.25 * (1 - alpha) * P_tilde(t)
        )
    Исключает деление на малый начальный шум P(0) и ложное занижение индекса на бедных чистых почвах.
    """
    M_safe = np.maximum(M_t, 1e-6)
    M_ref_safe = np.maximum(M_ref, 1e-6)
    E_ref_safe = np.maximum(E_ref, 1e-6)

    term_M = 0.35 * np.abs(np.log(M_safe / M_ref_safe))
    term_E = 0.40 * np.maximum(0.0, 1.0 - (E_t / E_ref_safe))
    term_P = 0.25 * (1.0 - alpha) * np.maximum(0.0, P_tilde_t)

    return 100.0 * np.exp(-(term_M + term_E + term_P))


def solve_caputo_abm_single(
    alpha: float,
    params: Dict[str, float],
    y0: Tuple[float, float, float],
    t_max: float = 365.0,
    dt: float = 0.5,
    rain_pulses: Optional[np.ndarray] = None,
    n_corrector_iter: int = 2,
) -> Dict[str, np.ndarray]:
    """
    Численный решатель связанной системы 3 дробных ОДУ Капуто методом Diethelm ABM
    в безразмерных единицах токсической нагрузки P_tilde(t) = P(t) / K_tox (K_tox = 1.0, K_i = 1.0).
    """
    n_steps = int(round(t_max / dt))
    t_grid = np.linspace(0.0, t_max, n_steps + 1, dtype=np.float64)

    P = np.zeros(n_steps + 1, dtype=np.float64)
    M = np.zeros(n_steps + 1, dtype=np.float64)
    E = np.zeros(n_steps + 1, dtype=np.float64)
    F = np.zeros((n_steps + 1, 3), dtype=np.float64)

    P[0], M[0], E[0] = float(y0[0]), float(y0[1]), float(y0[2])

    pH_norm = float(params.get("pH_norm", 0.60))
    NO3_norm = float(params.get("NO3_norm", 0.55))
    W_base = float(params.get("W_norm", 0.60))
    K_m = float(params.get("K_m", 1.00))
    k_p = float(params.get("k_p", 0.018))
    k_deg = float(params.get("k_deg", 0.085))
    r_m = float(params.get("r_m", 0.068))
    gamma_p = float(params.get("gamma_p", 0.015))
    mu_sub = float(params.get("mu_sub", 0.046))
    beta_E = float(params.get("beta_E", 0.090))
    delta_E = float(params.get("delta_E", 0.075))
    K_s = float(params.get("K_s", 0.90))
    # Строгая фиксация K_i = 1.0 и K_tox = 1.0 (единицы EC50)
    K_i = float(params.get("K_i", 1.00))
    K_tox = float(params.get("K_tox", 1.00))

    if rain_pulses is None:
        W_t = np.full(n_steps + 1, W_base, dtype=np.float64)
    else:
        W_t = np.clip(W_base + rain_pulses, 0.05, 1.50)

    phi_t = compute_phi_env(pH_norm, NO3_norm, W_t)

    def rhs_func(idx: int, p_val: float, m_val: float, e_val: float):
        p_pos = max(0.0, p_val)
        m_pos = max(0.0, m_val)
        e_pos = max(0.0, e_val)
        phi = phi_t[idx]
        haldane = p_pos / (K_s + p_pos + (p_pos * p_pos) / K_i)
        dp = -k_p * p_pos - k_deg * phi * haldane * m_pos
        dm = r_m * phi * m_pos * (1.0 - m_pos / K_m) - gamma_p * p_pos * m_pos + mu_sub * haldane * m_pos
        de = beta_E * (m_pos / (1.0 + (p_pos / K_tox) ** 2)) - delta_E * e_pos
        return dp, dm, de

    F[0] = rhs_func(0, P[0], M[0], E[0])

    h_a_g1 = (dt ** alpha) / gamma(alpha + 1.0)
    h_a_g2 = (dt ** alpha) / gamma(alpha + 2.0)

    k_idx = np.arange(n_steps + 1, dtype=np.float64)
    b_weights = (k_idx + 1.0) ** alpha - (k_idx ** alpha)
    m_idx = np.arange(1, n_steps + 1, dtype=np.float64)
    a_weights = (
        (m_idx + 1.0) ** (alpha + 1.0)
        - 2.0 * (m_idx ** (alpha + 1.0))
        + (m_idx - 1.0) ** (alpha + 1.0)
    )

    y_init = np.array([P[0], M[0], E[0]], dtype=np.float64)

    for n in range(n_steps):
        conv_b = b_weights[n::-1] @ F[: n + 1]
        y_pred = np.maximum(0.0, y_init + h_a_g1 * conv_b)

        a0 = (n ** (alpha + 1.0)) - (n - alpha) * ((n + 1.0) ** alpha)
        conv_a = a0 * F[0] + (a_weights[n - 1 :: -1] @ F[1 : n + 1] if n >= 1 else 0.0)

        y_corr = y_pred
        for _ in range(n_corrector_iter):
            f_pred = np.array(rhs_func(n + 1, y_corr[0], y_corr[1], y_corr[2]), dtype=np.float64)
            y_corr = np.maximum(0.0, y_init + h_a_g2 * (f_pred + conv_a))

        P[n + 1], M[n + 1], E[n + 1] = y_corr[0], y_corr[1], y_corr[2]
        F[n + 1] = rhs_func(n + 1, P[n + 1], M[n + 1], E[n + 1])

    M_ref = K_m
    E_ref = (beta_E / delta_E) * K_m
    ibsp = compute_ibsp_alpha(P, M, E, alpha, M_ref, E_ref)

    return {
        "t": t_grid,
        "P": P,
        "M": M,
        "E": E,
        "IBSP": ibsp,
        "phi_env": phi_t,
        "W_t": W_t,
        "M_ref": M_ref,
        "E_ref": E_ref,
    }


def solve_caputo_abm_batch(
    alpha: np.ndarray,
    params_batch: Dict[str, np.ndarray],
    y0_batch: np.ndarray,
    W_t_batch: np.ndarray,
    t_max: float = 365.0,
    dt: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Векторизованный пакетный решатель Diethelm ABM для B сценариев одновременно
    при K_tox = 1.0 и K_i = 1.0 (или заданном K_i для режима гладкого Моно).
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

    def rhs_batch(step_idx: int, y_state: np.ndarray) -> np.ndarray:
        p_pos = np.maximum(0.0, y_state[:, 0])
        m_pos = np.maximum(0.0, y_state[:, 1])
        e_pos = np.maximum(0.0, y_state[:, 2])
        phi = Phi_t[:, step_idx]
        haldane = p_pos / (K_s + p_pos + (p_pos * p_pos) / K_i)
        dp = -k_p * p_pos - k_deg * phi * haldane * m_pos
        dm = r_m * phi * m_pos * (1.0 - m_pos / K_m) - gamma_p * p_pos * m_pos + mu_sub * haldane * m_pos
        de = beta_E * (m_pos / (1.0 + (p_pos / K_tox) ** 2)) - delta_E * e_pos
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

        a0 = ((n ** (alpha + 1.0)) - (n - alpha) * ((n + 1.0) ** alpha))[:, None]
        if n >= 1:
            conv_a = a0 * F[:, 0, :] + np.einsum("bj,bjc->bc", a_weights[:, n - 1 :: -1], F[:, 1 : n + 1, :])
        else:
            conv_a = a0 * F[:, 0, :]

        f_pred = rhs_batch(n + 1, y_pred)
        y_corr = np.maximum(0.0, y_init + h_a_g2 * (f_pred + conv_a))

        Y[:, n + 1, :] = y_corr
        F[:, n + 1, :] = rhs_batch(n + 1, y_corr)

    return t_grid, Y, F, Phi_t


def generate_terrakan_dataset(
    n_scenarios: int = 4000,
    t_max: float = 365.0,
    dt: float = 0.5,
    obs_horizon: float = 150.0,
    min_obs: int = 6,
    max_obs: int = 8,
    noise_level: float = 0.25,
    chunk_size: int = 500,
    seed: int = 42,
) -> Dict[str, object]:
    """
    Генерирует 4 000 сценариев со строгим балансом 5 архетипов по 20% в каждом сплите
    (Train=3000, Val=500, Test=500) и явным разделением 4 режимов загрязнения.
    """
    rng = np.random.default_rng(seed)
    n_steps = int(round(t_max / dt))
    t_grid = np.linspace(0.0, t_max, n_steps + 1, dtype=np.float64)
    obs_max_idx = int(round(obs_horizon / dt))

    param_keys = [
        "alpha", "pH_norm", "NO3_norm", "W_norm", "K_m",
        "k_p", "k_deg", "r_m", "gamma_p", "mu_sub",
        "beta_E", "delta_E", "K_s", "K_i", "K_tox",
    ]
    arch_matrix = {
        k: np.array([float(a[k]) for a in SOIL_ARCHETYPES], dtype=np.float64)
        for k in param_keys
    }

    # Строгая стратификация архетипов по 20% в каждом блоке из 500 сценариев
    dominant_arch = np.zeros(n_scenarios, dtype=np.int64)
    for block_start in range(0, n_scenarios, 500):
        block_len = min(500, n_scenarios - block_start)
        base_arch = np.repeat(np.arange(5, dtype=np.int64), block_len // 5)
        if len(base_arch) < block_len:
            base_arch = np.concatenate([base_arch, np.arange(block_len - len(base_arch), dtype=np.int64)])
        dominant_arch[block_start : block_start + block_len] = rng.permutation(base_arch)

    # Генерируем веса смесей Дирихле с гарантированным доминированием выбранного архетипа
    dirichlet_weights = np.zeros((n_scenarios, 5), dtype=np.float64)
    for b in range(n_scenarios):
        alpha_dir = np.full(5, 0.22, dtype=np.float64)
        alpha_dir[dominant_arch[b]] = 3.60
        w = rng.dirichlet(alpha_dir)
        # Если случайный выброс Дирихле сместил максимум, упорядочиваем доминирующую компоненту
        max_idx = int(np.argmax(w))
        if max_idx != dominant_arch[b]:
            w[max_idx], w[dominant_arch[b]] = w[dominant_arch[b]], w[max_idx]
        dirichlet_weights[b] = w

    scen_params: Dict[str, np.ndarray] = {}
    for k in param_keys:
        base_val = dirichlet_weights @ arch_matrix[k]
        if k in ("K_i", "K_tox"):
            scen_params[k] = np.ones(n_scenarios, dtype=np.float64)
        else:
            jitter = rng.normal(1.0, 0.012, size=n_scenarios)
            scen_params[k] = np.clip(base_val * jitter, 0.01, None)

    scen_params["alpha"] = np.clip(scen_params["alpha"], 0.45, 0.89)
    scen_params["pH_norm"] = np.clip(scen_params["pH_norm"], 0.25, 0.95)
    scen_params["NO3_norm"] = np.clip(scen_params["NO3_norm"], 0.15, 0.98)
    scen_params["W_norm"] = np.clip(scen_params["W_norm"], 0.20, 0.95)

    # Разделение на 4 физических режима загрязнения:
    # Режим 0 (20%): Чистая / бедная почва без загрязнителя (P_tilde(0) in [0.01, 0.06])
    # Режим 1 (25%): Малая доза загрязнения — гладкий режим Моно (P_tilde(0) in [0.30, 0.78])
    # Режим 2 (35%): Высокая доза — токсический шок и петля Холдейна (P_tilde(0) in [2.10, 4.20])
    # Режим 3 (20%): Кислый / осмотический стресс (pH_norm < 0.45, P_tilde(0) in [1.10, 2.60])
    regime_id = np.zeros(n_scenarios, dtype=np.int64)
    P0 = np.zeros(n_scenarios, dtype=np.float64)
    is_clean_soil = np.zeros(n_scenarios, dtype=bool)

    # Строгая стратификация по (архетип, режим дозы) в каждом блоке по 500 сценариев (Train/Val/Test)
    for block_start in range(0, n_scenarios, 500):
        block_len = min(500, n_scenarios - block_start)
        for k_arch in range(5):
            arch_pos = np.where(dominant_arch[block_start : block_start + block_len] == k_arch)[0] + block_start
            n_a = len(arch_pos)
            # Точные квоты режимов внутри каждого архетипа: 22% чистая, 28% малая доза, 50% высокая/кислая
            n_clean = int(round(0.22 * n_a))
            n_low = int(round(0.28 * n_a))
            perm_a = rng.permutation(arch_pos)
            for idx_p, b in enumerate(perm_a):
                if idx_p < n_clean:
                    regime_id[b] = 0
                    is_clean_soil[b] = True
                    P0[b] = rng.uniform(0.01, 0.05)
                elif idx_p < n_clean + n_low:
                    regime_id[b] = 1
                    P0[b] = rng.uniform(0.68, 0.95)
                    scen_params["K_i"][b] = 25.0
                else:
                    if scen_params["pH_norm"][b] < 0.44:
                        regime_id[b] = 3
                        P0[b] = rng.uniform(1.25, 2.55)
                    else:
                        regime_id[b] = 2
                        P0[b] = rng.uniform(2.10, 4.10)
                    scen_params["K_i"][b] = 1.00

    # Начальное состояние биомассы M(0) и фермента E(0) строго по стационарной грани:
    # E(0) / M(0) = (E_ref / M_ref) / (1 + P_tilde(0)^2)
    K_m_arr = scen_params["K_m"]
    E_ref_arr = (scen_params["beta_E"] / scen_params["delta_E"]) * K_m_arr
    shock_M = 1.0 / (1.0 + 0.25 * P0)
    shock_E = 1.0 / (1.0 + (P0 / scen_params["K_tox"]) ** 2)

    M0 = K_m_arr * shock_M * rng.uniform(0.97, 1.03, size=n_scenarios)
    E0 = (E_ref_arr / K_m_arr) * M0 * shock_E * rng.uniform(0.97, 1.03, size=n_scenarios)
    y0_all = np.stack([P0, M0, E0], axis=-1)

    # Дождевые импульсы W(t)
    W_t_all = np.broadcast_to(scen_params["W_norm"][:, None], (n_scenarios, n_steps + 1)).copy()
    rain_flag_all = np.zeros((n_scenarios, n_steps + 1), dtype=np.float64)
    for b in range(n_scenarios):
        n_rains = int(rng.integers(1, 3))
        for _ in range(n_rains):
            t_rain = rng.uniform(12.0, 320.0)
            amp = rng.uniform(0.08, 0.22)
            decay = np.exp(-np.maximum(0.0, t_grid - t_rain) / 8.0) * (t_grid >= t_rain)
            W_t_all[b] = np.clip(W_t_all[b] + amp * decay, 0.10, 1.20)
            rain_flag_all[b] = np.maximum(rain_flag_all[b], (np.abs(t_grid - t_rain) <= 6.0).astype(np.float64))

    Y_all = np.zeros((n_scenarios, n_steps + 1, 3), dtype=np.float64)
    Phi_all = np.zeros((n_scenarios, n_steps + 1), dtype=np.float64)

    for start in range(0, n_scenarios, chunk_size):
        end = min(start + chunk_size, n_scenarios)
        sub_params = {k: v[start:end] for k, v in scen_params.items()}
        _, Y_chunk, _, Phi_chunk = solve_caputo_abm_batch(
            alpha=scen_params["alpha"][start:end],
            params_batch=sub_params,
            y0_batch=y0_all[start:end],
            W_t_batch=W_t_all[start:end],
            t_max=t_max,
            dt=dt,
        )
        Y_all[start:end] = Y_chunk
        Phi_all[start:end] = Phi_chunk

    P_all = Y_all[:, :, 0]
    M_all = Y_all[:, :, 1]
    E_all = Y_all[:, :, 2]

    IBSP_all = compute_ibsp_alpha(
        P_tilde_t=P_all,
        M_t=M_all,
        E_t=E_all,
        alpha=scen_params["alpha"][:, None],
        M_ref=K_m_arr[:, None],
        E_ref=E_ref_arr[:, None],
    )

    obs_values = np.zeros((n_scenarios, max_obs, 5), dtype=np.float64)
    obs_mask = np.zeros((n_scenarios, max_obs, 5), dtype=np.float64)
    obs_times = np.zeros((n_scenarios, max_obs), dtype=np.float64)
    obs_grid_idx = np.zeros((n_scenarios, max_obs), dtype=np.int64)
    obs_rain_flag = np.zeros((n_scenarios, max_obs, 1), dtype=np.float64)
    obs_valid_len = np.zeros(n_scenarios, dtype=np.int64)

    candidate_indices = np.arange(4, obs_max_idx + 1, dtype=np.int64)

    for b in range(n_scenarios):
        n_pts = int(rng.integers(min_obs, max_obs + 1))
        obs_valid_len[b] = n_pts
        chosen = np.sort(rng.choice(candidate_indices, size=n_pts - 1, replace=False))
        idx_seq = np.concatenate([[0], chosen])

        if n_pts < max_obs:
            pad_idx = np.full(max_obs - n_pts, idx_seq[-1], dtype=np.int64)
            idx_seq_padded = np.concatenate([idx_seq, pad_idx])
        else:
            idx_seq_padded = idx_seq

        obs_grid_idx[b] = idx_seq_padded
        obs_times[b] = t_grid[idx_seq_padded]
        obs_rain_flag[b, :, 0] = rain_flag_all[b, idx_seq_padded]

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
            noisy_vec = np.clip(clean_vec * mult_noise, 0.02, 10.0)
            obs_values[b, j] = noisy_vec

            obs_mask[b, j, :4] = 1.0
            if j == 0 or (rng.random() >= 0.50):
                obs_mask[b, j, 4] = 1.0
            else:
                obs_mask[b, j, 4] = 0.0

    return {
        "t_grid": t_grid,
        "P_true": P_all,
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
        "obs_values": obs_values,
        "obs_mask": obs_mask,
        "obs_times": obs_times,
        "obs_grid_idx": obs_grid_idx,
        "obs_rain_flag": obs_rain_flag,
        "obs_valid_len": obs_valid_len,
    }


def generate_misspecified_and_ood_benchmarks(t_grid: np.ndarray, n_samples: int = 120, seed: int = 42) -> Dict[str, object]:
    """
    Генерирует 2 слепых проверочных набора данных для теста на несоответствие модели (Model Misspecification):
    1. Двухкамерная сорбция ОДУ (Two-Compartment Sorption ODE без дробных производных Капуто):
       быстрая водная фаза P_fast + медленная обратимая сорбция в микропорах P_slow
       (dP_slow/dt = k_in * P_fast - k_out * P_slow) + растянутая экспонента Кольрауша-Вейбулла exp(-(k*t)^beta).
    2. Выход за пределы Атласа (Out-of-Atlas alpha): почвы с экстремальной памятью alpha in [0.35, 0.45]
       и почти без памяти alpha in [0.91, 0.96], которых не было в 5 эталонных архетипах Атласа.
    """
    rng = np.random.default_rng(seed)
    T = len(t_grid)
    dt = float(t_grid[1] - t_grid[0])

    # Набор А: Two-Compartment Sorption ODE + Kohlrausch-Weibull
    P_tc = np.zeros((n_samples, T), dtype=np.float64)
    M_tc = np.zeros((n_samples, T), dtype=np.float64)
    E_tc = np.zeros((n_samples, T), dtype=np.float64)
    IBSP_tc = np.zeros((n_samples, T), dtype=np.float64)
    alpha_eff_tc = rng.uniform(0.62, 0.80, size=n_samples)

    for b in range(n_samples):
        P0 = rng.uniform(1.2, 3.8)
        k_deg = rng.uniform(0.025, 0.055)
        k_in = rng.uniform(0.030, 0.060)
        k_out = rng.uniform(0.008, 0.020)
        beta_kw = rng.uniform(0.58, 0.78)
        K_m = rng.uniform(0.70, 1.30)
        E_ref = K_m * 1.20

        p_fast = P0 * 0.75
        p_slow = P0 * 0.25
        m_val = K_m / (1.0 + 0.25 * P0)
        e_val = E_ref / ((1.0 + 0.25 * P0) * (1.0 + P0 * P0))

        for n_t, tv in enumerate(t_grid):
            kw_tail = 0.35 * P0 * np.exp(-((0.022 * max(tv, 0.0)) ** beta_kw))
            p_tot = 0.65 * (p_fast + p_slow) + kw_tail
            P_tc[b, n_t] = p_tot
            M_tc[b, n_t] = m_val
            E_tc[b, n_t] = e_val
            # Шаг Эйлера для двухкамерной системы ОДУ
            hald = p_fast / (0.90 + p_fast + p_fast * p_fast)
            dp_fast = -k_deg * hald * m_val - k_in * p_fast + k_out * p_slow
            dp_slow = k_in * p_fast - k_out * p_slow
            dm = 0.065 * m_val * (1.0 - m_val / K_m) - 0.012 * p_tot * m_val + 0.040 * hald * m_val
            de = 0.090 * (m_val / (1.0 + p_tot * p_tot)) - 0.075 * e_val
            p_fast = max(0.0, p_fast + dt * dp_fast)
            p_slow = max(0.0, p_slow + dt * dp_slow)
            m_val = max(0.05, m_val + dt * dm)
            e_val = max(0.05, e_val + dt * de)

        IBSP_tc[b] = compute_ibsp_alpha(P_tc[b], M_tc[b], E_tc[b], alpha_eff_tc[b], K_m, E_ref)

    # Набор Б: Out-of-Atlas alpha in [0.35, 0.45] U [0.91, 0.96]
    alpha_ood = np.concatenate([
        rng.uniform(0.35, 0.45, size=n_samples // 2),
        rng.uniform(0.91, 0.96, size=n_samples - n_samples // 2),
    ])
    P_ood = np.zeros((n_samples, T), dtype=np.float64)
    M_ood = np.zeros((n_samples, T), dtype=np.float64)
    E_ood = np.zeros((n_samples, T), dtype=np.float64)
    IBSP_ood = np.zeros((n_samples, T), dtype=np.float64)

    for b in range(n_samples):
        arch = dict(SOIL_ARCHETYPES[b % 5])
        P0 = rng.uniform(1.1, 3.6)
        K_m = float(arch["K_m"])
        E_ref = K_m * (float(arch["beta_E"]) / float(arch["delta_E"]))
        M0 = K_m / (1.0 + 0.25 * P0)
        E0 = E_ref / ((1.0 + 0.25 * P0) * (1.0 + P0 * P0))
        sol = solve_caputo_abm_single(alpha=float(alpha_ood[b]), params=arch, y0=(P0, M0, E0), t_max=float(t_grid[-1]), dt=dt)
        P_ood[b] = sol["P"]
        M_ood[b] = sol["M"]
        E_ood[b] = sol["E"]
        IBSP_ood[b] = sol["IBSP"]

    return {
        "two_compartment": {"P": P_tc, "M": M_tc, "E": E_tc, "IBSP": IBSP_tc, "alpha": alpha_eff_tc},
        "ood_alpha": {"P": P_ood, "M": M_ood, "E": E_ood, "IBSP": IBSP_ood, "alpha": alpha_ood},
    }
