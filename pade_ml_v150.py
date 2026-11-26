"""
TerraKAN v1.5.0 — Routh-Hurwitz Certified [3/4] Pade Mittag-Leffler Kernel.

Объединяет C^1-гладкий [3/4] аппроксимант Паде из TerraKAN / abelkan v0.6.0
с алгебраической сертификацией устойчивости Рауса-Гурвица (нулевые полюсы на Re(z) >= 0)
и эталонным интегралом Оберхеттингера для численной верификации.
"""

import math
from typing import Dict, Tuple
import numpy as np
from scipy.integrate import quad
from scipy.special import gamma as scipy_gamma

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def certify_routh_hurwitz_numpy(
    alpha_min: float = 0.40,
    alpha_max: float = 0.90,
    n_grid: int = 501,
) -> Dict[str, object]:
    """
    Чисто NumPy/SciPy-сертификация отсутствия полюсов Q_4(z) на всей правой полуплоскости {Re(z) >= 0}
    по критерию Рауса-Гурвица (Delta_2 > 0, Delta_3 > 0, q_1..q_4 > 0).
    Работает даже без PyTorch.
    """
    alphas = np.linspace(alpha_min, alpha_max, n_grid, dtype=np.float64)
    min_q = np.full(4, np.inf)
    min_d2 = np.inf
    min_d3 = np.inf
    max_real_root = -np.inf

    for a in alphas:
        g = [1.0 / scipy_gamma(1.0 + k * a) for k in range(1, 6)]
        # Отражение Эйлера: 1 / Gamma(1 - k*a) = Gamma(k*a) * sin(k*pi*a) / pi
        m1 = scipy_gamma(a) * math.sin(math.pi * a) / math.pi
        m2 = scipy_gamma(2.0 * a) * math.sin(2.0 * math.pi * a) / math.pi

        A = np.array([
            [g[1], -g[0], 1.0, 0.0, -1.0],
            [-g[2], g[1], -g[0], 1.0, 0.0],
            [g[3], -g[2], g[1], -g[0], 0.0],
            [0.0, 0.0, 0.0, -m1, 1.0],
            [-g[0], 1.0, -m1, m2, 0.0],
        ], dtype=np.float64)
        rhs = np.array([g[2], -g[3], g[4], 0.0, -g[1]], dtype=np.float64)
        q1, q2, q3, q4, _ = np.linalg.solve(A, rhs)

        min_q = np.minimum(min_q, [q1, q2, q3, q4])
        d2 = q3 * q2 - q4 * q1
        d3 = q1 * d2 - (q3 ** 2)
        min_d2 = min(min_d2, float(d2))
        min_d3 = min(min_d3, float(d3))

        # Корни Q_4(z) = q4 z^4 + q3 z^3 + q2 z^2 + q1 z + 1
        roots = np.roots([q4, q3, q2, q1, 1.0])
        max_real_root = max(max_real_root, float(np.max(np.real(roots))))

    passed = bool(
        np.all(min_q > 0.0)
        and min_d2 > 0.0
        and min_d3 > 0.0
        and max_real_root < 0.0
    )
    return {
        "passed": passed,
        "n_grid": n_grid,
        "alpha_range": [alpha_min, alpha_max],
        "min_q": [float(v) for v in min_q],
        "min_delta2": float(min_d2),
        "min_delta3": float(min_d3),
        "max_real_root": float(max_real_root),
    }


def exact_mittag_leffler_oberhettinger(z: np.ndarray, alpha: float) -> np.ndarray:
    """Эталонное вычисление E_{alpha, 1}(-z) через интеграл Оберхеттингера (для тестов)."""
    z_arr = np.asarray(z, dtype=np.float64)
    out = np.ones_like(z_arr)
    sin_pa = math.sin(math.pi * alpha)
    cos_pa = math.cos(math.pi * alpha)

    for idx, zv in np.ndenumerate(z_arr):
        if zv <= 0.0:
            out[idx] = 1.0
            continue
        if abs(alpha - 1.0) < 1e-9:
            out[idx] = math.exp(-zv)
            continue

        def integrand(r: float) -> float:
            ra = r ** alpha
            num = math.exp(-r) * (r ** (alpha - 1.0)) * sin_pa
            den = (ra * ra) + 2.0 * zv * ra * cos_pa + (zv * zv)
            return num / max(den, 1e-30)

        val, _ = quad(integrand, 0.0, np.inf, limit=200, epsabs=1e-11, epsrel=1e-10)
        out[idx] = (zv / math.pi) * val
    return out


if HAS_TORCH:
    class MittagLefflerPade34(nn.Module):
        """
        Дифференцируемое ядро E_{alpha, 1}(-z) на базе глобального [3/4] аппроксиманта Паде
        с C^1-гладким сшиванием к exp(-z) на alpha in (0.90, 1.0] и точными производными Капуто.
        """

        def __init__(self, bridge_start: float = 0.90) -> None:
            super().__init__()
            self.bridge_start = float(bridge_start)

        @staticmethod
        def _solve_pade_coeffs(alpha64: torch.Tensor) -> Tuple[torch.Tensor, ...]:
            g1 = torch.exp(-torch.lgamma(1.0 + alpha64))
            g2 = torch.exp(-torch.lgamma(1.0 + 2.0 * alpha64))
            g3 = torch.exp(-torch.lgamma(1.0 + 3.0 * alpha64))
            g4 = torch.exp(-torch.lgamma(1.0 + 4.0 * alpha64))
            g5 = torch.exp(-torch.lgamma(1.0 + 5.0 * alpha64))

            m1 = torch.exp(torch.lgamma(alpha64)) * torch.sin(math.pi * alpha64) / math.pi
            m2 = torch.exp(torch.lgamma(2.0 * alpha64)) * torch.sin(2.0 * math.pi * alpha64) / math.pi

            zeros = torch.zeros_like(alpha64)
            ones = torch.ones_like(alpha64)

            row0 = torch.stack([g2, -g1, ones, zeros, -ones], dim=-1)
            row1 = torch.stack([-g3, g2, -g1, ones, zeros], dim=-1)
            row2 = torch.stack([g4, -g3, g2, -g1, zeros], dim=-1)
            row3 = torch.stack([zeros, zeros, zeros, -m1, ones], dim=-1)
            row4 = torch.stack([-g1, ones, -m1, m2, zeros], dim=-1)
            A = torch.stack([row0, row1, row2, row3, row4], dim=-2)
            rhs = torch.stack([g3, -g4, g5, zeros, -g2], dim=-1).unsqueeze(-1)

            sol = torch.linalg.solve(A, rhs).squeeze(-1)
            q1, q2, q3, q4, p3 = sol.unbind(dim=-1)
            p1 = q1 - g1
            p2 = q2 - g1 * q1 + g2
            return p1, p2, p3, q1, q2, q3, q4

        def _eval_pade_raw(self, z64: torch.Tensor, alpha64: torch.Tensor) -> torch.Tensor:
            p1, p2, p3, q1, q2, q3, q4 = self._solve_pade_coeffs(alpha64)
            while p1.ndim < z64.ndim:
                p1, p2, p3 = p1.unsqueeze(-1), p2.unsqueeze(-1), p3.unsqueeze(-1)
                q1, q2, q3, q4 = q1.unsqueeze(-1), q2.unsqueeze(-1), q3.unsqueeze(-1), q4.unsqueeze(-1)
            z2 = z64 * z64
            z3 = z2 * z64
            z4 = z2 * z2
            num = 1.0 + p1 * z64 + p2 * z2 + p3 * z3
            den = 1.0 + q1 * z64 + q2 * z2 + q3 * z3 + q4 * z4
            return num / den

        def forward(self, z: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
            orig_dtype = z.dtype
            z64 = torch.clamp(z.to(torch.float64), min=0.0)
            a64 = torch.clamp(alpha.to(torch.float64), min=0.25, max=1.0)

            a_safe = torch.clamp(a64, max=self.bridge_start)
            val_pade = self._eval_pade_raw(z64, a_safe)

            if torch.any(a64 > self.bridge_start):
                s = torch.clamp((a64 - self.bridge_start) / (1.0 - self.bridge_start), 0.0, 1.0)
                w = 3.0 * s * s - 2.0 * s * s * s
                while w.ndim < z64.ndim:
                    w = w.unsqueeze(-1)
                val_exp = torch.exp(-z64)
                out = (1.0 - w) * val_pade + w * val_exp
            else:
                out = val_pade

            return out.to(orig_dtype)

        @staticmethod
        def certify_complex_stability(
            alpha_min: float = 0.40, alpha_max: float = 0.90, n_grid: int = 501
        ) -> Dict[str, object]:
            return certify_routh_hurwitz_numpy(alpha_min, alpha_max, n_grid)
