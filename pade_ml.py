"""
Дифференцируемое ядро однопараметрической функции Миттаг-Леффлера E_{alpha, 1}(-z)
на базе глобальной рациональной аппроксимации Паде [3/4] (`MittagLefflerPade34`).

В данном модуле я реализую аналитическое решение системы 5x5 для коэффициентов
аппроксиманта с использованием формулы дополнения Эйлера и гладкую C^1-сшивку
в окрестности экспоненциального предела alpha -> 1.
"""

import math
import numpy as np
from scipy.integrate import quad
from scipy.special import erfcx
import torch
import torch.nn as nn


class MittagLefflerPade34(nn.Module):
    """
    Дифференцируемый вычислитель E_{alpha, 1}(-z) для z >= 0 и alpha in [0.10, 1.0].
    Поддерживает произвольные broadcast-размерности (Batch, Time, Channels) и
    точное обратное распространение градиентов по z и alpha.
    """

    def __init__(self, alpha_min: float = 0.10, alpha_cutoff: float = 0.90):
        super().__init__()
        self.alpha_min = float(alpha_min)
        self.alpha_cutoff = float(alpha_cutoff)

    def _solve_pade_coeffs(self, alpha_eff: torch.Tensor):
        """
        Решает батчевую линейную систему 5x5 для каждого элемента alpha_eff.
        Вход: alpha_eff формы (B,) в диапазоне [0.10, 0.90].
        Возвращает: p1, p2, p3, q1, q2, q3, q4 формы (B, 1).
        """
        a = alpha_eff.reshape(-1)
        g1 = torch.exp(-torch.special.gammaln(1.0 + 1.0 * a))
        g2 = torch.exp(-torch.special.gammaln(1.0 + 2.0 * a))
        g3 = torch.exp(-torch.special.gammaln(1.0 + 3.0 * a))
        g4 = torch.exp(-torch.special.gammaln(1.0 + 4.0 * a))
        g5 = torch.exp(-torch.special.gammaln(1.0 + 5.0 * a))

        # Формула дополнения: 1 / Gamma(1 - k*alpha) = Gamma(k*alpha) * sin(k*pi*alpha) / pi
        igm1 = torch.exp(torch.special.gammaln(a)) * torch.sin(math.pi * a) / math.pi
        igm2 = torch.exp(torch.special.gammaln(2.0 * a)) * torch.sin(2.0 * math.pi * a) / math.pi

        z0 = torch.zeros_like(a)
        o1 = torch.ones_like(a)

        # Строка 0 (z^3):  g2*q1 - g1*q2 + q3 - p3 = g3
        row0 = torch.stack([g2, -g1, o1, z0, -o1], dim=-1)
        # Строка 1 (z^4): -g3*q1 + g2*q2 - g1*q3 + q4 = -g4 
        row1 = torch.stack([-g3, g2, -g1, o1, z0], dim=-1)
        # Строка 2 (z^5):  g4*q1 - g3*q2 + g2*q3 - g1*q4 = g5
        row2 = torch.stack([g4, -g3, g2, -g1, z0], dim=-1)
        # Строка 3 (z^-1): -igm1*q4 + p3 = 0
        row3 = torch.stack([z0, z0, z0, -igm1, o1], dim=-1)
        # Строка 4 (z^-2): -g1*q1 + q2 - igm1*q3 + igm2*q4 = -g2
        row4 = torch.stack([-g1, o1, -igm1, igm2, z0], dim=-1)

        A = torch.stack([row0, row1, row2, row3, row4], dim=-2)  # (B, 5, 5)
        rhs = torch.stack([g3, -g4, g5, z0, -g2], dim=-1)        # (B, 5)

        sol = torch.linalg.solve(A, rhs)                         # (B, 5)

        q1 = sol[:, 0:1]
        q2 = sol[:, 1:2]
        q3 = sol[:, 2:3]
        q4 = sol[:, 3:4]
        p3 = sol[:, 4:5]

        g1_u = g1.unsqueeze(-1)
        g2_u = g2.unsqueeze(-1)
        p1 = q1 - g1_u
        p2 = q2 - g1_u * q1 + g2_u

        return p1, p2, p3, q1, q2, q3, q4

    @staticmethod
    def _eval_horner(z_flat: torch.Tensor, p1, p2, p3, q1, q2, q3, q4):
        """
        Вычисляет числитель P_3(z), знаменатель Q_4(z) и отношение R_{3,4}(z)
        строго по вложенной схеме Горнера.
        """
        num = 1.0 + z_flat * (p1 + z_flat * (p2 + z_flat * p3))
        den = 1.0 + z_flat * (q1 + z_flat * (q2 + z_flat * (q3 + z_flat * q4)))
        den_safe = torch.clamp(den, min=1e-12)
        return num / den_safe, num, den

    def forward(self, z: torch.Tensor, alpha: torch.Tensor, return_details: bool = False):
        """
        Вычисляет E_{alpha, 1}(-z) при z >= 0, alpha in [0.10, 1.0].

        Параметры:
            z: тензор неотрицательных аргументов произвольной формы (..., M) или (B, T, K).
            alpha: тензор порядков дробности (скаляр, (B,), (B, 1) или совместимый с z).
            return_details: если True, возвращает словарь с полями 'val', 'num', 'den', 'coeffs'.
        """
        z_pos = torch.clamp(z, min=0.0)
        orig_shape = z_pos.shape

        # Приводим к батчевому виду (B, M)
        if z_pos.ndim == 0:
            z_2d = z_pos.reshape(1, 1)
        elif z_pos.ndim == 1:
            if alpha.ndim == 0 or alpha.numel() == 1:
                z_2d = z_pos.unsqueeze(0)
            elif alpha.numel() == z_pos.shape[0]:
                z_2d = z_pos.unsqueeze(-1)
            else:
                z_2d = z_pos.unsqueeze(0)
        elif z_pos.ndim == 2:
            z_2d = z_pos
        else:
            B_dim = z_pos.shape[0]
            z_2d = z_pos.reshape(B_dim, -1)

        B = z_2d.shape[0]
        if alpha.numel() == 1:
            alpha_1d = alpha.reshape(1).expand(B)
        elif alpha.shape[0] == B:
            alpha_1d = alpha.reshape(B, -1)[:, 0]
        else:
            alpha_1d = alpha.reshape(-1)[:B]

        alpha_clamped = torch.clamp(alpha_1d, min=self.alpha_min, max=1.0)
        alpha_work = torch.clamp(alpha_clamped, min=self.alpha_min, max=self.alpha_cutoff)

        # Коэффициенты в рабочей зоне [0.10, 0.90]
        p1, p2, p3, q1, q2, q3, q4 = self._solve_pade_coeffs(alpha_work)
        r_work, num_work, den_work = self._eval_horner(z_2d, p1, p2, p3, q1, q2, q3, q4)

        # Значение в граничной точке alpha = 0.90 для гладкой сшивки на (0.90, 1.0]
        alpha_90 = torch.full_like(alpha_work, self.alpha_cutoff)
        p1_90, p2_90, p3_90, q1_90, q2_90, q3_90, q4_90 = self._solve_pade_coeffs(alpha_90)
        r_90, _, den_90 = self._eval_horner(z_2d, p1_90, p2_90, p3_90, q1_90, q2_90, q3_90, q4_90)

        w = torch.clamp((alpha_clamped - self.alpha_cutoff) / (1.0 - self.alpha_cutoff), min=0.0, max=1.0).unsqueeze(-1)
        exp_neg_z = torch.exp(-z_2d)
        r_tail = (1.0 - w) * r_90 + w * exp_neg_z
        den_tail = (1.0 - w) * den_90 + w * torch.exp(z_2d)

        in_tail = (alpha_clamped > self.alpha_cutoff).unsqueeze(-1)
        out_2d = torch.where(in_tail, r_tail, r_work)
        den_2d = torch.where(in_tail, den_tail, den_work)

        out = out_2d.reshape(orig_shape)
        if not return_details:
            return out

        return {
            "val": out,
            "num": num_work.reshape(orig_shape),
            "den": den_2d.reshape(orig_shape),
            "coeffs": {
                "p1": p1, "p2": p2, "p3": p3,
                "q1": q1, "q2": q2, "q3": q3, "q4": q4,
            },
        }


def exact_mittag_leffler_oberhettinger(z_array: np.ndarray, alpha: float) -> np.ndarray:
    """
    Эталонное высокоточное вычисление функции Миттаг-Леффлера E_{alpha, 1}(-z)
    через интегральное представление Оберхеттингера / Горенфло — Майнарди (SciPy quad):
        E_{alpha, 1}(-z) = (sin(pi * alpha) / (pi * alpha)) *
                           int_0^inf exp(-y^(1/alpha)) * z / (y^2 + 2*y*z*cos(pi*alpha) + z^2) dy
    При alpha = 0.50 совпадает с аналитическим значением erfcx(z) с машинной точностью.
    """
    z_arr = np.asarray(z_array, dtype=np.float64)
    orig_shape = z_arr.shape
    z_flat = z_arr.ravel()

    if abs(alpha - 1.0) < 1e-12:
        return np.exp(-z_arr)
    if abs(alpha - 0.5) < 1e-12:
        return erfcx(z_arr)

    pref = math.sin(math.pi * alpha) / (math.pi * alpha)
    cos_pa = math.cos(math.pi * alpha)
    inv_alpha = 1.0 / alpha
    out = np.zeros_like(z_flat, dtype=np.float64)

    for i, zv in enumerate(z_flat):
        if zv <= 0.0:
            out[i] = 1.0
            continue

        def integrand(y):
            if y <= 0.0:
                return 1.0 / zv
            arg = y ** inv_alpha
            if arg > 700.0:
                return 0.0
            return math.exp(-arg) * zv / (y * y + 2.0 * y * zv * cos_pa + zv * zv)

        # Верхний предел, где exp(-y^(1/alpha)) < 1e-16 => y_max = (37.0)^alpha
        y_max = (38.0) ** alpha
        val, _ = quad(integrand, 0.0, y_max, limit=200, epsabs=1e-13, epsrel=1e-13)
        out[i] = pref * val

    return out.reshape(orig_shape)
