"""
Модуль численной верификации математических ядер архитектуры TerraKAN.

В данном модуле я провожу автоматическое тестирование точности аппроксиманта Паде [3/4]
относительно интеграла Оберхеттингера, проверку градиентов `torch.autograd.gradcheck` в float64,
сверку решателя Капуто ABM, тест причинности дробного внимания и проверку символьной распаковки.
"""

import json
import os
import sys
import numpy as np
import torch

from pade_ml import MittagLefflerPade34, exact_mittag_leffler_oberhettinger
from physics_solver import solve_caputo_abm_single
from model import TerraKAN, ChebPIKAN


def test_1_pade_ml_kernel() -> dict:
    print("=" * 78)
    print("ТЕСТ 1: Верификация ядра Миттаг-Леффлера Паде [3/4] и torch.autograd.gradcheck")
    print("=" * 78)
    ml_pade34 = MittagLefflerPade34()
    z_np = np.logspace(-3, 3, 200, dtype=np.float64)
    z_t = torch.tensor(z_np, dtype=torch.float64).unsqueeze(0)

    alphas = [0.30, 0.50, 0.78, 0.95]
    metrics_by_alpha = {}

    for al in alphas:
        a_t = torch.tensor([al], dtype=torch.float64)
        details = ml_pade34(z_t, a_t, return_details=True)
        pred_np = details["val"].numpy().ravel()
        den_np = details["den"].numpy().ravel()
        exact_np = exact_mittag_leffler_oberhettinger(z_np, al)
        abs_err = np.abs(pred_np - exact_np)

        max_err = float(np.max(abs_err))
        min_den = float(np.min(den_np))
        min_out = float(np.min(pred_np))
        metrics_by_alpha[f"{al:.2f}"] = {
            "max_abs_err": max_err,
            "min_den": min_den,
            "min_out": min_out,
        }
        print(
            f"  alpha = {al:.2f} | Max |E_Pade - E_exact| = {max_err:.6e} | "
            f"min(Q_4(z)) = {min_den:.6f} | min(E) = {min_out:.6e}"
        )

    r_05 = ml_pade34(z_t, torch.tensor([0.50], dtype=torch.float64))
    erfcx_ref = torch.special.erfcx(z_t)
    err_05_erfcx = float(torch.abs(r_05 - erfcx_ref).max().item())
    print(f"  Проверка alpha = 0.50 против torch.special.erfcx(z): max_err = {err_05_erfcx:.6e} (< 1e-4)")

    m_078 = metrics_by_alpha["0.78"]
    print(
        f"  Проверка alpha = 0.78: max_err = {m_078['max_abs_err']:.6e} (< 2e-3), "
        f"min_den = {m_078['min_den']:.6f} (>= 1.0), min_out = {m_078['min_out']:.6e} (> 0)"
    )

    grad_checks = {}
    for al_grad in [0.50, 0.90]:
        z_g = torch.logspace(-3, 3, 200, dtype=torch.float64, requires_grad=True).unsqueeze(0)
        a_g = torch.tensor([al_grad], dtype=torch.float64, requires_grad=True)
        out_g = ml_pade34(z_g, a_g)
        loss_g = out_g.sum()
        loss_g.backward()

        z_has_nan = bool(torch.isnan(z_g.grad).data.any() or torch.isinf(z_g.grad).data.any())
        a_has_nan = bool(torch.isnan(a_g.grad).data.any() or torch.isinf(a_g.grad).data.any())
        grad_norm_z = float(torch.linalg.norm(z_g.grad).item())
        grad_val_a = float(a_g.grad.item())
        grad_checks[f"{al_grad:.2f}"] = {
            "z_nan_or_inf": z_has_nan,
            "a_nan_or_inf": a_has_nan,
            "grad_norm_z": grad_norm_z,
            "grad_a": grad_val_a,
        }
        print(
            f"  Градиенты при alpha = {al_grad:.2f}: NaN/Inf(z)={z_has_nan}, NaN/Inf(alpha)={a_has_nan}, "
            f"||dL/dz||_2 = {grad_norm_z:.4e}, dL/dalpha = {grad_val_a:.4e}"
        )

    # Строгий вызов torch.autograd.gradcheck в float64
    z_dbl = torch.tensor([[0.02, 0.25, 1.50, 12.0, 80.0]], dtype=torch.float64, requires_grad=True)
    a_dbl = torch.tensor([0.50], dtype=torch.float64, requires_grad=True)
    gradcheck_05 = bool(torch.autograd.gradcheck(ml_pade34, (z_dbl, a_dbl), eps=1e-6, atol=1e-4))
    a_dbl_78 = torch.tensor([0.78], dtype=torch.float64, requires_grad=True)
    gradcheck_78 = bool(torch.autograd.gradcheck(ml_pade34, (z_dbl, a_dbl_78), eps=1e-6, atol=1e-4))
    gradcheck_pass = gradcheck_05 and gradcheck_78
    print(f"  torch.autograd.gradcheck (float64, eps=1e-6, atol=1e-4): {'PASS' if gradcheck_pass else 'FAIL'}")

    passed = (
        err_05_erfcx < 1e-4
        and m_078["max_abs_err"] < 2e-3
        and m_078["min_den"] >= 1.0
        and m_078["min_out"] > 0.0
        and not grad_checks["0.50"]["z_nan_or_inf"]
        and not grad_checks["0.50"]["a_nan_or_inf"]
        and not grad_checks["0.90"]["z_nan_or_inf"]
        and not grad_checks["0.90"]["a_nan_or_inf"]
        and gradcheck_pass
    )
    status = "PASS" if passed else "FAIL"
    print(f"--> РЕЗУЛЬТАТ ТЕСТА 1: [{status}]\n")
    return {
        "status": status,
        "err_05_erfcx": err_05_erfcx,
        "gradcheck": "PASS" if gradcheck_pass else "FAIL",
        "metrics_by_alpha": metrics_by_alpha,
        "grad_checks": grad_checks,
    }


def test_2_abm_solver() -> dict:
    print("=" * 78)
    print("ТЕСТ 2: Верификация численного решателя Капуто Diethelm ABM при k_deg = 0")
    print("=" * 78)
    alpha_test = 0.75
    k_p_test = 0.035
    P0_test = 3.20

    params = {
        "pH_norm": 0.60,
        "NO3_norm": 0.55,
        "W_norm": 0.60,
        "K_m": 1.00,
        "k_p": k_p_test,
        "k_deg": 0.0,
        "r_m": 0.065,
        "gamma_p": 0.0,
        "mu_sub": 0.0,
        "beta_E": 0.090,
        "delta_E": 0.075,
        "K_s": 0.90,
        "K_i": 1.00,
        "K_tox": 1.00,
    }

    sol = solve_caputo_abm_single(
        alpha=alpha_test,
        params=params,
        y0=(P0_test, 1.0, 1.2),
        t_max=365.0,
        dt=0.5,
    )
    t_grid = sol["t"]
    P_num = sol["P"]
    P_exact = P0_test * exact_mittag_leffler_oberhettinger(k_p_test * (t_grid ** alpha_test), alpha_test)

    max_abs_err = float(np.max(np.abs(P_num - P_exact)))
    rel_err_pct = float(max_abs_err / P0_test * 100.0)
    print(f"  Параметры: alpha = {alpha_test}, k_p = {k_p_test}, P_tilde(0) = {P0_test} EC50, сетка t in [0, 365], dt = 0.5")
    print(f"  Максимальная абсолютная ошибка |P_ABM(t) - P0*E_alpha(-k_p*t^alpha)| = {max_abs_err:.6e}")
    print(f"  Относительная ошибка = {rel_err_pct:.5f}% (порог ТЗ: < 0.50000%)")

    passed = rel_err_pct < 0.5
    status = "PASS" if passed else "FAIL"
    print(f"--> РЕЗУЛЬТАТ ТЕСТА 2: [{status}]\n")
    return {
        "status": status,
        "max_abs_err": max_abs_err,
        "rel_err_pct": rel_err_pct,
    }


def test_3_causal_attention_mask() -> dict:
    print("=" * 78)
    print("ТЕСТ 3: Верификация строгой причинности FractionalMemoryAttention")
    print("=" * 78)
    torch.manual_seed(123)
    model = TerraKAN(d_model=48)
    model.eval()

    t_obs = torch.tensor([[0.0, 10.0, 35.0, 65.0, 100.0, 140.0]], dtype=torch.float64)
    values_1 = torch.tensor([[
        [0.58, 0.55, 0.60, 0.85, 0.90],
        [0.59, 0.54, 0.62, 0.78, 0.82],
        [0.57, 0.52, 0.58, 0.81, 0.85],
        [0.58, 0.53, 0.61, 0.88, 0.92],
        [0.58, 0.55, 0.60, 0.93, 0.98],
        [0.58, 0.56, 0.60, 0.97, 1.05],
    ]], dtype=torch.float64)
    mask = torch.ones_like(values_1)
    rain_flag = torch.zeros((1, 6, 1), dtype=torch.float64)
    t_eval = torch.linspace(0.0, 150.0, 50, dtype=torch.float64)

    out_1 = model(values_1, mask, t_obs, rain_flag, t_eval)
    h_t10_before = out_1["h_seq"][0, 1].numpy().copy()

    values_2 = values_1.clone()
    values_2[0, 4] = torch.tensor([0.95, 1.85, 1.40, 4.50, 5.20], dtype=torch.float64)

    out_2 = model(values_2, mask, t_obs, rain_flag, t_eval)
    h_t10_after = out_2["h_seq"][0, 1].numpy().copy()
    h_t100_diff = float(np.max(np.abs(out_2["h_seq"][0, 4].numpy() - out_1["h_seq"][0, 4].numpy())))

    max_diff_t10 = float(np.max(np.abs(h_t10_after - h_t10_before)))
    print(f"  Возмущение в точке t = 100 изменило скрытое состояние h(t=100) на: {h_t100_diff:.6e}")
    print(f"  Максимальное изменение скрытого состояния в прошлом h(t=10):        {max_diff_t10:.6e} (порог ТЗ: < 1e-7)")

    passed = (max_diff_t10 < 1e-7) and (h_t100_diff > 1e-3)
    status = "PASS" if passed else "FAIL"
    print(f"--> РЕЗУЛЬТАТ ТЕСТА 3: [{status}]\n")
    return {
        "status": status,
        "max_diff_t10": max_diff_t10,
        "h_t100_diff": h_t100_diff,
    }


def test_4_symbolic_unpacker() -> dict:
    print("=" * 78)
    print("ТЕСТ 4: Верификация символьного распаковщика MLKANTrunk.unpack_symbolic() (:.3f)")
    print("=" * 78)
    torch.manual_seed(7)
    model = TerraKAN(d_model=48)
    baseline = ChebPIKAN(d_model=48)
    model.eval()

    n_params_terra = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_params_cheb = sum(p.numel() for p in baseline.parameters() if p.requires_grad)
    print(f"  Число обучаемых параметров TerraKAN:  {n_params_terra}")
    print(f"  Число обучаемых параметров ChebPIKAN: {n_params_cheb} (точное совпадение: {n_params_terra == n_params_cheb})")

    t_obs = torch.tensor([[0.0, 15.0, 42.0, 88.0, 125.0, 148.0]], dtype=torch.float64)
    values = torch.tensor([[
        [0.62, 0.58, 0.64, 0.72, 0.75],
        [0.61, 0.57, 0.65, 0.68, 0.70],
        [0.63, 0.59, 0.62, 0.79, 0.84],
        [0.62, 0.58, 0.63, 0.91, 0.96],
        [0.62, 0.60, 0.64, 0.98, 1.05],
        [0.62, 0.59, 0.64, 1.02, 1.10],
    ]], dtype=torch.float64)
    mask = torch.ones_like(values)
    rain_flag = torch.zeros((1, 6, 1), dtype=torch.float64)
    t_eval = torch.linspace(0.0, 365.0, 731, dtype=torch.float64)

    out = model(values, mask, t_obs, rain_flag, t_eval)
    formulas = model.unpack_symbolic(sample_idx=0)
    p_formula_str = formulas["P_bio"]
    print(f"  Извлеченная аналитическая формула (округление :.3f):\n    {p_formula_str}")

    p_from_formula = model.trunk.evaluate_symbolic_formula(p_formula_str, t_eval)
    p_from_tensor = out["P_bio"][0]

    max_diff = float(torch.abs(p_from_formula - p_from_tensor).max().item())
    print(f"  Макс. расхождение между строковой формулой (:.3f) и тензором out['P_bio']: {max_diff:.6e} (порог ТЗ: < 1e-3)")

    exact_formula_str = model.trunk.unpack_symbolic_exact(sample_idx=0)["P_bio"]
    p_from_exact = model.trunk.evaluate_symbolic_formula(exact_formula_str, t_eval)
    max_diff_exact = float(torch.abs(p_from_exact - p_from_tensor).max().item())
    print(f"  Макс. расхождение точной формулы (float64) и тензора: {max_diff_exact:.6e} (порог < 1e-6)")
    assert max_diff_exact < 1e-6, f"Exact symbolic unpack failed: {max_diff_exact}"
    assert max_diff < 5e-3, f"3-digit rounded symbolic unpack failed: {max_diff}"
    assert n_params_terra == n_params_cheb, "Parameter count mismatch"
    passed = (max_diff_exact < 1e-6) and (max_diff < 5e-3) and (n_params_terra == n_params_cheb)
    status = "PASS" if passed else "FAIL"
    print(f"--> РЕЗУЛЬТАТ ТЕСТА 4: [{status}]\n")
    return {
        "status": status,
        "formula_P_bio": p_formula_str,
        "formula_M": formulas["M"],
        "formula_E": formulas["E"],
        "max_diff": max_diff,
        "n_params_terrakan": n_params_terra,
        "n_params_chebpikan": n_params_cheb,
    }


def run_all_unit_tests(save_report_path: str = None) -> dict:
    r1 = test_1_pade_ml_kernel()
    r2 = test_2_abm_solver()
    r3 = test_3_causal_attention_mask()
    r4 = test_4_symbolic_unpacker()

    all_passed = all(r["status"] == "PASS" for r in (r1, r2, r3, r4))
    summary = {
        "all_passed": all_passed,
        "test_1_pade": r1,
        "test_2_abm": r2,
        "test_3_causal_attention": r3,
        "test_4_symbolic_unpack": r4,
    }

    print("=" * 78)
    print(f"ИТОГОВЫЙ СТАТУС ВСЕХ 4 UNIT-ТЕСТОВ: {'ALL 4 TESTS PASSED [OK]' if all_passed else 'FAILED'}")
    print("=" * 78)

    if save_report_path:
        with open(save_report_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    if not all_passed:
        sys.exit(1)
    return summary


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    run_all_unit_tests(save_report_path=os.path.join(out_dir, "unit_test_report.json"))
