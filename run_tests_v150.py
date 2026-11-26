"""
TerraKAN v1.5.1 — Автоматические unit-тесты математических ядер и архитектуры.
"""

import json
import os
import numpy as np

from pade_ml_v150 import certify_routh_hurwitz_numpy
from physics_solver_v150 import (
    SOIL_ARCHETYPES,
    solve_caputo_abm_batch,
    generate_terrakan_v150_dataset,
    generate_extended_ood_benchmarks,
)
from wetlab_benchmarks_v150 import get_wetlab_suite_v150, pack_wetlab_scenario


def run_numpy_math_tests() -> dict:
    report = {}
    rh = certify_routh_hurwitz_numpy(0.40, 0.90, 501)
    assert rh["passed"], f"Routh-Hurwitz FAILED: {rh}"
    report["test_1_routh_hurwitz"] = rh
    print(f"[PASS] Test 1 (Routh-Hurwitz 501 pts): min_d2={rh['min_delta2']:.5f}, min_d3={rh['min_delta3']:.5f}, max_Re={rh['max_real_root']:.4f}")

    arch = SOIL_ARCHETYPES[0]
    params_b = {k: np.array([float(arch[k]), float(arch[k])]) for k in arch if isinstance(arch[k], (int, float))}
    params_b["k_deg"] = np.array([0.0, 0.35])
    params_b["k_p"] = np.array([0.0, 0.04])
    W_t = np.full((2, 201), 0.60)
    y0 = np.array([[2.0, 0.65, 0.25], [1.5, 0.85, 1.15]])
    _, Y, _, _ = solve_caputo_abm_batch(
        alpha=np.array([0.70, 0.82]),
        params_batch=params_b,
        y0_batch=y0,
        W_t_batch=W_t,
        t_max=100.0,
        dt=0.5,
        P_inf_batch=np.array([2.0, 0.0]),
        hormesis_amp=np.array([0.0, 0.85]),
    )
    recalc_drift = float(np.max(np.abs(Y[0, :, 0] - 2.0)))
    fast_deg_pct = float((1.0 - Y[1, 60, 0] / Y[1, 0, 0]) * 100.0)
    horm_peak_E = float(np.max(Y[1, :, 2]))
    assert recalc_drift < 1e-12
    assert fast_deg_pct > 75.0
    report["test_2_abm_v151"] = {
        "passed": True,
        "recalcitrant_P_inf_max_drift": recalc_drift,
        "fast_consortium_30d_deg_pct": round(fast_deg_pct, 2),
        "hormesis_peak_E": round(horm_peak_E, 3),
    }
    print(f"[PASS] Test 2 (Caputo ABM v1.5.1): P_inf drift={recalc_drift:.1e}, fast 30d deg={fast_deg_pct:.1f}%, hormesis peak E={horm_peak_E:.3f}")

    ds = generate_terrakan_v150_dataset(n_base=100, n_augmented=80, t_max=180.0, dt=1.0, seed=42)
    ood = generate_extended_ood_benchmarks(ds["t_grid"], n_samples=20, seed=123)
    report["test_3_simulators"] = {
        "passed": True,
        "regimes_present": sorted(int(x) for x in np.unique(ds["regime_id"])),
        "ood_suites": list(ood.keys()),
    }
    print(f"[PASS] Test 3 (Simulators v1.5.1): regimes={report['test_3_simulators']['regimes_present']}, OOD={list(ood.keys())}")

    suite = get_wetlab_suite_v150()
    for sc in suite:
        packed = pack_wetlab_scenario(sc)
        assert packed["values"].shape == (1, 8, 5)
    report["test_4_wetlab_loader"] = {"passed": True, "n_scenarios": len(suite)}
    print(f"[PASS] Test 4 (Wet-Lab Suite): {len(suite)} scenarios verified.")
    return report


def run_torch_arch_tests() -> dict:
    try:
        import torch
        from model_v150 import TerraKANv150, compute_terrakan_v150_loss
    except ImportError:
        print("[SKIP] PyTorch not installed in this environment — skipping Tests 5-7.")
        return {"torch_available": False}

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(42)
    m = TerraKANv150(d_model=64, n_heads=4, n_layers=2, n_loops=2)
    n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)

    B, N = 4, 8
    vals = torch.rand(B, N, 5) * 0.8 + 0.3
    mask = torch.ones(B, N, 5)
    t_obs = torch.linspace(0.0, 140.0, N).unsqueeze(0).expand(B, N)
    rain = torch.zeros(B, N, 1)
    t_eval = torch.linspace(0.0, 365.0, 50)

    vals_pert = vals.clone()
    vals_pert[:, 5:, :] += 5.0
    m.eval()
    with torch.no_grad():
        h1 = m.encoder(vals, mask, t_obs, rain)["h_seq"]
        h2 = m.encoder(vals_pert, mask, t_obs, rain)["h_seq"]
        out = m(vals, mask, t_obs, rain, t_eval)
    caus_diff = float((h1[:, :5, :] - h2[:, :5, :]).abs().max().item())
    assert caus_diff < 1e-12
    assert bool((out["P_quantiles"][..., 1:] > out["P_quantiles"][..., :-1]).all())
    assert bool((out["IBSP_quantiles"][..., 1:] > out["IBSP_quantiles"][..., :-1]).all())
    assert bool((out["M"] >= 0.05).all()) and bool((out["E"] >= 0.05).all())
    assert float(out["gamma_h"].item()) <= 2.40

    m.train()
    out_tr = m(vals, mask, t_obs, rain, t_eval, pollutant_class_hint=torch.tensor([-1.0, 0.0, 1.0, -1.0]))
    losses = compute_terrakan_v150_loss(
        out_tr,
        P_true=torch.rand(B, 50),
        P_inf_true=torch.rand(B) * 0.2,
        M_true=torch.rand(B, 50) + 0.3,
        E_true=torch.rand(B, 50) + 0.3,
        IBSP_true=torch.rand(B, 50) * 100.0,
        alpha_true=torch.full((B,), 0.72),
        phi_true=torch.full((B,), 0.65),
        dirichlet_targets=torch.full((B, 5), 0.2),
    )
    losses["total"].backward()
    assert m.trunk.raw_gamma_h.grad is not None and torch.isfinite(m.trunk.raw_gamma_h.grad)

    print(f"[PASS] Tests 5-7 (PyTorch v1.5.1): params={n_params:,}, causality_diff={caus_diff:.1e}, M/E>=0.05=True, H3_monotonic=True")
    return {"torch_available": True, "n_params": n_params, "causality_max_diff": caus_diff, "h3_strict_monotonic": True}


if __name__ == "__main__":
    res = run_numpy_math_tests()
    res["torch_tests"] = run_torch_arch_tests()
    out_path = os.path.join(os.path.dirname(__file__), "unit_test_report_v151.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
