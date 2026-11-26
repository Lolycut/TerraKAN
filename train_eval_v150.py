"""
TerraKAN v1.5.1 — Сквозное обучение (БЕЗ Ridge!) и полный 5-сидовый бенчмарк
(frozen Test 500, Slim N=1..7, 3 OOD-симулятора и 9 wet-lab сценариев с 5-seed ансамблем).

Запуск:
  - Быстрый smoke-тест (~10 сек):        python train_eval_v150.py --fast
  - Полный 5-сидовый прогон (~1 мин/сид): python train_eval_v150.py --seeds 42 101 777 1337 2026 --steps 2000
"""

import argparse
import json
import os
import time
from typing import Dict, List
import numpy as np
import torch

from model_v150 import TerraKANv150, compute_terrakan_v150_loss
from soe_adamw import SOEAdamW
from physics_solver_v150 import generate_terrakan_v150_dataset, generate_extended_ood_benchmarks
from wetlab_benchmarks_v150 import get_wetlab_suite_v150, pack_wetlab_scenario
from run_tests_v150 import run_numpy_math_tests, run_torch_arch_tests


def r2_rmse(y_true: np.ndarray, y_pred: np.ndarray):
    yt, yp = y_true.ravel(), y_pred.ravel()
    mse = float(np.mean((yt - yp) ** 2))
    return float(1.0 - mse / max(float(np.var(yt)), 1e-12)), float(np.sqrt(mse))


def calibrate_youden_threshold(min_dist: np.ndarray, is_clean: np.ndarray) -> Dict[str, float]:
    is_pos = ~is_clean
    grid = np.linspace(float(min_dist.min()) + 0.01, float(min_dist.max()) - 0.01, 200)
    best_j, best_th, best_tpr, best_fpr = -1.0, 0.42, 0.0, 0.0
    for th in grid:
        tpr = float(np.mean(min_dist[is_pos] >= th))
        fpr = float(np.mean(min_dist[is_clean] >= th))
        if tpr - fpr > best_j:
            best_j, best_th, best_tpr, best_fpr = tpr - fpr, float(th), tpr * 100.0, fpr * 100.0
    return {"d_star": best_th, "val_J": best_j, "val_tpr_pct": best_tpr, "val_fpr_pct": best_fpr}


def eval_ood_suites(model: TerraKANv150, t_grid: np.ndarray, device: str) -> Dict[str, Dict[str, float]]:
    """Zero-shot оценка модели на 3 академических OOD-симуляторах (25% шум, 7 точек в [0, 150] сут)."""
    ood = generate_extended_ood_benchmarks(t_grid, n_samples=60, seed=123)
    rng = np.random.default_rng(123)
    sub_idx = np.linspace(0, len(t_grid) - 1, 74, dtype=np.int64)
    t_eval_t = torch.tensor(t_grid[sub_idx], dtype=torch.float64, device=device)
    obs_idx = np.array([0, 20, 50, 90, 140, 210, 290], dtype=np.int64)
    obs_idx = np.clip(obs_idx, 0, len(t_grid) - 1)

    out_metrics = {}
    model.eval()
    with torch.no_grad():
        for name, d in ood.items():
            N = d["P"].shape[0]
            vals = np.zeros((N, 8, 5), dtype=np.float64)
            mask = np.zeros((N, 8, 5), dtype=np.float64)
            tobs = np.zeros((N, 8), dtype=np.float64)
            rain = np.zeros((N, 8, 1), dtype=np.float64)
            for b in range(N):
                for j, idx in enumerate(obs_idx):
                    vals[b, j] = np.array([0.62, 0.55, 0.60, d["M"][b, idx], d["E"][b, idx]]) * (
                        1.0 + rng.normal(0.0, 0.12, size=5)
                    )
                    mask[b, j, :] = 1.0
                    tobs[b, j] = t_grid[idx]
                tobs[b, 7] = tobs[b, 6]

            o = model(
                torch.tensor(vals, dtype=torch.float64, device=device),
                torch.tensor(mask, dtype=torch.float64, device=device),
                torch.tensor(tobs, dtype=torch.float64, device=device),
                torch.tensor(rain, dtype=torch.float64, device=device),
                t_eval_t,
                pollutant_class_hint=None,
            )
            r2_i, rmse_i = r2_rmse(d["IBSP"][:, sub_idx], o["IBSP"].cpu().numpy())
            r2_p, rmse_p = r2_rmse(d["P"][:, sub_idx], o["P_bio"].cpu().numpy())
            out_metrics[name] = {
                "r2_P": round(r2_p, 4),
                "rmse_P": round(rmse_p, 4),
                "r2_IBSP": round(r2_i, 4),
                "rmse_IBSP": round(rmse_i, 2),
            }
    return out_metrics


def train_single_seed(
    seed: int,
    ds: Dict[str, object],
    n_steps: int = 2000,
    batch_size: int = 64,
    device: str = "cpu",
) -> Dict[str, object]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    model = TerraKANv150(d_model=64, n_heads=4, n_layers=2, n_loops=2).to(device=device, dtype=torch.float64)
    opt = SOEAdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=3e-5)

    n_total = ds["P_true"].shape[0]
    train_idx = np.concatenate([np.arange(0, 3000), np.arange(4000, n_total)]) if n_total >= 4000 else np.arange(0, int(0.7 * n_total))
    val_idx = np.arange(3000, 3500) if n_total >= 4000 else np.arange(int(0.7 * n_total), int(0.85 * n_total))
    test_idx = np.arange(3500, 4000) if n_total >= 4000 else np.arange(int(0.85 * n_total), n_total)

    t_grid_full = ds["t_grid"]
    sub_idx = np.linspace(0, len(t_grid_full) - 1, 74, dtype=np.int64)
    t_eval_t = torch.tensor(t_grid_full[sub_idx], dtype=torch.float64, device=device)
    extrap_mask = t_grid_full[sub_idx] > 150.0

    t0 = time.time()
    for step in range(1, n_steps + 1):
        model.train()
        model.expert_bank.router_temp = max(0.6, 2.0 - 1.4 * (step - 1) / max(n_steps - 1, 1))
        b_ids = rng.choice(train_idx, size=min(batch_size, len(train_idx)), replace=False)

        vals = torch.tensor(ds["obs_values"][b_ids], dtype=torch.float64, device=device)
        mask = torch.tensor(ds["obs_mask"][b_ids], dtype=torch.float64, device=device)
        tobs = torch.tensor(ds["obs_times"][b_ids], dtype=torch.float64, device=device)
        rain = torch.tensor(ds["obs_rain_flag"][b_ids], dtype=torch.float64, device=device)

        for bi in range(len(b_ids)):
            if rng.random() < 0.15:
                keep_n = int(rng.integers(1, 4))
                mask[bi, keep_n:, :] = 0.0

        h_raw = ds["class_hint_true"][b_ids].copy()
        h_raw[rng.random(len(b_ids)) < 0.50] = 0.0
        hint_t = torch.tensor(h_raw, dtype=torch.float64, device=device)

        opt.zero_grad(set_to_none=True)
        out = model(vals, mask, tobs, rain, t_eval_t, pollutant_class_hint=hint_t)
        losses = compute_terrakan_v150_loss(
            out=out,
            P_true=torch.tensor(ds["P_true"][b_ids][:, sub_idx], dtype=torch.float64, device=device),
            P_inf_true=torch.tensor(ds["P_inf_true"][b_ids], dtype=torch.float64, device=device),
            M_true=torch.tensor(ds["M_true"][b_ids][:, sub_idx], dtype=torch.float64, device=device),
            E_true=torch.tensor(ds["E_true"][b_ids][:, sub_idx], dtype=torch.float64, device=device),
            IBSP_true=torch.tensor(ds["IBSP_true"][b_ids][:, sub_idx], dtype=torch.float64, device=device),
            alpha_true=torch.tensor(ds["alpha_true"][b_ids], dtype=torch.float64, device=device),
            phi_true=torch.tensor(ds["Phi_true"][b_ids, 0], dtype=torch.float64, device=device),
            dirichlet_targets=torch.tensor(ds["dirichlet_weights"][b_ids], dtype=torch.float64, device=device),
        )
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        val_out = model(
            torch.tensor(ds["obs_values"][val_idx], dtype=torch.float64, device=device),
            torch.tensor(ds["obs_mask"][val_idx], dtype=torch.float64, device=device),
            torch.tensor(ds["obs_times"][val_idx], dtype=torch.float64, device=device),
            torch.tensor(ds["obs_rain_flag"][val_idx], dtype=torch.float64, device=device),
            t_eval_t,
            pollutant_class_hint=None,
        )
        youden = calibrate_youden_threshold(val_out["min_dist"].cpu().numpy(), ds["is_clean_soil"][val_idx])
        model.atlas.d_star = youden["d_star"]

        test_vals = torch.tensor(ds["obs_values"][test_idx], dtype=torch.float64, device=device)
        test_mask = torch.tensor(ds["obs_mask"][test_idx], dtype=torch.float64, device=device)
        test_tobs = torch.tensor(ds["obs_times"][test_idx], dtype=torch.float64, device=device)
        test_rain = torch.tensor(ds["obs_rain_flag"][test_idx], dtype=torch.float64, device=device)

        test_out = model(test_vals, test_mask, test_tobs, test_rain, t_eval_t, pollutant_class_hint=None)

        # Slim curve N in {1, 3, 7}
        slim_curve = {}
        for k_pts in (1, 3, 7):
            m_k = test_mask.clone()
            if k_pts < 8:
                m_k[:, k_pts:, :] = 0.0
            o_k = model(test_vals, m_k, test_tobs, test_rain, t_eval_t, pollutant_class_hint=None)
            r2_ik, _ = r2_rmse(ds["IBSP_true"][test_idx][:, sub_idx][:, extrap_mask], o_k["IBSP"].cpu().numpy()[:, extrap_mask])
            pq_k = o_k["P_quantiles"].cpu().numpy()
            w80_k = float(np.mean(pq_k[..., 4] - pq_k[..., 0]))
            slim_curve[f"N={k_pts}"] = {"r2_IBSP_extrap": round(r2_ik, 4), "mean_q80_width_P": round(w80_k, 4)}

    P_t = ds["P_true"][test_idx][:, sub_idx]
    I_t = ds["IBSP_true"][test_idx][:, sub_idx]
    P_p = test_out["P_bio"].cpu().numpy()
    I_p = test_out["IBSP"].cpu().numpy()
    Pq = test_out["P_quantiles"].cpu().numpy()
    Iq = test_out["IBSP_quantiles"].cpu().numpy()

    r2_P_ext, rmse_P_ext = r2_rmse(P_t[:, extrap_mask], P_p[:, extrap_mask])
    r2_I_ext, rmse_I_ext = r2_rmse(I_t[:, extrap_mask], I_p[:, extrap_mask])
    cov80_P = float(np.mean((P_t >= Pq[..., 0]) & (P_t <= Pq[..., 4])) * 100.0)
    cov80_I = float(np.mean((I_t >= Iq[..., 0]) & (I_t <= Iq[..., 4])) * 100.0)

    # Arch_5 Clean FPR
    clean_arch5 = (ds["dominant_arch"][test_idx] == 4) & ds["is_clean_soil"][test_idx]
    arch5_fpr = float(np.mean(test_out["min_dist"].cpu().numpy()[clean_arch5] >= model.atlas.d_star) * 100.0) if np.any(clean_arch5) else 0.0

    # OOD 3 suites
    ood_metrics = eval_ood_suites(model, t_grid_full, device)

    # Wet-lab (fix #3: обнуляем deg на чистых почвах p0 < 0.05 или d_min < 0.72 * d_star)
    wetlab_preds = {}
    with torch.no_grad():
        for sc in get_wetlab_suite_v150():
            pk = pack_wetlab_scenario(sc)
            t_wl = torch.tensor([0.0, float(sc["horizon_days"])], dtype=torch.float64, device=device)
            o_wl = model(
                torch.tensor(pk["values"], dtype=torch.float64, device=device),
                torch.tensor(pk["mask"], dtype=torch.float64, device=device),
                torch.tensor(pk["t_obs"], dtype=torch.float64, device=device),
                torch.tensor(pk["rain_flag"], dtype=torch.float64, device=device),
                t_wl,
                pollutant_class_hint=torch.tensor(pk["hint"], dtype=torch.float64, device=device),
            )
            p0 = float(o_wl["P_bio"][0, 0].item())
            pend = float(o_wl["P_bio"][0, 1].item())
            d_min_v = float(o_wl["min_dist"][0].item())
            is_clean_pred = (p0 < 0.05) or (d_min_v < 0.72 * model.atlas.d_star)
            deg = 0.0 if is_clean_pred else float(np.clip((1.0 - pend / p0) * 100.0, 0.0, 100.0))
            wetlab_preds[sc["name"]] = {
                "pred_deg": round(deg, 2),
                "actual_deg": sc["actual_deg"],
                "P0": round(p0, 3),
                "pi_persist": round(float(o_wl["pi_persist"][0].item()), 3),
                "d_min": round(d_min_v, 3),
                "anom": bool(d_min_v >= model.atlas.d_star),
            }

    return {
        "seed": seed,
        "train_sec": round(time.time() - t0, 1),
        "youden": youden,
        "arch5_clean_fpr_pct": round(arch5_fpr, 2),
        "r2_P_extrap": round(r2_P_ext, 4),
        "rmse_P_extrap": round(rmse_P_ext, 4),
        "r2_IBSP_extrap": round(r2_I_ext, 4),
        "rmse_IBSP_extrap": round(rmse_I_ext, 2),
        "cov80_P_pct": round(cov80_P, 1),
        "cov80_IBSP_pct": round(cov80_I, 1),
        "gamma_h": round(float(model.trunk.get_gamma_h().item()), 3),
        "slim_curve": slim_curve,
        "ood_benchmarks": ood_metrics,
        "wetlab": wetlab_preds,
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Быстрый smoke-тест (150 шагов, 1 сид)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 101, 777, 1337, 2026])
    parser.add_argument("--steps", type=int, default=2000)
    args = parser.parse_args()

    run_numpy_math_tests()
    run_torch_arch_tests()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=== TerraKAN v1.5.1 Training & Full Benchmark (device={device}, fast={args.fast}) ===")
    if args.fast:
        ds = generate_terrakan_v150_dataset(n_base=200, n_augmented=120, t_max=365.0, dt=1.0, seed=42)
        seeds, n_steps = [42], 150
    else:
        ds = generate_terrakan_v150_dataset(n_base=4000, n_augmented=3500, t_max=365.0, dt=0.5, seed=42)
        seeds, n_steps = args.seeds, args.steps

    results = []
    ckpt_ensemble = {}
    for s in seeds:
        res = train_single_seed(s, ds, n_steps=n_steps, device=device)
        ckpt_ensemble[f"seed_{s}"] = res.pop("state_dict")
        results.append(res)
        print(
            f"Seed {s}: R2(P)={res['r2_P_extrap']:.4f} | R2(IBSP)={res['r2_IBSP_extrap']:.4f} | "
            f"Cov80(P/I)={res['cov80_P_pct']:.1f}/{res['cov80_IBSP_pct']:.1f}% | "
            f"gamma_h={res['gamma_h']:.3f} | J={res['youden']['val_J']:.4f} | Arch5_FPR={res['arch5_clean_fpr_pct']:.1f}%"
        )

    out_dir = os.path.dirname(__file__)
    torch.save(ckpt_ensemble, os.path.join(out_dir, "terrakan_v151_ensemble.pt"))
    with open(os.path.join(out_dir, "v151_eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("Saved terrakan_v151_ensemble.pt and v151_eval_summary.json")


if __name__ == "__main__":
    main()
