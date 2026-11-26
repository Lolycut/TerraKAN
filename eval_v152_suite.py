"""
TerraKAN v1.5.2 — Мгновенный 5-сидовый медианный ансамблевый бенчмарк БЕЗ переобучения!

Загружает уже обученный `terrakan_v151_ensemble.pt` (за 3 секунды!) и прогоняет:
  1. Все 15 реальных wet-lab сценариев (5 классических + 10 из новых open-access PMC-статей)
     с честным 5-сидовым медианным ансамблем (гасит одиночные выбросы сидов!).
  2. 36-сосудный dry-run протокол (`dry_run_synthetic_36pots.csv`) с per-series G0-нормализацией
     без утечки is_control_G0 (проверка 100% разделения G0 / G1 / G2+G3).
  3. Генерацию Детального Паспорта Почвы (`soil_passports_v152.json`) для всех 15 wet-lab полей.

Запуск (прямо рядом с terrakan_v151_ensemble.pt):
  python eval_v152_suite.py --ckpt terrakan_v151_ensemble.pt
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Dict, List
import numpy as np
import torch

from model_v150 import TerraKANv150
from wetlab_benchmarks_v150 import get_wetlab_suite_v150, pack_wetlab_scenario
from soil_passport import build_detailed_soil_passport


def load_ensemble(ckpt_path: str, device: str) -> List[TerraKANv150]:
    bundle = torch.load(ckpt_path, map_location=device)
    models = []
    for k, sd in bundle.items():
        m = TerraKANv150(d_model=64, n_heads=4, n_layers=2, n_loops=2).to(device=device, dtype=torch.float64)
        m.load_state_dict(sd)
        m.eval()
        models.append(m)
    print(f"Loaded {len(models)}-seed ensemble from {ckpt_path}")
    return models


def eval_wetlab_ensemble(models: List[TerraKANv150], device: str, d_star: float = 0.58):
    suite = get_wetlab_suite_v150()
    rows = []
    passports = []

    print("\n" + "=" * 98)
    print("TERRAKAN v1.5.2 — 5-SEED MEDIAN ENSEMBLE ON 15 REAL OPEN-ACCESS WET-LAB SCENARIOS")
    print("=" * 98)

    with torch.no_grad():
        for sc in suite:
            pk = pack_wetlab_scenario(sc)
            vals = torch.tensor(pk["values"], dtype=torch.float64, device=device)
            mask = torch.tensor(pk["mask"], dtype=torch.float64, device=device)
            tobs = torch.tensor(pk["t_obs"], dtype=torch.float64, device=device)
            rain = torch.tensor(pk["rain_flag"], dtype=torch.float64, device=device)
            hint = torch.tensor(pk["hint"], dtype=torch.float64, device=device)
            t_wl = torch.tensor([0.0, float(sc["horizon_days"])], dtype=torch.float64, device=device)

            degs, deg_q10s, deg_q90s, p0s, pis, dmins, ibsp0s, ibsp1s = [], [], [], [], [], [], [], []
            p_probs, d_vecs = [], []

            for m in models:
                o = m(vals, mask, tobs, rain, t_wl, pollutant_class_hint=hint)
                p0 = float(o["P_bio"][0, 0].item())
                pend = float(o["P_bio"][0, 1].item())
                d_min_v = float(o["min_dist"][0].item())
                is_clean = (p0 < 0.05) or (d_min_v < 0.72 * d_star)
                deg = 0.0 if is_clean else float(np.clip((1.0 - pend / p0) * 100.0, 0.0, 100.0))

                # 80% квантильный коридор деградации из H3
                pq = o["P_quantiles"][0].cpu().numpy()  # (2, 5)
                if is_clean:
                    dq10, dq90 = 0.0, 0.0
                else:
                    p_hi_end = max(0.0, float(pq[1, 4]))
                    p_lo_end = max(0.0, float(pq[1, 0]))
                    dq10 = float(np.clip((1.0 - p_hi_end / max(p0, 1e-4)) * 100.0, 0.0, 100.0))
                    dq90 = float(np.clip((1.0 - p_lo_end / max(p0, 1e-4)) * 100.0, 0.0, 100.0))

                degs.append(deg)
                deg_q10s.append(min(dq10, dq90))
                deg_q90s.append(max(dq10, dq90))
                p0s.append(p0)
                pis.append(float(o["pi_persist"][0].item()))
                dmins.append(d_min_v)
                ibsp0s.append(float(o["IBSP"][0, 0].item()))
                ibsp1s.append(float(o["IBSP"][0, 1].item()))
                p_probs.append(o["passport"][0].cpu().numpy())
                d_vecs.append(o["detailed_passport_vec"][0].cpu().numpy())

            deg_med = float(np.median(degs))
            deg_sd = float(np.std(degs))
            pi_med = float(np.median(pis))
            dmin_med = float(np.median(dmins))
            p0_med = float(np.median(p0s))
            act = float(sc["actual_deg"])
            err = abs(deg_med - act)
            horm_flag = (max(sc["E"]) > 1.26) or (max(sc["M"]) > 1.08)

            print(
                f"  {sc['name'][:48]:48s} | Ens={deg_med:5.1f}% (SD {deg_sd:4.1f}) | "
                f"Ref={act:5.1f}% | Err={err:4.1f} | pi={pi_med:.2f} | d={dmin_med:.2f}"
            )
            rows.append({
                "name": sc["name"],
                "source": sc["source"],
                "group": sc["group"],
                "actual_deg": act,
                "ensemble_median_deg": round(deg_med, 2),
                "seed_sd_deg": round(deg_sd, 2),
                "abs_err_pp": round(err, 2),
                "pi_persist_median": round(pi_med, 3),
                "d_min_median": round(dmin_med, 3),
            })
            passports.append(build_detailed_soil_passport(
                sample_name=sc["name"],
                passport_probs=np.mean(p_probs, axis=0),
                detailed_vec=np.mean(d_vecs, axis=0),
                P0_med=p0_med,
                deg_med=deg_med,
                deg_q10=float(np.median(deg_q10s)),
                deg_q90=float(np.median(deg_q90s)),
                pi_persist=pi_med,
                ibsp_start=float(np.median(ibsp0s)),
                ibsp_end=float(np.median(ibsp1s)),
                d_min=dmin_med,
                d_star=d_star,
                hormesis_flag=horm_flag,
            ))

    mae_all = float(np.mean([r["abs_err_pp"] for r in rows]))
    mae_persist = float(np.mean([r["abs_err_pp"] for r in rows if r["group"] in ("HeavyMetal-Persistent", "Clean-Control")]))
    print("-" * 98)
    print(f"  OVERALL 15-SCENARIO ENSEMBLE MAE: {mae_all:.2f} pp | PERSISTENT+CLEAN MAE: {mae_persist:.2f} pp")
    return rows, passports


def eval_36pots_dry_run(models: List[TerraKANv150], csv_path: str, device: str, d_star: float = 0.58) -> Dict[str, object]:
    if not os.path.exists(csv_path):
        print(f"[SKIP] {csv_path} not found.")
        return {}

    by_sample = defaultdict(list)
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_sample[row["sample_id"]].append(row)

    # Per-series G0 normalization (строго без утечки флага в модель!)
    series_g0_M = defaultdict(list)
    series_g0_E = defaultdict(list)
    for sid, rlist in by_sample.items():
        for r in rlist:
            if int(r["is_control_G0"]) == 1 and float(r["day"]) == 0.0:
                if r["respiration_CO2_mg"]:
                    series_g0_M[r["soil_series"]].append(float(r["respiration_CO2_mg"]))
                if r["catalase_O2_ml"]:
                    net_e = max(0.05, float(r["catalase_O2_ml"]) - float(r["blank_O2_ml"] or 0.0))
                    series_g0_E[r["soil_series"]].append(net_e)

    g0_ref_M = {s: float(np.mean(v)) for s, v in series_g0_M.items()}
    g0_ref_E = {s: float(np.mean(v)) for s, v in series_g0_E.items()}

    group_p0 = defaultdict(list)
    with torch.no_grad():
        for sid, rlist in sorted(by_sample.items()):
            rlist = sorted(rlist, key=lambda x: float(x["day"]))
            ser = rlist[0]["soil_series"]
            ref_m = g0_ref_M.get(ser, 15.0)
            ref_e = g0_ref_E.get(ser, 5.0)
            ref_p0 = float(rlist[0]["lab_toxin_ref_ec50"] or 0.0)
            is_g0 = int(rlist[0]["is_control_G0"]) == 1
            grp = "G0_Control" if is_g0 else ("G1_PoorClean" if ref_p0 < 0.15 else "G2G3_Contaminated")

            max_len = 8
            vals = np.zeros((1, max_len, 5), dtype=np.float64)
            mask = np.zeros((1, max_len, 5), dtype=np.float64)
            tobs = np.zeros((1, max_len), dtype=np.float64)
            rain = np.zeros((1, max_len, 1), dtype=np.float64)

            n_pts = min(len(rlist), max_len)
            for j in range(n_pts):
                r = rlist[j]
                vals[0, j, 0] = np.clip((float(r["pH"]) - 3.8) / 5.0, 0.1, 1.0)
                vals[0, j, 1] = np.clip(float(r["NO3_mg_l"]) / 80.0, 0.1, 1.0)
                vals[0, j, 2] = np.clip(float(r["moisture_pct"]) / 45.0, 0.1, 1.0)
                mask[0, j, :3] = 1.0
                if r["respiration_CO2_mg"]:
                    vals[0, j, 3] = float(r["respiration_CO2_mg"]) / ref_m
                    mask[0, j, 3] = 1.0
                if r["catalase_O2_ml"]:
                    net_e = max(0.05, float(r["catalase_O2_ml"]) - float(r["blank_O2_ml"] or 0.0))
                    vals[0, j, 4] = (net_e / ref_e) * 1.20
                    mask[0, j, 4] = 1.0
                tobs[0, j] = float(r["day"])
                rain[0, j, 0] = float(r["rain_flag"] or 0.0)
            if n_pts < max_len:
                tobs[0, n_pts:] = tobs[0, n_pts - 1]

            p0s = []
            t_ev = torch.tensor([0.0, 60.0], dtype=torch.float64, device=device)
            for m in models:
                o = m(
                    torch.tensor(vals, dtype=torch.float64, device=device),
                    torch.tensor(mask, dtype=torch.float64, device=device),
                    torch.tensor(tobs, dtype=torch.float64, device=device),
                    torch.tensor(rain, dtype=torch.float64, device=device),
                    t_ev,
                    pollutant_class_hint=None,
                )
                p0s.append(float(o["P_bio"][0, 0].item()))
            group_p0[grp].append(float(np.median(p0s)))

    summary = {
        g: {"n": len(v), "min_P0": round(min(v), 4), "max_P0": round(max(v), 4), "mean_P0": round(float(np.mean(v)), 4)}
        for g, v in group_p0.items()
    }
    clean_max = max(summary.get("G0_Control", {}).get("max_P0", 0.0), summary.get("G1_PoorClean", {}).get("max_P0", 0.0))
    contam_min = summary.get("G2G3_Contaminated", {}).get("min_P0", 0.0)
    summary["clean_vs_contam_separated"] = bool(contam_min > clean_max)
    print("\n=== 36-VESSEL DRY-RUN ENSEMBLE VERIFICATION ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="terrakan_v151_ensemble.pt")
    parser.add_argument("--csv36", type=str, default="dry_run_synthetic_36pots.csv")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = load_ensemble(args.ckpt, device)
    wetlab_rows, passports = eval_wetlab_ensemble(models, device)
    dry36 = eval_36pots_dry_run(models, args.csv36, device)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(out_dir, "v152_wetlab15_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"wetlab_15": wetlab_rows, "dry_run_36pots": dry36}, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "soil_passports_v152.json"), "w", encoding="utf-8") as f:
        json.dump(passports, f, indent=2, ensure_ascii=False)
    print("\nSaved v152_wetlab15_summary.json and soil_passports_v152.json!")


if __name__ == "__main__":
    main()
