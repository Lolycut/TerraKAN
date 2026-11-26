"""
TerraKAN v1.5.2 — Расширенная база реальных zero-shot wet-lab сценариев (15 серий, 9 классов поллютантов).

Включает:
  - 5 классических контрольных серий (Margesin 2001, Garousin 2021, Rafikova 2026 Cu2+, Kumar 2022 Chlorpyrifos+ERM, Chai 2023 CWS)
  - 10 расширенных open-access серий из PMC:
      * PMC5541004 (Lu et al. 2017, Sci. Rep.): Пирен + корневые экссудаты (глюкоза 54.3%, контроль 39.9%)
      * PMC12559167 (Omenna et al. 2025, Sci. Rep.): Сырая нефть 90 сут (T4 кенаф+рамнолипид 82.6%, BN3T6 95.5%)
      * PMC12238601 (Szpyrka et al. 2025, Sci. Rep.): Стойкий гербицид Aclonifen + EM-консорциум 67 сут (56.0% и контроль 26.0%)
      * PMC10303346 (2023): Шестивалентный хром Cr(VI) 40 мг/кг (стойкий металл 0.0%, подавление DHA -35%)
      * PMC10231758 (Dotaniya et al. 2023, PLOS ONE): Свинец Pb 300 мг/кг в индийском вертисоле (6.6% потерь / 93.4% удержание)
      * PMC9074702 (Zhang et al. 2019): Нефтяное месторождение Shengli — биочар+бактерии BIM (58.1%) и контроль CK (8.7%)
"""

from typing import Dict, List
import numpy as np


def get_wetlab_suite_v150() -> List[Dict[str, object]]:
    return [
        # --- 5 КЛАССИЧЕСКИХ КОНТРОЛЬНЫХ СЕРИЙ ---
        {
            "name": "Alpine Gravel-Sand (Biostim.)",
            "source": "Margesin & Schinner 2001 (PMC92991)",
            "group": "Hydrocarbons-Biostim",
            "hint": -1.0, "actual_deg": 70.4, "horizon_days": 135.0,
            "t_obs": [15.0, 75.0, 135.0],
            "pH_norm": 0.68, "NO3_norm": 0.72, "W_norm": 0.55,
            "M": [1.35, 1.05, 0.92], "E": [1.45, 0.96, 0.78],
            "mask_M": [1, 1, 1], "mask_E": [1, 1, 1],
        },
        {
            "name": "Saline-Sodic Sandy-Loam (SB, >60y Oil)",
            "source": "Garousin et al. 2021 (PMC7947215)",
            "group": "Hydrocarbons-Chronic",
            "hint": -1.0, "actual_deg": 38.2, "horizon_days": 60.0,
            "t_obs": [0.0, 20.0, 40.0, 60.0],
            "pH_norm": 0.80, "NO3_norm": 0.60, "W_norm": 0.65,
            "M": [0.42, 0.78, 0.50, 0.66], "E": [0.65, 1.10, 0.55, 0.48],
            "mask_M": [1, 1, 0, 1], "mask_E": [1, 1, 1, 1],
        },
        {
            "name": "Chernozem + Cu2+ (660 mg/kg)",
            "source": "Rafikova et al. 2026 (PMC12945044)",
            "group": "HeavyMetal-Persistent",
            "hint": 1.0, "actual_deg": 0.0, "horizon_days": 21.0,
            "t_obs": [0.0, 14.0, 21.0],
            "pH_norm": 0.62, "NO3_norm": 0.65, "W_norm": 0.60,
            "M": [0.85, 0.78, 0.74], "E": [0.78, 0.66, 0.63],
            "mask_M": [1, 1, 1], "mask_E": [1, 1, 1],
        },
        {
            "name": "Subtrop. Soil + Chlorpyrifos + ERM C-1",
            "source": "Kumar et al. 2022 (PMC9360973)",
            "group": "Pesticide-Consortium",
            "hint": -1.0, "actual_deg": 98.6, "horizon_days": 30.0,
            "t_obs": [0.0, 10.0, 20.0, 30.0],
            "pH_norm": 0.68, "NO3_norm": 0.65, "W_norm": 0.60,
            "M": [0.62, 0.95, 1.15, 1.02], "E": [0.68, 1.18, 1.25, 1.19],
            "mask_M": [1, 1, 1, 1], "mask_E": [1, 1, 1, 1],
        },
        {
            "name": "Clean Wetland Control (CWS, blind)",
            "source": "Chai et al. 2023",
            "group": "Clean-Control",
            "hint": 0.0, "actual_deg": 0.0, "horizon_days": 42.0,
            "t_obs": [0.0, 14.0, 28.0, 42.0],
            "pH_norm": 0.72, "NO3_norm": 0.55, "W_norm": 0.70,
            "M": [1.00, 1.01, 0.99, 1.00], "E": [1.20, 1.22, 1.18, 1.21],
            "mask_M": [1, 1, 1, 1], "mask_E": [1, 1, 1, 1],
        },

        # --- PMC5541004: Пирен + корневые экссудаты (Lu et al. 2017) ---
        {
            "name": "NEW: Pyrene + Glucose Exudate (21d)",
            "source": "Lu et al. 2017 (PMC5541004)",
            "group": "PAH-Rhizosphere",
            "hint": -1.0, "actual_deg": 54.3, "horizon_days": 21.0,
            "t_obs": [7.0, 14.0, 21.0],
            "pH_norm": 0.66, "NO3_norm": 0.52, "W_norm": 0.60,
            "M": [10.94 / 14.09, 4.24 / 14.09, 15.83 / 14.09],
            "E": [0.53 / 2.12 * 1.20, 1.15 / 2.12 * 1.20, 2.59 / 2.12 * 1.20],
            "mask_M": [1, 1, 1], "mask_E": [1, 0, 1],
        },
        {
            "name": "NEW: Pyrene Control No Exudate (21d)",
            "source": "Lu et al. 2017 (PMC5541004)",
            "group": "PAH-Rhizosphere",
            "hint": -1.0, "actual_deg": 39.9, "horizon_days": 21.0,
            "t_obs": [7.0, 14.0, 21.0],
            "pH_norm": 0.66, "NO3_norm": 0.52, "W_norm": 0.60,
            "M": [8.01 / 14.09, 4.00 / 14.09, 14.09 / 14.09],
            "E": [0.37 / 2.12 * 1.20, 0.90 / 2.12 * 1.20, 2.12 / 2.12 * 1.20],
            "mask_M": [1, 1, 1], "mask_E": [1, 0, 1],
        },

        # --- PMC12559167: Сырая нефть 90 сут (Omenna et al. 2025) ---
        {
            "name": "NEW: Crude Oil + Kenaf + Rhamnolipid T4 (90d)",
            "source": "Omenna et al. 2025 (PMC12559167)",
            "group": "Hydrocarbons-Biostim90d",
            "hint": -1.0, "actual_deg": 82.6, "horizon_days": 90.0,
            "t_obs": [15.0, 30.0, 45.0, 60.0, 75.0, 90.0],
            "pH_norm": 0.71, "NO3_norm": 0.65, "W_norm": 0.65,
            "M": [0.0] * 6,
            "E": [x / 18.45 * 1.20 for x in [9.51, 10.00, 9.20, 11.85, 14.21, 18.45]],
            "mask_M": [0] * 6, "mask_E": [1] * 6,
        },
        {
            "name": "NEW: Crude Oil + Benin-3 + Rhamnolipid BN3T6 (90d)",
            "source": "Omenna et al. 2025 (PMC12559167)",
            "group": "Hydrocarbons-Bioaug90d",
            "hint": -1.0, "actual_deg": 95.5, "horizon_days": 90.0,
            "t_obs": [15.0, 30.0, 45.0, 60.0, 75.0, 90.0],
            "pH_norm": 0.69, "NO3_norm": 0.65, "W_norm": 0.65,
            "M": [0.0] * 6,
            "E": [x / 17.30 * 1.20 for x in [10.32, 10.81, 10.00, 12.00, 13.75, 17.30]],
            "mask_M": [0] * 6, "mask_E": [1] * 6,
        },

        # --- PMC12238601: Гербицид Aclonifen + EM-консорциум 67 сут (Szpyrka et al. 2025) ---
        {
            "name": "NEW: Herbicide Aclonifen + 3-EM Consortium (67d)",
            "source": "Szpyrka et al. 2025, Sci. Rep. 15:24522 (PMC12238601)",
            "group": "Herbicide-Consortium",
            "hint": -1.0, "actual_deg": 56.0, "horizon_days": 67.0,
            "t_obs": [11.0, 27.0, 47.0, 67.0],
            "pH_norm": 0.76, "NO3_norm": 0.70, "W_norm": 0.70,
            "M": [0.0] * 4,  # DHA-only мониторинг
            "E": [9.5 / 27.6 * 1.20, 19.0 / 27.6 * 1.20, 28.5 / 27.6 * 1.20, 37.6 / 27.6 * 1.20],
            "mask_M": [0, 0, 0, 0], "mask_E": [1, 1, 1, 1],
        },
        {
            "name": "NEW: Herbicide Aclonifen Unamended Control (67d)",
            "source": "Szpyrka et al. 2025, Sci. Rep. 15:24522 (PMC12238601)",
            "group": "Herbicide-Control",
            "hint": -1.0, "actual_deg": 26.0, "horizon_days": 67.0,
            "t_obs": [11.0, 27.0, 47.0, 67.0],
            "pH_norm": 0.77, "NO3_norm": 0.65, "W_norm": 0.70,
            "M": [0.0] * 4,
            "E": [13.4 / 27.6 * 1.20, 17.5 / 27.6 * 1.20, 22.8 / 27.6 * 1.20, 27.6 / 27.6 * 1.20],
            "mask_M": [0, 0, 0, 0], "mask_E": [1, 1, 1, 1],
        },

        # --- PMC10303346: Тяжёлый металл Cr(VI) 40 мг/кг (2023) ---
        {
            "name": "NEW: Heavy Metal Cr(VI) 40 mg/kg (Persistent)",
            "source": "Wyszkowska et al. 2023 (PMC10303346)",
            "group": "HeavyMetal-Persistent",
            "hint": 1.0, "actual_deg": 0.0, "horizon_days": 60.0,
            "t_obs": [0.0, 30.0, 60.0],
            "pH_norm": 0.60, "NO3_norm": 0.55, "W_norm": 0.60,
            "M": [0.76, 0.70, 0.65],
            "E": [0.92 * 1.20, 0.78 * 1.20, 0.65 * 1.20],
            "mask_M": [1, 1, 1], "mask_E": [1, 1, 1],
        },

        # --- PMC10231758: Свинец Pb 300 мг/кг (Dotaniya et al. 2023) ---
        {
            "name": "NEW: Indian Vertisol + Pb 300 mg/kg (45d)",
            "source": "Dotaniya et al. 2023, PLOS ONE (PMC10231758)",
            "group": "HeavyMetal-Persistent",
            "hint": 1.0, "actual_deg": 6.6, "horizon_days": 45.0,
            "t_obs": [0.0, 15.0, 30.0, 45.0],
            "pH_norm": 0.78, "NO3_norm": 0.50, "W_norm": 0.55,
            "M": [0.0] * 4,
            "E": [28.0 / 41.99 * 1.20, 24.5 / 41.99 * 1.20, 21.8 / 41.99 * 1.20, 20.26 / 41.99 * 1.20],
            "mask_M": [0, 0, 0, 0], "mask_E": [1, 1, 1, 1],
        },

        # --- PMC9074702: Shengli Oilfield (Biochar + Immobilized Bacteria BIM vs CK, 60d) ---
        {
            "name": "NEW: Shengli Oilfield + Biochar-Immobilized Bact. BIM (60d)",
            "source": "Zhang et al. 2019 (PMC9074702)",
            "group": "Hydrocarbons-Biochar",
            "hint": -1.0, "actual_deg": 58.1, "horizon_days": 60.0,
            "t_obs": [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "pH_norm": 0.74, "NO3_norm": 0.45, "W_norm": 0.60,
            "M": [0.48, 0.72, 0.95, 1.18, 1.26, 1.10, 0.96],
            "E": [0.55, 0.80, 1.05, 1.25, 1.38, 1.42, 1.22],
            "mask_M": [1] * 7, "mask_E": [1] * 7,
        },
        {
            "name": "NEW: Shengli Oilfield Untreated Oil Control CK (60d)",
            "source": "Zhang et al. 2019 (PMC9074702)",
            "group": "Hydrocarbons-Untreated",
            "hint": 0.0, "actual_deg": 8.7, "horizon_days": 60.0,
            "t_obs": [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "pH_norm": 0.74, "NO3_norm": 0.35, "W_norm": 0.55,
            "M": [0.45, 0.52, 0.56, 0.58, 0.50, 0.46, 0.44],
            "E": [0.48, 0.54, 0.58, 0.56, 0.50, 0.46, 0.45],
            "mask_M": [1] * 7, "mask_E": [1] * 7,
        },
    ]


def pack_wetlab_scenario(scen: Dict[str, object], max_len: int = 8) -> Dict[str, np.ndarray]:
    t_raw = np.asarray(scen["t_obs"], dtype=np.float64)
    n_pts = min(len(t_raw), max_len)

    vals = np.zeros((1, max_len, 5), dtype=np.float64)
    mask = np.zeros((1, max_len, 5), dtype=np.float64)
    t_obs = np.zeros((1, max_len), dtype=np.float64)
    rain = np.zeros((1, max_len, 1), dtype=np.float64)

    for j in range(n_pts):
        vals[0, j, 0] = float(scen["pH_norm"])
        vals[0, j, 1] = float(scen["NO3_norm"])
        vals[0, j, 2] = float(scen["W_norm"])
        vals[0, j, 3] = float(scen["M"][j])
        vals[0, j, 4] = float(scen["E"][j])
        mask[0, j, :3] = 1.0
        mask[0, j, 3] = float(scen["mask_M"][j])
        mask[0, j, 4] = float(scen["mask_E"][j])
        t_obs[0, j] = float(t_raw[j])

    if n_pts < max_len:
        t_obs[0, n_pts:] = t_obs[0, n_pts - 1]

    return {
        "values": vals,
        "mask": mask,
        "t_obs": t_obs,
        "rain_flag": rain,
        "hint": np.array([float(scen["hint"])], dtype=np.float64),
    }
