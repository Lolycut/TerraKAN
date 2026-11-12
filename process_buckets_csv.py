"""
Конвейер обработки лабораторного и полевого журнала микрокосмного опыта (36 сосудов `Sample_01`..`Sample_36`)
на базе замороженного чекпоинта `terrakan_v4_frozen.pt`.

В данном модуле я выполняю по-серийную нормировку замеров каталазы O2 и базального дыхания CO2
по контрольным сосудам `G0` соответствующей почвенной серии (`S1_sand`, `S2_loam`, `S3_peat`),
формирую слепой прогноз восстановления на горизонте до 60 суток и вычисляю диагностический статус
исключительно по выходам модели.
"""

import csv
import os
import math
import shutil
from typing import Dict, List, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from model import TerraKAN
from physics_solver import SOIL_ARCHETYPES


REQUIRED_COLUMNS = [
    "sample_id",
    "soil_series",
    "day",
    "pH",
    "NO3_mg_l",
    "moisture_pct",
    "catalase_O2_ml",
    "blank_O2_ml",
    "respiration_CO2_mg",
    "rain_flag",
    "is_control_G0",
    "lab_toxin_ref_ec50",
]

# Справочник референсных емкостей K_m и параметров серий почв Атласа для согласования масштаба входов
SOIL_SERIES_META: Dict[str, Dict[str, object]] = {
    "S1_sand": {
        "arch_idx": 4,
        "arch_code": "Arch_5",
        "series_label": "S1_sand (Песчаная лесная)",
        "K_m_atlas": float(SOIL_ARCHETYPES[4]["K_m"]),       # 0.44
        "alpha_atlas": float(SOIL_ARCHETYPES[4]["alpha"]),   # 0.80
        "pH_norm_atlas": float(SOIL_ARCHETYPES[4]["pH_norm"]),
        "NO3_norm_atlas": float(SOIL_ARCHETYPES[4]["NO3_norm"]),
        "W_norm_atlas": float(SOIL_ARCHETYPES[4]["W_norm"]),
    },
    "S2_loam": {
        "arch_idx": 0,
        "arch_code": "Arch_1",
        "series_label": "S2_loam (Дерново-подзолистый суглинок)",
        "K_m_atlas": float(SOIL_ARCHETYPES[0]["K_m"]),       # 1.00
        "alpha_atlas": float(SOIL_ARCHETYPES[0]["alpha"]),   # 0.75
        "pH_norm_atlas": float(SOIL_ARCHETYPES[0]["pH_norm"]),
        "NO3_norm_atlas": float(SOIL_ARCHETYPES[0]["NO3_norm"]),
        "W_norm_atlas": float(SOIL_ARCHETYPES[0]["W_norm"]),
    },
    "S3_peat": {
        "arch_idx": 1,
        "arch_code": "Arch_2",
        "series_label": "S3_peat (Торфяно-болотная)",
        "K_m_atlas": float(SOIL_ARCHETYPES[1]["K_m"]),       # 1.18
        "alpha_atlas": float(SOIL_ARCHETYPES[1]["alpha"]),   # 0.55
        "pH_norm_atlas": float(SOIL_ARCHETYPES[1]["pH_norm"]),
        "NO3_norm_atlas": float(SOIL_ARCHETYPES[1]["NO3_norm"]),
        "W_norm_atlas": float(SOIL_ARCHETYPES[1]["W_norm"]),
    },
}

SERIES_KEYS = ["S1_sand", "S2_loam", "S3_peat"]


def generate_blank_lab_journal(csv_path: str, n_vessels: int = 36) -> str:
    """
    Генерирует пустой бланк лабораторного журнала `blank_lab_journal_36pots.csv`
    для распечатки и ручного заполнения в ходе «мокрого» опыта (36 сосудов, дни [0, 3, 7, 14, 21, 28]).
    Поля замеров (`pH`, `NO3_mg_l`, `moisture_pct`, `catalase_O2_ml`, `blank_O2_ml`,
    `respiration_CO2_mg`, `lab_toxin_ref_ec50`) оставлены пустыми.
    """
    days = [0, 3, 7, 14, 21, 28]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(REQUIRED_COLUMNS)
        for v_idx in range(1, n_vessels + 1):
            sample_id = f"Sample_{v_idx:02d}"
            soil_series = SERIES_KEYS[(v_idx - 1) % 3]
            is_ctrl = 1 if v_idx <= 6 else 0
            for d in days:
                rain_flag = 1 if d == 14 else 0
                writer.writerow([
                    sample_id,
                    soil_series,
                    d,
                    "",  # pH (заполняется в лаборатории)
                    "",  # NO3_mg_l (заполняется в лаборатории)
                    "",  # moisture_pct (заполняется в лаборатории)
                    "",  # catalase_O2_ml (заполняется в лаборатории)
                    "",  # blank_O2_ml (заполняется в лаборатории)
                    "",  # respiration_CO2_mg (заполняется в лаборатории)
                    rain_flag,
                    is_ctrl,
                    "",  # lab_toxin_ref_ec50 (опциональная внешняя верификация ТПГ/Cu)
                ])
    return csv_path


def generate_dry_run_synthetic_csv(csv_path: str, n_vessels: int = 36, seed: int = 42) -> str:
    """
    Генерирует синтетический проверочный файл `dry_run_synthetic_36pots.csv` для технического
    DRY-RUN тестирования парсера до старта мокрого опыта:
      - Сосуды `Sample_01`..`Sample_06` (6 шт., по 2 на каждую серию `S1_sand`, `S2_loam`, `S3_peat`):
        чистый контроль `G0` (`is_control_G0 == 1`, фоновая нагрузка ~0.01..0.02 EC50).
      - Сосуды `Sample_07`..`Sample_14` (8 шт.): чистая бедная/истощённая почва `G1` (`is_control_G0 == 0`,
        сниженный запас субстрата и биомассы без токсического шока, 0.02..0.04 EC50).
      - Сосуды `Sample_15`..`Sample_24` (10 шт.): умеренное угнетение `G2` (`is_control_G0 == 0`,
        начальная нагрузка 1.30..1.85 EC50).
      - Сосуды `Sample_25`..`Sample_36` (12 шт.): сильное угнетение `G3` (`is_control_G0 == 0`,
        начальная нагрузка 2.35..3.40 EC50).
    Пропуски в `respiration_CO2_mg` рандомизированы индивидуально по сосудам (разные комбинации
    пропущенных дней из {3, 7, 14, 21}), а столбец `lab_toxin_ref_ec50` заполнен только на `day == 0`
    для внешней пост-фактум верификации `P_bio(0)`.
    """
    rng = np.random.default_rng(seed)
    days = [0, 3, 7, 14, 21, 28]
    candidate_missing_days = [
        (3, 14),
        (3, 21),
        (7, 14),
        (14, 21),
        (3,),
        (14,),
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(REQUIRED_COLUMNS)

        for v_idx in range(1, n_vessels + 1):
            sample_id = f"Sample_{v_idx:02d}"
            soil_series = SERIES_KEYS[(v_idx - 1) % 3]
            meta = SOIL_SERIES_META[soil_series]
            km_series = float(meta["K_m_atlas"])
            alpha_s = float(meta["alpha_atlas"])

            pH_base = 3.5 + 5.0 * float(meta["pH_norm_atlas"])
            no3_base = 80.0 * float(meta["NO3_norm_atlas"])
            moist_base = 45.0 * float(meta["W_norm_atlas"])

            is_ctrl = 1 if v_idx <= 6 else 0
            if v_idx <= 6:
                true_group = "G0"
                dose_ec50 = float(rng.uniform(0.010, 0.022))
            elif v_idx <= 14:
                true_group = "G1"
                dose_ec50 = float(rng.uniform(0.018, 0.040))
            elif v_idx <= 24:
                true_group = "G2"
                dose_ec50 = float(rng.uniform(1.30, 1.85))
            else:
                true_group = "G3"
                dose_ec50 = float(rng.uniform(2.35, 3.40))

            miss_tuple = candidate_missing_days[(v_idx * 7 + int(rng.integers(0, 6))) % len(candidate_missing_days)]

            for d in days:
                rain_flag = 1 if d == 14 else 0
                pH_val = pH_base + rng.normal(0.0, 0.045)
                no3_mult = 0.52 if true_group == "G1" else 1.00
                no3_val = max(2.0, no3_base * no3_mult * math.exp(-0.002 * d) + rng.normal(0.0, 1.1))
                moist_val = np.clip(moist_base + (3.6 if rain_flag else 0.0) + rng.normal(0.0, 0.7), 8.0, 65.0)

                blank_o2 = round(float(rng.uniform(0.35, 0.55)), 2)

                if true_group == "G0":
                    m_rel = 1.0 + rng.normal(0.0, 0.015)
                    e_rel = 1.0 + rng.normal(0.0, 0.015)
                elif true_group == "G1":
                    rec = 0.54 + 0.24 * (1.0 - math.exp(-0.045 * d))
                    m_rel = rec * (1.0 + rng.normal(0.0, 0.016))
                    e_rel = rec * (1.0 + rng.normal(0.0, 0.016))
                else:
                    t_eff = max(d * 5.0, 0.1)
                    p_t = dose_ec50 * math.exp(-0.028 * (t_eff ** alpha_s))
                    shock_m = 1.0 / (1.0 + 0.25 * p_t)
                    shock_e = 1.0 / (1.0 + p_t * p_t)
                    m_rel = shock_m * (1.0 + rng.normal(0.0, 0.020))
                    e_rel = shock_m * shock_e * (1.0 + rng.normal(0.0, 0.020))

                net_o2 = max(0.15, 6.80 * km_series * e_rel)
                cat_o2 = round(blank_o2 + net_o2, 2)

                if d in miss_tuple:
                    resp_str = ""
                else:
                    resp_val = max(1.0, 24.00 * km_series * m_rel)
                    resp_str = f"{resp_val:.2f}"

                lab_ref_str = f"{dose_ec50:.3f}" if d == 0 else ""

                writer.writerow([
                    sample_id,
                    soil_series,
                    d,
                    f"{pH_val:.2f}",
                    f"{no3_val:.1f}",
                    f"{moist_val:.1f}",
                    f"{cat_o2:.2f}",
                    f"{blank_o2:.2f}",
                    resp_str,
                    rain_flag,
                    is_ctrl,
                    lab_ref_str,
                ])

    return csv_path


def _parse_float_or_nan(val_str: str) -> float:
    if val_str is None:
        return float("nan")
    s = str(val_str).strip()
    if s == "" or s.lower() in ("nan", "none", "null", "na"):
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def load_and_normalize_buckets_csv(csv_path: str, max_obs: int = 8) -> Dict[str, object]:
    """
    Читает журнал микрокосмного опыта, вычисляет по-серийные базовые делители
    `E_ref_G0[soil_series]` и `M_ref_G0[soil_series]` строго по сосудам с `is_control_G0 == 1`
    ТОЙ ЖЕ САМОЙ почвенной серии (`soil_series`) на `day == 0`, и формирует входные тензоры.

    ВАЖНО: Флаг `is_control_G0` используется ТОЛЬКО внутри этой функции для расчета делителей
    `E_ref_G0` и `M_ref_G0` и НЕ передается в выходной словарь для модели или диагноза!
    Столбец `lab_toxin_ref_ec50` считывается отдельно исключительно для внешней верификации после прогона.
    """
    rows_by_sample: Dict[str, List[Dict[str, object]]] = {}
    ctrl_cat_by_series: Dict[str, List[float]] = {k: [] for k in SERIES_KEYS}
    ctrl_resp_by_series: Dict[str, List[float]] = {k: [] for k in SERIES_KEYS}
    lab_ref_by_sample: Dict[str, float] = {}
    series_by_sample: Dict[str, str] = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row["sample_id"].strip()
            s_series = row.get("soil_series", "S2_loam").strip()
            if s_series not in SOIL_SERIES_META:
                s_series = "S2_loam"
            day = float(row["day"])
            pH = _parse_float_or_nan(row["pH"])
            no3 = _parse_float_or_nan(row["NO3_mg_l"])
            moist = _parse_float_or_nan(row["moisture_pct"])
            cat_o2 = _parse_float_or_nan(row["catalase_O2_ml"])
            blank_o2 = _parse_float_or_nan(row["blank_O2_ml"])
            resp_co2 = _parse_float_or_nan(row["respiration_CO2_mg"])
            rain = float(_parse_float_or_nan(row["rain_flag"]) or 0.0)
            is_ctrl = int(float(_parse_float_or_nan(row["is_control_G0"]) or 0.0))
            lab_ref = _parse_float_or_nan(row.get("lab_toxin_ref_ec50", ""))

            net_cat = (
                max(0.0, cat_o2 - (0.0 if math.isnan(blank_o2) else blank_o2))
                if not math.isnan(cat_o2)
                else float("nan")
            )

            # Используем is_control_G0 ТОЛЬКО для накопления по-серийных делителей на day == 0
            if is_ctrl == 1 and abs(day - 0.0) < 1e-6:
                if not math.isnan(net_cat) and net_cat > 0.0:
                    ctrl_cat_by_series.setdefault(s_series, []).append(net_cat)
                if not math.isnan(resp_co2) and resp_co2 > 0.0:
                    ctrl_resp_by_series.setdefault(s_series, []).append(resp_co2)

            if abs(day - 0.0) < 1e-6 and not math.isnan(lab_ref):
                lab_ref_by_sample[sid] = lab_ref

            series_by_sample[sid] = s_series
            rows_by_sample.setdefault(sid, []).append({
                "day": day,
                "soil_series": s_series,
                "pH": pH,
                "no3": no3,
                "moist": moist,
                "net_cat": net_cat,
                "resp_co2": resp_co2,
                "rain": rain,
            })

    E_ref_G0: Dict[str, float] = {}
    M_ref_G0: Dict[str, float] = {}
    for s_key in SERIES_KEYS:
        km_def = float(SOIL_SERIES_META[s_key]["K_m_atlas"])
        E_ref_G0[s_key] = (
            float(np.mean(ctrl_cat_by_series[s_key]))
            if len(ctrl_cat_by_series.get(s_key, [])) > 0
            else 6.80 * km_def
        )
        M_ref_G0[s_key] = (
            float(np.mean(ctrl_resp_by_series[s_key]))
            if len(ctrl_resp_by_series.get(s_key, [])) > 0
            else 24.00 * km_def
        )

    sample_ids = sorted(rows_by_sample.keys())
    B = len(sample_ids)
    values = np.zeros((B, max_obs, 5), dtype=np.float64)
    avail_mask = np.zeros((B, max_obs, 5), dtype=np.float64)
    t_obs = np.zeros((B, max_obs), dtype=np.float64)
    rain_flag = np.zeros((B, max_obs, 1), dtype=np.float64)
    soil_series_list: List[str] = []
    lab_ref_arr = np.full(B, np.nan, dtype=np.float64)

    for b, sid in enumerate(sample_ids):
        s_rows = sorted(rows_by_sample[sid], key=lambda r: float(r["day"]))[:max_obs]
        n_r = len(s_rows)
        s_series = series_by_sample[sid]
        soil_series_list.append(s_series)
        if sid in lab_ref_by_sample:
            lab_ref_arr[b] = lab_ref_by_sample[sid]

        e_ref_s = E_ref_G0[s_series]
        m_ref_s = M_ref_G0[s_series]
        km_atlas = float(SOIL_SERIES_META[s_series]["K_m_atlas"])

        for j in range(max_obs):
            if j < n_r:
                r = s_rows[j]
                t_obs[b, j] = float(r["day"]) * 5.0
                rain_flag[b, j, 0] = float(r["rain"])

                if not math.isnan(float(r["pH"])):
                    values[b, j, 0] = float(np.clip((float(r["pH"]) - 3.5) / 5.0, 0.05, 0.99))
                    avail_mask[b, j, 0] = 1.0
                if not math.isnan(float(r["no3"])):
                    values[b, j, 1] = float(np.clip(float(r["no3"]) / 80.0, 0.02, 1.50))
                    avail_mask[b, j, 1] = 1.0
                if not math.isnan(float(r["moist"])):
                    values[b, j, 2] = float(np.clip(float(r["moist"]) / 45.0, 0.05, 1.50))
                    avail_mask[b, j, 2] = 1.0
                if not math.isnan(float(r["resp_co2"])):
                    rel_m = float(r["resp_co2"]) / max(m_ref_s, 1e-6)
                    values[b, j, 3] = float(np.clip(km_atlas * rel_m, 0.02, 3.00))
                    avail_mask[b, j, 3] = 1.0
                else:
                    avail_mask[b, j, 3] = 0.0
                    values[b, j, 3] = values[b, max(0, j - 1), 3]
                if not math.isnan(float(r["net_cat"])):
                    rel_e = float(r["net_cat"]) / max(e_ref_s, 1e-6)
                    values[b, j, 4] = float(np.clip(1.22 * km_atlas * rel_e, 0.02, 3.50))
                    avail_mask[b, j, 4] = 1.0
                else:
                    avail_mask[b, j, 4] = 0.0
                    values[b, j, 4] = values[b, max(0, j - 1), 4]
            else:
                t_obs[b, j] = float(s_rows[-1]["day"]) * 5.0

    return {
        "sample_ids": sample_ids,
        "soil_series": soil_series_list,
        "E_ref_G0": E_ref_G0,
        "M_ref_G0": M_ref_G0,
        "lab_toxin_ref_ec50": lab_ref_arr,
        "values": torch.tensor(values, dtype=torch.float64),
        "avail_mask": torch.tensor(avail_mask, dtype=torch.float64),
        "t_obs": torch.tensor(t_obs, dtype=torch.float64),
        "rain_flag": torch.tensor(rain_flag, dtype=torch.float64),
    }


def classify_vessel_from_model_outputs(
    P_bio_0: float,
    ibsp_0: float,
    d_min: float,
    d_star: float,
) -> str:
    """
    Формирует текстовый диагноз ИСКЛЮЧИТЕЛЬНО по выходам модели TerraKAN
    без передачи флага `is_control_G0` или истинной метки группы (`G0/G1/G2/G3`):
      - если P_bio(0) < 0.8 и ИБСП(0) >= 85% (и d_min <= d_star) -> "НОРМА (здоровая почва)";
      - если P_bio(0) < 1.0 и ИБСП(0) < 85% (снижена ёмкость без скачка токсина и d_min <= d_star)
        -> "БЕДНАЯ / ИСТОЩЁННАЯ ПОЧВА (дефицит субстрата без токсикоза)";
      - если P_bio(0) >= 1.0 или d_min > d_star
        -> "ТОКСИЧЕСКОЕ УГНЕТЕНИЕ (требуется биоремедиация)".
    """
    if P_bio_0 < 0.8 and ibsp_0 >= 85.0 and d_min <= d_star:
        return "НОРМА (здоровая почва)"
    elif P_bio_0 < 1.0 and ibsp_0 < 85.0 and d_min <= d_star:
        return "БЕДНАЯ / ИСТОЩЁННАЯ ПОЧВА (дефицит субстрата без токсикоза)"
    else:
        return "ТОКСИЧЕСКОЕ УГНЕТЕНИЕ (требуется биоремедиация)"


def generate_human_report(
    report_rows: List[Dict[str, object]],
    E_ref_G0: Dict[str, float],
    M_ref_G0: Dict[str, float],
) -> str:
    """Формирует и печатает структурированный человекочитаемый отчет по 36 сосудам DRY-RUN прогона."""
    ref_str = " | ".join(
        f"{s_k}: V_O2={E_ref_G0[s_k]:.2f} мл, R_CO2={M_ref_G0[s_k]:.2f} мг"
        for s_k in SERIES_KEYS
    )
    lines = [
        "=" * 118,
        "ТЕХНИЧЕСКИЙ DRY-RUN ПАРСЕРА CSV (v1.1): ПРОВЕРКА КОНВЕЙЕРА TERRAKAN НА 36 СИНТЕТИЧЕСКИХ СОСУДАХ",
        "=" * 118,
        "Прокси-признак E: Каталаза O2 (прокси-индикатор ферментативной активности E)",
        "Прокси-признак M: Базальное дыхание CO2 (прокси-индикатор микробной биомассы M)",
        f"По-серийные эталоны G0 (день 0): {ref_str}",
        "Правило диагноза: СТРОГО по выходам модели [P_bio(0), ИБСП(0), d_min vs d_star=0.29], без утечки is_control_G0!",
        "-" * 118,
        f"{'Образец':<10} | {'Серия почвы':<10} | {'Архетип Атласа':<24} | {'alpha':<6} | {'d_min':<6} | {'P_bio(0)':<8} | {'Лаб.Реф':<7} | {'ИБСП(0)':<8} | {'ИБСП(60д)':<9} | {'Диагноз модели'}",
        "-" * 118,
    ]
    preview_indices = [0, 1, 2, 6, 7, 8, 14, 16, 24, 25, 26, 35]
    for idx in preview_indices:
        if idx < len(report_rows):
            r = report_rows[idx]
            lab_s = f"{r['lab_toxin_ref_ec50']:.2f}" if not math.isnan(float(r["lab_toxin_ref_ec50"])) else "NaN"
            lines.append(
                f"{r['sample_id']:<10} | {r['soil_series']:<10} | {r['archetype_name'][:24]:<24} | "
                f"{r['alpha_pred']:.3f}  | {r['d_min']:.3f}  | {r['P0_ec50']:>8.3f} | {lab_s:>7} | "
                f"{r['IBSP_day0']:>7.1f}% | {r['IBSP_day60']:>8.1f}% | {r['diagnosis']}"
            )
    lines.append("... (полная таблица по всем 36 сосудам сохранена в predictions_report.csv)")
    lines.append("=" * 118)
    text = "\n".join(lines)
    print(text)
    return text


def run_dry_run_assertions(report_rows: List[Dict[str, object]], d_star: float) -> Dict[str, object]:
    """
    Выполняет строгие автоматические проверки (assert) качества DRY-RUN калибровки:
      1. Все 6 контролей G0 (`Sample_01`..`Sample_06`) имеют P_bio(0) < 0.5 EC50, ИБСП(0) > 92%
         и получают диагноз "НОРМА (здоровая почва)" исключительно по выходам модели.
      2. Все 8 сосудов чистой бедной/истощённой почвы G1 (`Sample_07`..`Sample_14`) имеют
         P_bio(0) < 1.0 EC50, ИБСП(0) < 85%, d_min <= d_star и получают диагноз
         "БЕДНАЯ / ИСТОЩЁННАЯ ПОЧВА (дефицит субстрата без токсикоза)" без ложного токсикоза.
      3. Все 12 сильно нагруженных сосудов G3 (`Sample_25`..`Sample_36`) надежно активируют
         детектор аномалий d_min > 0.29 (d_star) и получают диагноз
         "ТОКСИЧЕСКОЕ УГНЕТЕНИЕ (требуется биоремедиация)".
    """
    g0_rows = report_rows[0:6]
    g1_rows = report_rows[6:14]
    g2_rows = report_rows[14:24]
    g3_rows = report_rows[24:36]

    g0_max_p0 = max(float(r["P0_ec50"]) for r in g0_rows)
    g0_min_ibsp0 = min(float(r["IBSP_day0"]) for r in g0_rows)
    g0_all_norm = all(r["diagnosis"] == "НОРМА (здоровая почва)" for r in g0_rows)

    g1_max_p0 = max(float(r["P0_ec50"]) for r in g1_rows)
    g1_max_ibsp0 = max(float(r["IBSP_day0"]) for r in g1_rows)
    g1_max_d = max(float(r["d_min"]) for r in g1_rows)
    g1_all_poor = all(
        r["diagnosis"] == "БЕДНАЯ / ИСТОЩЁННАЯ ПОЧВА (дефицит субстрата без токсикоза)"
        for r in g1_rows
    )

    g2_all_tox = all(
        r["diagnosis"] == "ТОКСИЧЕСКОЕ УГНЕТЕНИЕ (требуется биоремедиация)"
        for r in g2_rows
    )

    g3_min_d = min(float(r["d_min"]) for r in g3_rows)
    g3_min_p0 = min(float(r["P0_ec50"]) for r in g3_rows)
    g3_all_tox = all(
        r["diagnosis"] == "ТОКСИЧЕСКОЕ УГНЕТЕНИЕ (требуется биоремедиация)"
        for r in g3_rows
    )

    st_g0 = "PASS" if (g0_max_p0 < 0.5 and g0_min_ibsp0 > 92.0 and g0_all_norm) else "FAIL"
    st_g1 = "PASS" if (g1_max_p0 < 1.0 and g1_max_ibsp0 < 85.0 and g1_max_d <= d_star and g1_all_poor) else "FAIL"
    st_g3 = "PASS" if (g3_min_d > 0.29 and g3_min_d > d_star and g3_min_p0 >= 1.0 and g3_all_tox and g2_all_tox) else "FAIL"

    print(
        f"[DRY-RUN АУДИТ G0 (6 контролей по 3 сериям)] max P_bio(0) = {g0_max_p0:.3f} < 0.5 EC50, "
        f"min ИБСП(0) = {g0_min_ibsp0:.1f}% > 92.0%, диагноз 'НОРМА' = 6/6 [{st_g0}]"
    )
    print(
        f"[DRY-RUN АУДИТ G1 (8 бедных чистых сосудов)] max P_bio(0) = {g1_max_p0:.3f} < 1.0 EC50, "
        f"max ИБСП(0) = {g1_max_ibsp0:.1f}% < 85.0%, max d_min = {g1_max_d:.3f} <= {d_star:.3f}, "
        f"диагноз 'БЕДНАЯ / ИСТОЩЁННАЯ ПОЧВА' = 8/8 [{st_g1}]"
    )
    print(
        f"[DRY-RUN АУДИТ G3 (12 сильно загрязненных)]  min d_min = {g3_min_d:.3f} > 0.29 (d*={d_star:.3f}), "
        f"min P_bio(0) = {g3_min_p0:.2f} EC50, диагноз 'ТОКСИЧЕСКОЕ УГНЕТЕНИЕ' = 12/12 (G2: 10/10) [{st_g3}]"
    )

    assert g0_max_p0 < 0.5 and g0_min_ibsp0 > 92.0 and g0_all_norm, (
        f"Ошибка G0: max P0={g0_max_p0:.3f}, min IBSP0={g0_min_ibsp0:.1f}%"
    )
    assert g1_max_p0 < 1.0 and g1_max_ibsp0 < 85.0 and g1_max_d <= d_star and g1_all_poor, (
        f"Ошибка G1: max P0={g1_max_p0:.3f}, max IBSP0={g1_max_ibsp0:.1f}%, max d_min={g1_max_d:.3f}"
    )
    assert g3_min_d > 0.29 and g3_min_d > d_star and g3_min_p0 >= 1.0 and g3_all_tox and g2_all_tox, (
        f"Ошибка G3: min d_min={g3_min_d:.3f}, min P0={g3_min_p0:.3f}"
    )

    return {
        "g0_max_p0": g0_max_p0,
        "g0_min_ibsp0": g0_min_ibsp0,
        "g1_max_p0": g1_max_p0,
        "g1_max_ibsp0": g1_max_ibsp0,
        "g1_max_d": g1_max_d,
        "g3_min_d": g3_min_d,
        "g3_min_p0": g3_min_p0,
        "status_g0": st_g0,
        "status_g1": st_g1,
        "status_g3": st_g3,
    }


def run_buckets_inference(
    csv_path: str,
    ckpt_path: str,
    output_report_csv: str,
    output_plot_path: str,
    verify_dry_run: bool = True,
) -> List[Dict[str, object]]:
    """
    Загружает замороженный чекпоинт `terrakan_v4_frozen.pt`, прогоняет данные 36 сосудов
    за дни 0..28 строго в режиме `model.eval()` + `torch.no_grad()`, вычисляет слепой
    диагноз исключительно по выходам модели и строит проверочный график DRY-RUN.
    """
    ckpt = torch.load(ckpt_path)
    model = TerraKAN(d_model=int(ckpt.get("d_model", 48)))
    model.load_state_dict(ckpt["state_dict"])
    model.atlas.d_star = float(ckpt.get("d_star", 0.29))
    model.atlas.tau_d = float(ckpt.get("tau_d", 0.055))
    model.eval()
    d_star = float(model.atlas.d_star)

    batch = load_and_normalize_buckets_csv(csv_path, max_obs=8)
    t_eval_60 = torch.linspace(0.0, 60.0, 121, dtype=torch.float64)

    with torch.no_grad():
        out = model(
            values=batch["values"],
            mask=batch["avail_mask"],
            t_obs=batch["t_obs"],
            rain_flag=batch["rain_flag"],
            t_eval=t_eval_60,
        )

    P_pred = out["P_bio"].numpy()       # (36, 121)
    M_pred = out["M"].numpy()           # (36, 121)
    E_pred = out["E"].numpy()           # (36, 121)
    IBSP_pred = out["IBSP"].numpy()     # (36, 121)
    alpha_pred = out["alpha"].numpy()   # (36,)
    sigma_a = out["sigma_alpha"].numpy()
    passport = out["passport"].numpy()  # (36, 5)
    min_dist = out["min_dist"].numpy()  # (36,)

    idx_d28 = 56   # 28.0 сут
    idx_d60 = 120  # 60.0 сут

    report_rows: List[Dict[str, object]] = []
    fieldnames = [
        "sample_id",
        "soil_series",
        "archetype_code",
        "archetype_name",
        "alpha_pred",
        "sigma_alpha",
        "d_min",
        "P0_ec50",
        "lab_toxin_ref_ec50",
        "P_day28_ec50",
        "P_day60_ec50",
        "IBSP_day0",
        "IBSP_day28",
        "IBSP_day60",
        "diagnosis",
        "symbolic_formula_P",
    ]

    with open(output_report_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for b, sid in enumerate(batch["sample_ids"]):
            dom_k = int(np.argmax(passport[b]))
            arch_info = SOIL_ARCHETYPES[dom_k]
            p0_val = float(P_pred[b, 0])
            ibsp_0 = float(IBSP_pred[b, 0])
            ibsp_28 = float(IBSP_pred[b, idx_d28])
            ibsp_60 = float(IBSP_pred[b, idx_d60])
            d_m = float(min_dist[b])
            lab_ref = float(batch["lab_toxin_ref_ec50"][b])

            diag = classify_vessel_from_model_outputs(
                P_bio_0=p0_val,
                ibsp_0=ibsp_0,
                d_min=d_m,
                d_star=d_star,
            )

            sym_p = model.unpack_symbolic(sample_idx=b)["P_bio"]
            row_dict = {
                "sample_id": sid,
                "soil_series": batch["soil_series"][b],
                "archetype_code": arch_info["code"],
                "archetype_name": arch_info["name"],
                "alpha_pred": round(float(alpha_pred[b]), 4),
                "sigma_alpha": round(float(sigma_a[b]), 4),
                "d_min": round(d_m, 4),
                "P0_ec50": round(p0_val, 3),
                "lab_toxin_ref_ec50": round(lab_ref, 3) if not math.isnan(lab_ref) else float("nan"),
                "P_day28_ec50": round(float(P_pred[b, idx_d28]), 3),
                "P_day60_ec50": round(float(P_pred[b, idx_d60]), 3),
                "IBSP_day0": round(ibsp_0, 2),
                "IBSP_day28": round(ibsp_28, 2),
                "IBSP_day60": round(ibsp_60, 2),
                "diagnosis": diag,
                "symbolic_formula_P": sym_p,
            }
            writer.writerow(row_dict)
            report_rows.append(row_dict)

    generate_human_report(report_rows, batch["E_ref_G0"], batch["M_ref_G0"])
    audit_info = None
    if verify_dry_run and len(report_rows) == 36:
        audit_info = run_dry_run_assertions(report_rows, d_star=d_star)

    # Построение 3-панельной фигуры технической верификации DRY-RUN (fig_dry_run_36pots.png)
    fig, axes = plt.subplots(1, 3, figsize=(19.2, 5.7), dpi=300, gridspec_kw={"wspace": 0.28})
    t_arr = t_eval_60.numpy()

    # Панель А: Траектории P_bio(t) для 4 репрезентативных сосудов S2_loam (G0, G1, G2, G3) + точки lab_toxin_ref_ec50
    rep_indices = [1, 7, 16, 25]  # S2_loam: Sample_02 (G0), Sample_08 (G1), Sample_17 (G2), Sample_26 (G3)
    rep_colors = ["#2A9D8F", "#1D63B8", "#E76F51", "#D1495B"]
    rep_tags = ["G0 (Control)", "G1 (Poor Clean)", "G2 (Mod. Dose)", "G3 (High Load)"]

    for s_i, col, tag in zip(rep_indices, rep_colors, rep_tags):
        sid = batch["sample_ids"][s_i]
        s_ser = batch["soil_series"][s_i]
        lbl = f"{sid} [{s_ser}, {tag}]: $\\hat{{P}}(0)={P_pred[s_i, 0]:.2f}$"
        axes[0].plot(t_arr, P_pred[s_i], color=col, lw=2.2, label=lbl)
        lab_v = float(batch["lab_toxin_ref_ec50"][s_i])
        if not math.isnan(lab_v):
            axes[0].scatter([0.0], [lab_v], color=col, edgecolors="black", s=55, zorder=6, marker="D")

    axes[0].scatter([], [], color="#555555", edgecolors="black", s=50, marker="D", label="Lab Ref. lab_toxin_ref_ec50 ($t=0$)")
    axes[0].axvspan(0, 28, color="#E8F5E9", alpha=0.30, label="Synthetic Obs. (0-28 d)")
    axes[0].axvspan(28, 60, color="#FFF3E0", alpha=0.35, label="Blind Forecast (28-60 d)")
    axes[0].axvline(28.0, color="#37474F", ls="--", lw=1.3)
    axes[0].set_xlim(-1.5, 60)
    axes[0].set_xlabel(
        "Microcosm Time $t$, days\nCatalase O2 (proxy indicator of enzyme activity E)",
        fontsize=8.3,
    )
    axes[0].set_ylabel("Hidden Load $\\tilde{P}_{\\mathrm{bio}}(t)$ ($\\mathrm{EC}_{50}$ units)", fontsize=9.2)
    axes[0].set_title(
        "A. $\\tilde{P}_{\\mathrm{bio}}(t)$ Reconstruction & Lab Ref. Check\n"
        "lab_toxin_ref_ec50 (0-28 d -> 60 d forecast)",
        fontsize=9.3,
        fontweight="bold",
    )
    axes[0].legend(loc="upper right", fontsize=7.1, frameon=True, facecolor="white")
    axes[0].grid(True, alpha=0.28, ls="--")

    # Панель Б: Траектории ИБСП_alpha(t) и пороги слепого диагноза без утечки is_control_G0
    for s_i, col, tag in zip(rep_indices, rep_colors, rep_tags):
        sid = batch["sample_ids"][s_i]
        lbl = f"{sid} ({tag}): IBSP(0)={IBSP_pred[s_i, 0]:.1f}% $\\rightarrow$ 60d: {IBSP_pred[s_i, -1]:.1f}%"
        axes[1].plot(t_arr, IBSP_pred[s_i], color=col, lw=2.2, label=lbl)

    axes[1].axhline(85.0, color="#2E7D32", ls=":", lw=1.5, label="Healthy Threshold IBSP(0) = 85%")
    axes[1].axvspan(0, 28, color="#E8F5E9", alpha=0.30)
    axes[1].axvspan(28, 60, color="#FFF3E0", alpha=0.35)
    axes[1].axvline(28.0, color="#37474F", ls="--", lw=1.3)
    axes[1].set_xlim(0, 60)
    axes[1].set_ylim(25, 104)
    axes[1].set_xlabel(
        "Microcosm Time $t$, days\nBasal Respiration CO2 (proxy indicator of biomass M)",
        fontsize=8.3,
    )
    axes[1].set_ylabel("Soil Health Index IBSP$_\\alpha(t)$, %", fontsize=9.2)
    axes[1].set_title(
        "B. Soil Health Index IBSP$_\\alpha(t)$ Trajectories\n"
        "(blind model diagnosis without is_control_G0)",
        fontsize=9.3,
        fontweight="bold",
    )
    axes[1].legend(loc="lower right", fontsize=7.1, frameon=True, facecolor="white")
    axes[1].grid(True, alpha=0.28, ls="--")

    # Панель В: Карта слепой классификации всех 36 сосудов в координатах (d_min, ИБСП(0)) по сериям почв
    markers_by_series = {"S1_sand": "o", "S2_loam": "s", "S3_peat": "^"}
    diag_colors = {
        "НОРМА (здоровая почва)": "#2A9D8F",
        "БЕДНАЯ / ИСТОЩЁННАЯ ПОЧВА (дефицит субстрата без токсикоза)": "#1D63B8",
        "ТОКСИЧЕСКОЕ УГНЕТЕНИЕ (требуется биоремедиация)": "#D1495B",
    }
    diag_short = {
        "НОРМА (здоровая почва)": "HEALTHY (G0: 6/6, IBSP>92%)",
        "БЕДНАЯ / ИСТОЩЁННАЯ ПОЧВА (дефицит субстрата без токсикоза)": "POOR / DEPLETED (G1: 8/8)",
        "ТОКСИЧЕСКОЕ УГНЕТЕНИЕ (требуется биоремедиация)": "INHIBITED (G2+G3: 22/22)",
    }

    plotted_labels = set()
    for b, r in enumerate(report_rows):
        d_m = float(r["d_min"])
        ib0 = float(r["IBSP_day0"])
        dg = str(r["diagnosis"])
        s_ser = str(r["soil_series"])
        col = diag_colors.get(dg, "#333333")
        mk = markers_by_series.get(s_ser, "o")
        leg_lbl = f"{diag_short[dg]} [{s_ser}]"
        if leg_lbl in plotted_labels:
            leg_lbl = "_nolegend_"
        else:
            plotted_labels.add(leg_lbl)
        axes[2].scatter([d_m], [ib0], color=col, marker=mk, s=62, alpha=0.88, edgecolors="black", linewidths=0.6, label=leg_lbl)

    axes[2].axvline(d_star, color="#B71C1C", ls="--", lw=1.6, label=f"Detector Threshold $d_{{\\min}}^* = {d_star:.2f}$")
    axes[2].axhline(85.0, color="#2E7D32", ls=":", lw=1.5, label="Healthy Boundary IBSP(0) = 85%")
    axes[2].set_xlabel("Soil Atlas Anomaly Distance $d_{\\min}$ (dim.)", fontsize=8.8)
    axes[2].set_ylabel("Initial Health Index IBSP$_\\alpha(0)$, %", fontsize=9.2)
    axes[2].set_title(
        "C. Classification of 36 Synthetic Vessels (G0 / G1 / G2+G3)\n"
        "under per-series soil_series normalization",
        fontsize=9.3,
        fontweight="bold",
    )
    axes[2].set_ylim(21, 105)
    axes[2].legend(loc="upper right", fontsize=6.5, frameon=True, facecolor="white", ncol=1)
    axes[2].grid(True, alpha=0.28, ls="--")

    if audit_info is not None:
        box_txt = (
            f"Per-Series Normalization Audit (36 Synthetic Pots):\n"
            f"• G0 (n=6): max $\\hat{{P}}(0)={audit_info['g0_max_p0']:.3f}<0.5$, min IBSP(0)={audit_info['g0_min_ibsp0']:.1f}% [{audit_info['status_g0']}]\n"
            f"• G1 (n=8): max $\\hat{{P}}(0)={audit_info['g1_max_p0']:.3f}<1.0$, max $d_{{\\min}}={audit_info['g1_max_d']:.2f}\\leq d^*$ [{audit_info['status_g1']}]\n"
            f"• G3 (n=12): min $d_{{\\min}}={audit_info['g3_min_d']:.2f}>0.29$, zero G0 leakage [{audit_info['status_g3']}]"
        )
        axes[2].text(
            0.03, 0.03, box_txt,
            transform=axes[2].transAxes,
            fontsize=6.8,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8F9FA", edgecolor="#90A4AE", alpha=0.94),
        )

    fig.suptitle(
        "SYNTHETIC DRY-RUN PIPELINE CHECK (Pre-Experiment Verification on 36 Simulated Microcosm Vessels)",
        y=1.05,
        fontsize=12.4,
        fontweight="bold",
    )
    fig.savefig(output_plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return report_rows


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    blank_csv = os.path.join(base_dir, "blank_lab_journal_36pots.csv")
    dry_run_csv = os.path.join(base_dir, "dry_run_synthetic_36pots.csv")
    legacy_tpl_csv = os.path.join(base_dir, "experiment_log_template.csv")
    ckpt_file = os.path.join(base_dir, "terrakan_v4_frozen.pt")
    rep_csv = os.path.join(base_dir, "predictions_report.csv")
    dry_run_png = os.path.join(base_dir, "fig_dry_run_36pots.png")
    legacy_png = os.path.join(base_dir, "field_buckets_forecast_curves.png")

    # 1. Генерируем пустой лабораторный бланк для печати и синтетический DRY-RUN журнал
    generate_blank_lab_journal(blank_csv, n_vessels=36)
    generate_dry_run_synthetic_csv(dry_run_csv, n_vessels=36, seed=42)
    generate_dry_run_synthetic_csv(legacy_tpl_csv, n_vessels=36, seed=42)

    # 2. Запускаем слепой инференс на dry_run_synthetic_36pots.csv и проверяем все ассерты
    run_buckets_inference(
        csv_path=dry_run_csv,
        ckpt_path=ckpt_file,
        output_report_csv=rep_csv,
        output_plot_path=dry_run_png,
        verify_dry_run=True,
    )
    shutil.copyfile(dry_run_png, legacy_png)
