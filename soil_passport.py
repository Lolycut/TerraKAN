"""
TerraKAN v1.5.2 — Генератор детального паспорта почвы (Soil Passport v1.5.2).

Исправления v1.5.2:
  - Приоритет проверки стойкого загрязнения (pi_persist >= 0.60 и P0_med >= 0.20)
    вынесен ДО порога is_clean, чтобы тяжёлые металлы (Cu2+, Cr(VI), Pb) гарантированно
    получали статус СТОЙКОЕ НЕРАЗЛАГАЕМОЕ ЗАГРЯЗНЕНИЕ.
  - pH_n зажат в [0.0, 1.0], чтобы pH_estimated всегда оставался в физическом диапазоне [3.8, 8.8].
"""

from typing import Dict
import numpy as np

from physics_solver_v150 import SOIL_ARCHETYPES


TEXTURE_HINTS = {
    0: "Суглинистая / дерново-подзолистая (умеренная сорбция)",
    1: "Органогенная / торфяно-болотная (высокий TOC, сильное удержание)",
    2: "Карбонатный суглинок / чернозёмовидная (высокая биогенность)",
    3: "Засолённый солончак / щёлочной суглинок (осмотический стресс)",
    4: "Песчаная / супесчаная олиготрофная (быстрый дренаж, низкий K_m)",
}


def build_detailed_soil_passport(
    sample_name: str,
    passport_probs: np.ndarray,
    detailed_vec: np.ndarray,
    P0_med: float,
    deg_med: float,
    deg_q10: float,
    deg_q90: float,
    pi_persist: float,
    ibsp_start: float,
    ibsp_end: float,
    d_min: float,
    d_star: float,
    hormesis_flag: bool = False,
) -> Dict[str, object]:
    probs = np.asarray(passport_probs, dtype=np.float64)
    dom_idx = int(np.argmax(probs))
    dom_arch = SOIL_ARCHETYPES[dom_idx]

    pH_n, NO3_n, W_n, Km, alpha, buf_idx = [float(x) for x in detailed_vec]
    pH_real = round(3.8 + 5.0 * float(np.clip(pH_n, 0.0, 1.0)), 2)

    if alpha < 0.60:
        memory_class = "Глубокая микропоровая ловушка (сильная немарковость, высокий риск хвостов)"
    elif alpha < 0.78:
        memory_class = "Умеренная субдиффузия (стандартный органо-минеральный профиль)"
    else:
        memory_class = "Быстрый дренаж / слабая память (близко к марковской кинетике)"

    if buf_idx >= 0.65:
        buf_tier = "Высокая буферность"
    elif buf_idx >= 0.35:
        buf_tier = "Средняя буферность"
    else:
        buf_tier = "Низкая буферность (уязвима к сдвигу pH/токсичности)"

    # Сначала проверяем стойкое неразлагаемое загрязнение (металлы/соли/PFAS), затем чистый фон
    if pi_persist >= 0.60 and P0_med >= 0.20:
        status = "СТОЙКОЕ НЕРАЗЛАГАЕМОЕ ЗАГРЯЗНЕНИЕ (металлы / соли / PFAS)"
        rec = "Биоразложение неэффективно: рекомендована химическая иммобилизация (известь, биочар, сорбенты) или фитостабилизация."
    elif (P0_med < 0.05) or (d_min < 0.72 * d_star and P0_med < 0.30):
        status = "ЧИСТАЯ ПОЧВА (фон / норма)"
        rec = "Ремедиация не требуется; плановый мониторинг."
    elif hormesis_flag:
        status = "БИОРАЗЛАГАЕМЫЙ СТРЕСС С ГОРМЕТИЧЕСКОЙ СТИМУЛЯЦИЕЙ"
        rec = "Активный микробный отклик: поддерживать влажность и аэрацию, контролировать вторичный спад ферментов."
    elif deg_med >= 65.0:
        status = "АКТИВНОЕ САМООЧИЩЕНИЕ / БИОСТИМУЛЯЦИЯ"
        rec = "Высокий темп детоксикации: достаточно мониторинга контрольных точек."
    else:
        status = "УМЕРЕННОЕ / ХРОНИЧЕСКОЕ ЗАГРЯЗНЕНИЕ"
        rec = "Рекомендована биостимуляция (NPK, ПАВ) или инокуляция профильным консорциумом."

    return {
        "sample_id": sample_name,
        "diagnostic_status": status,
        "dominant_archetype": f"{dom_arch['code']} ({dom_arch['name']})",
        "archetype_mixture_pct": {
            SOIL_ARCHETYPES[i]["code"]: round(float(probs[i]) * 100.0, 1) for i in range(5)
        },
        "physicochemical_profile": {
            "pH_estimated": pH_real,
            "NO3_index": round(NO3_n, 3),
            "moisture_index": round(W_n, 3),
            "carrying_capacity_Km": round(Km, 3),
            "fractional_memory_alpha": round(alpha, 3),
            "buffering_index": round(buf_idx, 3),
            "buffering_tier": buf_tier,
            "inferred_texture_class": TEXTURE_HINTS[dom_idx],
            "micropore_memory_class": memory_class,
        },
        "pollution_and_forecast": {
            "P0_ec50_median": round(P0_med, 3),
            "persistence_fraction_pi": round(pi_persist, 3),
            "degradation_median_pct": round(deg_med, 1),
            "degradation_q80_interval_pct": [round(deg_q10, 1), round(deg_q90, 1)],
            "IBSP_start_pct": round(ibsp_start, 1),
            "IBSP_end_pct": round(ibsp_end, 1),
            "anomaly_score_d_min": round(d_min, 3),
            "threshold_d_star": round(d_star, 3),
        },
        "remediation_recommendation": rec,
    }
