"""
TerraKAN v1.5.1 — Полная архитектура без Ridge-костыля (с фиксами по итогам 5-seed прогона v1.5.0).

Ключевые исправления v1.5.1 относительно v1.5.0:
  1. Устранён провал R^2(IBSP) в минус: M_pos и E_pos используют гладкий softplus + 0.05
     вместо clamp(1e-4) (исключён взрыв ln(1e-4) = -9.21) + добавлен прямой H8-лосс на IBSP/100.
  2. Убран нецентрированный фон latent_vel из bi_shock: порог d_star возвращён в чистый диапазон
     ~0.40-0.46 для надёжной детекции биостимуляции/биоаугментации при 0/5 ложных тревог на контроле.
  3. Добавлена прямая BCE-супервизия логита pi_logit на загрязнённых сценариях + предел
     gamma_h <= 2.40 (достаточно для 0-2% ошибки на Cu2+/металлах без v1.3.1-взрыва 4.28).
"""

import math
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pade_ml_v150 import MittagLefflerPade34
from physics_solver_v150 import get_atlas_centroid_matrix


QUANTILE_LEVELS = (0.10, 0.25, 0.50, 0.75, 0.90)


class SubordinatedBioClock(nn.Module):
    """H1 (PlatinumQuant-4.1): субординированное биологическое время tau(t)."""

    def __init__(self, d_model: int = 64) -> None:
        super().__init__()
        self.clock_w = nn.Linear(3, 1)
        nn.init.zeros_(self.clock_w.weight)
        nn.init.constant_(self.clock_w.bias, 0.5413)
        self.freq = nn.Parameter(torch.linspace(0.1, 2.0, d_model // 2))
        self.proj = nn.Linear(d_model, d_model)
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        t_obs: torch.Tensor,
        W_obs: torch.Tensor,
        rain_flag: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dt = torch.zeros_like(t_obs)
        dt[:, 1:] = torch.clamp(t_obs[:, 1:] - t_obs[:, :-1], min=0.0)
        clock_feat = torch.stack([torch.log1p(dt), W_obs, rain_flag.squeeze(-1)], dim=-1)
        rate = F.softplus(self.clock_w(clock_feat)).squeeze(-1)
        tau = torch.cumsum(dt * rate, dim=-1)

        angles = torch.log1p(tau).unsqueeze(-1) * self.freq.view(1, 1, -1)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return tau, torch.tanh(self.gate) * self.proj(emb)


class AbelNoPEBlock(nn.Module):
    """Причинный блок внимания с тяжелым степенным ядром Абеля и без RoPE."""

    def __init__(self, d_model: int = 64, n_heads: int = 4, dropout: float = 0.05) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.raw_gamma = nn.Parameter(torch.full((n_heads,), 0.75))

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(
        self,
        h: torch.Tensor,
        tau: torch.Tensor,
        alpha_seq: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, N, D = h.shape
        x = self.norm1(h)
        qkv = self.qkv(x).view(B, N, 3, self.n_heads, self.head_dim)
        q, k, v = qkv[:, :, 0].transpose(1, 2), qkv[:, :, 1].transpose(1, 2), qkv[:, :, 2].transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        dtau = torch.clamp(tau.unsqueeze(2) - tau.unsqueeze(1), min=0.0)
        gamma_h = F.softplus(self.raw_gamma).view(1, self.n_heads, 1, 1)
        one_minus_a = (1.0 - alpha_seq).unsqueeze(1).unsqueeze(-1)
        scores = scores - gamma_h * one_minus_a * torch.log1p(dtau).unsqueeze(1)

        mask_4d = causal_mask.unsqueeze(1)
        scores = torch.where(mask_4d, scores, torch.full_like(scores, -1e9))
        attn = torch.softmax(scores, dim=-1) * mask_4d.to(scores.dtype)
        attn = attn / torch.clamp(attn.sum(dim=-1, keepdim=True), min=1e-12)

        ctx = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, D)
        h = h + self.out_proj(ctx)
        h = h + self.ffn(self.norm2(h))
        return h


class LoopedAbelNoPEEncoder(nn.Module):
    """2 физических слоя x R=2 рекуррентных прохода (эффективная глубина 4) с loop_embeds."""

    def __init__(self, d_in: int = 12, d_model: int = 64, n_heads: int = 4, n_layers: int = 2, n_loops: int = 2) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_loops = n_loops
        self.input_proj = nn.Linear(d_in, d_model)
        self.in_norm = nn.LayerNorm(d_model)
        self.bio_clock = SubordinatedBioClock(d_model=d_model)
        self.alpha_head = nn.Linear(d_model, 1)
        self.blocks = nn.ModuleList([AbelNoPEBlock(d_model, n_heads) for _ in range(n_layers)])
        self.loop_embeds = nn.Parameter(torch.zeros(n_loops, 1, 1, d_model))
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        t_obs: torch.Tensor,
        rain_flag: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        B, N, _ = values.shape
        masked_vals = values * mask
        v0 = values[:, 0:1, :] * mask[:, 0:1, :]
        delta_v0 = (masked_vals - v0) * mask
        log_t = torch.log1p(torch.clamp(t_obs, min=0.0)).unsqueeze(-1)
        x_in = torch.cat([masked_vals + 0.35 * delta_v0, mask, log_t, rain_flag], dim=-1)

        tau, tau_emb = self.bio_clock(t_obs, values[:, :, 2] * mask[:, :, 2], rain_flag)
        h = self.in_norm(F.gelu(self.input_proj(x_in)) + tau_emb)

        token_valid = (mask.sum(dim=-1) > 0.5)
        idx = torch.arange(N, device=values.device).unsqueeze(0)
        idx_causal = idx.unsqueeze(2) >= idx.unsqueeze(1)
        time_causal = (t_obs.unsqueeze(2) >= t_obs.unsqueeze(1) - 1e-9) & idx_causal
        causal_mask = time_causal & token_valid.unsqueeze(1)

        cw = causal_mask.to(h.dtype)
        prefix_mean = torch.matmul(cw, h) / torch.clamp(cw.sum(dim=-1, keepdim=True), min=1.0)
        alpha_seq = 0.38 + 0.57 * torch.sigmoid(self.alpha_head(prefix_mean).squeeze(-1))

        for r in range(self.n_loops):
            h = h + self.loop_embeds[r]
            for blk in self.blocks:
                h = blk(h, tau, alpha_seq, causal_mask)

        h = self.final_norm(h)
        valid_last = torch.clamp(token_valid.to(torch.int64).sum(dim=-1) - 1, min=0)
        b_idx = torch.arange(B, device=values.device)
        return {
            "h_seq": h,
            "h_final": h[b_idx, valid_last],
            "alpha": alpha_seq[b_idx, valid_last],
            "tau": tau,
        }


class MultKANEnvModuleV150(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        init_w = np.array([[0.01, 0.34, 0.39], [-0.01, 0.34, 0.39]], dtype=np.float64)
        self.W_power = nn.Parameter(torch.tensor(init_w))
        self.pH_opt = nn.Parameter(torch.tensor([0.65]))
        self.beta_pH = nn.Parameter(torch.tensor([2.45]))
        self.branch_weights = nn.Parameter(torch.tensor([0.60, 0.40]))

    def forward(self, env_3: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        env_safe = torch.clamp(env_3, min=0.02, max=2.0)
        pH = env_safe[:, 0]
        exps = torch.clamp(torch.matmul(torch.log(env_safe), self.W_power.T), min=-10.0, max=10.0)
        bw = torch.softmax(self.branch_weights, dim=-1)
        mult_term = torch.matmul(torch.exp(exps), bw)
        shelford = torch.exp(-F.softplus(self.beta_pH) * ((pH - self.pH_opt) ** 2))
        return shelford * mult_term, torch.abs(self.W_power).mean()


class SoilPassportV150(nn.Module):
    """
    Расширенный модуль паспортизации почвы + чистый двусторонний детектор аномалий (ингибирование + гормезис).
    """

    def __init__(self, d_model: int = 64) -> None:
        super().__init__()
        self.register_buffer("centroids", torch.tensor(get_atlas_centroid_matrix()))
        self.feature_extractor = nn.Linear(d_model, 4)
        self.log_temp = nn.Parameter(torch.tensor([3.2]))
        self.d_star = 0.42

    def forward(
        self,
        h_final: torch.Tensor,
        alpha: torch.Tensor,
        obs_env_4: torch.Tensor,
        P0_boundary: torch.Tensor,
        hormesis_score: torch.Tensor,
        drop_severity: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        delta_feat = self.feature_extractor(h_final)
        pred_4 = torch.clamp(obs_env_4 + delta_feat, min=0.05, max=2.2)
        soil_vec = torch.cat([pred_4, alpha.unsqueeze(-1)], dim=-1)

        diff = soil_vec.unsqueeze(1) - self.centroids.to(soil_vec.dtype).unsqueeze(0)
        feat_w = torch.tensor([7.5, 7.5, 5.0, 2.5, 14.0], device=soil_vec.device, dtype=soil_vec.dtype).view(1, 1, 5)
        dist_sq = (diff * diff * feat_w).sum(dim=-1)
        raw_min_dist = torch.sqrt(dist_sq.min(dim=-1).values / 10.0 + 1e-8)

        # Чистый двусторонний скор без нецентрированного шума latent_vel
        bi_shock = torch.tanh(0.42 * P0_boundary + 0.50 * hormesis_score + 0.80 * drop_severity)
        min_dist = 0.24 * raw_min_dist + 0.76 * bi_shock

        temp = F.softplus(self.log_temp) + 1.0
        logits = -temp * dist_sq
        passport = torch.softmax(logits, dim=-1)
        log_passport = F.log_softmax(logits, dim=-1)

        pH_est, NO3_est, W_est, Km_est = pred_4[:, 0], pred_4[:, 1], pred_4[:, 2], pred_4[:, 3]
        buffering_idx = torch.exp(-2.0 * ((pH_est - 0.65) ** 2)) * torch.clamp(Km_est / 1.38, 0.0, 1.0)

        return {
            "passport": passport,
            "log_passport": log_passport,
            "logits": logits,
            "min_dist": min_dist,
            "soil_vec": soil_vec,
            "K_m_est": Km_est,
            "buffering_index": buffering_idx,
            "detailed_passport_vec": torch.stack([pH_est, NO3_est, W_est, Km_est, alpha, buffering_idx], dim=-1),
        }


class PhysicsExpertBankV150(nn.Module):
    """6-экспертный Sparse Top-2 MoE (включая Эксперт 5: Быстрая биоаугментация / гормезис)."""

    def __init__(self, d_model: int = 64, n_experts: int = 6) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.router_temp = 1.0
        self.router = nn.Linear(d_model + 8, n_experts)
        self.experts = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_experts)])

    def forward(
        self,
        h_final: torch.Tensor,
        phi_env: torch.Tensor,
        dyn_feat: torch.Tensor,
        class_hint: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        router_in = torch.cat([h_final, phi_env.unsqueeze(-1), dyn_feat, class_hint.unsqueeze(-1)], dim=-1)
        raw_logits = self.router(router_in)

        pH_v, dE, P0_b, horm, p_cue = dyn_feat[:, 0], dyn_feat[:, 1], dyn_feat[:, 3], dyn_feat[:, 4], dyn_feat[:, 5]
        b_hald = 1.6 * torch.sigmoid(2.0 * (P0_b - 1.45)) + 1.0 * (1.0 - dE)
        b_monod = 1.6 * torch.exp(-2.8 * ((P0_b - 0.55) ** 2))
        b_clean = 2.0 * torch.exp(-10.0 * (P0_b ** 2 + horm ** 2)) + 0.5 * dE
        b_acid = 1.8 * torch.sigmoid(10.0 * (0.44 - pH_v))
        b_recalc = 1.5 * p_cue + 1.5 * class_hint * torch.sigmoid(3.0 * (P0_b - 0.15))
        b_bioaug = 1.6 * torch.tanh(horm) - 0.9 * class_hint * torch.sigmoid(3.0 * (P0_b - 0.15))

        prior_logits = torch.stack([b_hald, b_monod, b_clean, b_acid, b_recalc, b_bioaug], dim=-1)
        tau = max(0.6, float(self.router_temp))
        gate_probs = torch.softmax((raw_logits + prior_logits) / tau, dim=-1)

        top2 = torch.topk(gate_probs, k=2, dim=-1)
        sparse_mask = torch.zeros_like(gate_probs).scatter(-1, top2.indices, 1.0)
        eff_w = gate_probs * sparse_mask
        eff_w = eff_w / torch.clamp(eff_w.sum(dim=-1, keepdim=True), min=1e-8)

        f_e = sparse_mask.mean(dim=0).detach()
        lb_loss = float(self.n_experts) * (f_e * gate_probs.mean(dim=0)).sum()

        exp_outs = torch.stack([F.gelu(layer(h_final)) for layer in self.experts], dim=1)
        h_phys = (exp_outs * eff_w.unsqueeze(-1)).sum(dim=1)
        return {
            "h_physics": h_final + h_phys,
            "expert_weights": eff_w,
            "gate_probs": gate_probs,
            "load_balance_loss": lb_loss,
        }


class UnboundedMLTrunkV150(nn.Module):
    """Спектральный генератор v1.5.1 с лог-шкалой lambda_k, ограниченным gamma_h <= 2.4 и H3-квантилями."""

    def __init__(self, d_model: int = 64) -> None:
        super().__init__()
        self.ml_kernel = MittagLefflerPade34()
        self.param_head = nn.Linear(d_model, 22)
        self.persist_head = nn.Sequential(
            nn.Linear(d_model + 4, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        # Ограниченный вес подсказки: gamma_h = 2.40 * sigmoid(raw_gamma_h) in (0, 2.40)
        self.raw_gamma_h = nn.Parameter(torch.tensor(0.5))
        self.q_head_P = nn.Linear(d_model + 2, 4)
        self.q_head_IBSP = nn.Linear(d_model + 2, 4)

    def get_gamma_h(self) -> torch.Tensor:
        return 2.40 * torch.sigmoid(self.raw_gamma_h)

    def forward(
        self,
        h_ctx: torch.Tensor,
        alpha: torch.Tensor,
        t_eval: torch.Tensor,
        obs_summary: torch.Tensor,
        phi_env: torch.Tensor,
        u_persist: torch.Tensor,
        class_hint: torch.Tensor,
        gate_d: torch.Tensor,
        n_obs_norm: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        B = h_ctx.shape[0]
        raw = self.param_head(h_ctx)

        log_lam1 = -5.5 + 4.3 * torch.sigmoid(raw[:, 0])
        lam1 = torch.exp(log_lam1)
        lam2 = lam1 + torch.exp(-4.0 + 3.0 * torch.sigmoid(raw[:, 1]))
        lam3 = lam2 + torch.exp(-3.4 + 2.8 * torch.sigmoid(raw[:, 2]))
        lambdas = torch.stack([lam1, lam2, lam3], dim=-1)

        cP_raw = F.softplus(raw[:, 3:6], beta=3.0)
        P_tot = cP_raw.sum(dim=-1) * gate_d

        gamma_h = self.get_gamma_h()
        pi_logit = self.persist_head(torch.cat([h_ctx, u_persist], dim=-1)).squeeze(-1) + gamma_h * class_hint
        pi_persist = torch.sigmoid(pi_logit)

        cP_weights = cP_raw / torch.clamp(cP_raw.sum(dim=-1, keepdim=True), min=1e-6)
        P_inf = pi_persist * P_tot
        cP_eff = cP_weights * ((1.0 - pi_persist) * P_tot).unsqueeze(-1)

        M_max_obs = obs_summary[:, 4]
        M_inf = torch.clamp(M_max_obs + raw[:, 6], min=0.15, max=2.8)
        cM = torch.tanh(raw[:, 7:10] * 0.5) * 2.0  # защита от нефизических выбросов амплитуд
        E_inf = torch.clamp(M_inf * 1.18 + raw[:, 10], min=0.15, max=3.4)
        cE = torch.tanh(raw[:, 11:14] * 0.5) * 2.5

        k_p_eff = torch.exp(-6.5 + 4.0 * torch.sigmoid(raw[:, 14]))
        k_deg_eff = torch.exp(-5.0 + 4.5 * torch.sigmoid(raw[:, 15])) * phi_env
        r_m_eff = (0.035 + 0.085 * torch.sigmoid(raw[:, 16])) * phi_env
        beta_e_eff = 0.060 + 0.065 * torch.sigmoid(raw[:, 17])
        gamma_p_eff = 0.008 + 0.022 * torch.sigmoid(raw[:, 18])
        mu_sub_eff = 0.010 + 0.055 * torch.sigmoid(raw[:, 19])
        K_s_eff = 0.70 + 0.55 * torch.sigmoid(raw[:, 20])
        delta_e_eff = 0.060 + 0.035 * torch.sigmoid(raw[:, 21])

        if t_eval.ndim == 1:
            t_b = t_eval.unsqueeze(0).expand(B, t_eval.shape[0])
        else:
            t_b = t_eval
        t_pos = torch.clamp(t_b, min=0.0)
        t_pow = torch.where(t_pos > 0.0, t_pos ** alpha.unsqueeze(-1), torch.zeros_like(t_pos))

        modes = [self.ml_kernel(lambdas[:, k : k + 1] * t_pow, alpha) for k in range(3)]
        basis = torch.stack(modes, dim=-1)

        P_deg_t = (basis * cP_eff.unsqueeze(1)).sum(dim=-1)
        P_tot_t = P_inf.unsqueeze(1) + P_deg_t
        M_pred = F.softplus(M_inf.unsqueeze(1) + (basis * cM.unsqueeze(1)).sum(dim=-1) - 0.05, beta=5.0) + 0.05
        E_pred = F.softplus(E_inf.unsqueeze(1) + (basis * cE.unsqueeze(1)).sum(dim=-1) - 0.05, beta=5.0) + 0.05

        D_alpha_P = -(basis * (lambdas * cP_eff).unsqueeze(1)).sum(dim=-1)
        D_alpha_M = -(basis * (lambdas * cM).unsqueeze(1)).sum(dim=-1)
        D_alpha_E = -(basis * (lambdas * cE).unsqueeze(1)).sum(dim=-1)

        q_in = torch.cat([h_ctx, n_obs_norm.unsqueeze(-1), pi_persist.unsqueeze(-1)], dim=-1)
        dP = F.softplus(self.q_head_P(q_in)) + 1e-3
        t_scale = 0.6 + 0.4 * torch.clamp(t_pos / 150.0, 0.0, 2.5)
        P_q25 = P_tot_t - dP[:, 0:1] * t_scale
        P_q10 = P_q25 - dP[:, 1:2] * t_scale
        P_q75 = P_tot_t + dP[:, 2:3] * t_scale
        P_q90 = P_q75 + dP[:, 3:4] * t_scale
        P_quantiles = torch.stack([P_q10, P_q25, P_tot_t, P_q75, P_q90], dim=-1)

        dI = (F.softplus(self.q_head_IBSP(q_in)) + 1e-3) * 8.0

        return {
            "P_bio": P_tot_t,
            "P_deg": P_deg_t,
            "P_inf": P_inf,
            "pi_logit": pi_logit,
            "pi_persist": pi_persist,
            "gamma_h": gamma_h,
            "M": M_pred,
            "E": E_pred,
            "D_alpha_P": D_alpha_P,
            "D_alpha_M": D_alpha_M,
            "D_alpha_E": D_alpha_E,
            "P_quantiles": P_quantiles,
            "dI_quantiles": dI,
            "t_scale": t_scale,
            "decoded": {
                "lambdas": lambdas,
                "cP_eff": cP_eff,
                "P_inf": P_inf,
                "M_inf": M_inf,
                "E_inf": E_inf,
                "k_p_eff": k_p_eff,
                "k_deg_eff": k_deg_eff,
                "r_m_eff": r_m_eff,
                "beta_e_eff": beta_e_eff,
                "gamma_p_eff": gamma_p_eff,
                "mu_sub_eff": mu_sub_eff,
                "K_s_eff": K_s_eff,
                "delta_e_eff": delta_e_eff,
            },
        }


class TerraKANv150(nn.Module):
    """TerraKAN v1.5.1 (~102K параметров, 100% сквозное обучение без Ridge)."""

    def __init__(self, d_model: int = 64, n_heads: int = 4, n_layers: int = 2, n_loops: int = 2) -> None:
        super().__init__()
        self.encoder = LoopedAbelNoPEEncoder(d_in=12, d_model=d_model, n_heads=n_heads, n_layers=n_layers, n_loops=n_loops)
        self.atlas = SoilPassportV150(d_model=d_model)
        self.multkan = MultKANEnvModuleV150()
        self.expert_bank = PhysicsExpertBankV150(d_model=d_model, n_experts=6)
        self.trunk = UnboundedMLTrunkV150(d_model=d_model)

    @staticmethod
    def _extract_features(values: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, N, _ = values.shape
        masked = values * mask
        counts = torch.clamp(mask.sum(dim=1), min=1.0)
        means = masked.sum(dim=1) / counts

        pH_m, NO3_m, W_m = means[:, 0], means[:, 1], means[:, 2]
        m_valid = mask[:, :, 3] > 0.5
        e_valid = mask[:, :, 4] > 0.5

        M0 = torch.where(m_valid[:, 0], values[:, 0, 3], means[:, 3])
        E0 = torch.where(e_valid[:, 0], values[:, 0, 4], means[:, 4])
        M_max = torch.clamp((values[:, :, 3] * mask[:, :, 3]).max(dim=1).values, min=0.05)
        E_max = torch.clamp((values[:, :, 4] * mask[:, :, 4]).max(dim=1).values, min=0.05)

        M_min = torch.where(m_valid, values[:, :, 3], torch.full_like(values[:, :, 3], 99.0)).min(dim=1).values
        E_min = torch.where(e_valid, values[:, :, 4], torch.full_like(values[:, :, 4], 99.0)).min(dim=1).values
        M_min = torch.where(M_min > 50.0, M_max, M_min)
        E_min = torch.where(E_min > 50.0, E_max, E_min)

        dM_drop = torch.clamp(M_min / M_max, 0.05, 1.0)
        dE_drop = torch.clamp(E_min / E_max, 0.05, 1.0)
        drop_sev = torch.clamp(0.86 - torch.minimum(dM_drop, dE_drop), 0.0, 0.80)

        ratio_EM = torch.clamp(E0 / torch.clamp(M0, min=0.05), 0.05, 3.0)
        p_ratio = torch.sqrt(torch.clamp(1.20 / ratio_EM - 1.0, 0.0, 25.0))
        p_M_inh = 0.55 * torch.sqrt(torch.clamp(1.0 / torch.clamp(M0, min=0.05) - 1.0, 0.0, 25.0)) * m_valid[:, 0].to(values.dtype)
        p_E_inh = 0.50 * torch.sqrt(torch.clamp(1.20 / torch.clamp(E0, min=0.05) - 1.0, 0.0, 25.0)) * e_valid[:, 0].to(values.dtype)
        P0_boundary = torch.maximum(p_ratio, torch.maximum(p_M_inh, p_E_inh))

        s_horm_M = torch.clamp(torch.log(torch.clamp(M_max / 1.05, min=1e-3)), min=0.0)
        s_horm_E = torch.clamp(torch.log(torch.clamp(E_max / 1.26, min=1e-3)), min=0.0)
        horm_score = 0.5 * s_horm_M + 0.5 * s_horm_E

        last_idx = torch.clamp((mask.sum(dim=-1) > 0.5).to(torch.int64).sum(dim=-1) - 1, min=0)
        b_idx = torch.arange(B, device=values.device)
        E_end = torch.where(e_valid[b_idx, last_idx], values[b_idx, last_idx, 4], means[:, 4])
        end_deficit = torch.clamp(1.0 - E_end / 1.20, 0.0, 1.0)
        rebound_ratio = torch.clamp(E_end / torch.clamp(E0, min=0.05), 0.0, 3.0)
        monotone_decline = torch.clamp((E0 - E_end) / torch.clamp(E_max, min=0.05), 0.0, 1.0)
        persist_cue = torch.clamp(0.55 * end_deficit + 0.45 * monotone_decline - 0.25 * torch.clamp(rebound_ratio - 1.0, 0.0, 1.0), 0.0, 1.0)
        u_persist = torch.stack([end_deficit, rebound_ratio, monotone_decline, persist_cue], dim=-1)

        n_obs_norm = (mask.sum(dim=-1) > 0.5).to(values.dtype).sum(dim=-1) / 8.0
        obs_summary = torch.stack([pH_m, NO3_m, W_m, M0, M_max, E0, dE_drop, dM_drop, P0_boundary], dim=-1)
        return {
            "obs_summary": obs_summary,
            "env_3": obs_summary[:, :3],
            "obs_env_4": torch.stack([pH_m, NO3_m, W_m, M_max], dim=-1),
            "P0_boundary": P0_boundary,
            "horm_score": horm_score,
            "drop_sev": drop_sev,
            "u_persist": u_persist,
            "dyn_feat": torch.stack([pH_m, dE_drop, dM_drop, P0_boundary, horm_score, persist_cue], dim=-1),
            "n_obs_norm": n_obs_norm,
        }

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        t_obs: torch.Tensor,
        rain_flag: torch.Tensor,
        t_eval: torch.Tensor,
        pollutant_class_hint: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B = values.shape[0]
        if pollutant_class_hint is None:
            class_hint = torch.zeros(B, device=values.device, dtype=values.dtype)
        else:
            class_hint = torch.clamp(pollutant_class_hint.to(values.dtype), -1.0, 1.0)

        feats = self._extract_features(values, mask)
        enc = self.encoder(values, mask, t_obs, rain_flag)
        alpha, h_final = enc["alpha"], enc["h_final"]

        atlas_out = self.atlas(
            h_final=h_final,
            alpha=alpha,
            obs_env_4=feats["obs_env_4"],
            P0_boundary=feats["P0_boundary"],
            hormesis_score=feats["horm_score"],
            drop_severity=feats["drop_sev"],
        )
        phi_env, l1_kan = self.multkan(feats["env_3"])
        exp_out = self.expert_bank(h_final, phi_env, feats["dyn_feat"], class_hint)
        gate_d = torch.sigmoid((atlas_out["min_dist"] - 0.55 * self.atlas.d_star) / 0.045)

        trunk_out = self.trunk(
            h_ctx=exp_out["h_physics"],
            alpha=alpha,
            t_eval=t_eval,
            obs_summary=feats["obs_summary"],
            phi_env=phi_env,
            u_persist=feats["u_persist"],
            class_hint=class_hint,
            gate_d=gate_d,
            n_obs_norm=feats["n_obs_norm"],
        )

        P_pos = torch.clamp(trunk_out["P_bio"], min=0.0)
        M_pos = trunk_out["M"]  # уже >= 0.05 через softplus
        E_pos = trunk_out["E"]  # уже >= 0.05 через softplus
        M_ref = torch.clamp(trunk_out["decoded"]["M_inf"], min=0.20).unsqueeze(-1)
        E_ref = torch.clamp(trunk_out["decoded"]["E_inf"], min=0.20).unsqueeze(-1)

        term_M = 0.35 * torch.abs(torch.log(M_pos / M_ref))
        term_E = 0.40 * torch.clamp(1.0 - E_pos / E_ref, min=0.0)
        term_P = 0.25 * (1.0 - alpha.unsqueeze(-1)) * P_pos
        ibsp_med = 100.0 * torch.exp(-(term_M + term_E + term_P))

        dI = trunk_out["dI_quantiles"]
        ts = trunk_out["t_scale"]
        I_q25 = ibsp_med - dI[:, 0:1] * ts
        I_q10 = I_q25 - dI[:, 1:2] * ts
        I_q75 = ibsp_med + dI[:, 2:3] * ts
        I_q90 = I_q75 + dI[:, 3:4] * ts
        ibsp_quantiles = torch.stack([I_q10, I_q25, ibsp_med, I_q75, I_q90], dim=-1)

        return {
            **trunk_out,
            "IBSP": ibsp_med,
            "IBSP_quantiles": ibsp_quantiles,
            "alpha": alpha,
            "passport": atlas_out["passport"],
            "log_passport": atlas_out["log_passport"],
            "min_dist": atlas_out["min_dist"],
            "K_m_est": atlas_out["K_m_est"],
            "detailed_passport_vec": atlas_out["detailed_passport_vec"],
            "phi_env": phi_env,
            "l1_kan": l1_kan,
            "expert_weights": exp_out["expert_weights"],
            "load_balance_loss": exp_out["load_balance_loss"],
            "gate_d": gate_d,
        }


def _hybrid_h8_loss(pred: torch.Tensor, target: torch.Tensor, beta: float = 0.25) -> torch.Tensor:
    return 0.75 * F.mse_loss(pred, target) + 0.25 * F.smooth_l1_loss(pred, target, beta=beta)


def _pinball_loss(q_preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    losses = []
    for i, q in enumerate(QUANTILE_LEVELS):
        err = target - q_preds[..., i]
        losses.append(torch.maximum(q * err, (q - 1.0) * err).mean())
    return sum(losses) / len(losses)


def compute_terrakan_v150_loss(
    out: Dict[str, torch.Tensor],
    P_true: torch.Tensor,
    P_inf_true: torch.Tensor,
    M_true: torch.Tensor,
    E_true: torch.Tensor,
    IBSP_true: torch.Tensor,
    alpha_true: torch.Tensor,
    phi_true: torch.Tensor,
    dirichlet_targets: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    loss_P = _hybrid_h8_loss(out["P_bio"], P_true)
    loss_Pinf = _hybrid_h8_loss(out["P_inf"], P_inf_true)
    loss_M = _hybrid_h8_loss(out["M"], M_true)
    loss_E = _hybrid_h8_loss(out["E"], E_true)
    loss_IBSP = _hybrid_h8_loss(out["IBSP"] / 100.0, IBSP_true / 100.0, beta=0.10)
    loss_alpha = _hybrid_h8_loss(out["alpha"], alpha_true, beta=0.05)

    # Прямая BCE-супервизия рекальцитрантности (только на загрязнённых примерах P0 > 0.15)
    contam_mask = P_true[:, 0] > 0.15
    if torch.any(contam_mask):
        is_recalc = (P_inf_true[contam_mask] > 0.25 * P_true[contam_mask, 0]).to(P_true.dtype)
        loss_pi_bce = F.binary_cross_entropy_with_logits(out["pi_logit"][contam_mask], is_recalc)
    else:
        loss_pi_bce = torch.tensor(0.0, device=P_true.device, dtype=P_true.dtype)

    loss_q_P = _pinball_loss(out["P_quantiles"], P_true)
    loss_q_I = _pinball_loss(out["IBSP_quantiles"] / 100.0, IBSP_true / 100.0)

    dec = out["decoded"]
    P_tot = torch.clamp(out["P_bio"], min=0.0)
    P_bio = torch.clamp(out["P_deg"], min=0.0)
    M_pos = out["M"]
    E_pos = out["E"]
    K_m = torch.clamp(out["K_m_est"], min=0.25).unsqueeze(-1)
    K_s = dec["K_s_eff"].unsqueeze(-1)

    haldane = P_bio / (K_s + P_bio + P_bio * P_bio)
    rhs_P = -dec["k_p_eff"].unsqueeze(-1) * P_bio - dec["k_deg_eff"].unsqueeze(-1) * haldane * M_pos
    rhs_M = (
        dec["r_m_eff"].unsqueeze(-1) * M_pos * (1.0 - M_pos / K_m)
        - dec["gamma_p_eff"].unsqueeze(-1) * P_tot * M_pos
        + dec["mu_sub_eff"].unsqueeze(-1) * haldane * M_pos
    )
    rhs_E = dec["beta_e_eff"].unsqueeze(-1) * (M_pos / (1.0 + P_tot * P_tot)) - dec["delta_e_eff"].unsqueeze(-1) * E_pos

    loss_phys = (
        F.smooth_l1_loss(out["D_alpha_P"], rhs_P, beta=0.1)
        + F.smooth_l1_loss(out["D_alpha_M"], rhs_M, beta=0.1)
        + F.smooth_l1_loss(out["D_alpha_E"], rhs_E, beta=0.1)
    )

    loss_atlas = F.kl_div(out["log_passport"], dirichlet_targets, reduction="batchmean")
    loss_kan = F.mse_loss(out["phi_env"], phi_true)

    total = (
        1.25 * loss_P
        + 0.70 * loss_Pinf
        + 0.75 * loss_pi_bce
        + 0.85 * loss_M
        + 0.85 * loss_E
        + 0.80 * loss_IBSP
        + 1.00 * loss_alpha
        + 0.40 * loss_q_P
        + 0.35 * loss_q_I
        + 0.12 * loss_phys
        + 0.15 * loss_atlas
        + 0.25 * loss_kan
        + 0.02 * out["l1_kan"]
        + 0.04 * out["load_balance_loss"]
    )
    return {
        "total": total,
        "P": loss_P,
        "IBSP": loss_IBSP,
        "pi_bce": loss_pi_bce,
        "phys": loss_phys,
    }
