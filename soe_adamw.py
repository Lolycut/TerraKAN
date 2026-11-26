"""
TerraKAN v1.5.0 — Multi-Timescale SOE-AdamW Optimizer (портирован из abelkan v0.8.0).

Отслеживает первый момент по трём временным шкалам beta = (0.80, 0.985, 0.999),
стабилизируя одновременное обучение быстрых весов энкодера и медленных кинетических параметров
(alpha, log_lambda, gamma_h, W_power) без костыля периодического Ridge Alignment.
"""

import math
from typing import Iterable, Tuple
import torch
from torch.optim import Optimizer


class SOEAdamW(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 2.0e-3,
        betas1: Tuple[float, float, float] = (0.80, 0.985, 0.999),
        scale_weights: Tuple[float, float, float] = (0.50, 0.35, 0.15),
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 1e-4,
    ) -> None:
        defaults = dict(
            lr=lr,
            betas1=betas1,
            scale_weights=scale_weights,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            b1_fast, b1_mid, b1_slow = group["betas1"]
            w_fast, w_mid, w_slow = group["scale_weights"]
            b2 = group["beta2"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if wd != 0.0 and p.ndim >= 2:
                    p.mul_(1.0 - lr * wd)

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m1"] = torch.zeros_like(p)
                    state["m2"] = torch.zeros_like(p)
                    state["m3"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                state["step"] += 1
                t = state["step"]

                state["m1"].mul_(b1_fast).add_(grad, alpha=1.0 - b1_fast)
                state["m2"].mul_(b1_mid).add_(grad, alpha=1.0 - b1_mid)
                state["m3"].mul_(b1_slow).add_(grad, alpha=1.0 - b1_slow)
                state["v"].mul_(b2).addcmul_(grad, grad, value=1.0 - b2)

                m1_hat = state["m1"] / (1.0 - b1_fast ** t)
                m2_hat = state["m2"] / (1.0 - b1_mid ** t)
                m3_hat = state["m3"] / (1.0 - b1_slow ** t)
                m_mix = w_fast * m1_hat + w_mid * m2_hat + w_slow * m3_hat

                v_hat = state["v"] / (1.0 - b2 ** t)
                p.addcdiv_(m_mix, v_hat.sqrt().add_(eps), value=-lr)

        return loss
