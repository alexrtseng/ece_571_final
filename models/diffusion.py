"""
DDPM training loss and DDPM / DDIM samplers.
Both samplers operate on the same trained UNet1D weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .schedule import NoiseSchedule


class DDPMTrainer(nn.Module):
    def __init__(self, model: nn.Module, schedule: NoiseSchedule):
        super().__init__()
        self.model = model
        self.schedule = schedule

    def loss(
        self,
        rt: torch.Tensor,
        da_cond: torch.Tensor,
        cal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        rt:      (B, 1, 288) normalized RT prices (x0)
        da_cond: (B, 1, 288) normalized DA prices (conditioning)
        cal:     (B, 4) long calendar features, or None
        Returns scalar MSE loss.
        """
        B = rt.shape[0]
        t = torch.randint(0, self.schedule.T, (B,), device=rt.device)
        x_t, noise = self.schedule.q_sample(rt, t)
        model_input = torch.cat([x_t, da_cond], dim=1)  # (B, 2, 288)
        pred_noise = self.model(model_input, t, cal=cal)
        return F.mse_loss(pred_noise, noise)


@torch.no_grad()
def sample_ddpm(
    model: nn.Module,
    schedule: NoiseSchedule,
    da_cond: torch.Tensor,
    n_samples: int = 1,
    cal: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    DDPM ancestral sampling (T steps, stochastic).

    da_cond: (1, 1, 288) or (B, 1, 288) normalized DA conditioning
    cal:     (1, 4) or (B, 4) long calendar features, or None
    Returns: (n_samples, 1, 288)
    """
    model.eval()
    device = da_cond.device
    cond = da_cond.expand(n_samples, -1, -1)   # (n_samples, 1, 288)
    cal_exp = cal.expand(n_samples, -1).to(device) if cal is not None else None

    x = torch.randn(n_samples, 1, 288, device=device)

    for t_val in reversed(range(schedule.T)):
        t = torch.full((n_samples,), t_val, device=device, dtype=torch.long)
        model_input = torch.cat([x, cond], dim=1)
        pred_noise = model(model_input, t, cal=cal_exp)

        # Posterior mean
        coef1 = schedule.posterior_mean_coef1[t_val]
        coef2 = schedule.posterior_mean_coef2[t_val]
        sqrt_abar = schedule.sqrt_alphas_bar[t_val]
        sqrt_1mabar = schedule.sqrt_one_minus_alphas_bar[t_val]

        x0_pred = (x - sqrt_1mabar * pred_noise) / sqrt_abar
        x0_pred = x0_pred.clamp(-5, 5)  # numerical stability
        mu = coef1 * x0_pred + coef2 * x

        if t_val > 0:
            noise = torch.randn_like(x)
            log_var = schedule.posterior_log_var[t_val]
            x = mu + (0.5 * log_var).exp() * noise
        else:
            x = mu

    return x  # (n_samples, 1, 288)


@torch.no_grad()
def sample_ddim(
    model: nn.Module,
    schedule: NoiseSchedule,
    da_cond: torch.Tensor,
    n_samples: int = 1,
    S: int = 50,
    eta: float = 0.0,
    cal: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    DDIM deterministic sub-sequence sampling (S steps).
    eta=0 → fully deterministic; eta=1 → matches DDPM variance.

    da_cond: (1, 1, 288) or (B, 1, 288)
    cal:     (1, 4) or (B, 4) long calendar features, or None
    Returns: (n_samples, 1, 288)
    """
    model.eval()
    device = da_cond.device
    cond = da_cond.expand(n_samples, -1, -1)
    cal_exp = cal.expand(n_samples, -1).to(device) if cal is not None else None

    # Select S evenly-spaced timesteps in [0, T-1]
    step_indices = np.linspace(0, schedule.T - 1, S, dtype=int)
    timesteps = list(reversed(step_indices.tolist()))

    x = torch.randn(n_samples, 1, 288, device=device)

    for i, t_val in enumerate(timesteps):
        t = torch.full((n_samples,), t_val, device=device, dtype=torch.long)
        model_input = torch.cat([x, cond], dim=1)
        pred_noise = model(model_input, t, cal=cal_exp)

        alpha_t = schedule.alphas_bar[t_val]
        alpha_prev = schedule.alphas_bar[timesteps[i + 1]] if i + 1 < len(timesteps) else torch.tensor(1.0)
        alpha_prev = alpha_prev.to(device) if isinstance(alpha_prev, torch.Tensor) else torch.tensor(alpha_prev, device=device)

        sqrt_abar = alpha_t.sqrt()
        sqrt_1mabar = (1.0 - alpha_t).sqrt()

        x0_pred = (x - sqrt_1mabar * pred_noise) / sqrt_abar
        x0_pred = x0_pred.clamp(-5, 5)

        # DDIM update
        sigma = eta * ((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)).sqrt()
        direction = (1 - alpha_prev - sigma ** 2).sqrt() * pred_noise

        x = alpha_prev.sqrt() * x0_pred + direction
        if eta > 0 and i + 1 < len(timesteps):
            x = x + sigma * torch.randn_like(x)

    return x  # (n_samples, 1, 288)
