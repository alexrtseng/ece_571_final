import math
import torch


def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    steps = T + 1
    t = torch.linspace(0, T, steps) / T
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    alphas_bar = f / f[0]
    betas = 1 - alphas_bar[1:] / alphas_bar[:-1]
    return betas.clamp(1e-4, 0.9999)


def linear_beta_schedule(T: int) -> torch.Tensor:
    return torch.linspace(1e-4, 0.02, T)


class NoiseSchedule:
    def __init__(self, T: int = 1000, schedule: str = "cosine"):
        if schedule == "cosine":
            betas = cosine_beta_schedule(T)
        elif schedule == "linear":
            betas = linear_beta_schedule(T)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = torch.cat([torch.ones(1), alphas_bar[:-1]])

        self.T = T
        self.betas = betas
        self.alphas = alphas
        self.alphas_bar = alphas_bar
        self.alphas_bar_prev = alphas_bar_prev
        self.sqrt_alphas_bar = alphas_bar.sqrt()
        self.sqrt_one_minus_alphas_bar = (1.0 - alphas_bar).sqrt()
        self.sqrt_recip_alphas = alphas.rsqrt()

        # DDPM posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar)
        self.posterior_log_var = torch.log(self.posterior_variance.clamp(min=1e-20))
        self.posterior_mean_coef1 = betas * alphas_bar_prev.sqrt() / (1.0 - alphas_bar)
        self.posterior_mean_coef2 = (1.0 - alphas_bar_prev) * alphas.sqrt() / (1.0 - alphas_bar)

    def to(self, device):
        for attr in [
            "betas", "alphas", "alphas_bar", "alphas_bar_prev",
            "sqrt_alphas_bar", "sqrt_one_minus_alphas_bar", "sqrt_recip_alphas",
            "posterior_variance", "posterior_log_var",
            "posterior_mean_coef1", "posterior_mean_coef2",
        ]:
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    def _gather(self, tensor, t):
        """Gather schedule values at timestep indices t, shaped for broadcasting."""
        return tensor[t][:, None, None]  # (B, 1, 1) for (B, C, L) tensors

    def q_sample(self, x0, t, noise=None):
        """Forward diffusion: x_t = sqrt(ā_t) * x0 + sqrt(1 - ā_t) * ε"""
        if noise is None:
            noise = torch.randn_like(x0)
        return (
            self._gather(self.sqrt_alphas_bar, t) * x0
            + self._gather(self.sqrt_one_minus_alphas_bar, t) * noise
        ), noise
