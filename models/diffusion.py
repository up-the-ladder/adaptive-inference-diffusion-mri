"""
models/diffusion.py
-------------------
DDPM for MRI reconstruction.

Forward process: q(x_t | x_0) — adds Gaussian noise to clean image
Reverse process: p_theta(x_{t-1} | x_t) — denoises using U-Net

Two reverse-step variants:
  p_sample_step    — standard DDPM step (used during training eval)
  p_sample_step_dc — DDPM step + data consistency projection onto
                     observed k-space measurements (used at inference)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from models.unet import UNet


class DDPM(nn.Module):

    def __init__(
        self,
        unet: UNet,
        T: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule: str = "cosine",
    ):
        super().__init__()
        self.unet = unet
        self.T = T

        # Noise schedule
        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, T)
        elif schedule == "cosine":
            steps = T + 1
            x = torch.linspace(0, T, steps)
            alphas_cumprod = torch.cos(((x / T) + 0.008) / 1.008 * torch.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

    # ── Forward process ────────────────────────────────────────────────────────

    def q_sample(
        self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        x_t = sqrt_alpha * x_0 + sqrt_one_minus * noise
        return x_t, noise

    def training_loss(
        self,
        x_0: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        B = x_0.shape[0]
        device = x_0.device
        t = torch.randint(0, self.T, (B,), device=device)
        x_t, noise = self.q_sample(x_0, t)
        noise_pred = self.unet(x_t, condition, t)
        return nn.functional.mse_loss(noise_pred, noise)

    # ── Reverse process ────────────────────────────────────────────────────────

    @torch.no_grad()
    def p_sample_step(
        self,
        x_t: torch.Tensor,
        condition: torch.Tensor,
        t: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single DDPM reverse step: x_t → x_{t-1}.
        Returns (x_{t-1}, noise_pred).
        """
        B = x_t.shape[0]
        device = x_t.device
        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)

        noise_pred = self.unet(x_t, condition, t_tensor)

        beta_t = self.betas[t]
        sqrt_recip_alpha = self.sqrt_recip_alphas[t]
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[t]

        mean = sqrt_recip_alpha * (
            x_t - beta_t / sqrt_one_minus_alpha_cumprod * noise_pred
        )

        if t == 0:
            return mean, noise_pred
        else:
            posterior_var = self.posterior_variance[t]
            noise = torch.randn_like(x_t)
            return mean + torch.sqrt(posterior_var) * noise, noise_pred

    @torch.no_grad()
    def p_sample_step_dc(
        self,
        x_t: torch.Tensor,
        condition: torch.Tensor,
        t: int,
        mask: torch.Tensor,
        observed_kspace: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        DDPM reverse step with data consistency (DC) projection.

        After each denoising step, the estimate is projected back onto the
        observed k-space measurements — enforcing that the reconstruction
        agrees with what the scanner actually acquired.

        DC projection:
            x_k = FFT(x_{t-1})
            x_k_dc = mask * y + (1 - mask) * x_k
            x_{t-1}_dc = IFFT(x_k_dc)

        where y = observed_kspace (true measured frequencies).

        Args:
            x_t:             noisy image at step t,        (B, 1, H, W)
            condition:       undersampled image,           (B, 1, H, W)
            t:               current timestep (int)
            mask:            binary k-space mask,          (B, 1, H, W)
            observed_kspace: true measurements [real,imag],(B, 2, H, W)

        Returns:
            x_dc:       DC-projected denoised image,       (B, 1, H, W)
            noise_pred: predicted noise,                   (B, 1, H, W)
        """
        x_next, noise_pred = self.p_sample_step(x_t, condition, t)

        # Reconstruct complex observed k-space from real/imag channels
        obs_complex = (
            observed_kspace[:, 0] + 1j * observed_kspace[:, 1]
        ).to(torch.complex64)  # (B, H, W)

        m = mask.squeeze(1)  # (B, H, W)

        # FFT using uncentered convention matching dataset storage.
        # Dataset stores: masked_kspace = kspace * mask (raw, no fftshift applied).
        # So we use fft2 (uncentered) to stay in the same frequency domain as obs_kspace.
        # DC is approximate because dataset images use abs(), which loses phase.
        x_k = torch.fft.fft2(x_next.squeeze(1).to(torch.complex64))  # (B, H, W)

        # Replace measured frequencies with true observations; keep model
        # prediction at unobserved locations
        x_k_dc = m * obs_complex + (1.0 - m) * x_k

        # Back to image: abs(ifft2) matches dataset reconstruction convention
        x_dc = torch.fft.ifft2(x_k_dc).abs().float().unsqueeze(1)  # (B, 1, H, W)

        return x_dc, noise_pred

    @torch.no_grad()
    def sample_fixed(
        self,
        condition: torch.Tensor,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Full fixed-step DDPM reverse diffusion (baseline, no DC)."""
        B, _, H, W = condition.shape
        device = condition.device
        steps = num_steps or self.T
        x = torch.randn(B, 1, H, W, device=device)
        for t in range(steps - 1, -1, -1):
            x, _ = self.p_sample_step(x, condition, t)
        return x
