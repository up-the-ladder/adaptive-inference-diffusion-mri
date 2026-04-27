"""
inference/fixed.py
------------------
Fixed-step inference baselines: DDIM (with and without DC projection).

DDIM (Song et al. 2020): deterministic sampling with fewer steps than DDPM.
Two variants are provided:
  ddim_sample     — standard DDIM, no DC projection (pure baseline)
  ddim_sample_dc  — DDIM + data consistency projection (fair comparison
                    against adaptive_sample which also uses DC)
"""

import torch
import time
import numpy as np
from typing import Optional, List, Tuple
from models.diffusion import DDPM


@torch.no_grad()
def ddim_sample(
    ddpm: DDPM,
    condition: torch.Tensor,
    num_steps: int = 250,
    eta: float = 0.0,
) -> Tuple[torch.Tensor, float]:
    """
    Standard DDIM sampling — no DC projection.

    Args:
        ddpm:      trained DDPM model
        condition: undersampled image, (B, 1, H, W)
        num_steps: inference steps
        eta:       stochasticity (0=deterministic, 1=DDPM)

    Returns:
        (reconstructed image (B, 1, H, W), wall_time_seconds)
    """
    B, _, H, W = condition.shape
    device = condition.device
    T = ddpm.T
    step_indices = _get_ddim_steps(T, num_steps)
    x = torch.randn(B, 1, H, W, device=device)

    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.perf_counter()

    for i, t in enumerate(step_indices):
        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
        noise_pred = ddpm.unet(x, condition, t_tensor)

        alpha_cumprod_t = ddpm.alphas_cumprod[t]
        alpha_cumprod_prev = (
            ddpm.alphas_cumprod[step_indices[i + 1]]
            if i + 1 < len(step_indices)
            else torch.tensor(1.0, device=device)   # fix: keep on same device
        )

        x0_pred = (x - torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_cumprod_t)
        x0_pred = torch.clamp(x0_pred, -1, 1)

        sigma = eta * torch.sqrt(
            (1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t) *
            (1 - alpha_cumprod_t / alpha_cumprod_prev)
        )
        dir_xt = torch.sqrt(1 - alpha_cumprod_prev - sigma**2) * noise_pred
        noise   = sigma * torch.randn_like(x)
        x = torch.sqrt(alpha_cumprod_prev) * x0_pred + dir_xt + noise

    torch.cuda.synchronize() if device.type == "cuda" else None
    elapsed = time.perf_counter() - t0

    return x, elapsed


@torch.no_grad()
def ddim_sample_dc(
    ddpm: DDPM,
    condition: torch.Tensor,
    mask: torch.Tensor,
    observed_kspace: torch.Tensor,
    num_steps: int = 50,
    eta: float = 0.0,
) -> Tuple[torch.Tensor, float]:
    """
    DDIM + data consistency projection at every step.

    This is the fair baseline to compare against adaptive_sample, since
    both apply DC projection. The only difference is adaptive_sample
    stops early per-sample; this always runs num_steps.

    Args:
        ddpm:            trained DDPM model
        condition:       undersampled image,          (B, 1, H, W)
        mask:            k-space mask,                (B, 1, H, W)
        observed_kspace: true measurements [re, im],  (B, 2, H, W)
        num_steps:       number of inference steps
        eta:             stochasticity

    Returns:
        (reconstructed image (B, 1, H, W), wall_time_seconds)
    """
    B, _, H, W = condition.shape
    device = condition.device
    T = ddpm.T
    step_indices = _get_ddim_steps(T, num_steps)
    x = torch.randn(B, 1, H, W, device=device)

    obs_complex = (
        observed_kspace[:, 0] + 1j * observed_kspace[:, 1]
    ).to(torch.complex64)
    m = mask.squeeze(1)

    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.perf_counter()

    for i, t in enumerate(step_indices):
        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
        noise_pred = ddpm.unet(x, condition, t_tensor)

        alpha_cumprod_t = ddpm.alphas_cumprod[t]
        alpha_cumprod_prev = (
            ddpm.alphas_cumprod[step_indices[i + 1]]
            if i + 1 < len(step_indices)
            else torch.tensor(1.0, device=device)
        )

        x0_pred = (x - torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_cumprod_t)
        x0_pred = torch.clamp(x0_pred, -1, 1)

        sigma = eta * torch.sqrt(
            (1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t) *
            (1 - alpha_cumprod_t / alpha_cumprod_prev)
        )
        dir_xt = torch.sqrt(1 - alpha_cumprod_prev - sigma**2) * noise_pred
        noise   = sigma * torch.randn_like(x)
        x = torch.sqrt(alpha_cumprod_prev) * x0_pred + dir_xt + noise

        # DC projection — uncentered fft2 matches dataset k-space storage convention
        x_k = torch.fft.fft2(x.squeeze(1).to(torch.complex64))
        x_k = m * obs_complex + (1.0 - m) * x_k
        x   = torch.fft.ifft2(x_k).abs().float().unsqueeze(1)

    torch.cuda.synchronize() if device.type == "cuda" else None
    elapsed = time.perf_counter() - t0

    return x, elapsed


@torch.no_grad()
def ddpm_sample_fixed(
    ddpm: DDPM,
    condition: torch.Tensor,
    num_steps: Optional[int] = None,
) -> Tuple[torch.Tensor, float]:
    """
    Standard DDPM fixed-step sampling. Returns (image, wall_time_s).
    """
    torch.cuda.synchronize() if condition.device.type == "cuda" else None
    t0 = time.perf_counter()
    result = ddpm.sample_fixed(condition, num_steps=num_steps)
    torch.cuda.synchronize() if condition.device.type == "cuda" else None
    elapsed = time.perf_counter() - t0
    return result, elapsed


def _get_ddim_steps(T: int, num_steps: int) -> List[int]:
    """Evenly spaced timestep indices for DDIM, descending."""
    step_ratio = T // num_steps
    steps = list(range(0, T, step_ratio))[:num_steps]
    return list(reversed(steps))
