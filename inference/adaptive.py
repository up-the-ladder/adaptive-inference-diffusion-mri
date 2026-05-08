"""
inference/adaptive.py

Here we are using DDIM (eta=0, deterministic) instead of stochastic DDPM:
  - x0_pred converges monotonically in DDIM
  - No noise injection means x0_pred trajectory is smooth
  - Early stopping is meaningful because reconstruction quality
    improves consistently step by step

WHY DDIM NOT DDPM:
  - DDPM adds stochastic noise at each step → x0_pred oscillates
  - DDIM is deterministic (eta=0) → x0_pred converges smoothly
  - Early stopping on oscillating signal = stopping at wrong point
  - Early stopping on smooth signal = stopping when truly converged
"""

import torch
import time
from typing import Dict, Tuple, Optional, List
from models.diffusion import DDPM


def _get_adaptive_steps(T: int, max_steps: int) -> List[int]:
    """
    Evenly spaced timestep indices from T-1 down to 0.
    e.g. T=1000, max_steps=50 → [980, 960, ..., 20, 0]
    """
    step_ratio = T // max_steps
    steps = list(range(0, T, step_ratio))[:max_steps]
    return list(reversed(steps))


def _ddim_step(
    ddpm: DDPM,
    x: torch.Tensor,
    condition: torch.Tensor,
    t: int,
    t_prev: Optional[int],
    eta: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Single deterministic DDIM step (eta=0).
    Returns: (x_{t-1}, x0_pred)
    """
    B      = x.shape[0]
    device = x.device
    t_tensor = torch.full((B,), t, device=device, dtype=torch.long)

    noise_pred = ddpm.unet(x, condition, t_tensor)

    alpha_t = ddpm.alphas_cumprod[t]
    alpha_prev = (
        ddpm.alphas_cumprod[t_prev]
        if t_prev is not None
        else torch.tensor(1.0, device=device)
    )

    x0_pred = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
    x0_pred = torch.clamp(x0_pred, -1, 1)

    dir_xt = torch.sqrt(1 - alpha_prev) * noise_pred
    x_prev = torch.sqrt(alpha_prev) * x0_pred + dir_xt

    return x_prev, x0_pred


def _compute_relative_change(
    x0_curr: torch.Tensor,
    x0_prev: torch.Tensor,
) -> torch.Tensor:
    """
    delta = || x0_curr - x0_prev || / (|| x0_prev || + eps)
    Returns: (B,)
    """
    diff = (x0_curr - x0_prev).pow(2).mean(dim=[1, 2, 3]).sqrt()
    norm = x0_prev.pow(2).mean(dim=[1, 2, 3]).sqrt() + 1e-8
    return diff / norm


class AdaptiveStopper:
    """
    Stops DDIM inference when x0_pred has stabilised.

    Records baseline delta d0 at step min_steps.
    Stops each sample when: smooth(delta_t) < relative_threshold * d0
    """

    def __init__(
        self,
        relative_threshold: float = 0.5,
        min_steps: int = 5,
        patience: int = 3,
        window_size: int = 3,
    ):
        self.relative_threshold = relative_threshold
        self.min_steps          = min_steps
        self.patience           = patience
        self.window_size        = window_size

    def check_convergence(
        self,
        x0_curr: torch.Tensor,
        x0_prev: torch.Tensor,
        patience_counters: torch.Tensor,
        baseline_delta: torch.Tensor,
        delta_history: list,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:

        delta = _compute_relative_change(x0_curr, x0_prev)

        delta_history.append(delta.detach())
        if len(delta_history) >= self.window_size:
            window       = torch.stack(delta_history[-self.window_size:], dim=0)
            smooth_delta = window.mean(dim=0)
        else:
            smooth_delta = delta

        threshold = self.relative_threshold * baseline_delta
        converged = smooth_delta < threshold

        patience_counters = torch.where(
            converged,
            patience_counters + 1,
            torch.zeros_like(patience_counters),
        )
        should_stop = patience_counters >= self.patience

        metrics = {
            "delta":             delta,
            "smooth_delta":      smooth_delta,
            "baseline_delta":    baseline_delta,
            "threshold":         threshold,
            "ratio":             smooth_delta / (baseline_delta + 1e-8),
            "converged":         converged.float(),
            "patience_counters": patience_counters,
        }

        return should_stop, patience_counters, metrics


@torch.no_grad()
def adaptive_sample(
    ddpm: DDPM,
    condition: torch.Tensor,
    mask: torch.Tensor,
    observed_kspace: torch.Tensor,
    stopper: AdaptiveStopper,
    max_steps: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict]:
    """
    Adaptive stopping DDIM inference.

    Uses deterministic DDIM (eta=0) so x0_pred converges smoothly.
    Stops each sample independently when x0_pred has stabilised.
    """
    B, _, H, W = condition.shape
    device     = condition.device
    max_steps  = max_steps or ddpm.T

    step_indices = _get_adaptive_steps(ddpm.T, max_steps)

    x                 = torch.randn(B, 1, H, W, device=device)
    stopped           = torch.zeros(B, dtype=torch.bool, device=device)
    steps_used        = torch.full((B,), max_steps, dtype=torch.long, device=device)
    patience_counters = torch.zeros(B, device=device)
    final_x           = x.clone()
    delta_history     = []
    metrics_history   = []
    baseline_delta    = None
    x0_pred_prev      = torch.zeros_like(x)

    torch.cuda.synchronize() if device.type == "cuda" else None
    t_start = time.perf_counter()

    for step_num, t in enumerate(step_indices):
        active = ~stopped
        if not active.any():
            break

        t_prev = step_indices[step_num + 1] if step_num + 1 < len(step_indices) else None

        x_new        = x.clone()
        x0_pred_full = torch.zeros_like(x)

        x_active, x0_active = _ddim_step(
            ddpm, x[active], condition[active], t, t_prev, eta=0.0
        )
        x_new[active]        = x_active
        x0_pred_full[active] = x0_active

        # Record baseline at min_steps
        if step_num == stopper.min_steps:
            baseline_delta = _compute_relative_change(
                x0_pred_full, x0_pred_prev
            ).clone()

        # Check convergence after baseline
        if step_num > stopper.min_steps and baseline_delta is not None:
            should_stop, patience_counters, metrics = stopper.check_convergence(
                x0_curr=x0_pred_full,
                x0_prev=x0_pred_prev,
                patience_counters=patience_counters,
                baseline_delta=baseline_delta,
                delta_history=delta_history,
            )
            should_stop               = should_stop & (~stopped)
            newly_stopped             = should_stop
            steps_used[newly_stopped] = step_num + 1
            stopped                   = stopped | should_stop
            metrics_history.append({k: v.mean().item() for k, v in metrics.items()})

        final_x[active]  = x0_pred_full[active]
        x0_pred_prev     = x0_pred_full.clone()
        x                = x_new

    torch.cuda.synchronize() if device.type == "cuda" else None
    elapsed = time.perf_counter() - t_start

    info = {
        "steps_used":         steps_used.cpu().numpy(),
        "max_steps":          max_steps,
        "mean_steps":         steps_used.float().mean().item(),
        "step_reduction_pct": (1 - steps_used.float().mean().item() / max_steps) * 100,
        "wall_time_s":        elapsed,
        "metrics_history":    metrics_history,
        "all_stopped_early":  stopped.all().item(),
        "baseline_delta":     baseline_delta.mean().item() if baseline_delta is not None else None,
    }

    return final_x, info
