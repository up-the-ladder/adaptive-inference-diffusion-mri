"""
data/masks.py
-------------
K-space undersampling masks for simulating accelerated MRI acquisition.

Three mask types:
  - random:           uniformly random phase-encode lines
  - radial:           radial spokes through k-space center
  - variable_density: higher density near center (where most signal is)

All masks always keep the center fraction of k-space (low frequencies)
because the center contains most of the image energy/contrast.
"""

import numpy as np
from typing import Tuple


def get_mask(
    shape: Tuple[int, int],
    acceleration: int,
    mask_type: str = "random",
    center_fraction: float = 0.08,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate a binary k-space undersampling mask.

    Args:
        shape:            (H, W) spatial dimensions of k-space
        acceleration:     R=4 or R=8 — fraction kept = 1/R
        mask_type:        'random', 'radial', or 'variable_density'
        center_fraction:  fraction of center lines always kept
        seed:             random seed

    Returns:
        mask: binary float array of shape (H, W), 1=keep, 0=zero
    """
    if mask_type == "random":
        return _random_mask(shape, acceleration, center_fraction, seed)
    elif mask_type == "radial":
        return _radial_mask(shape, acceleration, center_fraction, seed)
    elif mask_type == "variable_density":
        return _variable_density_mask(shape, acceleration, center_fraction, seed)
    else:
        raise ValueError(f"Unknown mask type: {mask_type}. Choose from: random, radial, variable_density")


def _random_mask(
    shape: Tuple[int, int],
    acceleration: int,
    center_fraction: float,
    seed: int,
) -> np.ndarray:
    """
    Random uniform undersampling along phase-encode (vertical) direction.
    Standard baseline mask used in fastMRI and Calgary-Campinas challenges.
    """
    rng = np.random.RandomState(seed)
    H, W = shape
    mask = np.zeros((H, W), dtype=np.float32)

    # Always keep center lines
    num_center = int(H * center_fraction)
    center_start = H // 2 - num_center // 2
    center_end = center_start + num_center
    mask[center_start:center_end, :] = 1.0

    # Randomly select remaining lines to reach target acceleration
    target_lines = int(H / acceleration)
    remaining = target_lines - num_center
    remaining = max(remaining, 0)

    # Available lines (excluding center)
    available = list(range(0, center_start)) + list(range(center_end, H))
    selected = rng.choice(available, size=min(remaining, len(available)), replace=False)
    mask[selected, :] = 1.0

    return mask


def _radial_mask(
    shape: Tuple[int, int],
    acceleration: int,
    center_fraction: float,
    seed: int,
) -> np.ndarray:
    """
    Radial undersampling — spokes radiating from k-space center.
    More robust to motion than Cartesian sampling.
    """
    rng = np.random.RandomState(seed)
    H, W = shape
    mask = np.zeros((H, W), dtype=np.float32)

    cy, cx = H // 2, W // 2

    # Number of radial spokes
    num_spokes = int((H * W) / (acceleration * max(H, W)))
    num_spokes = max(num_spokes, 8)

    # Random spoke angles
    angles = rng.uniform(0, np.pi, num_spokes)

    for angle in angles:
        # Draw line through center at this angle
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        length = int(np.sqrt(H**2 + W**2))
        for t in range(-length, length):
            y = int(cy + t * sin_a)
            x = int(cx + t * cos_a)
            if 0 <= y < H and 0 <= x < W:
                mask[y, x] = 1.0

    # Always include center region
    num_center = int(min(H, W) * center_fraction)
    for y in range(H):
        for x in range(W):
            if (y - cy)**2 + (x - cx)**2 <= (num_center // 2)**2:
                mask[y, x] = 1.0

    return mask


def _variable_density_mask(
    shape: Tuple[int, int],
    acceleration: int,
    center_fraction: float,
    seed: int,
) -> np.ndarray:
    """
    Variable density undersampling — higher probability of keeping
    k-space lines near the center (low frequencies = most image energy).
    Polynomial density: p(r) ∝ (1 - r)^2 where r is normalized distance from center.
    """
    rng = np.random.RandomState(seed)
    H, W = shape
    mask = np.zeros((H, W), dtype=np.float32)

    # Compute normalized distance from center for each row
    center = H // 2
    distances = np.abs(np.arange(H) - center) / (H / 2)  # 0=center, 1=edge

    # Variable density probability: higher near center
    # Polynomial decay: p(d) = (1 - d)^3
    probs = (1 - distances) ** 3
    probs = probs / probs.sum()  # normalize

    # Always keep center
    num_center = int(H * center_fraction)
    center_start = H // 2 - num_center // 2
    center_end = center_start + num_center
    mask[center_start:center_end, :] = 1.0

    # Sample remaining lines according to variable density
    target_lines = int(H / acceleration)
    remaining = target_lines - num_center
    remaining = max(remaining, 0)

    available = list(range(0, center_start)) + list(range(center_end, H))
    avail_probs = probs[available]
    avail_probs = avail_probs / avail_probs.sum()

    n_select = min(remaining, len(available))
    selected = rng.choice(available, size=n_select, replace=False, p=avail_probs)
    mask[selected, :] = 1.0

    return mask


def compute_actual_acceleration(mask: np.ndarray) -> float:
    """Compute the actual acceleration factor from a mask."""
    return mask.size / mask.sum()
