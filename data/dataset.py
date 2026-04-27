"""
data/dataset.py
---------------
Calgary-Campinas Single-channel dataset loader.

Each .npy file contains one subject's brain MRI k-space data.
Shape: (170, 256, 256, 2) — 170 slices, 256x256, [real, imaginary]

Returns 4 tensors per sample:
  - undersampled_image:  aliased image from masked k-space  (1, H, W)
  - target_image:        clean ground truth image           (1, H, W)
  - mask:                binary k-space mask                (1, H, W)
  - observed_kspace:     raw masked complex k-space         (2, H, W) [real, imag]

FIX v3: Mask seed is randomized per epoch to prevent the model from
memorizing fixed mask patterns per slice. Call set_epoch(epoch) from
the training loop each epoch.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple
from data.masks import get_mask


class CalgaryDataset(Dataset):

    def __init__(
        self,
        root_dir: str,
        acceleration: int = 4,
        mask_type: str = "random",
        split: str = "train",
        center_fraction: float = 0.08,
        seed: int = 42,
    ):
        self.root_dir = root_dir
        self.acceleration = acceleration
        self.mask_type = mask_type
        self.split = split
        self.center_fraction = center_fraction
        self.base_seed = seed
        self.epoch = 0   # updated each epoch via set_epoch()

        split_dir = os.path.join(root_dir, "Train" if split == "train" else "Val")
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Could not find split directory: {split_dir}")

        self.npy_files = sorted([
            os.path.join(split_dir, f)
            for f in os.listdir(split_dir)
            if f.endswith(".npy")
        ])
        if len(self.npy_files) == 0:
            raise FileNotFoundError(f"No .npy files found in {split_dir}")

        print(f"[{split}] Found {len(self.npy_files)} subjects")

        self.slice_index = []
        for fpath in self.npy_files:
            data = np.load(fpath, mmap_mode="r")
            n_slices = data.shape[0]
            for s in range(50, n_slices - 50):
                self.slice_index.append((fpath, s))

        print(f"[{split}] Total usable slices (excl. edge 50): {len(self.slice_index)}")

    def set_epoch(self, epoch: int):
        """
        Call this at the start of each training epoch.
        Changes the mask seed so each epoch sees different undersampling
        patterns — prevents the model from memorizing fixed mask+slice combos.
        Val dataset does NOT need this (fixed mask for reproducible eval).
        """
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.slice_index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fpath, slice_idx = self.slice_index[idx]

        slice_data = np.load(fpath, mmap_mode="r")[slice_idx]
        kspace = slice_data[..., 0] + 1j * slice_data[..., 1]

        # Ground truth: IFFT of full k-space
        target = np.abs(
            np.fft.ifft2(np.fft.ifftshift(kspace))
        ).astype(np.float32)
        target = self._normalize(target)

        H, W = kspace.shape

        # FIX: seed varies per epoch so mask is different each epoch
        # Val set always uses epoch=0 (fixed mask for reproducibility)
        mask_seed = self.base_seed + idx + self.epoch * len(self.slice_index)
        mask = get_mask(
            shape=(H, W),
            acceleration=self.acceleration,
            mask_type=self.mask_type,
            center_fraction=self.center_fraction,
            seed=mask_seed,
        )

        masked_kspace = kspace * mask

        undersampled_image = np.abs(
            np.fft.ifft2(np.fft.ifftshift(masked_kspace))
        ).astype(np.float32)
        undersampled_image = self._normalize(undersampled_image)

        masked_kspace_tensor = torch.stack([
            torch.from_numpy(masked_kspace.real.astype(np.float32)),
            torch.from_numpy(masked_kspace.imag.astype(np.float32)),
        ], dim=0)

        undersampled_tensor = torch.from_numpy(undersampled_image).unsqueeze(0)
        target_tensor       = torch.from_numpy(target).unsqueeze(0)
        mask_tensor         = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)

        return undersampled_tensor, target_tensor, mask_tensor, masked_kspace_tensor

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        p99 = np.percentile(x, 99)
        if p99 > 0:
            x = x / p99
        return np.clip(x, 0, 1).astype(np.float32)
