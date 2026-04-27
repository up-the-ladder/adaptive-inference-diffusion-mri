"""
train.py
--------
Training script for diffusion-based MRI reconstruction.

v3 fixes:
  - Calls train_dataset.set_epoch(epoch) each epoch so mask seed
    varies — prevents memorization of fixed mask+slice patterns.
  - Adds SSIM/PSNR validation every 10 epochs using a small number
    of full reverse diffusion passes (DDIM-50) so we can track
    actual reconstruction quality, not just noise-prediction MSE.
  - Val SSIM/PSNR logged to training_log.csv and loss_curve.png.
  - best.pt now saved on best val_ssim (not best val_loss).

Usage:
    python train.py --config configs/r4.yaml
    python train.py --config configs/r4.yaml \\
        --resume /scratch/ks8413/mri_diffusion/runs/r4_v3/latest.pt
"""

import os
import csv
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from data.dataset import CalgaryDataset
from models.unet import UNet, count_parameters
from models.diffusion import DDPM
from inference.fixed import ddim_sample


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     type=str, default="configs/r4.yaml")
    parser.add_argument("--data_root",  type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--resume",     type=str, default=None)
    parser.add_argument("--epochs",     type=int, default=None)
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def compute_batch_ssim_psnr(pred: torch.Tensor, target: torch.Tensor):
    """Compute mean SSIM and PSNR for a batch."""
    pred_np   = pred.squeeze(1).cpu().numpy()
    target_np = target.squeeze(1).cpu().numpy()
    ssim_vals, psnr_vals = [], []
    for p, t in zip(pred_np, target_np):
        p = np.clip(p, 0, 1)
        t = np.clip(t, 0, 1)
        ssim_vals.append(ssim(t, p, data_range=1.0))
        psnr_vals.append(psnr(t, p, data_range=1.0))
    return float(np.mean(ssim_vals)), float(np.mean(psnr_vals))


def run_reconstruction_eval(ddpm, val_loader, device, n_batches=10):
    """
    Run DDIM-50 on a subset of val set and compute SSIM/PSNR.
    Called every 10 epochs to track actual reconstruction quality.
    """
    ddpm.eval()
    all_ssim, all_psnr = [], []
    with torch.no_grad():
        for i, (undersampled, target, _mask, _kspace) in enumerate(val_loader):
            if i >= n_batches:
                break
            undersampled = undersampled.to(device)
            target       = target.to(device)
            pred, _      = ddim_sample(ddpm, undersampled, num_steps=50)
            s, p = compute_batch_ssim_psnr(pred, target)
            all_ssim.append(s)
            all_psnr.append(p)
    return float(np.mean(all_ssim)), float(np.mean(all_psnr))


def save_loss_curve(log_path: str, out_path: str):
    epochs, train_losses, val_losses = [], [], []
    val_ssims, val_psnrs = [], []

    with open(log_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))
            val_ssims.append(float(row["val_ssim"]) if row["val_ssim"] else None)
            val_psnrs.append(float(row["val_psnr"]) if row["val_psnr"] else None)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, train_losses, label="Train loss", color="steelblue")
    axes[0].plot(epochs, val_losses,   label="Val loss",   color="tomato")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Noise prediction loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    ssim_epochs = [e for e, s in zip(epochs, val_ssims) if s is not None]
    ssim_vals   = [s for s in val_ssims if s is not None]
    if ssim_vals:
        axes[1].plot(ssim_epochs, ssim_vals, color="green", marker="o", markersize=4)
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("SSIM")
        axes[1].set_title("Val SSIM (DDIM-50) — actual reconstruction quality")
        axes[1].grid(alpha=0.3)

    psnr_vals = [p for p in val_psnrs if p is not None]
    if psnr_vals:
        psnr_epochs = [e for e, p in zip(epochs, val_psnrs) if p is not None]
        axes[2].plot(psnr_epochs, psnr_vals, color="purple", marker="o", markersize=4)
        axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("PSNR (dB)")
        axes[2].set_title("Val PSNR (DDIM-50) — actual reconstruction quality")
        axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def train(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # ── Data ───────────────────────────────────────────────────────────────────
    train_dataset = CalgaryDataset(
        root_dir=cfg["data_root"],
        acceleration=cfg["acceleration"],
        mask_type=cfg["mask_type"],
        split="train",
        center_fraction=cfg.get("center_fraction", 0.08),
    )
    val_dataset = CalgaryDataset(
        root_dir=cfg["data_root"],
        acceleration=cfg["acceleration"],
        mask_type=cfg["mask_type"],
        split="val",
        center_fraction=cfg.get("center_fraction", 0.08),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg.get("num_workers", 4), pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=4, shuffle=False,
        num_workers=cfg.get("num_workers", 4), pin_memory=(device.type == "cuda"),
    )

    print(f"Train: {len(train_dataset)} slices | Val: {len(val_dataset)} slices")

    # ── Model ──────────────────────────────────────────────────────────────────
    unet = UNet(
        in_channels=2,
        out_channels=1,
        base_channels=cfg.get("base_channels", 64),
        channel_mults=cfg.get("channel_mults", [1, 2, 4, 8]),
        time_embed_dim=cfg.get("time_embed_dim", 128),
    ).to(device)

    ddpm = DDPM(
        unet=unet,
        T=cfg.get("T", 1000),
        schedule=cfg.get("schedule", "cosine"),
    ).to(device)

    print(f"Model parameters: {count_parameters(unet):,}")

    lr = float(cfg.get("lr", 1e-4))
    optimizer = AdamW(ddpm.parameters(), lr=lr, weight_decay=1e-4)

    # ── Resume ─────────────────────────────────────────────────────────────────
    start_epoch    = 0
    best_val_loss  = float("inf")
    best_val_ssim  = 0.0

    if cfg.get("resume"):
        ckpt = torch.load(cfg["resume"], map_location=device)
        ddpm.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        best_val_ssim = ckpt.get("best_val_ssim", 0.0)
        print(f"Resumed from epoch {start_epoch}  (best_val_ssim={best_val_ssim:.4f})")

    remaining_epochs = cfg["epochs"] - start_epoch
    scheduler = CosineAnnealingLR(
        optimizer, T_max=max(remaining_epochs, 1), eta_min=1e-6
    )

    # ── CSV log ─────────────────────────────────────────────────────────────────
    log_path   = os.path.join(output_dir, "training_log.csv")
    curve_path = os.path.join(output_dir, "loss_curve.png")
    log_fields = ["epoch", "train_loss", "val_loss", "val_ssim", "val_psnr", "lr"]
    checkpoint_interval  = cfg.get("checkpoint_interval", 25)
    ssim_eval_interval   = cfg.get("ssim_eval_interval", 10)
    ssim_eval_batches    = cfg.get("ssim_eval_batches", 10)

    if start_epoch == 0:
        with open(log_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=log_fields).writeheader()

    # ── Training loop ───────────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg["epochs"]):

        # FIX: update mask seed each epoch to prevent mask memorization
        train_dataset.set_epoch(epoch)

        ddpm.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}")
        for undersampled, target, _mask, _kspace in pbar:
            undersampled = undersampled.to(device)
            target       = target.to(device)

            optimizer.zero_grad()
            loss = ddpm.training_loss(x_0=target, condition=undersampled)
            loss.backward()
            nn.utils.clip_grad_norm_(ddpm.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        current_lr     = optimizer.param_groups[0]["lr"]
        avg_train_loss = np.mean(train_losses)

        # ── Noise-prediction val loss ───────────────────────────────────────────
        ddpm.eval()
        val_losses = []
        with torch.no_grad():
            for undersampled, target, _mask, _kspace in val_loader:
                undersampled = undersampled.to(device)
                target       = target.to(device)
                loss = ddpm.training_loss(x_0=target, condition=undersampled)
                val_losses.append(loss.item())
        avg_val_loss = np.mean(val_losses)

        # ── SSIM/PSNR reconstruction eval every N epochs ───────────────────────
        val_ssim_str, val_psnr_str = "", ""
        avg_val_ssim, avg_val_psnr = None, None

        if (epoch + 1) % ssim_eval_interval == 0 or epoch == 0:
            print(f"  Running DDIM-50 reconstruction eval ({ssim_eval_batches} batches)...")
            avg_val_ssim, avg_val_psnr = run_reconstruction_eval(
                ddpm, val_loader, device, n_batches=ssim_eval_batches
            )
            val_ssim_str = f"{avg_val_ssim:.4f}"
            val_psnr_str = f"{avg_val_psnr:.2f}"
            print(f"  Val SSIM={avg_val_ssim:.4f}  PSNR={avg_val_psnr:.2f} dB")

        print(
            f"Epoch {epoch+1:>4}/{cfg['epochs']}  "
            f"train={avg_train_loss:.4f}  val={avg_val_loss:.4f}  "
            f"lr={current_lr:.2e}"
            + (f"  ssim={avg_val_ssim:.4f}  psnr={avg_val_psnr:.2f}" if avg_val_ssim else "")
        )

        # ── CSV log ─────────────────────────────────────────────────────────────
        row = {
            "epoch":      epoch + 1,
            "train_loss": f"{avg_train_loss:.6f}",
            "val_loss":   f"{avg_val_loss:.6f}",
            "val_ssim":   val_ssim_str,
            "val_psnr":   val_psnr_str,
            "lr":         f"{current_lr:.2e}",
        }
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=log_fields).writerow(row)

        save_loss_curve(log_path, curve_path)

        # ── Checkpoints ─────────────────────────────────────────────────────────
        ckpt = {
            "epoch":         epoch,
            "model":         ddpm.state_dict(),
            "optimizer":     optimizer.state_dict(),
            "train_loss":    avg_train_loss,
            "val_loss":      avg_val_loss,
            "val_ssim":      avg_val_ssim,
            "val_psnr":      avg_val_psnr,
            "best_val_loss": best_val_loss,
            "best_val_ssim": best_val_ssim,
            "config":        cfg,
        }

        torch.save(ckpt, os.path.join(output_dir, "latest.pt"))

        if (epoch + 1) % checkpoint_interval == 0:
            periodic_path = os.path.join(output_dir, f"epoch_{epoch+1:04d}.pt")
            torch.save(ckpt, periodic_path)
            print(f"  → Periodic checkpoint: epoch_{epoch+1:04d}.pt")

        # Save best on val_loss (every epoch)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(ckpt, os.path.join(output_dir, "best_loss.pt"))
            print(f"  ✓ New best loss: {best_val_loss:.4f}")

        # Save best on val_ssim (only when measured)
        if avg_val_ssim is not None and avg_val_ssim > best_val_ssim:
            best_val_ssim = avg_val_ssim
            torch.save(ckpt, os.path.join(output_dir, "best_ssim.pt"))
            print(f"  ✓ New best SSIM: {best_val_ssim:.4f}")

    print(f"\nTraining complete. Outputs saved to: {output_dir}")
    print(f"  best_loss.pt     — best noise-prediction loss checkpoint")
    print(f"  best_ssim.pt     — best reconstruction SSIM checkpoint")
    print(f"  latest.pt        — final checkpoint")
    print(f"  training_log.csv — full loss + SSIM/PSNR history")
    print(f"  loss_curve.png   — loss + SSIM + PSNR plots")
    return ddpm


if __name__ == "__main__":
    args = get_args()
    cfg  = load_config(args.config)

    if args.data_root:  cfg["data_root"]  = args.data_root
    if args.output_dir: cfg["output_dir"] = args.output_dir
    if args.resume:     cfg["resume"]     = args.resume
    if args.epochs:     cfg["epochs"]     = args.epochs

    train(cfg)
