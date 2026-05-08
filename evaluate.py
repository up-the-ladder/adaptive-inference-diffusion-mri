import os
import json
import argparse
import yaml
import time
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from data.dataset import CalgaryDataset
from models.unet import UNet
from models.diffusion import DDPM
from inference.fixed import ddim_sample
from inference.adaptive import adaptive_sample, AdaptiveStopper


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    pred   = np.clip(pred, 0, 1)
    target = np.clip(target, 0, 1)
    return {
        "ssim": float(ssim(target, pred, data_range=1.0)),
        "psnr": float(psnr(target, pred, data_range=1.0)),
        "nmse": float(np.sum((pred - target)**2) / (np.sum(target**2) + 1e-8)),
    }


def save_reconstruction_figure(
    undersampled, pred, target, metrics, method, steps, save_path
):
    error = np.abs(pred - target)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.suptitle(
        f"Method: {method}  |  Steps: {steps}  |  "
        f"SSIM: {metrics['ssim']:.4f}  PSNR: {metrics['psnr']:.2f} dB  "
        f"NMSE: {metrics['nmse']:.4f}",
        fontsize=11, fontweight="bold"
    )
    axes[0].imshow(undersampled, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Undersampled input", fontsize=10); axes[0].axis("off")
    axes[1].imshow(pred,         cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Reconstruction",     fontsize=10); axes[1].axis("off")
    axes[2].imshow(target,       cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Ground truth",        fontsize=10); axes[2].axis("off")
    im = axes[3].imshow(error, cmap="hot", vmin=0, vmax=0.3)
    axes[3].set_title("Error map |pred - gt|", fontsize=10); axes[3].axis("off")
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()


def evaluate(cfg, checkpoint_path, eval_batches, save_images):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = cfg.get("output_dir", ".")
    eval_dir   = os.path.join(output_dir, "eval_results")
    recon_dir  = os.path.join(eval_dir, "reconstructions")
    os.makedirs(eval_dir,  exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)

    # Load model
    unet = UNet(
        in_channels=2, out_channels=1,
        base_channels=cfg.get("base_channels", 64),
        channel_mults=cfg.get("channel_mults", [1, 2, 4, 8]),
        time_embed_dim=cfg.get("time_embed_dim", 128),
    )
    ddpm = DDPM(unet=unet, T=cfg.get("T", 1000), schedule=cfg.get("schedule", "cosine"))
    ckpt = torch.load(checkpoint_path, map_location=device)
    ddpm.load_state_dict(ckpt["model"])
    ddpm = ddpm.to(device).eval()
    trained_epoch = ckpt["epoch"]
    print(f"Loaded checkpoint: epoch {trained_epoch}  val_loss={ckpt['val_loss']:.4f}")
    if ckpt.get("val_ssim"):
        print(f"  val_ssim={ckpt['val_ssim']:.4f}  val_psnr={ckpt['val_psnr']:.2f}")

    val_dataset = CalgaryDataset(
        root_dir=cfg["data_root"],
        acceleration=cfg["acceleration"],
        mask_type=cfg["mask_type"],
        split="val",
    )
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

    stopper = AdaptiveStopper(
        relative_threshold=cfg.get("relative_threshold", 0.15),
        min_steps=cfg.get("min_steps", 5),
        patience=cfg.get("patience", 3),
        window_size=cfg.get("window_size", 3),
    )

    methods = ["ddim_100", "ddim_50", "ddim_25", "adaptive"]
    results = {m: {"ssim": [], "psnr": [], "nmse": [], "steps": [], "time_s": []} for m in methods}

    n_batches    = min(eval_batches, len(val_loader))
    slice_idx    = 0
    images_saved = {m: 0 for m in methods}

    print(f"\nEvaluating {n_batches} batches...")

    for i, (undersampled, target, mask, observed_kspace) in enumerate(tqdm(val_loader)):
        if i >= n_batches:
            break

        undersampled    = undersampled.to(device)
        target          = target.to(device)
        mask            = mask.to(device)
        observed_kspace = observed_kspace.to(device)

        batch_preds = {}

        # DDIM 100 — upper quality reference
        pred, t = ddim_sample(ddpm, undersampled, num_steps=100)
        _record(results["ddim_100"], pred, target, 100, t)
        batch_preds["ddim_100"] = (pred, [100] * pred.shape[0])

        # DDIM 50 — practical baseline
        pred, t = ddim_sample(ddpm, undersampled, num_steps=50)
        _record(results["ddim_50"], pred, target, 50, t)
        batch_preds["ddim_50"] = (pred, [50] * pred.shape[0])

        # DDIM 25 — fast baseline
        pred, t = ddim_sample(ddpm, undersampled, num_steps=25)
        _record(results["ddim_25"], pred, target, 25, t)
        batch_preds["ddim_25"] = (pred, [25] * pred.shape[0])

        # Adaptive DDIM — our contribution (max_steps=100, fair vs ddim_100)
        pred, info = adaptive_sample(
            ddpm, undersampled, mask, observed_kspace, stopper, max_steps=100
        )
        _record_adaptive(results["adaptive"], pred, target, info)
        batch_preds["adaptive"] = (pred, info["steps_used"].tolist())

        # Save reconstruction figures
        if any(images_saved[m] < save_images for m in methods):
            B = undersampled.shape[0]
            for b in range(B):
                u_np = undersampled[b, 0].cpu().numpy()
                t_np = target[b, 0].cpu().numpy()
                for method in methods:
                    if images_saved[method] >= save_images:
                        continue
                    p_np  = batch_preds[method][0][b, 0].cpu().numpy()
                    steps = batch_preds[method][1][b]
                    m_vals = compute_metrics(p_np, t_np)
                    save_reconstruction_figure(
                        u_np, p_np, t_np, m_vals, method, steps,
                        os.path.join(recon_dir, f"slice_{slice_idx:04d}_{method}.png")
                    )
                    images_saved[method] += 1
                slice_idx += 1

    # Results table
    table_lines = ["=" * 80,
                   f"{'Method':<16} {'SSIM':>8} {'PSNR':>8} {'NMSE':>10} {'Steps':>8} {'Time(s)':>10}",
                   "=" * 80]
    for method in methods:
        r = results[method]
        table_lines.append(
            f"{method:<16} {np.mean(r['ssim']):>8.4f} {np.mean(r['psnr']):>8.2f} "
            f"{np.mean(r['nmse']):>10.6f} {np.mean(r['steps']):>8.1f} "
            f"{np.mean(r['time_s']):>10.3f}"
        )
    table_lines.append("=" * 80)

    if results["adaptive"]["steps"]:
        adap_steps = np.mean(results["adaptive"]["steps"])
        adap_time  = np.mean(results["adaptive"]["time_s"])
        base_time  = np.mean(results["ddim_100"]["time_s"])
        table_lines += [
            f"\nAdaptive vs DDIM-100:",
            f"  Step reduction : {(1 - adap_steps/100)*100:.1f}%  ({adap_steps:.1f} vs 100)",
            f"  Time reduction : {(1 - adap_time/base_time)*100:.1f}%  ({adap_time:.3f}s vs {base_time:.3f}s)",
        ]

    table_str = "\n".join(table_lines)
    print("\n" + table_str)
    with open(os.path.join(eval_dir, "metrics_table.txt"), "w") as f:
        f.write(table_str)

    # Save results.json
    results_json = {
        method: {
            "ssim_mean": float(np.mean(r["ssim"])),
            "psnr_mean": float(np.mean(r["psnr"])),
            "nmse_mean": float(np.mean(r["nmse"])),
            "ssim_std":  float(np.std(r["ssim"])),
            "psnr_std":  float(np.std(r["psnr"])),
            "steps_mean": float(np.mean(r["steps"])),
            "time_mean":  float(np.mean(r["time_s"])),
            "n_samples":  len(r["ssim"]),
        }
        for method, r in results.items()
    }
    results_json["_meta"] = {
        "checkpoint": checkpoint_path, "epoch": trained_epoch,
        "acceleration": cfg["acceleration"], "mask_type": cfg["mask_type"],
        "eval_batches": n_batches,
    }
    with open(os.path.join(eval_dir, "results.json"), "w") as f:
        json.dump(results_json, f, indent=2)

    _plot_metrics_comparison(results, eval_dir)
    _plot_quality_vs_steps(results, eval_dir)
    _plot_step_distribution(results["adaptive"]["steps"], eval_dir, 100)
    _plot_boxplots(results, eval_dir)
    _plot_per_sample_scatter(results, eval_dir)
    _save_per_sample_data(results, eval_dir)

    print(f"\nAll outputs saved to: {eval_dir}/")
    return results


def _record(d, pred, target, steps, t):
    for p, tgt in zip(pred.squeeze(1).cpu().numpy(), target.squeeze(1).cpu().numpy()):
        m = compute_metrics(p, tgt)
        d["ssim"].append(m["ssim"]); d["psnr"].append(m["psnr"]); d["nmse"].append(m["nmse"])
    d["steps"].extend([steps] * pred.shape[0])
    d["time_s"].append(t)


def _record_adaptive(d, pred, target, info):
    for p, tgt in zip(pred.squeeze(1).cpu().numpy(), target.squeeze(1).cpu().numpy()):
        m = compute_metrics(p, tgt)
        d["ssim"].append(m["ssim"]); d["psnr"].append(m["psnr"]); d["nmse"].append(m["nmse"])
    d["steps"].extend(info["steps_used"].tolist())
    d["time_s"].append(info["wall_time_s"])


def _plot_metrics_comparison(results, out_dir):
    methods = list(results.keys())
    colors  = ["#555555", "#f0a500", "#ff7043", "#2171b5"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, metric, ylabel, title in [
        (axes[0], "ssim", "SSIM",     "SSIM (higher is better)"),
        (axes[1], "psnr", "PSNR (dB)","PSNR (higher is better)"),
    ]:
        vals = [np.mean(results[m][metric]) for m in methods]
        bars = ax.bar(methods, vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_ylabel(ylabel); ax.set_title(title)
        if metric == "ssim": ax.set_ylim(0, 1)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticklabels(methods, rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "metrics_comparison.png"), dpi=150); plt.close()


def _plot_quality_vs_steps(results, out_dir):
    colors  = {"ddim_100": "#555555", "ddim_50": "#f0a500", "ddim_25": "#ff7043", "adaptive": "#d62728"}
    markers = {"ddim_100": "^",       "ddim_50": "D",       "ddim_25": "s",       "adaptive": "*"}
    sizes   = {"ddim_100": 120,       "ddim_50": 120,       "ddim_25": 120,       "adaptive": 300}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for metric, ax, ylabel in [("ssim", axes[0], "SSIM"), ("psnr", axes[1], "PSNR (dB)")]:
        for method, r in results.items():
            if r["steps"] and r[metric]:
                ax.scatter(np.mean(r["steps"]), np.mean(r[metric]),
                           color=colors[method], marker=markers[method],
                           s=sizes[method], zorder=5, label=method,
                           edgecolors="black", linewidth=0.5)
                ax.annotate(method, (np.mean(r["steps"]), np.mean(r[metric])),
                            textcoords="offset points", xytext=(8, 4), fontsize=9)
        ax.set_xlabel("Average steps used"); ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs. compute — Pareto frontier")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "quality_vs_steps.png"), dpi=150); plt.close()


def _plot_step_distribution(steps, out_dir, max_steps):
    if not steps: return
    plt.figure(figsize=(8, 4))
    plt.hist(steps, bins=max_steps, edgecolor="black", alpha=0.75, color="steelblue")
    plt.axvline(np.mean(steps),   color="red",   linestyle="--", linewidth=1.5,
                label=f"Mean: {np.mean(steps):.1f}")
    plt.axvline(np.median(steps), color="orange", linestyle="--", linewidth=1.5,
                label=f"Median: {np.median(steps):.1f}")
    plt.axvline(max_steps, color="green", linestyle=":", linewidth=1.5,
                label=f"DDIM-100 ({max_steps})")
    plt.xlabel("Steps used per sample"); plt.ylabel("Count")
    plt.title("Adaptive stopping: step distribution")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "step_distribution.png"), dpi=150); plt.close()


def _plot_boxplots(results, out_dir):
    methods = list(results.keys())
    colors  = ["#555555", "#f0a500", "#ff7043", "#2171b5"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric, ylabel in [(axes[0], "ssim", "SSIM"), (axes[1], "psnr", "PSNR (dB)")]:
        bp = ax.boxplot([results[m][metric] for m in methods], patch_artist=True,
                        medianprops=dict(color="black", linewidth=2))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
        ax.set_xticklabels(methods, rotation=15, ha="right")
        ax.set_ylabel(ylabel); ax.set_title(f"{ylabel} distribution")
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "metrics_boxplot.png"), dpi=150); plt.close()


def _plot_per_sample_scatter(results, out_dir):
    """
    Three plots answering Professor Rajesh and TA Nhi's questions.

    Plot 1 (Rajesh): Step distribution histogram — proves different samples
                     stop at different steps (spread = per-sample adaptation)

    Plot 2 (Nhi #1): SSIM vs steps scatter — tight band = dependable,
                     broad scatter = not dependable

    Plot 3 (Nhi #2): Lower bound curve — at X steps budget, what is the
                     minimum SSIM adaptive achieves? Shows reliability floor.
    """
    adaptive = results["adaptive"]
    steps    = np.array(adaptive["steps"], dtype=float)
    ssims    = np.array(adaptive["ssim"])
    ddim100_ssim = np.mean(results["ddim_100"]["ssim"])
    ddim50_ssim  = np.mean(results["ddim_50"]["ssim"])
    ddim25_ssim  = np.mean(results["ddim_25"]["ssim"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Adaptive stopping — dependability analysis", fontsize=13, fontweight="bold")

    # ── Plot 1: Step distribution histogram (for Professor Rajesh) ─────────
    ax = axes[0]
    ax.hist(steps, bins=30, color="#2171b5", edgecolor="black", alpha=0.75)
    ax.axvline(np.mean(steps),   color="red",    linestyle="--", linewidth=1.5,
               label=f"Mean: {np.mean(steps):.1f}")
    ax.axvline(np.median(steps), color="orange", linestyle="--", linewidth=1.5,
               label=f"Median: {np.median(steps):.1f}")
    ax.axvline(100, color="gray", linestyle=":", linewidth=1.5,
               label="Max (100 steps)")
    ax.set_xlabel("Steps used per sample", fontsize=11)
    ax.set_ylabel("Number of samples", fontsize=11)
    ax.set_title("Do different samples stop\nat different steps?", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    # annotation
    ax.text(0.05, 0.92, f"Range: {int(np.min(steps))}–{int(np.max(steps))} steps\nStd: {np.std(steps):.1f}",
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # ── Plot 2: SSIM vs steps scatter (for TA — thin vs broad) ────────────
    ax2 = axes[1]
    ax2.scatter(steps, ssims, alpha=0.5, s=30, color="#2171b5",
                edgecolors="none", label="Per-sample adaptive")
    ax2.axhline(ddim100_ssim, color="#555555", linestyle="--", linewidth=1.5,
                label=f"DDIM-100 mean ({ddim100_ssim:.3f})")
    ax2.axhline(ddim50_ssim,  color="#f0a500", linestyle="--", linewidth=1.5,
                label=f"DDIM-50 mean ({ddim50_ssim:.3f})")
    ax2.axhline(np.mean(ssims), color="#d62728", linestyle="-", linewidth=1.5,
                label=f"Adaptive mean ({np.mean(ssims):.3f})")
    # shaded band showing ± 1 std
    ax2.axhspan(np.mean(ssims) - np.std(ssims),
                np.mean(ssims) + np.std(ssims),
                alpha=0.1, color="#d62728", label=f"±1 std ({np.std(ssims):.3f})")
    ax2.set_xlabel("Steps used by adaptive", fontsize=11)
    ax2.set_ylabel("SSIM", fontsize=11)
    ax2.set_title("Is quality consistent regardless\nof steps? (tight = dependable)", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # ── Plot 3: Lower bound curve (for TA — reliability floor) ────────────
    ax3 = axes[2]

    # Sort samples by steps used
    sort_idx    = np.argsort(steps)
    sorted_steps = steps[sort_idx]
    sorted_ssims = ssims[sort_idx]

    # Compute running statistics as we include more steps in budget
    step_budgets = np.arange(int(np.min(steps)), 101, 1)
    mean_ssim_at_budget   = []
    lower_bound_at_budget = []   # 10th percentile
    n_samples_at_budget   = []

    for budget in step_budgets:
        # samples that stopped AT OR BEFORE this budget
        mask = sorted_steps <= budget
        if mask.sum() > 0:
            ssims_here = sorted_ssims[mask]
            mean_ssim_at_budget.append(np.mean(ssims_here))
            lower_bound_at_budget.append(np.percentile(ssims_here, 10))
            n_samples_at_budget.append(mask.sum())
        else:
            mean_ssim_at_budget.append(np.nan)
            lower_bound_at_budget.append(np.nan)
            n_samples_at_budget.append(0)

    mean_ssim_at_budget   = np.array(mean_ssim_at_budget)
    lower_bound_at_budget = np.array(lower_bound_at_budget)

    ax3.plot(step_budgets, mean_ssim_at_budget, color="#2171b5", linewidth=2,
             label="Mean SSIM")
    ax3.plot(step_budgets, lower_bound_at_budget, color="#d62728", linewidth=2,
             linestyle="--", label="Lower bound (10th percentile)")
    ax3.fill_between(step_budgets, lower_bound_at_budget, mean_ssim_at_budget,
                     alpha=0.15, color="#2171b5")

    # Reference lines
    ax3.axhline(ddim100_ssim, color="#555555", linestyle=":", linewidth=1.2,
                label=f"DDIM-100 ({ddim100_ssim:.3f})")
    ax3.axhline(ddim50_ssim,  color="#f0a500", linestyle=":", linewidth=1.2,
                label=f"DDIM-50 ({ddim50_ssim:.3f})")

    ax3.set_xlabel("Step budget", fontsize=11)
    ax3.set_ylabel("SSIM", fontsize=11)
    ax3.set_title("Lower bound — minimum SSIM\nat each step budget", fontsize=11, fontweight="bold")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)
    ax3.set_xlim(int(np.min(steps)) - 2, 102)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "per_sample_scatter.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: per_sample_scatter.png")


def _save_per_sample_data(results, out_dir):
    """
    Save per-sample SSIM, PSNR, NMSE, steps for all methods.
    Useful for custom analysis and answering professor/TA questions.
    """
    per_sample = {}
    for method, r in results.items():
        if method == "_meta":
            continue
        per_sample[method] = {
            "ssim":  r["ssim"],
            "psnr":  r["psnr"],
            "nmse":  r["nmse"],
            "steps": r["steps"],
        }

    # Summary stats for adaptive
    adap = results["adaptive"]
    steps = np.array(adap["steps"])
    ssims = np.array(adap["ssim"])

    per_sample["_adaptive_summary"] = {
        "steps_mean":   float(np.mean(steps)),
        "steps_median": float(np.median(steps)),
        "steps_min":    float(np.min(steps)),
        "steps_max":    float(np.max(steps)),
        "steps_std":    float(np.std(steps)),
        "ssim_mean":    float(np.mean(ssims)),
        "ssim_std":     float(np.std(ssims)),
        "ssim_min":     float(np.min(ssims)),
        "ssim_max":     float(np.max(ssims)),
        "n_stopped_early": int(np.sum(steps < 100)),
        "n_used_all":      int(np.sum(steps == 100)),
        "pct_stopped_early": float(np.mean(steps < 100) * 100),
    }

    path = os.path.join(out_dir, "per_sample_data.json")
    with open(path, "w") as f:
        json.dump(per_sample, f, indent=2)
    print(f"  Saved: per_sample_data.json")
    print(f"\n  Adaptive per-sample summary:")
    s = per_sample["_adaptive_summary"]
    print(f"    Steps: mean={s['steps_mean']:.1f}  median={s['steps_median']:.1f}  "
          f"min={s['steps_min']:.0f}  max={s['steps_max']:.0f}  std={s['steps_std']:.1f}")
    print(f"    SSIM:  mean={s['ssim_mean']:.4f}  std={s['ssim_std']:.4f}  "
          f"min={s['ssim_min']:.4f}  max={s['ssim_max']:.4f}")
    print(f"    Stopped early: {s['n_stopped_early']}/{s['n_stopped_early']+s['n_used_all']} "
          f"samples ({s['pct_stopped_early']:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   type=str, required=True)
    parser.add_argument("--config",       type=str, default="configs/r4.yaml")
    parser.add_argument("--eval_batches", type=int, default=50)
    parser.add_argument("--save_images",  type=int, default=20)
    args = parser.parse_args()
    cfg  = load_config(args.config)
    evaluate(cfg, args.checkpoint, args.eval_batches, args.save_images)
