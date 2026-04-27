# Adaptive Inference Scheduling for Diffusion-Based MRI Reconstruction

## Setup (NYU HPC)
```bash
singularity shell --nv --overlay /scratch/ks8413/pytorch-env/my_pytorch2.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
source /ext3/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /scratch/ks8413/mri_diffusion/mri_diffusion_v2
```

## Train

### R=4 (fresh run)
```bash
python train.py --config configs/r4.yaml
```

### R=8 (run in parallel once R=4 is stable)
```bash
python train.py --config configs/r8.yaml
```

### Resume
```bash
python train.py --config configs/r4.yaml \
    --resume /scratch/ks8413/mri_diffusion/runs/r4_v2/latest.pt
```

Training outputs saved to `output_dir`:
- `best.pt`            — best val loss checkpoint
- `latest.pt`          — most recent checkpoint  
- `epoch_0025.pt`      — periodic checkpoint every 25 epochs
- `training_log.csv`   — epoch, train_loss, val_loss, lr
- `loss_curve.png`     — loss plot (updated every epoch)

## Evaluate
```bash
python evaluate.py \
    --checkpoint /scratch/ks8413/mri_diffusion/runs/r4_v2/best.pt \
    --config configs/r4.yaml \
    --eval_batches 50 \
    --save_images 20
```

Evaluation outputs saved to `output_dir/eval_results/`:
- `results.json`             — all metrics (reload without re-running)
- `metrics_table.txt`        — printed results table
- `metrics_comparison.png`   — SSIM/PSNR bar chart
- `quality_vs_steps.png`     — Pareto frontier (SSIM/PSNR vs steps)
- `step_distribution.png`    — histogram of adaptive steps used
- `metrics_boxplot.png`      — box plots per method
- `reconstructions/`         — per-slice figures:
                               [undersampled | prediction | ground truth | error map]

## Methods compared
| Method       | Steps | DC | Notes                          |
|--------------|-------|----|--------------------------------|
| ddpm_1000    | 1000  | No | Upper-bound quality reference  |
| ddim_250     | 250   | No | Strong fixed baseline          |
| ddim_50      | 50    | No | Practical fixed baseline       |
| ddim_50_dc   | 50    | Yes| Fair comparison vs adaptive    |
| adaptive     | ≤50   | Yes| **Our contribution**           |
