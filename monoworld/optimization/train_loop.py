"""3DGS training loop.

The core optimization: given an initial GaussianParams and a target image,
minimize photometric loss via Adam for N iterations, periodically densifying.

Logs PSNR, LPIPS, gradient norms, Gaussian count, wall-clock time every
`log_every` iterations. The log is both the H1 (convergence) and H2
(gradient trajectory) evidence.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .rasterizer import GaussianParams, render, save_splat_ply
from .losses import compute_photometric_loss, compute_depth_loss, psnr
from .densify import densify_and_prune


# ----------------------------------------------------------------------
# Config + log records
# ----------------------------------------------------------------------

@dataclass
class TrainConfig:
    # Optimization horizon.
    iterations: int = 2000
    log_every: int = 50
    save_every: int = 500          # checkpoint every N iters
    # Loss weights.
    lambda_l1: float = 0.8
    lambda_ssim: float = 0.2
    lambda_lpips: float = 0.0      # set > 0 to enable LPIPS
    lambda_depth: float = 0.0      # set > 0 for H3 ablation (depth-reg)
    # Learning rates (3DGS paper).
    lr_means: float = 1.6e-4
    lr_scales: float = 5e-3
    lr_quats: float = 1e-3
    lr_opacities: float = 5e-2
    lr_sh_dc: float = 2.5e-3
    # LR decay on means.
    means_lr_decay_factor: float = 0.01  # final = init * this
    # Densification schedule (iterations).
    densify_from: int = 500
    densify_until: int = 1500
    densify_every: int = 100
    # Scene scale — approximately the diagonal of the input bbox.
    scene_extent: float = 5.0


@dataclass
class LogRecord:
    """Per-iteration training metric. Appended to a JSONL log."""
    iter: int = 0
    psnr: float = 0.0
    l1: float = 0.0
    ssim: float = 0.0
    total_loss: float = 0.0
    grad_norm_means: float = 0.0
    grad_norm_total: float = 0.0
    num_gaussians: int = 0
    wall_seconds: float = 0.0


# ----------------------------------------------------------------------
# Main train function
# ----------------------------------------------------------------------

def train(
    params: GaussianParams,
    target_image: np.ndarray,          # (H, W, 3) uint8
    intrinsics: np.ndarray,            # (3, 3)
    cfg: TrainConfig,
    depth_target: np.ndarray | None = None,  # (H, W) float, for depth-reg
    out_dir: Path | None = None,
    tag: str = "depth_init",
) -> tuple[GaussianParams, list[LogRecord]]:
    """Run Adam optimization on the Gaussians.

    Args:
        params: initial GaussianParams (from depth-init OR random-init).
        target_image: the single training-view image, uint8 in [0, 255].
        intrinsics: 3x3 pinhole.
        cfg: TrainConfig.
        depth_target: optional depth map for the depth-reg loss (H3).
        out_dir: save checkpoints + log here. Optional.
        tag: identifier for logs/checkpoints (e.g. 'depth_init', 'random_init').

    Returns:
        (final_params, log_records).
    """
    device = params.means.device
    H, W = target_image.shape[:2]

    target = torch.from_numpy(target_image).float().to(device) / 255.0
    if depth_target is not None:
        depth_gt = torch.from_numpy(depth_target).float().to(device)
    else:
        depth_gt = None

    # Build optimizer with per-parameter learning rates.
    optimizer = torch.optim.Adam([
        {"params": [params.means], "lr": cfg.lr_means, "name": "means"},
        {"params": [params.scales], "lr": cfg.lr_scales, "name": "scales"},
        {"params": [params.quats], "lr": cfg.lr_quats, "name": "quats"},
        {"params": [params.opacities], "lr": cfg.lr_opacities, "name": "opacities"},
        {"params": [params.sh_dc], "lr": cfg.lr_sh_dc, "name": "sh_dc"},
        {"params": [params.sh_rest], "lr": cfg.lr_sh_dc, "name": "sh_rest"},
    ], eps=1e-15)

    # Gradient-norm tracking for densification (EMA of ||grad(means)||).
    N0 = params.means.shape[0]
    grad_accum = torch.zeros(N0, device=device)
    vis_accum = torch.zeros(N0, device=device)

    records: list[LogRecord] = []
    t0 = time.time()

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / f"log_{tag}.jsonl"
        log_f = open(log_path, "w")
    else:
        log_f = None

    for it in range(1, cfg.iterations + 1):
        # ---- Decay means LR ----
        t = it / cfg.iterations
        decay = cfg.means_lr_decay_factor ** t
        for g in optimizer.param_groups:
            if g["name"] == "means":
                g["lr"] = cfg.lr_means * decay

        # ---- Forward render ----
        render_out = render(
            params,
            intrinsics=intrinsics,
            width=W, height=H,
            c2w=torch.eye(4, device=device, dtype=torch.float32),
            return_depth=(cfg.lambda_depth > 0),
        )
        rendered = render_out["rgb"]

        # ---- Loss ----
        loss_dict = compute_photometric_loss(
            rendered, target,
            lambda_l1=cfg.lambda_l1,
            lambda_ssim=cfg.lambda_ssim,
            lambda_lpips=cfg.lambda_lpips,
        )
        total = loss_dict["total"]
        if cfg.lambda_depth > 0 and depth_gt is not None and "depth" in render_out:
            d_loss = compute_depth_loss(render_out["depth"], depth_gt)
            total = total + cfg.lambda_depth * d_loss
            loss_dict["depth"] = d_loss

        # ---- Backward ----
        optimizer.zero_grad(set_to_none=True)
        total.backward()

        # ---- Accumulate positional-gradient stats for densification ----
        with torch.no_grad():
            grad = params.means.grad
            if grad is not None and params.means.shape[0] == grad_accum.shape[0]:
                grad_norm = grad.norm(dim=-1)
                grad_accum += grad_norm
                vis_accum += 1.0

        # ---- Step ----
        optimizer.step()

        # ---- Log ----
        if it % cfg.log_every == 0 or it == 1 or it == cfg.iterations:
            with torch.no_grad():
                p = psnr(rendered, target)
                gnorm_means = float(
                    params.means.grad.norm().item() if params.means.grad is not None else 0
                )
                gnorm_total = sum(
                    float(pt.grad.norm().item())
                    for pt in [params.means, params.scales, params.quats,
                               params.opacities, params.sh_dc]
                    if pt.grad is not None
                )
            rec = LogRecord(
                iter=it,
                psnr=p,
                l1=float(loss_dict["l1"].item()),
                ssim=float(loss_dict["ssim"].item()),
                total_loss=float(total.item()),
                grad_norm_means=gnorm_means,
                grad_norm_total=gnorm_total,
                num_gaussians=int(params.means.shape[0]),
                wall_seconds=time.time() - t0,
            )
            records.append(rec)
            print(
                f"[iter {it:4d}/{cfg.iterations}] "
                f"psnr={p:5.2f} "
                f"loss={rec.total_loss:.4f} "
                f"N={rec.num_gaussians:6d} "
                f"grad_means={gnorm_means:.2e} "
                f"t={rec.wall_seconds:6.1f}s"
            )
            if log_f is not None:
                log_f.write(json.dumps(asdict(rec)) + "\n")
                log_f.flush()

        # ---- Densify ----
        if (
            cfg.densify_from <= it <= cfg.densify_until
            and it % cfg.densify_every == 0
        ):
            params = densify_and_prune(
                params, optimizer,
                positional_grad_accum=grad_accum,
                visibility_count=vis_accum,
                scene_extent=cfg.scene_extent,
            )
            # Reset accumulators to match new param size.
            N = params.means.shape[0]
            grad_accum = torch.zeros(N, device=device)
            vis_accum = torch.zeros(N, device=device)

        # ---- Checkpoint ----
        if out_dir is not None and it % cfg.save_every == 0:
            save_splat_ply(params, out_dir / f"splat_{tag}_iter{it:05d}.ply")

    if out_dir is not None:
        save_splat_ply(params, out_dir / f"splat_{tag}_final.ply")
    if log_f is not None:
        log_f.close()

    return params, records
