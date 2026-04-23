"""Optimize a 3DGS scene — the Path A systems contribution entrypoint.

Usage:
    python optimize.py --scene data/outputs/<scene_id>/ --init depth
    python optimize.py --scene data/outputs/<scene_id>/ --init random
    python optimize.py --scene data/outputs/<scene_id>/ --init depth --depth-reg

The --scene directory must contain at least:
    - metadata.json           (intrinsics, image path)
    - splat.ply               (init produced by run.py, for --init depth)
    - depth_raw.npy           (for --depth-reg)
    - the original input image (path from metadata.json)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from monoworld.optimization.rasterizer import (
    load_splat_ply,
    random_init_gaussians,
)
from monoworld.optimization.train_loop import TrainConfig, train
from monoworld.utils import get_logger, load_image


log = get_logger("monoworld.optimize")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="3DGS optimization.")
    p.add_argument("--scene", required=True, type=str,
                   help="Path to data/outputs/<scene_id>/ directory.")
    p.add_argument("--init", choices=["depth", "random"], default="depth",
                   help="Initialization strategy.")
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--depth-reg", action="store_true",
                   help="Enable depth-regularization loss (H3 ablation).")
    p.add_argument("--lambda-depth", type=float, default=0.1)
    p.add_argument("--lambda-lpips", type=float, default=0.0)
    p.add_argument("--device", default="auto",
                   choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--n-random", type=int, default=200_000,
                   help="Number of points for --init random.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", type=str, default=None,
                   help="Override auto-tag (used for output filenames).")
    return p.parse_args()


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    log.info("Device: %s", device)
    if device == "cpu":
        log.warning(
            "CPU optimization is extremely slow. For real experiments use a "
            "CUDA GPU — a single 2000-iter run will take hours on CPU."
        )
    if device == "mps":
        log.warning(
            "MPS will use the PyTorch fallback rasterizer (no gsplat on Mac). "
            "Results are correct but much slower than CUDA. "
            "For headline experiments, use a cloud CUDA GPU."
        )

    scene_dir = Path(args.scene)
    meta_path = scene_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {scene_dir}")
    with open(meta_path) as f:
        meta = json.load(f)

    # ---- Load target image ----
    image_path = Path(meta["input_image"])
    if not image_path.exists():
        # Try relative to repo root.
        image_path = Path.cwd() / meta["input_image"]
    target_image = load_image(image_path)
    H, W = target_image.shape[:2]
    log.info("Target image: %dx%d from %s", W, H, image_path)

    # ---- Build intrinsics matrix ----
    K = meta["intrinsics"]
    intrinsics = np.array(
        [[K["fx"], 0.0, K["cx"]],
         [0.0, K["fy"], K["cy"]],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    # ---- Initialize Gaussians ----
    if args.init == "depth":
        splat_path = scene_dir / "splat.ply"
        if not splat_path.exists():
            raise FileNotFoundError(
                f"{splat_path} not found. Run `python run.py` first."
            )
        log.info("Loading depth-init splats from %s", splat_path)
        params = load_splat_ply(splat_path, device=device)
        # Estimate scene extent from the existing splats.
        xyz = params.means.detach().cpu().numpy()
        scene_extent = float(np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0)))
    else:
        # Match depth-init scene bounds (both should occupy the same volume).
        splat_path = scene_dir / "splat.ply"
        if splat_path.exists():
            init_params = load_splat_ply(splat_path, device="cpu")
            xyz = init_params.means.detach().cpu().numpy()
            bmin, bmax = xyz.min(axis=0), xyz.max(axis=0)
        else:
            # No depth-init available. Use camera frustum estimate.
            log.warning("No splat.ply found. Using heuristic bbox.")
            bmin = np.array([-5.0, -5.0, 0.5], dtype=np.float32)
            bmax = np.array([5.0, 5.0, 20.0], dtype=np.float32)
        log.info(
            "Random-init: %d points in bbox [%s .. %s]",
            args.n_random,
            bmin.round(2).tolist(),
            bmax.round(2).tolist(),
        )
        params = random_init_gaussians(
            args.n_random, bmin, bmax, device=device, seed=args.seed
        )
        scene_extent = float(np.linalg.norm(bmax - bmin))

    log.info(
        "Initial state: %d Gaussians, scene_extent=%.2f",
        params.means.shape[0], scene_extent,
    )

    # ---- Optional depth target for depth-reg ----
    depth_target = None
    if args.depth_reg:
        depth_path = scene_dir / "depth_raw.npy"
        if depth_path.exists():
            depth_target = np.load(depth_path)
            log.info("Depth-reg enabled: using %s", depth_path)
        else:
            log.warning(
                "--depth-reg set but depth_raw.npy not found. Disabling."
            )

    # ---- Train ----
    cfg = TrainConfig(
        iterations=args.iters,
        log_every=args.log_every,
        save_every=args.save_every,
        lambda_lpips=args.lambda_lpips,
        lambda_depth=args.lambda_depth if args.depth_reg else 0.0,
        scene_extent=scene_extent,
    )

    tag = args.tag or (
        f"{args.init}{'_depthreg' if args.depth_reg else ''}"
    )
    out_dir = scene_dir / "optimization" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Writing logs + checkpoints to %s", out_dir)
    final_params, records = train(
        params=params,
        target_image=target_image,
        intrinsics=intrinsics,
        cfg=cfg,
        depth_target=depth_target,
        out_dir=out_dir,
        tag=tag,
    )

    # ---- Summary ----
    final_psnr = records[-1].psnr if records else 0.0
    total_time = records[-1].wall_seconds if records else 0.0
    log.info(
        "Done. Final PSNR=%.2f, Gaussians=%d, total time=%.1fs",
        final_psnr, final_params.means.shape[0], total_time,
    )

    # Update scene's metadata.json with the optimization result.
    meta.setdefault("optimization", {})[tag] = {
        "iterations": cfg.iterations,
        "final_psnr": final_psnr,
        "final_num_gaussians": int(final_params.means.shape[0]),
        "wall_seconds": total_time,
        "log_file": str(out_dir / f"log_{tag}.jsonl"),
        "final_ply": str(out_dir / f"splat_{tag}_final.ply"),
        "depth_reg": args.depth_reg,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
